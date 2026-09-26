"""Synthesized soundtrack for the demo (numpy only, no samples).

Layers, timed in storyboard frames (frame numbers -> seconds at storyboard fps):
  * workshop ambience  — filtered brown+pink noise, -32 dBFS RMS, slightly decorrelated L/R;
  * servo whine        — soft swept sawtooth (200-400 Hz, few harmonics) during robot/positioner
                         moves: ``audio.servo_intervals`` (frame pairs, exact motion spans) when
                         present, otherwise the captions whose TAG is in ``servo_tags`` ±0.25 s;
  * MIG arc crackle    — white-noise bursts at a random 30-120 Hz burst rate, band-passed 1-6 kHz,
                         plus a 50/100 Hz hum, 0.3 s fades at the interval edges and a hard
                         ignition click at every arc start: ``audio.weld_intervals`` (frame pairs =
                         the real arc-on intervals of the animation) when present, otherwise the
                         captions whose TAG starts with ``weld_prefix``.
Everything is deterministic (``audio.seed``).  The result is written as a 48 kHz stereo 16-bit
WAV normalised to -3 dBFS peak.

CLI:  python3 post/audio.py --storyboard post/storyboard.json --out out/soundtrack.wav
"""
import argparse
import json
import math
import wave
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

SR = 48000
Interval = Tuple[float, float]           # (start_s, end_s)

# Defaults; storyboard["audio"] may override any of them.
AUDIO_DEFAULTS = {
    "weld_intervals": None,       # [[first_frame, last_frame], ...] arc-on spans; None -> from caption tags
    "servo_intervals": None,      # [[first_frame, last_frame], ...] move spans;   None -> from caption tags
    "servo_tags": ["ПОДГОТОВКА", "АДАПТАЦИЯ", "ПЕРЕМЕЩЕНИЕ ПО ТРЕКУ", "ПЕРЕНАЛАДКА",
                   "ИНДЕКСАЦИЯ 180°", "ЗАВЕРШЕНИЕ"],
    "servo_pad_s": 0.25,
    "weld_prefix": "ШОВ",
    "ambience_db": -32.0,
    "servo_db": -27.0,
    "weld_db": -20.0,
    "hum_db": -32.0,
    "peak_db": -3.0,
    "seed": 7,
}


# ----------------------------------------------------------------------------- helpers
def db(x: float) -> float:
    return 10.0 ** (x / 20.0)


def fft_bandpass(x: np.ndarray, lo: float, hi: float, sr: int = SR, soft: float = 0.15) -> np.ndarray:
    """Zero-phase band-pass via rFFT with raised-cosine edges (``soft`` = relative transition width)."""
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    mask = np.ones_like(f)
    if lo > 0:
        w = lo * soft
        mask *= np.clip((f - (lo - w)) / (2 * w), 0, 1)
    if hi < sr / 2:
        w = hi * soft
        mask *= 1 - np.clip((f - (hi - w)) / (2 * w), 0, 1)
    mask = 0.5 - 0.5 * np.cos(np.pi * mask)     # smooth the ramps
    return np.fft.irfft(spec * mask, n)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise by spectral shaping of white noise."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1]
    spec /= np.sqrt(f)
    y = np.fft.irfft(spec, n)
    return y / (np.std(y) + 1e-12)


def brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Leaky-integrated white noise (~1/f^2)."""
    white = rng.standard_normal(n)
    y = np.empty(n)
    acc = 0.0
    leak = 0.998
    # Vectorised leaky integrator y[i] = leak*y[i-1] + white[i], evaluated block-wise so the
    # 1/leak^k weights stay in a safe float range.
    block = 4096
    for s in range(0, n, block):
        seg = white[s:s + block]
        k = np.arange(len(seg))
        decay = leak ** k
        # y[s+j] = acc*leak^(j+1) + sum_{i<=j} white[s+i]*leak^(j-i)
        conv = np.cumsum(seg / decay) * decay
        y[s:s + len(seg)] = acc * decay * leak + conv
        acc = y[s + len(seg) - 1]
    return y / (np.std(y) + 1e-12)


def envelope(n: int, intervals: Iterable[Interval], fade_s: float) -> np.ndarray:
    """0..1 gate with linear fades of ``fade_s`` at each interval edge (fades lie inside the interval)."""
    env = np.zeros(n)
    t = np.arange(n) / SR
    for a, b in intervals:
        a, b = max(0.0, a), min(n / SR, b)
        if b <= a:
            continue
        seg = np.clip(np.minimum((t - a) / fade_s, (b - t) / fade_s), 0, 1)
        env = np.maximum(env, seg)
    return env


def rms_to(x: np.ndarray, level_db: float) -> np.ndarray:
    return x * (db(level_db) / (np.sqrt(np.mean(x ** 2)) + 1e-12))


def rms_in(x: np.ndarray, intervals: Iterable[Interval]) -> float:
    """RMS of ``x`` measured only inside the given intervals (so sparse layers get a sensible level)."""
    n = len(x)
    acc, cnt = 0.0, 0
    for a, b in intervals:
        i0, i1 = max(0, int(a * SR)), min(n, int(b * SR))
        if i1 > i0:
            acc += float(np.sum(x[i0:i1] ** 2))
            cnt += i1 - i0
    return math.sqrt(acc / cnt) if cnt else 1e-12


def soft_clip(x: np.ndarray, ceiling: float) -> np.ndarray:
    """tanh limiter: keeps peaks below ``ceiling`` (linear amplitude) without hard clipping."""
    return ceiling * np.tanh(x / ceiling)


# ----------------------------------------------------------------------------- layers
def ambience(n: int, rng: np.random.Generator, level_db: float) -> np.ndarray:
    """Low workshop rumble: brown noise < 250 Hz + pink noise < 2 kHz, slow amplitude wander.
    Returns (n, 2)."""
    out = np.zeros((n, 2))
    for ch in range(2):
        b = fft_bandpass(brown_noise(n, rng), 0, 250)
        p = fft_bandpass(pink_noise(n, rng), 40, 2000)
        mix = 1.0 * b / (np.std(b) + 1e-12) + 0.35 * p
        # slow wander (ventilation surges)
        t = np.arange(n) / SR
        wander = 1 + 0.15 * np.sin(2 * np.pi * 0.07 * t + ch) + 0.08 * np.sin(2 * np.pi * 0.23 * t)
        out[:, ch] = rms_to(mix * wander, level_db)
    # partially correlate the channels so the bed is not "too wide"
    m = out.mean(axis=1, keepdims=True)
    out = 0.6 * out + 0.4 * m
    return out * (db(level_db) / (np.sqrt(np.mean(out ** 2)) + 1e-12))


def servo(n: int, intervals: Sequence[Interval], rng: np.random.Generator, level_db: float) -> np.ndarray:
    """Soft sawtooth whine, pitch swept between 200 and 400 Hz per move (accelerate/decelerate)."""
    t = np.arange(n) / SR
    out = np.zeros(n)
    for a, b in intervals:
        a, b = max(0.0, a), min(n / SR, b)
        if b <= a:
            continue
        i0, i1 = int(a * SR), int(b * SR)
        tt = t[i0:i1] - a
        dur = b - a
        u = tt / dur
        # pitch: ramp up over the first third, hold, ramp down; small vibrato for "motor" life
        prof = np.clip(np.minimum(u / 0.3, (1 - u) / 0.3), 0, 1)
        f0 = 200 + 200 * (prof ** 0.7) + 4 * np.sin(2 * np.pi * 6.0 * tt)
        phase = 2 * np.pi * np.cumsum(f0) / SR
        # band-limited sawtooth: 5 harmonics with 1/k amplitude (soft), slight second voice detuned
        saw = sum(np.sin(k * phase) / k for k in range(1, 6))
        saw2 = sum(np.sin(k * phase * 1.004 + 0.3) / k for k in range(1, 4))
        sig = saw + 0.4 * saw2
        # amplitude follows the pitch profile (louder while moving fast), plus edge fades
        env = (0.35 + 0.65 * prof) * np.clip(np.minimum(tt / 0.3, (dur - tt) / 0.3), 0, 1)
        out[i0:i1] += sig * env
    out = fft_bandpass(out, 80, 4000)
    return out * (db(level_db) / rms_in(out, intervals))


def weld(n: int, intervals: Sequence[Interval], rng: np.random.Generator,
         level_db: float, hum_db: float) -> np.ndarray:
    """MIG short-arc crackle: noise bursts at 30-120 Hz random rate, band-passed 1-6 kHz,
    plus a mains hum (50/100 Hz) and an ignition click at each arc start."""
    t = np.arange(n) / SR
    crackle = np.zeros(n)
    for a, b in intervals:
        a, b = max(0.0, a), min(n / SR, b)
        if b <= a:
            continue
        pos = a
        while pos < b:
            rate = rng.uniform(30, 120)                         # bursts per second, re-drawn per burst
            pos += 1.0 / rate * rng.uniform(0.7, 1.3)
            i0 = int(pos * SR)
            length = int(SR * rng.uniform(0.002, 0.009))       # 2..9 ms burst
            if i0 + length >= n:
                break
            burst = rng.standard_normal(length) * np.exp(-np.linspace(0, 5, length))
            crackle[i0:i0 + length] += burst * rng.uniform(0.4, 1.0)
        # ignition click: 6 ms broadband pop + short 1.5 kHz ring
        i0 = int(a * SR)
        length = int(0.006 * SR)
        if i0 + length < n:
            crackle[i0:i0 + length] += 3.0 * rng.standard_normal(length) * np.exp(-np.linspace(0, 6, length))
        ring = int(0.05 * SR)
        if i0 + ring < n:
            tr = np.arange(ring) / SR
            crackle[i0:i0 + ring] += 1.5 * np.sin(2 * np.pi * 1500 * tr) * np.exp(-tr * 90)
    crackle = fft_bandpass(crackle, 1000, 6000)
    gate = envelope(n, intervals, 0.3)
    crackle *= gate
    # keep the click un-faded: the gate starts at 0 at the interval edge, so add the click again
    # on top with a hard onset (it is short and inside the first 50 ms).
    hum = np.zeros(n)
    for a, b in intervals:
        i0, i1 = int(max(0, a) * SR), int(min(n / SR, b) * SR)
        tt = t[i0:i1]
        hum[i0:i1] = (np.sin(2 * np.pi * 50 * tt) + 0.7 * np.sin(2 * np.pi * 100 * tt)
                      + 0.25 * np.sin(2 * np.pi * 150 * tt))
    hum *= envelope(n, intervals, 0.3)
    # level the crackle by its RMS inside the weld intervals, then tame the spiky peaks
    out = crackle * (db(level_db) / rms_in(crackle, intervals))
    out = soft_clip(out, db(level_db + 10))
    out += hum * (db(hum_db) / rms_in(hum, intervals))
    # hard-onset ignition click at every arc start (not affected by the 0.3 s fade-in): a 4 ms
    # broadband pop followed by a short damped 1.8 kHz "ping" of the contact tip.
    for a, _ in intervals:
        i0 = int(max(0, a) * SR)
        length = int(0.004 * SR)
        if i0 + length < n:
            click = rng.standard_normal(length) * np.exp(-np.linspace(0, 5, length))
            out[i0:i0 + length] += click / (np.abs(click).max() + 1e-12) * db(level_db + 8)
        ring = int(0.04 * SR)
        if i0 + ring < n:
            tr = np.arange(ring) / SR
            out[i0:i0 + ring] += np.sin(2 * np.pi * 1800 * tr) * np.exp(-tr * 120) * db(level_db + 2)
    return out


# ----------------------------------------------------------------------------- storyboard -> intervals
def frames_to_seconds(pairs: Sequence[Sequence[float]], fps: float) -> List[Interval]:
    """[[first_frame, last_frame], ...] (1-based, inclusive) -> [(start_s, end_s), ...]."""
    out = []
    for a, b in pairs:
        if b >= a:
            out.append(((float(a) - 1) / fps, float(b) / fps))
    return out


def caption_intervals(sb: dict, fps: float, pred) -> List[Interval]:
    """Seconds intervals of the captions whose ``tag`` satisfies ``pred``."""
    return frames_to_seconds([(c["start"], c["end"]) for c in sb.get("captions", []) if pred(c.get("tag", ""))],
                             fps)


def plan_intervals(sb: dict) -> Tuple[List[Interval], List[Interval]]:
    """(servo_intervals, weld_intervals) in seconds for the storyboard: explicit frame pairs from
    ``audio.servo_intervals`` / ``audio.weld_intervals`` when present, otherwise derived from the
    caption tags (servo ones padded by ``servo_pad_s`` because captions trail the motion)."""
    cfg = dict(AUDIO_DEFAULTS)
    cfg.update(sb.get("audio") or {})
    fps = float(sb.get("fps", 24))
    if cfg.get("servo_intervals"):
        servo_iv = frames_to_seconds(cfg["servo_intervals"], fps)
    else:
        pad = cfg["servo_pad_s"]
        tags = [s.upper() for s in cfg["servo_tags"]]
        servo_iv = [(a - pad, b + pad) for a, b in caption_intervals(sb, fps, lambda tag: tag.upper() in tags)]
    if cfg.get("weld_intervals"):
        weld_iv = frames_to_seconds(cfg["weld_intervals"], fps)
    else:
        weld_iv = caption_intervals(sb, fps, lambda tag: tag.upper().startswith(cfg["weld_prefix"].upper()))
    return servo_iv, weld_iv


def synthesize(sb: dict, duration_s: float = None) -> np.ndarray:
    """Build the stereo float track (n, 2) in [-1, 1] for the storyboard (peak at ``peak_db``)."""
    cfg = dict(AUDIO_DEFAULTS)
    cfg.update(sb.get("audio") or {})
    fps = float(sb.get("fps", 24))
    if duration_s is None:
        duration_s = sb["frames"] / fps
    n = int(round(duration_s * SR))
    rng = np.random.default_rng(cfg["seed"])

    servo_iv, weld_iv = plan_intervals(sb)

    mix = ambience(n, rng, cfg["ambience_db"])
    s = servo(n, servo_iv, rng, cfg["servo_db"])
    w = weld(n, weld_iv, rng, cfg["weld_db"], cfg["hum_db"])
    # servo slightly left of centre (robot side), arc centred with a touch of width
    mix[:, 0] += 1.0 * s + 0.95 * w
    mix[:, 1] += 0.8 * s + 1.0 * w
    peak = np.max(np.abs(mix)) + 1e-12
    return mix / peak * db(cfg["peak_db"])


def write_wav(path: str, data: np.ndarray, sr: int = SR) -> None:
    """Write (n, 2) float [-1, 1] as 16-bit PCM stereo WAV."""
    pcm = np.clip(data, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def build_soundtrack(storyboard: dict, out_path: str, duration_s: float = None) -> str:
    """Synthesize and write the WAV; returns the path."""
    write_wav(out_path, synthesize(storyboard, duration_s))
    return out_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Synthesize the demo soundtrack from the storyboard.")
    ap.add_argument("--storyboard", default=str(Path(__file__).with_name("storyboard.json")))
    ap.add_argument("--out", default="out/soundtrack.wav")
    ap.add_argument("--duration", type=float, default=None, help="seconds (default: frames/fps)")
    a = ap.parse_args(argv)
    sb = json.load(open(a.storyboard, encoding="utf-8"))
    build_soundtrack(sb, a.out, a.duration)


if __name__ == "__main__":
    main()
