#!/usr/bin/env python3
"""Stage-2 soundtrack for the OUTPUT timeline of the edit (numpy only, no samples, deterministic).

    python3 stage2_logistics/post/audio2.py [--storyboard post/storyboard2.json] [--out out/soundtrack2.wav] [--report]

Everything is read from storyboard2.json "audio" (output frames, written by storyboard2.py):

  stage-1 layers (demo_video/post/audio.py, reused unchanged)
    ambience        audio.ambience, whole video, -32 dBFS RMS (one continuous room tone across the splices)
    servo_intervals audio.servo, -27 dBFS: stage-1 robot/positioner moves inside the stage-1 segments (mapped with
                    edl.map_intervals(src="s1")) + the stage-1 choreography seen in stage-2 shots (return home)
    weld_intervals  audio.weld, -20 dBFS arc crackle + 50/100 Hz hum -32 dBFS, ignition clicks
  stage-2 layers ("layers": scene intervals of edl.json mapped with edl.map_intervals(src="s2"); each piece is
  [out0, out1, real_start, real_end] - real_* = 0 where the edit cuts the interval)
    servo      handler / tack robot / positioner / QC axis: audio.servo transposed by "pitch" (resampling);
               pieces cut by a picture cut on both sides are joined so a move continues across the cut
    tack_arc   short MIG tack arcs: dense crackle 1.2-7 kHz + arc sizzle, ignition pop + 2.2 kHz ping, break pop
    pneumatic  clamps / locating studs (/ gripper jaws, "soft"): valve hiss (fast attack, decaying to a sustain),
               cylinder "clack" (thump + metallic ring) + exhaust puff at the real end of each stroke
    conveyor   roller conveyor: low rumble, drive hum 100/200 Hz, roller ticks ~7/s
    marker     laser marker: galvo whine hopping 1.8-3.4 kHz at ~30 steps/s + driver buzz + extraction hiss
    scan       profilometer: soft 1.6 kHz blips every 0.5 s + sweeping whir
    agv        AGV warning beeper (1.9 kHz, 0.18 s every 0.8 s) + faint drive whine
  chime        two-tone "OK" (E6 -> A6) at the qc_ok event, ~-24 dBFS RMS over the chime
Levels ("db") are RMS inside each layer's own intervals before the final normalisation of the whole mix to
``peak_db`` (-3 dBFS), exactly like stage 1.  Output: 48 kHz stereo 16-bit WAV of frames/fps seconds.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
DEMO = HERE.parents[1] / "demo_video"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))
from post import audio                      # noqa: E402  (stage-1 module: demo_video/post/audio.py)

SR = audio.SR
Piece = Tuple[float, float, bool, bool]     # (start_s, end_s, real_start, real_end)

AUDIO2_DEFAULTS = {"ambience_db": -32.0, "servo_db": -27.0, "weld_db": -20.0, "hum_db": -32.0, "peak_db": -3.0,
                   "seed": 11, "weld_intervals": [], "servo_intervals": [], "layers": [], "chime": {}}


# ----------------------------------------------------------------------------- helpers
def pan_gains(p: float) -> Tuple[float, float]:
    """Constant-power pan normalised to (1, 1) at the centre; p in -1 (left) .. +1 (right)."""
    th = (max(-1.0, min(1.0, p)) + 1) * math.pi / 4
    return math.cos(th) * math.sqrt(2), math.sin(th) * math.sqrt(2)


def join_cuts(intervals: Sequence[Sequence[int]]) -> List[List[int]]:
    """Join a piece that ends at a picture cut with the piece that starts at the cut right after it."""
    out: List[List[int]] = []
    for iv in sorted((list(iv) + [1, 1])[:4] for iv in intervals):     # 2-element entries = real edges
        if out and out[-1][3] == 0 and iv[2] == 0 and iv[0] == out[-1][1] + 1:
            out[-1][1], out[-1][3] = iv[1], iv[3]
        else:
            out.append(iv)
    return out


def to_pieces(intervals: Sequence[Sequence[int]], fps: float, min_frames: int = 1) -> List[Piece]:
    """[[out0, out1(, real_start, real_end)], ...] (1-based inclusive frames) -> seconds pieces."""
    out = []
    for iv in intervals:
        o0, o1 = iv[0], iv[1]
        rs, re = (bool(iv[2]), bool(iv[3])) if len(iv) >= 4 else (True, True)
        if o1 - o0 + 1 >= min_frames:
            out.append(((o0 - 1) / fps, o1 / fps, rs, re))
    return out


def gate(n: int, pieces: Sequence[Piece], fade_real: float, fade_cut: float) -> np.ndarray:
    """0..1 envelope: linear fades inside each piece, ``fade_real`` at real edges, ``fade_cut`` at cut edges."""
    env = np.zeros(n)
    for a, b, rs, re in pieces:
        i0, i1 = max(0, int(round(a * SR))), min(n, int(round(b * SR)))
        if i1 <= i0:
            continue
        d = (i1 - i0) / SR
        t = np.arange(i1 - i0) / SR
        fi = min(fade_real if rs else fade_cut, d / 2)
        fo = min(fade_real if re else fade_cut, d / 2)
        seg = np.ones(i1 - i0)
        if fi > 0:
            seg = np.minimum(seg, np.clip(t / fi, 0, 1))
        if fo > 0:
            seg = np.minimum(seg, np.clip((d - t) / fo, 0, 1))
        env[i0:i1] = np.maximum(env[i0:i1], seg)
    return env


def fft_size(n: int) -> int:
    """Smallest 2^a 3^b 5^c >= n: numpy's FFT is ~20x slower on lengths with large prime factors (the layers use
    full-length FFT filters), so everything is synthesised on this length and cut to the video length at the end."""
    best = 1 << max(0, (n - 1).bit_length())
    p5 = 1
    while p5 < 2 * n:
        p35 = p5
        while p35 < 2 * n:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best


def secs(pieces: Sequence[Piece]) -> List[Tuple[float, float]]:
    return [(a, b) for a, b, _, _ in pieces]


def level(x: np.ndarray, pieces: Sequence[Piece], level_db: float) -> np.ndarray:
    """Scale ``x`` so its RMS inside the pieces is ``level_db`` dBFS."""
    return x * (audio.db(level_db) / audio.rms_in(x, secs(pieces)))


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.std(x) + 1e-12)


def _tone(t: np.ndarray, f: float, harmonics=(1.0,)) -> np.ndarray:
    return sum(h * np.sin(2 * np.pi * f * (k + 1) * t) for k, h in enumerate(harmonics))


# ----------------------------------------------------------------------------- layers (mono, float)
def servo_layer(n: int, pieces: Sequence[Piece], rng, level_db: float, pitch: float = 1.0) -> np.ndarray:
    """Stage-1 servo whine (audio.servo, 200-400 Hz sweep per move) transposed by ``pitch``: synthesised on a
    timeline stretched by ``pitch`` and read back ``pitch`` times faster (frequencies x pitch, same timing)."""
    iv = secs(pieces)
    if not iv:
        return np.zeros(n)
    if abs(pitch - 1.0) < 1e-6:
        return audio.servo(n, iv, rng, level_db)
    m = fft_size(int(math.ceil(n * pitch)) + 2)
    y = audio.servo(m, [(a * pitch, b * pitch) for a, b in iv], rng, level_db)
    out = np.interp(np.arange(n) * pitch, np.arange(m), y)
    return level(out, pieces, level_db)


def tack_arc_layer(n: int, pieces: Sequence[Piece], rng, level_db: float) -> np.ndarray:
    """Short MIG tack arcs (0.3-0.5 s): dense crackle bursts + sizzle, fast 10 ms gate, ignition pop + 2.2 kHz ping
    at real starts and a small break pop at real ends."""
    if not pieces:
        return np.zeros(n)
    crackle = np.zeros(n)
    for a, b, _, _ in pieces:
        pos, stop = a, min(n, int(b * SR))
        while True:
            pos += 1.0 / rng.uniform(60, 160) * rng.uniform(0.6, 1.4)
            i0 = int(pos * SR)
            ln = int(SR * rng.uniform(0.0015, 0.007))
            if i0 + ln >= stop:
                break
            crackle[i0:i0 + ln] += rng.standard_normal(ln) * np.exp(-np.linspace(0, 5, ln)) * rng.uniform(0.3, 1.0)
    iv = secs(pieces)
    crackle = audio.fft_bandpass(crackle, 1200, 7000)
    sizzle = audio.fft_bandpass(rng.standard_normal(n), 2500, 9000)
    x = crackle / audio.rms_in(crackle, iv) + 0.4 * sizzle / audio.rms_in(sizzle, iv)
    x *= gate(n, pieces, 0.01, 0.01)
    x = level(x, pieces, level_db)
    x = audio.soft_clip(x, audio.db(level_db + 10))
    for a, b, rs, re in pieces:
        if rs:
            i0, ln, ring = int(a * SR), int(0.003 * SR), int(0.03 * SR)
            if i0 + ring < n:
                click = rng.standard_normal(ln) * np.exp(-np.linspace(0, 5, ln))
                x[i0:i0 + ln] += click / (np.abs(click).max() + 1e-12) * audio.db(level_db + 6)
                tr = np.arange(ring) / SR
                x[i0:i0 + ring] += np.sin(2 * np.pi * 2200 * tr) * np.exp(-tr * 110) * audio.db(level_db)
        if re:
            i1, ln = int(b * SR), int(0.002 * SR)
            if 0 <= i1 - ln and i1 < n:
                pop = rng.standard_normal(ln) * np.exp(-np.linspace(0, 4, ln))
                x[i1 - ln:i1] += pop / (np.abs(pop).max() + 1e-12) * audio.db(level_db + 2)
    return x


def pneumatic_layer(n: int, pieces: Sequence[Piece], rng, level_db: float, soft: bool = False) -> np.ndarray:
    """Pneumatic stroke: valve hiss (4 ms attack, decays to a 35 % sustain over ~0.1 s; a stroke entered at a
    picture cut starts on the sustain) and at the real end of the stroke the cylinder clack (160 Hz thump + 2.3 /
    3.7 kHz metallic ring + click) followed by a short exhaust puff.  ``soft`` = gripper jaws (quieter, no ring)."""
    if not pieces:
        return np.zeros(n)
    hiss = unit(audio.fft_bandpass(rng.standard_normal(n), 2500, 11000))
    env = np.zeros(n)
    clack = np.zeros(n)
    for a, b, rs, re in pieces:
        i0, i1 = max(0, int(a * SR)), min(n, int(b * SR))
        if i1 <= i0:
            continue
        t = np.arange(i1 - i0) / SR
        d = (i1 - i0) / SR
        if rs:
            e = (0.35 + 0.65 * np.exp(-t / 0.10)) * np.clip(t / 0.004, 0, 1)
        else:
            e = 0.35 * np.clip(t / 0.02, 0, 1)
        e *= np.clip((d - t) / 0.03, 0, 1)
        env[i0:i1] = np.maximum(env[i0:i1], e * (0.5 if soft else 1.0))
        if re:
            c0 = max(0, i1 - int(0.008 * SR))
            ln = min(int(0.15 * SR), n - c0)
            tc = np.arange(ln) / SR
            s = (np.sin(2 * np.pi * 160 * tc) * np.exp(-tc / 0.025)
                 + 0.9 * rng.standard_normal(ln) * np.exp(-tc / 0.003))
            if not soft:
                s += (0.5 * np.sin(2 * np.pi * 2300 * tc) * np.exp(-tc / 0.012)
                      + 0.3 * np.sin(2 * np.pi * 3700 * tc) * np.exp(-tc / 0.008))
            clack[c0:c0 + ln] += s * (0.45 if soft else 1.0)
            # exhaust puff right after the stroke
            p0, pl = i1, min(int(0.2 * SR), n - i1)
            if pl > 0:
                env[p0:p0 + pl] = np.maximum(env[p0:p0 + pl],
                                             (0.25 if soft else 0.5) * np.exp(-np.arange(pl) / SR / 0.05))
    h = hiss * env
    x = h + clack / (np.abs(clack).max() + 1e-12) * 4.0 * audio.rms_in(h, secs(pieces))   # clack peak +12 dB
    x = level(x, pieces, level_db)
    return audio.soft_clip(x, audio.db(level_db + 14))


def conveyor_layer(n: int, pieces: Sequence[Piece], rng, level_db: float) -> np.ndarray:
    """Roller conveyor: brown-noise rumble 25-220 Hz, drive hum 100/200/300 Hz, roller ticks ~7 per second."""
    if not pieces:
        return np.zeros(n)
    t = np.arange(n) / SR
    rumble = unit(audio.fft_bandpass(audio.brown_noise(n, rng), 25, 220))
    hum = unit(_tone(t, 100, (1.0, 0.4, 0.2)))
    ticks = np.zeros(n)
    for a, b, _, _ in pieces:
        pos = a + rng.uniform(0, 0.1)
        while pos < b:
            i0, ln = int(pos * SR), int(0.004 * SR)
            if i0 + ln >= n:
                break
            ticks[i0:i0 + ln] += rng.standard_normal(ln) * np.exp(-np.linspace(0, 4, ln)) * rng.uniform(0.5, 1.0)
            pos += 1.0 / 7.0 * rng.uniform(0.85, 1.15)
    ticks = unit(audio.fft_bandpass(ticks, 800, 4000)) if ticks.any() else ticks
    x = (rumble + 0.3 * hum + 0.6 * ticks) * gate(n, pieces, 0.5, 0.03)
    return level(x, pieces, level_db)


def marker_layer(n: int, pieces: Sequence[Piece], rng, level_db: float) -> np.ndarray:
    """Laser marker: galvo whine hopping between 1.8 and 3.4 kHz (~30 steps/s, 4 ms glides) with step-wise
    amplitude, 120 Hz driver buzz and fume-extraction hiss."""
    if not pieces:
        return np.zeros(n)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for a, b, _, _ in pieces:
        i0, i1 = max(0, int(a * SR)), min(n, int(b * SR))
        if i1 <= i0:
            continue
        m = i1 - i0
        step = int(SR / 30)
        k = m // step + 2
        f_steps = rng.uniform(1800, 3400, k)
        a_steps = rng.uniform(0.55, 1.0, k)
        f = np.repeat(f_steps, step)[:m]
        amp = np.repeat(a_steps, step)[:m]
        w = int(0.004 * SR)
        kern = np.ones(w) / w
        f = np.convolve(f, kern, mode="same")
        amp = np.convolve(amp, kern, mode="same")
        ph = 2 * np.pi * np.cumsum(f) / SR
        x[i0:i1] += amp * (np.sin(ph) + 0.25 * np.sin(2 * ph))
    buzz = unit(audio.fft_bandpass(np.sign(np.sin(2 * np.pi * 120 * t)), 300, 3000))
    air = unit(audio.fft_bandpass(rng.standard_normal(n), 400, 3000))
    x = unit(x) + 0.25 * buzz + 0.3 * air
    x *= gate(n, pieces, 0.05, 0.02)
    return level(x, pieces, level_db)


def scan_layer(n: int, pieces: Sequence[Piece], rng, level_db: float) -> np.ndarray:
    """Profilometer scan: soft 1.6 kHz blips (45 ms, every 0.5 s) + a whir sweeping at 2.5 Hz."""
    if not pieces:
        return np.zeros(n)
    t = np.arange(n) / SR
    beeps = np.zeros(n)
    for a, b, _, _ in pieces:
        pos = a + 0.05
        while pos + 0.045 < b:
            i0, ln = int(pos * SR), int(0.045 * SR)
            if i0 + ln >= n:
                break
            tb = np.arange(ln) / SR
            ramp = np.clip(np.minimum(tb / 0.006, (0.045 - tb) / 0.006), 0, 1)
            beeps[i0:i0 + ln] += _tone(tb, 1600, (1.0, 0.15)) * ramp
            pos += 0.5
    whir = unit(audio.fft_bandpass(rng.standard_normal(n), 600, 1800)) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.5 * t))
    x = (unit(beeps) if beeps.any() else beeps) + 0.35 * whir
    x *= gate(n, pieces, 0.05, 0.02)
    return level(x, pieces, level_db)


def agv_layer(n: int, pieces: Sequence[Piece], rng, level_db: float) -> np.ndarray:
    """AGV warning beeper (1.9 kHz with a soft 3rd harmonic, 0.18 s on / 0.8 s period, 8 ms ramps) + faint drive
    whine; the beep rhythm runs on the output clock so it continues across a cut."""
    if not pieces:
        return np.zeros(n)
    t = np.arange(n) / SR
    ph = np.mod(t, 0.8)
    on = np.clip(np.minimum(ph / 0.008, (0.18 - ph) / 0.008), 0, 1)
    beep = _tone(t, 1900, (1.0, 0.0, 0.3)) * on
    whine = _tone(t, 700, (1.0, 0.3)) * (1 + 0.1 * np.sin(2 * np.pi * 0.7 * t))
    x = (beep + 0.12 * whine) * gate(n, pieces, 0.2, 0.02)
    return level(x, pieces, level_db)


def chime_layer(n: int, times_s: Sequence[float], level_db: float) -> np.ndarray:
    """Two-tone "OK" chime (E6 1318.5 Hz, then A6 1760 Hz 0.15 s later), bell-like decays."""
    x = np.zeros(n)
    dur = 0.65
    for t0 in times_s:
        i0 = int(t0 * SR)
        ln = min(int(dur * SR), n - i0)
        if ln <= 0:
            continue
        tt = np.arange(ln) / SR
        s = _tone(tt, 1318.5, (1.0, 0.25, 0.08)) * np.exp(-tt / 0.15) * np.clip(tt / 0.004, 0, 1)
        t2 = tt - 0.15
        s += np.where(t2 >= 0, _tone(np.maximum(t2, 0), 1760.0, (1.0, 0.25, 0.08))
                      * np.exp(-np.maximum(t2, 0) / 0.25) * np.clip(t2 / 0.004, 0, 1), 0)
        s *= np.clip((dur - tt) / 0.05, 0, 1)
        x[i0:i0 + ln] += s
    if not x.any():
        return x
    return x * (audio.db(level_db) / audio.rms_in(x, [(t0, t0 + dur) for t0 in times_s]))


LAYER_TYPES = {"servo", "tack_arc", "pneumatic", "conveyor", "marker", "scan", "agv"}


def render_layer(n: int, lay: dict, rng, fps: float) -> np.ndarray:
    kind = lay["type"]
    ivs = lay.get("intervals", [])
    lv = float(lay.get("db", -30.0))
    if kind == "servo":
        # join moves continued across a picture cut; drop slivers (< 3 frames) that would only click
        return servo_layer(n, to_pieces(join_cuts(ivs), fps, min_frames=3), rng, lv, float(lay.get("pitch", 1.0)))
    pieces = to_pieces(ivs, fps)
    if kind == "tack_arc":
        return tack_arc_layer(n, pieces, rng, lv)
    if kind == "pneumatic":
        return pneumatic_layer(n, pieces, rng, lv, soft=bool(lay.get("soft")))
    if kind == "conveyor":
        return conveyor_layer(n, to_pieces(join_cuts(ivs), fps), rng, lv)
    if kind == "marker":
        return marker_layer(n, to_pieces(join_cuts(ivs), fps), rng, lv)
    if kind == "scan":
        return scan_layer(n, to_pieces(join_cuts(ivs), fps), rng, lv)
    if kind == "agv":
        return agv_layer(n, to_pieces(join_cuts(ivs), fps), rng, lv)
    raise ValueError(f"unknown audio layer type {kind!r} (known: {sorted(LAYER_TYPES)})")


# ----------------------------------------------------------------------------- mix
def synthesize2(sb: dict, duration_s: float = None, stems_out: Dict[str, np.ndarray] = None) -> np.ndarray:
    """Stereo float track (n, 2) for the stage-2 storyboard, normalised to ``peak_db``.  If ``stems_out`` is a
    dict it receives every layer (after the same normalisation gain) plus "_gain" and "_intervals" (seconds)."""
    au = dict(AUDIO2_DEFAULTS)
    au.update(sb.get("audio") or {})
    fps = float(sb.get("fps", 24))
    if duration_s is None:
        duration_s = sb["frames"] / fps
    n_out = int(round(duration_s * SR))
    n = fft_size(n_out)                     # synthesis length (fast FFTs); cut to n_out below
    seed = int(au["seed"])

    def rng(k: int):
        return np.random.default_rng([seed, k])

    stems: Dict[str, np.ndarray] = {}
    where: Dict[str, List[Tuple[float, float]]] = {}
    stems["ambience"] = audio.ambience(n, rng(0), au["ambience_db"])
    where["ambience"] = [(0.0, n_out / SR)]
    servo_iv = audio.frames_to_seconds(au["servo_intervals"], fps)
    weld_iv = audio.frames_to_seconds(au["weld_intervals"], fps)
    if servo_iv:
        s = audio.servo(n, servo_iv, rng(1), au["servo_db"])
        stems["servo_s1"] = np.stack([1.0 * s, 0.8 * s], axis=1)     # stage-1 panning
        where["servo_s1"] = servo_iv
    if weld_iv:
        w = audio.weld(n, weld_iv, rng(2), au["weld_db"], au["hum_db"])
        stems["weld_s1"] = np.stack([0.95 * w, 1.0 * w], axis=1)
        where["weld_s1"] = weld_iv
    for k, lay in enumerate(au["layers"]):
        if not lay.get("intervals"):
            continue
        mono = render_layer(n, lay, rng(10 + k), fps)
        gl, gr = pan_gains(float(lay.get("pan", 0.0)))
        stems[lay["name"]] = np.stack([gl * mono, gr * mono], axis=1)
        where[lay["name"]] = secs(to_pieces(lay["intervals"], fps))
    ch = au.get("chime") or {}
    if ch.get("frames"):
        times = [(f - 1) / fps for f in ch["frames"]]
        c = chime_layer(n, times, float(ch.get("db", -24.0)))
        gl, gr = pan_gains(float(ch.get("pan", 0.0)))
        stems["chime"] = np.stack([gl * c, gr * c], axis=1)
        where["chime"] = [(t, t + 0.65) for t in times]

    mix = np.zeros((n_out, 2))
    for v in stems.values():
        mix += v[:n_out]
    gain = audio.db(au["peak_db"]) / (np.max(np.abs(mix)) + 1e-12)
    if stems_out is not None:
        stems_out.update({k: v[:n_out] * gain for k, v in stems.items()})
        stems_out["_gain"] = gain
        stems_out["_intervals"] = where
    return mix * gain


def layer_report(stems: Dict[str, np.ndarray]) -> List[Tuple[str, float, float]]:
    """(name, RMS dBFS inside the layer's intervals, peak dBFS) for every stem of synthesize2(stems_out=...)."""
    out = []
    for name, v in stems.items():
        if name.startswith("_"):
            continue
        iv = stems["_intervals"][name]             # RMS over both channels (pan-independent, like audio.ambience)
        r = math.sqrt((audio.rms_in(v[:, 0], iv) ** 2 + audio.rms_in(v[:, 1], iv) ** 2) / 2)
        out.append((name, 20 * math.log10(r + 1e-12), 20 * math.log10(np.max(np.abs(v)) + 1e-12)))
    return out


def build_soundtrack2(sb: dict, out_path: str, duration_s: float = None) -> str:
    audio.write_wav(out_path, synthesize2(sb, duration_s))
    return out_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Synthesize the stage-2 soundtrack from storyboard2.json.")
    ap.add_argument("--storyboard", default=str(HERE / "storyboard2.json"))
    ap.add_argument("--out", default=str(HERE.parent / "out" / "soundtrack2.wav"))
    ap.add_argument("--duration", type=float, default=None, help="seconds (default: frames/fps)")
    ap.add_argument("--report", action="store_true", help="print per-layer levels")
    a = ap.parse_args(argv)
    with open(a.storyboard, encoding="utf-8") as fh:
        sb = json.load(fh)
    stems: Dict[str, np.ndarray] = {}
    data = synthesize2(sb, a.duration, stems_out=stems if a.report else None)
    audio.write_wav(a.out, data)
    print("written", a.out, f"({len(data) / SR:.2f} s)")
    if a.report:
        print(f"normalisation gain {20 * math.log10(stems['_gain']):+.1f} dB")
        for name, rms, pk in layer_report(stems):
            print(f"  {name:<12s} rms {rms:6.1f} dBFS (in its intervals)   peak {pk:6.1f} dBFS")


if __name__ == "__main__":
    main()
