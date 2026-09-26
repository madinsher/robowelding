#!/usr/bin/env python3
"""Compose the final demo mp4 from the rendered frame sequence: overlays (title, lower-third
captions, end card, corner label) + synthesized soundtrack.

    python3 post/compose.py --frames out/frames --out out/demo_final.mp4 [--storyboard post/storyboard.json]
                            [--no-audio] [--preview] [--keep-temp]

Input frames: ``frame_%04d.png`` numbered in storyboard frame space (1..1008 at 24 fps).  The
sequence may be sparse (e.g. every 8th frame from a preview render) or an excerpt (e.g. frames
930..1008): timing is derived from the frame numbers, so overlays and audio always line up with
the storyboard.  Output: 1920x1080 H.264 (libx264, yuv420p, crf 18, faststart) + AAC, 24 fps.

Pipeline (ffmpeg only, overlays are pre-rendered RGBA PNGs from ``overlays.py``):
    main = scale(1920x1080) -> fps=24
    for each overlay: loop PNG -> fade(alpha in/out) -> overlay(enable='between(t,a,b)')
    -> format yuv420p -> libx264;  audio = soundtrack.wav trimmed to the excerpt -> aac
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # demo_video/ on the path
from post import overlays as ov_mod          # noqa: E402
from post import audio as audio_mod          # noqa: E402

FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
OUT_W, OUT_H = 1920, 1080
FRAME_RE = re.compile(r"frame_(\d+)\.png$")


# ----------------------------------------------------------------------------- frames
def list_frames(frames_dir: Path) -> List[Tuple[int, Path]]:
    """Sorted (frame_number, path) for every frame_NNNN.png in the folder."""
    found = []
    for p in frames_dir.iterdir():
        m = FRAME_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort()
    if not found:
        raise SystemExit(f"no frame_NNNN.png files in {frames_dir}")
    return found


def sequence_info(frames: List[Tuple[int, Path]], fps: float):
    """Return (step, first, last, regular): step = spacing between frame numbers."""
    nums = [n for n, _ in frames]
    diffs = sorted(set(b - a for a, b in zip(nums, nums[1:]))) or [1]
    step = diffs[0]
    regular = len(diffs) == 1
    return step, nums[0], nums[-1], regular


def stage_sequence(frames: List[Tuple[int, Path]], tmp: Path, fps: float):
    """Symlink the frames into ``tmp/seq/seq_%05d.png`` as a contiguous sequence and return
    (input_fps, n_frames).  Regular spacing -> the sequence runs at fps/step (a 3 fps sequence
    for every-8th-frame previews); irregular spacing -> each frame is held (duplicated) at the
    full frame rate so timing still matches the storyboard."""
    seq = tmp / "seq"
    seq.mkdir()
    step, first, last, regular = sequence_info(frames, fps)
    if regular:
        for k, (_, p) in enumerate(frames):
            os.symlink(p.resolve(), seq / f"seq_{k:05d}.png")
        return fps / step, len(frames)
    # hold-mode: frame n shows the last rendered frame with number <= n
    k = 0
    idx = 0
    for n in range(first, last + 1):
        while idx + 1 < len(frames) and frames[idx + 1][0] <= n:
            idx += 1
        os.symlink(frames[idx][1].resolve(), seq / f"seq_{k:05d}.png")
        k += 1
    return fps, k


# ----------------------------------------------------------------------------- ffmpeg graph
def fmt(x: float) -> str:
    return f"{x:.4f}"


def build_filter(ovs: List[ov_mod.Overlay], t_offset: float, duration: float, fps: float,
                 out_fps: float) -> Tuple[str, List[ov_mod.Overlay]]:
    """Build the filter_complex string; returns it and the overlays actually used (input order)."""
    parts = [f"[0:v]scale={OUT_W}:{OUT_H}:flags=lanczos,fps={out_fps},format=yuv444p,setsar=1[base]"]
    used = []
    cur = "base"
    for ov in ovs:
        a = (ov.start - 1) / fps - t_offset          # first visible second (output timeline)
        b = ov.end / fps - t_offset                  # end of the last visible frame
        if b <= 0 or a >= duration:
            continue                                 # not in this excerpt
        used.append(ov)
        i = len(used)                                # ffmpeg input index (0 = frames)
        fade = ov.fade / fps
        chain = ["format=rgba"]
        if ov.opacity < 1.0:
            chain.append(f"colorchannelmixer=aa={ov.opacity:.3f}")
        if fade > 0:
            chain.append(f"fade=t=in:st={fmt(max(a, 0))}:d={fmt(fade)}:alpha=1")
            chain.append(f"fade=t=out:st={fmt(b - fade)}:d={fmt(fade)}:alpha=1")
        parts.append(f"[{i}:v]{','.join(chain)}[o{i}]")
        parts.append(f"[{cur}][o{i}]overlay=x={ov.x}:y={ov.y}:format=yuv444:eof_action=pass"
                     f":enable='between(t,{fmt(max(a, 0))},{fmt(b)})'[v{i}]")
        cur = f"v{i}"
    parts.append(f"[{cur}]format=yuv420p[vout]")
    return ";".join(parts), used


def run(cmd: List[str], verbose: bool) -> None:
    if verbose:
        print(" ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-4000:])
        raise SystemExit(f"ffmpeg failed ({res.returncode})")


# ----------------------------------------------------------------------------- main
def compose(frames_dir: str, out_path: str, storyboard: str, audio: bool = True,
            preview: bool = False, keep_temp: bool = False, verbose: bool = False) -> str:
    """Compose the video; returns the output path."""
    sb = json.load(open(storyboard, encoding="utf-8"))
    fps = float(sb.get("fps", 24))
    frames = list_frames(Path(frames_dir))
    step, first, last, regular = sequence_info(frames, fps)
    t_offset = (first - 1) / fps                                   # excerpt start on the storyboard timeline
    duration = (last - first + step) / fps if regular else (last - first + 1) / fps
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="post_compose_"))
    try:
        in_fps, n_in = stage_sequence(frames, tmp_root, fps)
        ovs = ov_mod.render_all(sb, tmp_root / "overlays")
        graph, used = build_filter(ovs, t_offset, duration, fps, fps)

        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-stats" if verbose else "-nostats",
               "-framerate", fmt(in_fps), "-start_number", "0",
               "-i", str(tmp_root / "seq" / "seq_%05d.png")]
        for ov in used:
            cmd += ["-loop", "1", "-framerate", fmt(fps), "-t", fmt(duration), "-i", ov.path]
        wav = None
        if audio:
            wav = tmp_root / "soundtrack.wav"
            audio_mod.build_soundtrack(sb, str(wav))
            cmd += ["-ss", fmt(t_offset), "-t", fmt(duration), "-i", str(wav)]
        cmd += ["-filter_complex", graph, "-map", "[vout]"]
        if audio:
            cmd += ["-map", f"{len(used) + 1}:a", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast" if preview else "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-r", fmt(fps), "-t", fmt(duration),
                "-movflags", "+faststart", out_path]
        run(cmd, verbose)
        if keep_temp:
            keep = Path(out_path).with_suffix("").as_posix() + "_tmp"
            shutil.copytree(tmp_root, keep, dirs_exist_ok=True, ignore=shutil.ignore_patterns("seq"))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return out_path


def main(argv=None) -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default=str(here.parent / "out" / "frames"), help="folder with frame_NNNN.png")
    ap.add_argument("--out", default=str(here.parent / "out" / "demo_final.mp4"))
    ap.add_argument("--storyboard", default=str(here / "storyboard.json"))
    ap.add_argument("--no-audio", action="store_true", help="skip the synthesized soundtrack")
    ap.add_argument("--preview", action="store_true",
                    help="fast x264 preset; frames may be sparse (e.g. every 8th) and/or low-res")
    ap.add_argument("--keep-temp", action="store_true", help="keep overlays/wav next to the output (<out>_tmp/)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    out = compose(a.frames, a.out, a.storyboard, audio=not a.no_audio, preview=a.preview,
                  keep_temp=a.keep_temp, verbose=a.verbose)
    print(out)


if __name__ == "__main__":
    main()
