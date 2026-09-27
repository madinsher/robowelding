#!/usr/bin/env python3
"""Compose the stage-2 video from the EDL: new stage-2 shots (rendered PNG frames) + segments of the finished stage-1
video (cut frame-accurately, not re-rendered), overlays over the stage-2 parts, synthesized soundtrack.

    python3 stage2_logistics/post/compose2.py --frames <dir with frame_NNNN.png, SCENE numbering> --out <mp4>
            [--edl post/edl.json] [--storyboard post/storyboard2.json] [--stage1 <stage-1 mp4>]
            [--preview] [--no-audio] [--compact <mp4>] [--keep-temp] [--verify] [--threads N] [-v]

Pipeline (one ffmpeg run, no intermediate encodes):
  * edl.json segments in output order.  Consecutive stage-2 segments form a block; each block is one ffconcat list
    with one entry per OUTPUT frame -> the PNG of that scene frame.  A scene frame that was not rendered holds the
    nearest rendered frame <= it inside the same shot, or the first one after it (sparse preview renders, e.g.
    every 6th frame at 640x360, work; a final render should have them all - missing ones are reported).
    Block: setpts=N (exact 24 fps) -> scale 1920x1080 lanczos -> yuv444p -> overlays -> yuv420p.
  * stage-1 segment: the stage-1 mp4 decoded and cut with trim=start_frame=f0-1:end_frame=f1 (frame indices, so the
    cut is exact), setpts=N; its frames are passed through untouched (no overlays: the stage-1 title, captions and
    corner label are burned in).
  * concat of all parts -> libx264 crf 18 (medium; veryfast with --preview) yuv420p +faststart, AAC 192 kbit/s
    from audio2.py; --compact adds a second output from the same graph (crf 25, preset slow, AAC 128 kbit/s, like
    the stage-1 compact deliverable).
  * overlays (overlays2.render_all2): title / captions / end card with the stage-1 fades, one corner label per
    stage-2 block (hard cut at the splices, where the stage-1 label continues).
Runs on Linux and Windows: only python (numpy, PIL) and ffmpeg/ffprobe on PATH; temp files in the system temp dir
(tempfile), no symlinks.
"""
import argparse
import bisect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent             # stage2_logistics/post
S2_ROOT = HERE.parent
DEMO = S2_ROOT.parent / "demo_video"
for _p in (str(DEMO), str(S2_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import overlays2                                   # noqa: E402
import storyboard2                                 # noqa: E402

EDL_JSON = HERE / "edl.json"
SB2_JSON = HERE / "storyboard2.json"
OUT_W, OUT_H = 1920, 1080
FRAME_RE = re.compile(r"frame_(\d+)\.png$", re.IGNORECASE)
MAIN_CRF, COMPACT_CRF = 18, 25


def resolve(p) -> Optional[Path]:
    """A relative input path is taken from the current directory, else from stage2_logistics/ (so the documented
    ``--edl post/edl.json`` works from the repository root too)."""
    if p is None:
        return None
    q = Path(p)
    if q.is_absolute() or q.exists():
        return q
    alt = S2_ROOT / q
    return alt if alt.exists() else q


def tool(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise SystemExit(f"{name} not found on PATH (install ffmpeg and add its bin folder to PATH)")
    return p


# ----------------------------------------------------------------------------- inputs
def load_json(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def list_frames(frames_dir) -> Dict[int, Path]:
    """{scene frame number: path} for every frame_NNNN.png in the folder."""
    d = Path(frames_dir)
    if not d.is_dir():
        raise SystemExit(f"frames folder not found: {d}")
    out = {}
    for p in d.iterdir():
        m = FRAME_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    if not out:
        raise SystemExit(f"no frame_NNNN.png files in {d}")
    return out


def hold_sources(seg: dict, available: Sequence[int]) -> Tuple[List[int], int]:
    """Scene frame shown at each output frame of the stage-2 segment ``seg`` (f0..f1) and the number of scene frames
    that were not rendered: nearest rendered frame <= f inside the shot, else the first rendered one after f."""
    lo = bisect.bisect_left(available, seg["f0"])
    hi = bisect.bisect_right(available, seg["f1"])
    inside = list(available[lo:hi])
    if not inside:
        raise SystemExit(f"shot {seg.get('shot')}: no rendered frames in scene frames {seg['f0']}..{seg['f1']}")
    src = []
    for f in range(seg["f0"], seg["f1"] + 1):
        i = bisect.bisect_right(inside, f) - 1
        src.append(inside[i] if i >= 0 else inside[0])
    missing = (seg["f1"] - seg["f0"] + 1) - len(inside)
    return src, missing


def timeline(E: dict) -> List[dict]:
    """Parts in output order: {"src": "s2", "out0", "out1", "segments": [...]} blocks of consecutive stage-2 segments
    and {"src": "s1", "f0", "f1", "out0", "out1"} stage-1 segments.  Checks that the EDL tiles 1..frames."""
    parts: List[dict] = []
    expect = 1
    for s in E["segments"]:
        if s["out0"] != expect or s["out1"] - s["out0"] != s["f1"] - s["f0"]:
            raise SystemExit(f"EDL segment {s} does not continue the timeline at output frame {expect}")
        expect = s["out1"] + 1
        if s["src"] == "s2":
            if parts and parts[-1]["src"] == "s2":
                parts[-1]["out1"] = s["out1"]
                parts[-1]["segments"].append(s)
            else:
                parts.append({"src": "s2", "out0": s["out0"], "out1": s["out1"], "segments": [s]})
        elif s["src"] == "s1":
            parts.append(dict(s))
        else:
            raise SystemExit(f"unknown segment source {s['src']!r}")
    if expect - 1 != E["frames"]:
        raise SystemExit(f"EDL segments end at {expect - 1}, but frames = {E['frames']}")
    return parts


def probe_video(path) -> dict:
    r = subprocess.run([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(path)],
                       capture_output=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"ffprobe failed on {path}: {r.stderr.strip()[-500:]}")
    j = json.loads(r.stdout)
    st = j["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    nb = int(st.get("nb_frames") or 0) or int(round(float(j["format"]["duration"]) * fps))
    return {"width": int(st["width"]), "height": int(st["height"]), "fps": fps, "frames": nb}


def load_storyboard(path, E: dict) -> dict:
    """storyboard2.json if present (must be made for this EDL), else built in memory from the EDL."""
    if path and Path(path).is_file():
        sb = load_json(path)
        if sb.get("edl_segments") != storyboard2.edl_signature(E):
            raise SystemExit(f"{path} was generated for a different EDL - run: python3 {HERE / 'storyboard2.py'}")
        return sb
    if path and Path(path) != SB2_JSON:
        raise SystemExit(f"storyboard not found: {path}")
    print("storyboard2.json not found - building it in memory from the EDL")
    return storyboard2.build(E)


def ffconcat_quote(p: Path) -> str:
    s = Path(p).resolve().as_posix()
    return "'" + s.replace("'", "'\\''") + "'"


# ----------------------------------------------------------------------------- graph
def fmt(x: float) -> str:
    return f"{x:.6f}"


class Graph:
    """ffmpeg inputs + filter_complex chains."""

    def __init__(self):
        self.inputs: List[List[str]] = []
        self.chains: List[str] = []

    def add_input(self, args: List[str]) -> int:
        self.inputs.append(args)
        return len(self.inputs) - 1

    def args(self) -> List[str]:
        out = []
        for a in self.inputs:
            out += a
        return out


def add_overlays(g: Graph, cur: str, ovs: Sequence[overlays2.Overlay], block: dict, fps: int, tag: str) -> str:
    """Chain the overlays on the block stream ``cur`` (block-local time, frame 0 = block out0); returns the label."""
    for j, ov in enumerate(ovs):
        k0 = ov.start - block["out0"]                 # first visible frame, block-local
        nfr = ov.end - ov.start + 1
        a, b = k0 / fps, (k0 + nfr) / fps
        i = g.add_input(["-loop", "1", "-framerate", str(fps), "-t", fmt((nfr + 0.5) / fps), "-i", ov.path])
        chain = [f"trim=end_frame={nfr}", "format=rgba"]      # exactly nfr frames
        if ov.opacity < 1.0:
            chain.append(f"colorchannelmixer=aa={ov.opacity:.3f}")
        chain += [f"settb=1/{fps}", f"setpts=N+{k0}"]
        fi, fo = overlays2.fades(ov)
        if fi > 0:                                    # frame-exact version of the stage-1 fades (alpha 0 -> 1)
            chain.append(f"fade=t=in:s=0:n={fi}:alpha=1")
        if fo > 0:
            chain.append(f"fade=t=out:s={nfr - fo}:n={fo}:alpha=1")
        lab = f"o{tag}_{j}"
        g.chains.append(f"[{i}:v]{','.join(chain)}[{lab}]")
        nxt = f"v{tag}_{j}"
        q = 0.25 / fps
        g.chains.append(f"[{cur}][{lab}]overlay=x={ov.x}:y={ov.y}:format=yuv444:eof_action=pass"
                        f":enable='between(t,{fmt(a - q)},{fmt(b - q)})'[{nxt}]")
        cur = nxt
    return cur


def assign_overlays(ovs: Sequence[overlays2.Overlay], parts: List[dict]) -> Dict[int, list]:
    out: Dict[int, list] = {k: [] for k, p in enumerate(parts) if p["src"] == "s2"}
    for ov in ovs:
        for k, p in enumerate(parts):
            if p["src"] == "s2" and p["out0"] <= ov.start and ov.end <= p["out1"]:
                out[k].append(ov)
                break
        else:
            raise SystemExit(f"overlay {ov.name} ({ov.start}..{ov.end}) is not inside one stage-2 block "
                             f"(overlays never go over the stage-1 segments)")
    return out


# ----------------------------------------------------------------------------- compose
def check_pngs(paths) -> List[str]:
    """Paths that are not complete, decodable PNGs (PIL verify: header, chunk CRCs, end of stream)."""
    from PIL import Image
    bad = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            bad.append(f"{p} ({type(e).__name__}: {e})")
    return bad


def compose2(frames_dir, out_path, edl_path=EDL_JSON, storyboard=SB2_JSON, stage1=None, audio: bool = True,
             preview: bool = False, compact=None, keep_temp: bool = False, threads: int = 0, verbose: bool = False,
             soundtrack: Optional[str] = None, allow_missing: bool = False) -> str:
    """Compose the stage-2 video; returns the output path.  ``soundtrack``: use this WAV instead of synthesizing."""
    frames_dir, edl_path, storyboard = resolve(frames_dir), resolve(edl_path), resolve(storyboard)
    stage1, soundtrack = resolve(stage1), resolve(soundtrack)
    E = load_json(edl_path)
    sb = load_storyboard(storyboard, E)
    fps = int(E.get("fps", 24))
    n_frames = int(E["frames"])
    duration = n_frames / fps
    parts = timeline(E)
    stage1 = Path(stage1) if stage1 else (S2_ROOT / E["stage1_video"]).resolve()
    need_s1 = max([p["f1"] for p in parts if p["src"] == "s1"] or [0])
    if need_s1:
        if not stage1.is_file():
            raise SystemExit(f"stage-1 video not found: {stage1}")
        info = probe_video(stage1)
        if (info["width"], info["height"]) != (OUT_W, OUT_H) or abs(info["fps"] - fps) > 1e-3:
            raise SystemExit(f"stage-1 video {stage1} is {info}, expected {OUT_W}x{OUT_H} at {fps} fps")
        if info["frames"] < need_s1:
            raise SystemExit(f"stage-1 video has {info['frames']} frames, the EDL needs frame {need_s1}")

    frames = list_frames(frames_dir)
    available = sorted(frames)
    sources: Dict[int, List[int]] = {}
    report = []
    for k, p in enumerate(parts):
        if p["src"] != "s2":
            continue
        src = []
        for s in p["segments"]:
            seg_src, missing = hold_sources(s, available)
            src += seg_src
            if missing:
                report.append(f"{s['shot']}: {missing}/{s['f1'] - s['f0'] + 1}")
        sources[k] = src
    if report:
        msg = "scene frames not rendered (held): " + ", ".join(report)
        if not preview and not allow_missing:
            raise SystemExit("ERROR: " + msg + "\nRender them (build2.py --render resumes) or pass --allow-missing / --preview.")
        print(("preview: " if preview else "WARNING: ") + msg)
    if not preview:
        bad = check_pngs([frames[f] for f in sorted({f for v in sources.values() for f in v})])
        if bad:
            raise SystemExit("ERROR: unreadable / truncated frame files (delete them and re-run build2.py --render):\n  "
                             + "\n  ".join(bad[:30]) + ("" if len(bad) <= 30 else f"\n  ... {len(bad)} in total"))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        Path(compact).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="post2_compose_"))
    t0 = time.time()
    try:
        ovs = overlays2.render_all2(sb, tmp / "overlays")
        by_block = assign_overlays(ovs, parts)
        g = Graph()
        labels = []
        for k, p in enumerate(parts):
            if p["src"] == "s2":
                lst = tmp / f"block{k}.ffconcat"
                lines = ["ffconcat version 1.0"]
                for f in sources[k]:
                    lines += [f"file {ffconcat_quote(frames[f])}", f"duration {fmt(1 / fps)}"]
                lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
                i = g.add_input(["-f", "concat", "-safe", "0", "-i", str(lst)])
                g.chains.append(f"[{i}:v]settb=1/{fps},setpts=N,scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1,"
                                f"format=yuv444p[b{k}]")
                cur = add_overlays(g, f"b{k}", by_block[k], p, fps, str(k))
                g.chains.append(f"[{cur}]format=yuv420p[p{k}]")
            else:
                i = g.add_input(["-an", "-sn", "-i", str(stage1)])
                g.chains.append(f"[{i}:v]trim=start_frame={p['f0'] - 1}:end_frame={p['f1']},settb=1/{fps},"
                                f"setpts=N,setsar=1,format=yuv420p[p{k}]")
            labels.append(f"[p{k}]")
        cat = f"{''.join(labels)}concat=n={len(parts)}:v=1:a=0"
        g.chains.append(cat + ("[vcat];[vcat]split=2[vout][vcmp]" if compact else "[vout]"))

        a_idx = None
        if audio:
            wav = Path(soundtrack) if soundtrack else tmp / "soundtrack2.wav"
            if not soundtrack:
                import audio2                          # numpy synthesis only when needed
                ta = time.time()
                audio2.build_soundtrack2(sb, str(wav))
                if verbose:
                    print(f"soundtrack: {time.time() - ta:.1f} s")
            a_idx = g.add_input(["-i", str(wav)])

        ff = tool("ffmpeg")
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-stats" if verbose else "-nostats"] + g.args()
        cmd += ["-filter_complex", ";".join(g.chains)]
        preset = "veryfast" if preview else "medium"
        thr = ["-threads", str(threads)] if threads else []

        def output(label, crf, preset_, abr, path):
            o = ["-map", f"[{label}]"]
            if a_idx is not None:
                o += ["-map", f"{a_idx}:a", "-c:a", "aac", "-b:a", abr, "-ar", "48000", "-ac", "2"]
            o += ["-c:v", "libx264", "-preset", preset_, "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps)]
            o += thr + ["-t", fmt(duration), "-movflags", "+faststart", str(path)]
            return o

        cmd += output("vout", MAIN_CRF, preset, "192k", out_path)
        if compact:
            cmd += output("vcmp", COMPACT_CRF, "veryfast" if preview else "slow", "128k", compact)
        (tmp / "ffmpeg_cmd.txt").write_text(" ".join(f'"{c}"' if " " in c else c for c in cmd), encoding="utf-8")
        if verbose:
            print(f"ffmpeg: {len(g.inputs)} inputs, {len(g.chains)} filter chains")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None if verbose else subprocess.PIPE,
                             encoding="utf-8", errors="replace")
        if res.returncode != 0:
            if res.stderr:
                sys.stderr.write(res.stderr[-4000:])
            raise SystemExit(f"ffmpeg failed ({res.returncode}); command in {tmp / 'ffmpeg_cmd.txt'}"
                             + ("" if keep_temp else " (use --keep-temp to keep it)"))
        if verbose:
            print(f"encoded in {time.time() - t0:.1f} s")
        for path in [out_path] + ([Path(compact)] if compact else []):
            got = probe_video(path)["frames"]
            if got != n_frames:
                raise SystemExit(f"ERROR: {path} has {got} frames, the edit has {n_frames} (a broken input frame?)")
    finally:
        if keep_temp:
            keep = Path(str(out_path.with_suffix("")) + "_tmp")
            shutil.copytree(tmp, keep, dirs_exist_ok=True)
            print("temp files kept in", keep)
        shutil.rmtree(tmp, ignore_errors=True)
    return str(out_path)


# ----------------------------------------------------------------------------- verification
def extract_frames(mp4, indices: Sequence[int]):
    """{0-based frame index: HxWx3 uint8 RGB} decoded frame-accurately (select by decode order index n)."""
    import numpy as np
    idx = sorted(set(int(i) for i in indices))
    info = probe_video(mp4)
    w, h = info["width"], info["height"]
    expr = "+".join(f"eq(n\\,{i})" for i in idx)
    r = subprocess.run([tool("ffmpeg"), "-v", "error", "-i", str(mp4), "-an", "-vf", f"select='{expr}'",
                        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace")[-2000:])
    arr = np.frombuffer(r.stdout, np.uint8)
    n = arr.size // (w * h * 3)
    if n != len(idx):
        raise RuntimeError(f"extracted {n} frames from {mp4}, expected {len(idx)} ({idx[:5]}...)")
    arr = arr.reshape(n, h, w, 3)
    return {i: arr[k] for k, i in enumerate(idx)}


def psnr(a, b) -> float:
    import numpy as np
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-10 else 10 * math.log10(255.0 ** 2 / mse)


def verify_splices(out_mp4, edl_path=EDL_JSON, stage1=None, min_psnr: float = 30.0) -> List[dict]:
    """Compare the first and last output frame of every stage-1 segment with stage-1 frames f0/f1 and their
    neighbours: the matching frame must reach ``min_psnr`` and beat both neighbours (frame-accurate cut)."""
    E = load_json(resolve(edl_path))
    stage1 = resolve(stage1) if stage1 else (S2_ROOT / E["stage1_video"]).resolve()
    n1 = probe_video(stage1)["frames"]
    segs = [s for s in E["segments"] if s["src"] == "s1"]
    want_out, want_s1 = [], []
    for s in segs:
        want_out += [s["out0"] - 1, s["out1"] - 1]
        want_s1 += [f - 1 for f in (s["f0"] - 1, s["f0"], s["f0"] + 1, s["f1"] - 1, s["f1"], s["f1"] + 1)
                    if 1 <= f <= n1]
    fo = extract_frames(out_mp4, want_out)
    fs = extract_frames(stage1, want_s1)
    res = []
    for s in segs:
        for o, f in ((s["out0"], s["f0"]), (s["out1"], s["f1"])):
            p = psnr(fo[o - 1], fs[f - 1])
            nb = {d: psnr(fo[o - 1], fs[f - 1 + d]) for d in (-1, 1) if (f - 1 + d) in fs}
            ok = p >= min_psnr and all(p >= v for v in nb.values())
            res.append({"out": o, "s1": f, "psnr": p, "neighbours": nb, "ok": ok})
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder with stage-2 frame_NNNN.png (scene frame numbers)")
    ap.add_argument("--out", required=True, help="output mp4")
    ap.add_argument("--edl", default=str(EDL_JSON))
    ap.add_argument("--storyboard", default=str(SB2_JSON))
    ap.add_argument("--stage1", default=None, help="stage-1 mp4 (default: edl.json stage1_video)")
    ap.add_argument("--preview", action="store_true", help="fast x264 preset; sparse / low-res frames expected")
    ap.add_argument("--no-audio", action="store_true", help="no soundtrack")
    ap.add_argument("--compact", default=None, metavar="MP4", help="also write a compact encode (crf 25, slow, AAC 128k)")
    ap.add_argument("--keep-temp", action="store_true", help="keep overlays / ffconcat lists / wav in <out>_tmp/")
    ap.add_argument("--verify", action="store_true", help="check the stage-1 splices (PSNR) after encoding")
    ap.add_argument("--soundtrack", default=None, metavar="WAV", help="use this WAV instead of synthesizing")
    ap.add_argument("--threads", type=int, default=0, help="x264 threads (0 = auto)")
    ap.add_argument("--allow-missing", action="store_true", help="final encode even if some stage-2 frames are missing (held)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    out = compose2(a.frames, a.out, a.edl, a.storyboard, a.stage1, audio=not a.no_audio, preview=a.preview,
                   compact=a.compact, keep_temp=a.keep_temp, threads=a.threads, verbose=a.verbose,
                   soundtrack=a.soundtrack, allow_missing=a.allow_missing)
    print(out)
    if a.compact:
        print(a.compact)
    if a.verify:
        bad = 0
        for r in verify_splices(out, a.edl, a.stage1):
            nb = ", ".join(f"{d:+d}: {v:.1f}" for d, v in r["neighbours"].items())
            print(f"  out {r['out']:5d} = stage-1 {r['s1']:4d}: PSNR {r['psnr']:.1f} dB (neighbours {nb})"
                  f" {'ok' if r['ok'] else 'FAIL'}")
            bad += not r["ok"]
        if bad:
            raise SystemExit(f"{bad} splice check(s) failed")


if __name__ == "__main__":
    main()
