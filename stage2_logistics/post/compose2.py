#!/usr/bin/env python3
"""Compose the stage-2 video of an edition from its EDL: stage-2 shots (rendered PNG frames) + ("video" mode only)
segments of the finished stage-1 video (cut frame-accurately, not re-rendered), overlays, logos, synthesized soundtrack.

    python3 stage2_logistics/post/compose2.py --edition ru|en --final             the deliverables of the edition:
            frames out/frames_<ed> -> deliverables/demo_full_cycle_<ed>_<brand>_1080p.mp4 + _compact.mp4 (+ --verify
            of the stage-1 splices when the edition has them)
    python3 stage2_logistics/post/compose2.py --edition ru|en --raw               the edition's pre-montage cut: the same
            frames in the same order, no overlays and no soundtrack -> deliverables/..._1080p_raw.mp4 (+ --verify)
    python3 stage2_logistics/post/compose2.py --edition en --preview              out/preview_<ed> -> out/preview_<ed>.mp4
    python3 stage2_logistics/post/compose2.py [--edition ru|en] --frames <dir with frame_NNNN.png, SCENE numbering>
            --out <mp4> [--edl post/edl_<ed>.json] [--storyboard post/storyboard2_<ed>.json] [--stage1 <stage-1 mp4>]
            [--preview] [--no-audio] [--no-overlays] [--compact <mp4>] [--keep-temp] [--verify] [--threads N] [-v]

--edition (default ru, editions.py) gives the defaults of --frames (frames_dir, --preview: preview_dir), --edl,
--storyboard, --out (video, --preview: preview_video); explicit flags win.  A storyboard made for another edition or
another EDL is refused.  Both editions are in "render" mode: only stage-2 segments (the stage-1 shots are rendered
from the stage-2 scene), so no stage-1 mp4 is needed and there are no splices to verify.  The "video" mode (editions.py
stage1="video") splices the stage-1 video instead.

Pipeline (one ffmpeg run, no intermediate encodes):
  * edl_<ed>.json segments in output order.  Consecutive stage-2 segments form a block; each block is one ffconcat list
    with one entry per OUTPUT frame -> the PNG of that scene frame.  A scene frame that was not rendered holds the
    nearest rendered frame <= it inside the same shot, or the first one after it (sparse preview renders, e.g.
    every 6th frame at 640x360, work; a final render should have them all - missing ones are reported).
    Block: setpts=N (exact 24 fps) -> scale 1920x1080 lanczos -> yuv444p -> overlays -> yuv420p.
  * stage-1 segment: the stage-1 mp4 decoded and cut with trim=start_frame=f0-1:end_frame=f1 (frame indices, so the
    cut is exact), setpts=N; its frames are passed through untouched (no block overlays: the stage-1 title,
    captions and corner label are burned in; only the corner logo goes over them, after the concat).
  * concat of all parts -> the "whole" overlays (the corner logo: over the stage-2 blocks AND the stage-1 segments;
    overlay in yuv420 on the concatenated stream, so only the logo pixels of the stage-1 frames change)
    -> libx264 crf 18 (medium; veryfast with --preview) yuv420p +faststart, AAC 192 kbit/s
    from audio2.py; --compact adds a second output from the same graph (crf 25, preset slow, AAC 128 kbit/s, like
    the stage-1 compact deliverable).
  * overlays (overlays2.render_all2): title + title logo / captions (render mode: also the translated stage-1
    captions over the stage-1 shots) / end card with its logo, with the stage-1 fades; one corner label per stage-2
    block (hard cut at the splices, where the stage-1 label continues; render mode: one over the whole video).
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
import editions                                    # noqa: E402  (stage2_logistics/editions.py)
import overlays2                                   # noqa: E402
import storyboard2                                 # noqa: E402

EDL_JSON = Path(editions.get()["edl_json"])        # default edition; per edition: editions.get(ed)["edl_json"]
SB2_JSON = Path(editions.get()["storyboard_json"])
OUT_W, OUT_H = 1920, 1080
FRAME_RE = re.compile(r"frame_(\d+)\.png$", re.IGNORECASE)
MAIN_CRF, COMPACT_CRF = 18, 25


def resolve(p) -> Optional[Path]:
    """A relative input path is taken from the current directory, else from stage2_logistics/ (so the documented
    ``--edl post/edl_ru.json`` works from the repository root too)."""
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


def edl_edition(E: dict) -> str:
    return E.get("edition") or editions.DEFAULT


def load_storyboard(path, E: dict) -> dict:
    """The storyboard of the EDL's edition (``path`` None -> its storyboard_json; relative -> resolve()): it must be
    made for this edition and this EDL and pass storyboard2.check() (brand, logos, timing: a storyboard written
    before the editions has no logos and would give a video without the brand).  The edition's default file
    missing -> built in memory from the EDL."""
    default = Path(editions.get(edl_edition(E))["storyboard_json"])
    path = resolve(path) if path else default
    if path.is_file():
        sb = load_json(path)
        ed_sb, ed_e = sb.get("edition", editions.DEFAULT), edl_edition(E)
        regen = f"regenerate: python3 {S2_ROOT / 'edl.py'} --edition {ed_e}"
        if ed_sb != ed_e:
            raise SystemExit(f"{path} is the storyboard of the {ed_sb!r} edition, the EDL is {ed_e!r} "
                             f"(use --edition {ed_e} or --storyboard {default})")
        if sb.get("edl_segments") != storyboard2.edl_signature(E):
            raise SystemExit(f"{path} was generated for a different EDL - {regen}")
        problems = storyboard2.check(sb, E)
        if problems:
            raise SystemExit(f"{path}: " + "; ".join(problems) + f" - {regen}")
        return sb
    if path.resolve() != default.resolve():
        raise SystemExit(f"storyboard not found: {path}")
    print(f"{path.name} not found - building it in memory from the EDL")
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


def add_overlays(g: Graph, cur: str, ovs: Sequence[overlays2.Overlay], block: dict, fps: int, tag: str,
                 fmt_: str = "yuv444") -> str:
    """Chain the overlays on the block stream ``cur`` (block-local time, frame 0 = block out0); returns the label.
    ``fmt_``: the overlay filter's working format (yuv444 in the stage-2 blocks; yuv420 on the concatenated video,
    whose stage-1 frames are yuv420p and must stay untouched outside the overlay)."""
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
        g.chains.append(f"[{cur}][{lab}]overlay=x={ov.x}:y={ov.y}:format={fmt_}:eof_action=pass"
                        f":enable='between(t,{fmt(a - q)},{fmt(b - q)})'[{nxt}]")
        cur = nxt
    return cur


def assign_overlays(ovs: Sequence[overlays2.Overlay], parts: List[dict]) -> Dict[int, list]:
    """Block overlays -> the stage-2 block containing them (the "whole" overlays are drawn after the concat)."""
    out: Dict[int, list] = {k: [] for k, p in enumerate(parts) if p["src"] == "s2"}
    for ov in ovs:
        if getattr(ov, "whole", False):
            continue
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


def compose2(frames_dir, out_path, edl_path=None, storyboard=None, stage1=None, audio: bool = True,
             preview: bool = False, compact=None, keep_temp: bool = False, threads: int = 0, verbose: bool = False,
             soundtrack: Optional[str] = None, allow_missing: bool = False, edition: Optional[str] = None,
             overlays: bool = True) -> str:
    """Compose the video; returns the output path.  ``edl_path`` None -> the EDL of ``edition`` (None: default
    edition); ``storyboard`` None -> the storyboard of the EDL's edition.  ``soundtrack``: this WAV instead of
    synthesizing.  ``overlays`` False: no title, captions, logos, corner label or end card (the pre-montage cut; the
    stage-1 segments of ru keep their burned-in stage-1 texts)."""
    edl_path = resolve(edl_path) if edl_path else Path(editions.get(edition)["edl_json"])
    frames_dir, storyboard = resolve(frames_dir), resolve(storyboard)
    stage1, soundtrack = resolve(stage1), resolve(soundtrack)
    E = load_json(edl_path)
    if edition and edl_edition(E) != editions.get(edition)["name"]:
        raise SystemExit(f"{edl_path} is the EDL of the {edl_edition(E)!r} edition, not {edition!r}")
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
        ovs = overlays2.render_all2(sb, tmp / "overlays") if overlays else []
        by_block = assign_overlays(ovs, parts)
        whole = [ov for ov in ovs if getattr(ov, "whole", False)]
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
        g.chains.append(f"{''.join(labels)}concat=n={len(parts)}:v=1:a=0[vcat0]")
        # whole-video overlays (corner logo): output time, frame 0 = output frame 1; yuv420 keeps the stage-1
        # frames bit-identical outside the logo
        vcat = add_overlays(g, "vcat0", whole, {"out0": 1}, fps, "w", fmt_="yuv420") if whole else "vcat0"
        g.chains.append(f"[{vcat}]" + ("split=2[vout][vcmp]" if compact else "null[vout]"))

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


def mask_boxes(sb: dict, pad: int = 8) -> List[Tuple[int, int, int, int]]:
    """Pixel boxes drawn over the stage-1 video segments (the corner logo), padded for the chroma subsampling: the
    splice check compares the rest of the frame."""
    boxes = overlays2.logo_boxes(sb)
    return [(max(0, x0 - pad), max(0, y0 - pad), min(OUT_W, x1 + pad), min(OUT_H, y1 + pad))
            for name, (x0, y0, x1, y1) in boxes.items() if name == "corner"]


def verify_splices(out_mp4, edl_path=EDL_JSON, stage1=None, min_psnr: float = 30.0,
                   exclude: Sequence[Tuple[int, int, int, int]] = ()) -> List[dict]:
    """Compare the first and last output frame of every stage-1 segment with stage-1 frames f0/f1 and their
    neighbours: the matching frame must reach ``min_psnr`` and beat both neighbours (frame-accurate cut).  Pixels in
    the ``exclude`` boxes (x0, y0, x1, y1) - the corner logo drawn over the stage-1 video - are not compared.
    An EDL without stage-1 video segments (render mode) gives []."""
    E = load_json(resolve(edl_path))
    if not any(s["src"] == "s1" for s in E["segments"]):
        return []
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
    if exclude:
        import numpy as np
        keep = np.ones((OUT_H, OUT_W), bool)
        for x0, y0, x1, y1 in exclude:
            keep[y0:y1, x0:x1] = False
        fo = {k: v[keep] for k, v in fo.items()}
        fs = {k: v[keep] for k, v in fs.items()}
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
    editions.add_argument(ap)
    ap.add_argument("--final", action="store_true", help="the edition's deliverables: --out <video> --compact <compact>"
                    " (+ --verify when the edition splices the stage-1 video)")
    ap.add_argument("--raw", action="store_true", help="the edition's pre-montage cut: --out <raw>, --no-overlays"
                    " --no-audio (+ --verify when the edition splices the stage-1 video)")
    ap.add_argument("--frames", default=None, help="folder with stage-2 frame_NNNN.png (scene frame numbers); default "
                    "the edition's frames_dir (--preview: preview_dir)")
    ap.add_argument("--out", default=None, help="output mp4 (default: the edition's video; --preview: preview_video)")
    ap.add_argument("--edl", default=None, help="EDL (default: the edition's post/edl_<ed>.json)")
    ap.add_argument("--storyboard", default=None, help="storyboard (default: the edition's post/storyboard2_<ed>.json)")
    ap.add_argument("--stage1", default=None, help="stage-1 mp4 (default: the EDL's stage1_video; video mode only)")
    ap.add_argument("--preview", action="store_true", help="fast x264 preset; sparse / low-res frames expected")
    ap.add_argument("--no-audio", action="store_true", help="no soundtrack")
    ap.add_argument("--no-overlays", action="store_true", help="no title, captions, logos, corner label, end card")
    ap.add_argument("--compact", default=None, metavar="MP4", help="also write a compact encode (crf 25, slow, AAC 128k)")
    ap.add_argument("--keep-temp", action="store_true", help="keep overlays / ffconcat lists / wav in <out>_tmp/")
    ap.add_argument("--verify", action="store_true", help="check the stage-1 splices (PSNR) after encoding")
    ap.add_argument("--soundtrack", default=None, metavar="WAV", help="use this WAV instead of synthesizing")
    ap.add_argument("--threads", type=int, default=0, help="x264 threads (0 = auto)")
    ap.add_argument("--allow-missing", action="store_true", help="final encode even if some stage-2 frames are missing (held)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    ed = editions.get(a.edition)
    if a.final + a.preview + a.raw > 1:
        raise SystemExit("--final, --raw and --preview exclude each other")
    edl_path = resolve(a.edl) if a.edl else Path(ed["edl_json"])
    E = load_json(edl_path)
    if edl_edition(E) != ed["name"]:
        raise SystemExit(f"{edl_path} is the EDL of the {edl_edition(E)!r} edition; pass --edition {edl_edition(E)}")
    splices = any(s["src"] == "s1" for s in E["segments"])
    storyboard = resolve(a.storyboard)            # once: the encode and the --verify step read the same file
    frames = a.frames or (ed["preview_dir"] if a.preview else ed["frames_dir"])
    out = a.out or (ed["preview_video"] if a.preview else ed["raw"] if a.raw else ed["video"])
    compact = a.compact or (ed["compact"] if a.final else None)
    verify = a.verify or ((a.final or a.raw) and splices)
    overlays, audio = not (a.no_overlays or a.raw), not (a.no_audio or a.raw)
    print(f"edition {ed['name']}: {ed['label']}\n  frames {frames}\n  edl {edl_path}\n  out {out}"
          + (f"\n  compact {compact}" if compact else "")
          + ("" if overlays else "\n  no overlays") + ("" if audio else "\n  no soundtrack"))
    out = compose2(frames, out, edl_path, storyboard, a.stage1, audio=audio, preview=a.preview,
                   compact=compact, keep_temp=a.keep_temp, threads=a.threads, verbose=a.verbose,
                   soundtrack=a.soundtrack, allow_missing=a.allow_missing, overlays=overlays)
    print(out)
    if compact:
        print(compact)
    if verify:
        if not splices:
            print(f"verify: the {ed['name']} edition has no stage-1 video splices (the stage-1 shots are rendered from "
                  f"the stage-2 scene) - nothing to check")
            return
        sb = load_storyboard(storyboard, E)
        bad = 0
        for r in verify_splices(out, edl_path, a.stage1, exclude=mask_boxes(sb) if overlays else ()):
            nb = ", ".join(f"{d:+d}: {v:.1f}" for d, v in r["neighbours"].items())
            print(f"  out {r['out']:5d} = stage-1 {r['s1']:4d}: PSNR {r['psnr']:.1f} dB (neighbours {nb})"
                  f" {'ok' if r['ok'] else 'FAIL'}")
            bad += not r["ok"]
        if bad:
            raise SystemExit(f"{bad} splice check(s) failed")


if __name__ == "__main__":
    main()
