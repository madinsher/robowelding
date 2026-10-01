"""Edit decision list of the stage-2 video: new stage-2 shots + segments of the finished stage-1 video (not
re-rendered).  The splices are continuous in scene time: the last "pre" shot ends at scene frame WELD_OFFSET + 96 and
the first "post" shot starts at WELD_OFFSET + 931, the stage-1 segments cover stage-1 frames 97..456 and 541..930
(the track-move shot 457..540 is dropped: its wide background predates the logistics zone), so the welding robot,
the positioner and the spool are in the same state on both sides of each cut.

    python3 stage2_logistics/edl.py            print the EDLs and write stage2_logistics/post/edl_<edition>.json

edl_<edition>.json (consumed by post/compose2.py, no bpy needed):
    {"fps": 24, "frames": N_out, "stage1_video": "...mp4", "weld_offset": W,
     "segments": [{"src": "s2", "shot": name, "f0": scene_f0, "f1": scene_f1, "out0": o0, "out1": o1, "caption": key},
                  {"src": "s1", "f0": 97, "f1": 456, "out0": ..., "out1": ...}, ...],
     "render_frames": [scene frames to render], "events": {name: scene frame}, "intervals": {...}}

Editions (editions.py): "ru" splices the stage-1 video segments (src "s1"); "en" renders the same stage-1 frames
from this scene with the stage-1 cameras (cameras2.S1_SHOTS) - segments {"src": "s2", "shot": "S1_...",
"f0": WELD_OFFSET + a, "f1": WELD_OFFSET + b, "caption": null, "s1": [a, b]} that cover the stage-1 frames exactly
like the video segments, so both editions have the same length and cut points.

    python3 stage2_logistics/edl.py [--edition ru|en|all]     (default all: post/edl_<ed>.json + post/storyboard2_<ed>.json)
    python3 stage2_logistics/edl.py --edition en --check-frames stage2_logistics/out/frames_en
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE1_VIDEO = os.path.join(os.path.dirname(HERE), "demo_video", "deliverables", "demo_welding_cell_1080p.mp4")
S1_SEGMENTS = [(97, 456), (541, 930)]
EDL_JSON = os.path.join(HERE, "post", "edl_ru.json")     # default edition (editions.DEFAULT); see edl_path()


def edl_path(edition=None):
    import editions
    return editions.get(edition)["edl_json"]


def stage1_shots(ev, off):
    """The stage-1 camera shots inside S1_SEGMENTS as s2 segments (English edition); they must tile S1_SEGMENTS."""
    import cameras2
    segs = []
    for shot in cameras2.S1_SHOTS:
        g0, g1 = cameras2.shot_range(shot, ev)
        for a, b in S1_SEGMENTS:
            lo, hi = max(g0 - off, a), min(g1 - off, b)
            if lo <= hi:
                segs.append(dict(src="s2", shot=shot[0], f0=lo + off, f1=hi + off, caption=None, s1=[lo, hi]))
    covered = sorted(f for s in segs for f in range(s["s1"][0], s["s1"][1] + 1))
    wanted = sorted(f for a, b in S1_SEGMENTS for f in range(a, b + 1))
    assert covered == wanted, "the stage-1 shots do not tile the stage-1 segments of the edit exactly"
    return segs


def build_edl(plan=None, edition=None):
    import cameras2
    import editions
    import plan2
    ed = editions.get(edition)
    P = plan or plan2.solve()
    ev = cameras2.events_of(P)
    off = P["weld_offset"]
    segs = []
    for shot in cameras2.PRE_SHOTS:
        f0, f1 = cameras2.shot_range(shot, ev)
        segs.append(dict(src="s2", shot=shot[0], f0=f0, f1=f1, caption=shot[9]))
    if ed["stage1"] == "video":
        for a, b in S1_SEGMENTS:
            segs.append(dict(src="s1", f0=a, f1=b))
    else:
        segs += stage1_shots(ev, off)
    for shot in cameras2.POST_SHOTS:
        f0, f1 = cameras2.shot_range(shot, ev)
        segs.append(dict(src="s2", shot=shot[0], f0=f0, f1=f1, caption=shot[9]))
    # checks: s2 shots strictly increasing in scene time; splices continuous
    last = 0
    for s in segs:
        if s["src"] != "s2":
            continue
        assert s["f1"] > s["f0"], s
        assert s["f0"] > last, f"shot {s['shot']} starts at {s['f0']} before the previous one ended ({last})"
        last = s["f1"]
    pre = [s for s in segs if s["src"] == "s2" and "s1" not in s and s["f1"] <= off + 96]
    post = [s for s in segs if s["src"] == "s2" and "s1" not in s and s["f0"] >= off + 931]
    assert pre[-1]["f1"] == off + 96, f"last pre shot must end at WELD_OFFSET+96 = {off + 96}, ends {pre[-1]['f1']}"
    assert post[0]["f0"] == off + 931, f"first post shot must start at WELD_OFFSET+931 = {off + 931}, starts {post[0]['f0']}"
    o = 1
    for s in segs:
        n = s["f1"] - s["f0"] + 1
        s["out0"], s["out1"] = o, o + n - 1
        o += n
    render = sorted({f for s in segs if s["src"] == "s2" for f in range(s["f0"], s["f1"] + 1)})
    return dict(edition=ed["name"], lang=ed["lang"], brand=ed["brand"], stage1_mode=ed["stage1"],
                fps=24, frames=o - 1, stage1_video=os.path.relpath(STAGE1_VIDEO, HERE).replace(os.sep, "/"),
                weld_offset=int(off),             # posix path: the json is the same on Windows and Linux
                n_scene_frames=int(P["n_frames"]), segments=segs, render_frames=render,
                events={k: int(v) for k, v in ev.items()},
                intervals={k: [[int(a), int(b)] for a, b in v] for k, v in P["intervals"].items()})


def scene_to_out(edl, f):
    """Output frame showing scene frame f (None if f is not in any stage-2 shot)."""
    for s in edl["segments"]:
        if s["src"] == "s2" and s["f0"] <= f <= s["f1"]:
            return s["out0"] + f - s["f0"]
    return None


def s1_to_out(edl, f):
    """Output frame of stage-1 frame f in a spliced stage-1 VIDEO segment (src "s1"; None in the English edition)."""
    for s in edl["segments"]:
        if s["src"] == "s1" and s["f0"] <= f <= s["f1"]:
            return s["out0"] + f - s["f0"]
    return None


def _stage1_parts(edl):
    """(stage-1 f0, f1, out0) of every part of the edit that shows stage-1 frames: spliced video segments ("s1") or
    re-rendered stage-1 shots (s2 segments with an "s1" range)."""
    out = []
    for s in edl["segments"]:
        if s["src"] == "s1":
            out.append((s["f0"], s["f1"], s["out0"]))
        elif s.get("s1"):
            out.append((s["s1"][0], s["s1"][1], s["out0"]))
    return out


def stage1_to_out(edl, f):
    """Output frame showing stage-1 frame f in either edition (video segment or re-rendered stage-1 shot)."""
    for a, b, o in _stage1_parts(edl):
        if a <= f <= b:
            return o + f - a
    return None


def map_stage1_intervals(edl, intervals):
    """Stage-1 frame intervals -> output frames in either edition (e.g. the stage-1 captions).  Pieces that are
    contiguous in the output (an interval running across a camera cut of the re-rendered stage-1 shots) are merged."""
    out = []
    for a, b in intervals:
        pieces = []
        for s0, s1, o in _stage1_parts(edl):
            lo, hi = max(a, s0), min(b, s1)
            if lo <= hi:
                pieces.append([o + lo - s0, o + hi - s0])
        for p in sorted(pieces):
            if out and p[0] == out[-1][1] + 1 and out[-1][2] == (a, b):
                out[-1][1] = p[1]
            else:
                out.append([p[0], p[1], (a, b)])
    return sorted([o0, o1] for o0, o1, _ in out)


def map_intervals(edl, intervals, src="s2"):
    """Clip scene (or stage-1) frame intervals to the segments of `src` and map them to output frames."""
    out = []
    for a, b in intervals:
        for s in edl["segments"]:
            if s["src"] != src:
                continue
            lo, hi = max(a, s["f0"]), min(b, s["f1"])
            if lo <= hi:
                out.append([s["out0"] + lo - s["f0"], s["out0"] + hi - s["f0"]])
    return sorted(out)


def write(edl, path=None):
    path = path or edl_path(edl.get("edition"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(edl, fh, ensure_ascii=False, indent=1)
    return path


def check_frames(edl, frames_dir):
    """Missing / empty frames of the edit in a render folder (frame_NNNN.png, scene numbering)."""
    missing, empty = [], []
    for f in edl["render_frames"]:
        p = os.path.join(frames_dir, f"frame_{f:04d}.png")
        if not os.path.exists(p):
            missing.append(f)
        elif os.path.getsize(p) < 10000:
            empty.append(f)
    return missing, empty


def _ranges(frames):
    out = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in out)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    import editions
    editions.safe_console()
    argv = sys.argv[1:]
    which = argv[argv.index("--edition") + 1] if "--edition" in argv else "all"
    eds = editions.names() if which == "all" else [editions.get(which)["name"]]
    if "--check-frames" in argv:
        d = argv[argv.index("--check-frames") + 1]
        if which == "all":
            raise SystemExit("--check-frames needs --edition ru|en")
        E = json.load(open(edl_path(eds[0]), encoding="utf-8"))
        miss, empty = check_frames(E, d)
        print(f"[{eds[0]}] {len(E['render_frames'])} frames in the edit; missing {len(miss)}: {_ranges(miss) or '-'}; "
              f"suspiciously small {len(empty)}: {_ranges(empty) or '-'}")
        sys.exit(1 if miss or empty else 0)
    import plan2
    P = plan2.solve()
    sys.path.insert(0, os.path.join(HERE, "post"))
    import storyboard2
    for name in eds:
        ed = editions.get(name)
        E = build_edl(P, name)
        print(f"== edition {name}: {ed['label']}")
        for s in E["segments"]:
            n = s["f1"] - s["f0"] + 1
            lab = s.get("shot", "stage-1 video")
            print(f"  out {s['out0']:5d}-{s['out1']:5d}  {s['src']}  {s['f0']:5d}-{s['f1']:5d}  {n / 24:5.1f} s  {lab}")
        print(f"total {E['frames']} frames = {E['frames'] / 24:.1f} s; frames to render: {len(E['render_frames'])}")
        print("written", write(E))
        print("written", storyboard2.write(storyboard2.build(E), ed["storyboard_json"]))
