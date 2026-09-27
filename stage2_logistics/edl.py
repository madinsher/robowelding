"""Edit decision list of the stage-2 video: new stage-2 shots + segments of the finished stage-1 video (not
re-rendered).  The splices are continuous in scene time: the last "pre" shot ends at scene frame WELD_OFFSET + 96 and
the first "post" shot starts at WELD_OFFSET + 931, the stage-1 segments cover stage-1 frames 97..456 and 541..930
(the track-move shot 457..540 is dropped: its wide background predates the logistics zone), so the welding robot,
the positioner and the spool are in the same state on both sides of each cut.

    python3 stage2_logistics/edl.py            print the EDL and write stage2_logistics/post/edl.json

edl.json (consumed by post/compose2.py, no bpy needed):
    {"fps": 24, "frames": N_out, "stage1_video": "...mp4", "weld_offset": W,
     "segments": [{"src": "s2", "shot": name, "f0": scene_f0, "f1": scene_f1, "out0": o0, "out1": o1, "caption": key},
                  {"src": "s1", "f0": 97, "f1": 456, "out0": ..., "out1": ...}, ...],
     "render_frames": [scene frames to render], "events": {name: scene frame}, "intervals": {...}}
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE1_VIDEO = os.path.join(os.path.dirname(HERE), "demo_video", "deliverables", "demo_welding_cell_1080p.mp4")
S1_SEGMENTS = [(97, 456), (541, 930)]
EDL_JSON = os.path.join(HERE, "post", "edl.json")


def build_edl(plan=None):
    import cameras2
    import plan2
    P = plan or plan2.solve()
    ev = P["events"]
    off = P["weld_offset"]
    segs = []
    prev_end = 0
    for shot in cameras2.PRE_SHOTS:
        f0, f1 = cameras2.shot_range(shot, ev)
        segs.append(dict(src="s2", shot=shot[0], f0=f0, f1=f1, caption=shot[9]))
    for a, b in S1_SEGMENTS:
        segs.append(dict(src="s1", f0=a, f1=b))
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
    pre = [s for s in segs if s["src"] == "s2" and s["f1"] <= off + 96]
    post = [s for s in segs if s["src"] == "s2" and s["f0"] >= off + 931]
    assert pre[-1]["f1"] == off + 96, f"last pre shot must end at WELD_OFFSET+96 = {off + 96}, ends {pre[-1]['f1']}"
    assert post[0]["f0"] == off + 931, f"first post shot must start at WELD_OFFSET+931 = {off + 931}, starts {post[0]['f0']}"
    o = 1
    for s in segs:
        n = s["f1"] - s["f0"] + 1
        s["out0"], s["out1"] = o, o + n - 1
        o += n
    render = sorted({f for s in segs if s["src"] == "s2" for f in range(s["f0"], s["f1"] + 1)})
    return dict(fps=24, frames=o - 1, stage1_video=os.path.relpath(STAGE1_VIDEO, HERE), weld_offset=int(off),
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
    for s in edl["segments"]:
        if s["src"] == "s1" and s["f0"] <= f <= s["f1"]:
            return s["out0"] + f - s["f0"]
    return None


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


def write(edl, path=EDL_JSON):
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
    if "--check-frames" in sys.argv:
        d = sys.argv[sys.argv.index("--check-frames") + 1]
        E = json.load(open(EDL_JSON, encoding="utf-8"))
        miss, empty = check_frames(E, d)
        print(f"{len(E['render_frames'])} frames in the edit; missing {len(miss)}: {_ranges(miss) or '-'}; "
              f"suspiciously small {len(empty)}: {_ranges(empty) or '-'}")
        sys.exit(1 if miss or empty else 0)
    E = build_edl()
    for s in E["segments"]:
        n = s["f1"] - s["f0"] + 1
        lab = s.get("shot", "stage-1 video")
        print(f"  out {s['out0']:5d}-{s['out1']:5d}  {s['src']}  {s['f0']:5d}-{s['f1']:5d}  {n / 24:5.1f} s  {lab}")
    print(f"total {E['frames']} frames = {E['frames'] / 24:.1f} s; stage-2 frames to render: {len(E['render_frames'])}")
    print("written", write(E))
    sys.path.insert(0, os.path.join(HERE, "post"))
    import storyboard2
    print("written", storyboard2.write(storyboard2.build(E)))
