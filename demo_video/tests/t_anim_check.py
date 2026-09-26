"""Build the full scene (no env/vfx) and report IK failures and joint continuity."""
import sys, os, numpy as np, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build as B
from cell import animation as A
S = B.build_scene(with_env=False, with_vfx=False)
Q = S["anim"]["Q"]
steps = np.degrees(np.abs(np.diff(Q, axis=0)))
worst = steps.max(axis=1)
bad = np.where(worst > 6.0)[0]
print("IK fails:", len(A.IK_FAILS), A.IK_FAILS[:8])
print("max joint step per frame (deg):", worst.max().round(2), "frames with step > 6 deg:", [(int(i+1), float(worst[i].round(1))) for i in bad][:20])
print("q ranges (deg):", np.degrees(Q.min(axis=0)).round(0), np.degrees(Q.max(axis=0)).round(0))
tl = S["anim"]["timeline"]
for k in ("weld_A","sector1","sector2","sector3","sector4"):
    a,b = tl[k]; print(k, "max step", worst[a-1:b-1].max().round(2))
B.setup_render(S["scene"], "CYCLES", (960, 540), samples=16)
import bpy
for f in (60, 210, 300, 420, 570, 660, 730, 800, 930):
    S["scene"].frame_set(f)
    S["scene"].render.filepath = f"/home/user/robowelding/demo_video/out/anim_f{f:04d}.png"
    bpy.ops.render.render(write_still=True)
print("rendered")
