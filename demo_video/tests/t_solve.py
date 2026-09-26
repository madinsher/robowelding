"""Solve (and cache) the trajectory; print continuity stats. No rendering."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build as B
from cell import animation as A
S = B.build_scene(with_env=False, with_vfx=False)
Q = S["anim"]["Q"]
steps = np.degrees(np.abs(np.diff(Q, axis=0))); worst = steps.max(axis=1)
bad = np.where(worst > 6.0)[0]
print("IK fails:", len(A.IK_FAILS), "max step:", worst.max().round(2), "frames>6deg:", [(int(i+1), float(worst[i].round(1))) for i in bad][:20], flush=True)
