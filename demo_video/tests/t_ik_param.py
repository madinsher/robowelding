import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell import layout as L
from cell import animation as A
from cell import robot_build as RB
arm = RB.load_arm(); TOOL = RB.tool_transform()
QREF = np.array([0,0,0,0,1.0,0])
def run(track_x, dy, toward_plus_x, rot=A.ROT_A1, seam_y=0.381):
    ytrack = seam_y + dy
    def base(y):
        c, s = math.cos(L.ROBOT_YAW), math.sin(L.ROBOT_YAW)
        return np.array([[c,-s,0,track_x],[s,c,0,y],[0,0,1,L.TRACK_TOP_Z],[0,0,0,1.0]])
    Ts = A.spool_world(90.0, rot); sf = A.seam_frame("B", Ts); a_top = A.top_angle(sf)
    p5,_,_ = A.seam_point(sf, a_top+5); sgn = 1.0 if ((p5[0] > sf[0][0]) == toward_plus_x) else -1.0
    targets=[]
    for k in range(0, 93, 4):
        p,n,t = A.seam_point(sf, a_top + sgn*k)
        lean = np.array([track_x-p[0], ytrack-p[1], 0.0]); lean/=np.linalg.norm(lean); lean += np.array([0,0,0.9])
        targets.append(A.target_frame(p, n, t*sgn, lean, 10.0))
    q = np.array(L.ROBOT_Q_HOME); fails=0; Q=[]
    for T in targets:
        q, ok, err = arm.ik(T, q, base(ytrack), TOOL, free_spin=True, q_ref=QREF, iters=200)
        fails += (not ok); Q.append(q.copy())
    Q=np.array(Q); step=np.degrees(np.abs(np.diff(Q,axis=0)).max())
    return fails, step, np.degrees(Q[:,3]).round(0).tolist()[::6], np.degrees(Q[:,4]).round(0).tolist()[::6]
for tx in (2.0, 2.3, 2.6):
    for dy in (0.0, 0.6, 1.0, 1.4):
        f, st, q4, q5 = run(tx, dy, True)
        print(f"+X side  track_x={tx} dy={dy}: fails={f} maxstep={st:.1f}  q4={q4} q5={q5}")
for tx in (2.0, 2.3, 2.6):
    for dy in (0.0, 0.6, 1.0):
        f, st, q4, q5 = run(tx, dy, False)
        print(f"-X side  track_x={tx} dy={dy}: fails={f} maxstep={st:.1f}  q4={q4} q5={q5}")
