"""Reachability in the new layout (faceplate faces +Y at tilt +90, pipe along +X during seam-B sectors)."""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell import layout as L
from cell import robot_build as RB
arm = RB.load_arm(); TOOL = RB.tool_transform(); DEG = math.pi/180
QREF = np.array([0,0,0,0,1.2,0])

def spool_world(tilt, rot):
    zt = L.POS_TILT_AXIS_Z
    c, s = math.cos(-tilt*DEG), math.sin(-tilt*DEG)
    Rx = np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1.0]])
    c2, s2 = math.cos(rot*DEG), math.sin(rot*DEG)
    Rz = np.array([[c2,-s2,0,0],[s2,c2,0,0],[0,0,1,0],[0,0,0,1.0]])
    Tt = np.eye(4); Tt[2,3]=zt; Tf = np.eye(4); Tf[2,3]=L.POS_FACEPLATE_OFFSET; Tm = np.eye(4); Tm[2,3]=L.POS_FIXTURE_THICK
    return Tt @ Rx @ Tf @ Rz @ Tm
def seam_frame(which, T):
    if which=="A": c=np.array([0,0,L.SEAM_A_Z,1.0]); ax=np.array([0,0,1.0]); u=np.array([1.0,0,0]); v=np.array([0,1.0,0])
    else: c=np.array([*L.SEAM_B_CENTER,1.0]); ax=np.array([1.0,0,0]); u=np.array([0,1.0,0]); v=np.array([0,0,1.0])
    R=T[:3,:3]; return (T@c)[:3], R@ax, R@u, R@v
def seam_point(sf, a):
    c,ax,u,v = sf; n = math.cos(a*DEG)*u + math.sin(a*DEG)*v; return c + L.SEAM_RADIUS*n, n, np.cross(ax, n)
def top_angle(sf): return math.degrees(math.atan2(sf[3][2], sf[2][2]))
def target(p, n, t, lean, push=10.0):
    n=n/np.linalg.norm(n); t=t/np.linalg.norm(t); z = -n*math.cos(push*DEG) + t*math.sin(push*DEG)
    x = lean - np.dot(lean,z)*z; x/=np.linalg.norm(x); y=np.cross(z,x)
    T=np.eye(4); T[:3,0]=x; T[:3,1]=y; T[:3,2]=z; T[:3,3]=p; return T
def lean(p, ytrack, up=0.9):
    d = np.array([L.TRACK_X-p[0], ytrack-p[1], 0.0]); d/=np.linalg.norm(d); return d + np.array([0,0,up])
def run(targets, ytrack):
    base = RB.base_matrix(ytrack); q = np.array(L.ROBOT_Q_HOME); fails=0; Q=[]
    for T in targets:
        q, ok, err = arm.ik(T, q, base, TOOL, free_spin=True, q_ref=QREF, iters=200); fails += (not ok); Q.append(q.copy())
    Q=np.array(Q); step = np.degrees(np.abs(np.diff(Q,axis=0)).max()) if len(Q)>1 else 0
    return fails, step, np.degrees(Q[:,4]).round(0).tolist()[::8], np.degrees(Q[:,3]).round(0).tolist()[::8]
# seam A: tilt 90, torch at the top, part rotates
Ts = spool_world(90, -20); sf = seam_frame("A", Ts); print("seam A centre", sf[0].round(3), "axis", sf[1].round(3))
p = sf[0] + L.SEAM_RADIUS*np.array([0,0,1.0]); n=np.array([0,0,1.0]); t=np.array([1.0,0,0])
for yt in (0.0, 0.2, 0.4):
    print("seam A top, track y=%.1f:"%yt, run([target(p,n,t,lean(p,yt))]*3, yt))
# seam B sectors: pipe along +X at rot=0, tilt +90 (seam at y=+0.76) and tilt -90 (y=-0.76)
for tilt in (90, -90):
    Ts = spool_world(tilt, 0); sf = seam_frame("B", Ts); a_top = top_angle(sf)
    print(f"tilt {tilt}: seam B centre", sf[0].round(3), "axis", sf[1].round(3))
    for sgn in (+1, -1):
        p5,_,_ = seam_point(sf, a_top + sgn*5); side = "+Y" if p5[1] > sf[0][1] else "-Y"
        for dy in (0.0, 0.5, 0.9):
            yt = sf[0][1] + dy*(1 if side=="+Y" else -1)   # robot offset toward the sector side
            tg=[]
            for k in range(0, 97, 4):
                pp,nn,tt = seam_point(sf, a_top + sgn*(k-4)); tg.append(target(pp,nn,tt*sgn,lean(pp,yt)))
            f, st, q5, q4 = run(tg, yt)
            print(f"   sector toward {side}, track y={yt:+.2f}: fails={f} maxstep={st:.1f} q5={q5} q4={q4}")
