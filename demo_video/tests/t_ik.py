"""Numpy-only reachability check of the welding targets (no bpy)."""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell import layout as L
from cell.robot_urdf import URDFArm
from cell import robot_build as RB

arm = URDFArm(RB.URDF, RB.MESH_ROOT)
TOOL = RB.tool_transform()
DEG = math.pi/180

def frame_from(p, n, t, lean, push_deg=10.0, spin_deg=0.0):
    """Target TCP frame: z into the surface (-n) leaned by push angle toward travel t; x toward `lean`."""
    n = n/np.linalg.norm(n); t = t/np.linalg.norm(t)
    z = -n*math.cos(push_deg*DEG) + t*math.sin(push_deg*DEG)
    x = lean - np.dot(lean, z)*z
    x /= np.linalg.norm(x)
    # spin about z
    y = np.cross(z, x)
    c, s = math.cos(spin_deg*DEG), math.sin(spin_deg*DEG)
    x, y = c*x + s*y, -s*x + c*y
    T = np.eye(4); T[:3,0]=x; T[:3,1]=y; T[:3,2]=z; T[:3,3]=p
    return T

def spool_world(tilt_deg, rot_deg):
    """World transform of the spool root for given positioner angles."""
    zt = L.POS_TILT_AXIS_Z
    c, s = math.cos(tilt_deg*DEG), math.sin(tilt_deg*DEG)
    Ry = np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])
    c2, s2 = math.cos(rot_deg*DEG), math.sin(rot_deg*DEG)
    Rz = np.array([[c2,-s2,0,0],[s2,c2,0,0],[0,0,1,0],[0,0,0,1]])
    T = np.eye(4); T[2,3]=zt
    Tf = np.eye(4); Tf[2,3]=L.POS_FACEPLATE_OFFSET
    Tm = np.eye(4); Tm[2,3]=L.POS_FIXTURE_THICK
    return T @ Ry @ Tf @ Rz @ Tm

def seam_points(which, T_spool, angles_deg):
    if which=="A":
        c = np.array([0,0,L.SEAM_A_Z,1.0]); ax=np.array([0,0,1.0]); u=np.array([1.0,0,0]); v=np.array([0,1.0,0])
    else:
        c = np.array([*L.SEAM_B_CENTER,1.0]); ax=np.array([1.0,0,0]); u=np.array([0,1.0,0]); v=np.array([0,0,1.0])
    R = T_spool[:3,:3]; cw = (T_spool@c)[:3]; axw=R@ax; uw=R@u; vw=R@v
    out=[]
    for a in angles_deg:
        n = math.cos(a*DEG)*uw + math.sin(a*DEG)*vw
        p = cw + L.SEAM_RADIUS*n
        t = np.cross(axw, n)   # tangent (direction of increasing angle)
        out.append((p, n, t))
    return out, cw, axw

def check(label, targets, q0, base_y):
    base = RB.base_matrix(base_y)
    Q, oks, errs = arm.solve_path(targets, q0, base, TOOL, free_spin=True, q_ref=np.array([0,0,0,0,1.0,0]))
    bad = [i for i,o in enumerate(oks) if not o]
    lim = [(i,j) for i,q in enumerate(Q) for j in range(6) if abs(q[j]-arm.limits[j][0])<1e-3 or abs(q[j]-arm.limits[j][1])<1e-3]
    print(f"{label}: n={len(targets)} fails={len(bad)} at_limit={len(lim)} maxerr={max(errs):.4f}  q_first(deg)={np.degrees(Q[0]).round(1)}  q_last={np.degrees(Q[-1]).round(1)}")
    return Q

q_home = np.array(L.ROBOT_Q_HOME)
def robot_dir(p, y):
    d = np.array([L.TRACK_X - p[0], y - p[1], 0.0]); d /= np.linalg.norm(d)
    return d + np.array([0, 0, 0.9])   # toward the robot base AND upward -> wrist stays above the seam
# Seam A: tilt 90, torch fixed at 12 o'clock; part rotates -> torch target constant; travel tangent = surface motion direction
Ts = spool_world(90, 0)
pts, cw, axw = seam_points("A", Ts, [0])
print("seam A centre world", cw.round(3), "axis", axw.round(3))
# 12 o'clock point = centre + R*(0,0,1)
p = cw + L.SEAM_RADIUS*np.array([0,0,1.0]); n=np.array([0,0,1.0]); t=np.array([0,1.0,0])
TA = frame_from(p, n, t, robot_dir(p, 0.0), push_deg=10)
QA = check("seam A 12h (weave test 5 pts)", [TA]*5, q_home, 0.0)
# approach 0.12 m above
TA_up = TA.copy(); TA_up[:3,3] += np.array([0,0,0.12])
check("seam A approach", [TA_up], q_home, 0.0)
# Seam B: tilt 90, rot=+90 -> pipe along +Y; sectors on the top half
for rot, ytrack, name in ((90, 0.4, "pipe +Y"), (-90, -0.4, "pipe -Y")):
    Ts = spool_world(90, rot)
    pts, cw, axw = seam_points("B", Ts, [0])
    print(f"seam B centre world ({name})", cw.round(3), "axis", axw.round(3))
    # find angle param of the top point: n = cos a*u + sin a*v ; want n=(0,0,1)
    R = Ts[:3,:3]; uw=R@np.array([0,1.0,0]); vw=R@np.array([0,0,1.0])
    a_top = math.degrees(math.atan2(np.dot(vw,[0,0,1]), np.dot(uw,[0,0,1])))
    for sgn, sec in ((+1,"12->3 (+X side?)"), (-1,"12->9")):
        angs = [a_top + sgn*k for k in range(0, 95, 5)]
        pts,_,_ = seam_points("B", Ts, angs)
        targets=[]
        for (p,n,t) in pts:
            tt = t*sgn
            targets.append(frame_from(p, n, tt, robot_dir(p, ytrack), push_deg=10))
        side = "+X" if pts[-1][0][0] > cw[0] else "-X"
        Q = check(f"seam B {name} sector {sec} ends on {side} side", targets, q_home, ytrack)
        jmp = np.abs(np.diff(Q, axis=0)).max()
        print("   max joint step (deg):", round(math.degrees(jmp),2), " wrist q5 range:", np.degrees(Q[:,4]).min().round(1), np.degrees(Q[:,4]).max().round(1))
