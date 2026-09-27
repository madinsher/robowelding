"""Kinematics helpers for stage 2 (pure numpy, no bpy): scaled URDF arms, pose/frame helpers, IK wrappers.

The handler and the tack robot reuse the IRB 4600 URDF of stage 1, scaled uniformly (joint origins and link meshes):
    handler  x1.25  -> reach ~3.2 m, proportions of the IRB 6700-150/3.20 class
    tack     x0.57  -> reach ~1.45 m, IRB 1600-class arc-welding robot
A uniform scale keeps the kinematics exact (same joint axes, scaled lengths), so the stage-1 FK/IK code applies as is.
"""
import math
import numpy as np

import tools  # noqa: F401  (puts demo_video/ on sys.path)
from cell import robot_build as RB
from cell.robot_urdf import URDFArm  # noqa: F401

DEG = math.pi / 180.0


# ------------------------------------------------------------------ elementary transforms
def tr(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def rotx(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[1:3, 1:3] = [[c, -s], [s, c]]
    return T


def roty(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
    return T


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def frame(x_axis, z_axis, origin):
    """Right-handed frame from an x direction and a z direction (x is orthogonalised against z)."""
    z = np.asarray(z_axis, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_axis, float)
    x = x - np.dot(x, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, origin
    return T


def planar_frame(origin, x_dir_xy):
    """Frame with Z up and X along the horizontal direction x_dir_xy (a spool 'standing' pose)."""
    x = np.array([x_dir_xy[0], x_dir_xy[1], 0.0])
    return frame(x, (0, 0, 1), origin)


def inv(T):
    Ti = np.eye(4)
    R = T[:3, :3]
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def rot_angle(Ra, Rb):
    """Angle (rad) of the relative rotation Ra^T Rb."""
    c = (np.trace(Ra[:3, :3].T @ Rb[:3, :3]) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


# ------------------------------------------------------------------ quaternion blend (Cartesian moves)
def mat2quat(R):
    m = R
    tr_ = m[0, 0] + m[1, 1] + m[2, 2]
    if tr_ > 0:
        S = math.sqrt(tr_ + 1.0) * 2
        q = [0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S]
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, s):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def blend_pose(T0, T1, s):
    T = np.eye(4)
    T[:3, :3] = quat2mat(slerp(mat2quat(T0[:3, :3]), mat2quat(T1[:3, :3]), s))
    T[:3, 3] = T0[:3, 3] * (1 - s) + T1[:3, 3] * s
    return T


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def smootherstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


# ------------------------------------------------------------------ scaled URDF arm
def scaled_arm(scale):
    """IRB 4600 URDF chain with every joint origin and link visual origin scaled by `scale` (meshes are scaled by the
    Blender builder with the same factor).  Returns a URDFArm instance (FK/IK as in stage 1)."""
    arm = RB.load_arm()
    for j in arm.chain:
        j["origin"] = j["origin"].copy()
        j["origin"][:3, 3] *= scale
    for v in arm.links.values():
        v["vis_origin"] = v["vis_origin"].copy()
        v["vis_origin"][:3, 3] *= scale
    arm.scale = scale
    return arm


def ik6(arm, target, q0, base, tool, seeds=(), iters=300, tol_ok=2e-3):
    """Full 6-DoF damped-LS IK (orientation fully constrained).  Tries q0 first, then each seed; returns
    (q, ok, err).  Joint limits are clamped inside URDFArm.ik."""
    best = None
    for qs in [q0] + list(seeds):
        q, ok, err = arm.ik(target, np.asarray(qs, float), base, tool, iters=iters)
        if best is None or err < best[2]:
            best = (q, ok, err)
        if ok and err < tol_ok:
            break
    return best


def unwrap_to(q, q_ref):
    """Shift the unlimited wrist joints (4 and 6, range +-400 deg) by 2*pi so they stay close to q_ref."""
    q = np.array(q, float)
    for j in (3, 5):
        while q[j] - q_ref[j] > math.pi:
            q[j] -= 2 * math.pi
        while q[j] - q_ref[j] < -math.pi:
            q[j] += 2 * math.pi
    return q
