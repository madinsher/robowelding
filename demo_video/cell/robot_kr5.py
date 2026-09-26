"""KUKA KR 5 arc as a standard-DH chain with the rtb-data link meshes.

Frames follow standard DH (Craig-style "distal" frames, as in the Robotics
Toolbox KR5 model): frame j is attached at the far end of link j, and the
visual mesh link{j}.stl is expressed in frame j.  A_j = Rz(q_j) Tz(d_j) Tx(a_j) Rx(alpha_j).
"""
import math
import numpy as np

DEG = math.pi / 180.0

# a, d, alpha, (qmin, qmax)   -- rtb KR5 (ARTE) parameters, metres / rad
DH = [
    (0.18, 0.40, -math.pi / 2, (-155 * DEG, 155 * DEG)),
    (0.60, 0.00, 0.0, (-180 * DEG, 65 * DEG)),
    (0.12, 0.00, math.pi / 2, (-15 * DEG, 158 * DEG)),
    (0.00, -0.62, -math.pi / 2, (-350 * DEG, 350 * DEG)),
    (0.00, 0.00, math.pi / 2, (-130 * DEG, 130 * DEG)),
    (0.00, -0.115, math.pi, (-350 * DEG, 350 * DEG)),
]


def rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def tr(x=0.0, y=0.0, z=0.0):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def fixed_part(j):
    """Constant part of A_j (applied after the joint rotation)."""
    a, d, al, _ = DH[j]
    return tr(z=d) @ tr(x=a) @ rx(al)


def fk_all(q, base=np.eye(4), tool=np.eye(4)):
    """Return list of 4x4 frames [T0 (base), T1..T6, T_tcp]."""
    T = base.copy()
    out = [T.copy()]
    for j in range(6):
        T = T @ rz(q[j]) @ fixed_part(j)
        out.append(T.copy())
    out.append(T @ tool)
    return out


def fk(q, base=np.eye(4), tool=np.eye(4)):
    return fk_all(q, base, tool)[-1]


def _rot_err(Rd, R):
    """Small-angle rotation error vector taking R to Rd."""
    E = Rd @ R.T
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    c = (np.trace(E) - 1) / 2
    c = max(-1.0, min(1.0, c))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    return v / (2 * math.sin(ang)) * ang


def jacobian(q, base, tool, eps=1e-6):
    """Geometric Jacobian by finite differences (6x6): [dp; dtheta] in base frame."""
    T0 = fk(q, base, tool)
    J = np.zeros((6, 6))
    for j in range(6):
        dq = np.array(q, dtype=float)
        dq[j] += eps
        T1 = fk(dq, base, tool)
        J[:3, j] = (T1[:3, 3] - T0[:3, 3]) / eps
        J[3:, j] = _rot_err(T1[:3, :3], T0[:3, :3]) / eps
    return J


def ik(target, q0, base=np.eye(4), tool=np.eye(4), iters=120, tol=1e-5,
       lam=0.05, w_rot=1.0, clamp=True):
    """Damped least-squares IK for the full 6-DoF pose. Returns (q, ok, err)."""
    q = np.array(q0, dtype=float)
    err_n = 1e9
    for _ in range(iters):
        T = fk(q, base, tool)
        e = np.concatenate([target[:3, 3] - T[:3, 3],
                            w_rot * _rot_err(target[:3, :3], T[:3, :3])])
        err_n = np.linalg.norm(e)
        if err_n < tol:
            break
        J = jacobian(q, base, tool)
        J[3:, :] *= w_rot
        dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), e)
        step = np.linalg.norm(dq)
        if step > 0.35:
            dq *= 0.35 / step
        q = q + dq
        if clamp:
            for j in range(6):
                lo, hi = DH[j][3]
                q[j] = min(max(q[j], lo), hi)
    return q, err_n < 1e-3, err_n


def solve_path(targets, q0, base=np.eye(4), tool=np.eye(4)):
    """Solve a sequence of poses with warm-starting; returns (Q, list_ok)."""
    Q, oks = [], []
    q = np.array(q0, dtype=float)
    for T in targets:
        q, ok, err = ik(T, q, base, tool)
        Q.append(q.copy())
        oks.append(ok)
    return np.array(Q), oks
