"""Minimal URDF serial-chain loader (revolute joints) with FK / damped-LS IK.

Link k visual mesh is expressed in link k's frame; joint k origin is in the
parent link frame: T_child = T_parent @ Origin_k @ Rot(axis_k, q_k).
"""
import math
import os
import xml.etree.ElementTree as ET
import numpy as np


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _origin(el):
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    T[:3, :3] = _rpy(*rpy)
    T[:3, 3] = xyz
    return T


def axis_rot(axis, t):
    a = np.array(axis, dtype=float)
    a /= np.linalg.norm(a)
    x, y, z = a
    c, s, C = math.cos(t), math.sin(t), 1 - math.cos(t)
    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])
    T = np.eye(4)
    T[:3, :3] = R
    return T


class URDFArm:
    def __init__(self, urdf_path, mesh_root, joint_names=None, tool_link="tool0"):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        self.links = {}
        for l in root.findall("link"):
            vis = l.find("visual")
            mesh = None
            vis_origin = np.eye(4)
            if vis is not None:
                geo = vis.find("geometry/mesh")
                if geo is not None:
                    fn = geo.get("filename")
                    fn = fn.replace("package://", "")
                    mesh = os.path.join(mesh_root, fn)
                vis_origin = _origin(vis.find("origin"))
            self.links[l.get("name")] = dict(mesh=mesh, vis_origin=vis_origin)
        joints = []
        for j in root.findall("joint"):
            joints.append(dict(
                name=j.get("name"), type=j.get("type"),
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                origin=_origin(j.find("origin")),
                axis=[float(v) for v in (j.find("axis").get("xyz").split() if j.find("axis") is not None else "1 0 0".split())],
                limit=(float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))) if j.find("limit") is not None else (-math.pi, math.pi),
            ))
        # order chain from base_link to tool link
        by_child = {j["child"]: j for j in joints}
        chain = []
        cur = tool_link
        while cur in by_child:
            j = by_child[cur]
            chain.append(j)
            cur = j["parent"]
        chain.reverse()
        self.chain = chain
        self.base_link = chain[0]["parent"]
        self.rev = [j for j in chain if j["type"] == "revolute"]
        assert len(self.rev) == 6, [j["name"] for j in self.rev]
        self.limits = [j["limit"] for j in self.rev]

    def fk_frames(self, q, base=np.eye(4)):
        """Returns dict link_name -> world 4x4 for every link in the chain (incl. tool link)."""
        frames = {self.base_link: base.copy()}
        T = base.copy()
        qi = 0
        for j in self.chain:
            T = T @ j["origin"]
            if j["type"] == "revolute":
                T = T @ axis_rot(j["axis"], q[qi])
                qi += 1
            frames[j["child"]] = T.copy()
        return frames

    def fk(self, q, base=np.eye(4), tool=np.eye(4)):
        return self.fk_frames(q, base)[self.chain[-1]["child"]] @ tool

    @staticmethod
    def _rot_err(Rd, R):
        E = Rd @ R.T
        v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
        c = max(-1.0, min(1.0, (np.trace(E) - 1) / 2))
        ang = math.acos(c)
        if ang < 1e-9:
            return np.zeros(3)
        return v / (2 * math.sin(ang)) * ang

    def jacobian(self, q, base, tool, eps=1e-6):
        T0 = self.fk(q, base, tool)
        J = np.zeros((6, 6))
        for j in range(6):
            dq = np.array(q, dtype=float)
            dq[j] += eps
            T1 = self.fk(dq, base, tool)
            J[:3, j] = (T1[:3, 3] - T0[:3, 3]) / eps
            J[3:, j] = self._rot_err(T1[:3, :3], T0[:3, :3]) / eps
        return J

    def ik(self, target, q0, base=np.eye(4), tool=np.eye(4), iters=150, tol=1e-5, lam=0.04,
           w_rot=0.5, clamp=True, joint_weights=None, free_spin=False, q_ref=None, k_null=0.25):
        """Damped least-squares IK.

        free_spin: ignore the rotation about the tool Z axis (symmetric welding torch) -> 5-DoF task;
        q_ref/k_null: null-space posture bias (keeps the wrist away from flips) — only meaningful with free_spin.
        """
        q = np.array(q0, dtype=float)
        err_n = 1e9
        W = np.ones(6) if joint_weights is None else np.array(joint_weights, dtype=float)
        qr = None if q_ref is None else np.array(q_ref, dtype=float)
        for _ in range(iters):
            T = self.fk(q, base, tool)
            er = self._rot_err(target[:3, :3], T[:3, :3])
            if free_spin:
                zt = T[:3, 2]
                er = er - np.dot(er, zt) * zt
            e = np.concatenate([target[:3, 3] - T[:3, 3], w_rot * er])
            err_n = np.linalg.norm(e)
            if err_n < tol:
                break
            J = self.jacobian(q, base, tool)
            if free_spin:
                zt = T[:3, 2]
                Jr = J[3:, :]
                J = np.vstack([J[:3, :], Jr - np.outer(zt, zt @ Jr)])
            J[3:, :] *= w_rot
            Jw = J * W[None, :]
            JJt = Jw @ Jw.T + (lam ** 2) * np.eye(6)
            Jp = Jw.T @ np.linalg.inv(JJt)             # damped pseudo-inverse
            dq = W * (Jp @ e)
            if qr is not None and free_spin:
                N = np.eye(6) - Jp @ Jw
                dq = dq + W * (N @ (k_null * (qr - q)))
            step = np.linalg.norm(dq)
            if step > 0.3:
                dq *= 0.3 / step
            q = q + dq
            if clamp:
                for j in range(6):
                    lo, hi = self.limits[j]
                    q[j] = min(max(q[j], lo), hi)
        return q, err_n < 2e-3, err_n

    def solve_path(self, targets, q0, base=np.eye(4), tool=np.eye(4), **kw):
        Q, oks, errs = [], [], []
        q = np.array(q0, dtype=float)
        for T in targets:
            q, ok, err = self.ik(T, q, base, tool, **kw)
            Q.append(q.copy()); oks.append(ok); errs.append(err)
        return np.array(Q), oks, errs
