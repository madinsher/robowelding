"""Pneumatic swing clamps on the positioner faceplate: they hold the spool flange while the spool is welded.

Four clamps stand on T-slots of the faceplate around the fixture ring (0 / 90 / 180 / 270 deg of the faceplate: each
between two bolt holes of the flange).  Open: the arm is turned 90 deg away from the flange and lifted, so the spool is
lowered onto the ring past it.  Closing: the arm swings over the rim of the flange and is pulled down onto it - the
usual swing-clamp stroke.  Nothing passes through the flange (the stage-1 bolts and washers, which stage 2 first used
as studs rising from below through the bolt holes, are hidden: a bolt head cannot come up through its hole).

The plan drives the clamps with its 0..1 "stud" channel (0 = open, 1 = clamped; plan2 keeps the name so the shipped
plan and its hash stay as they are): animate() turns the first SWING_PART of the stroke into the swing, the rest into
the pull-down.

    clamps = pos_clamps.build(pos)                 # pos: the stage-1 positioner dict (cell.positioner.build)
    pos_clamps.animate(clamps, P["stud"], bake)    # bake: animation2.bake(idb, data_path, index, values)

Faceplate frame (the positioner's "rot" object): z = 0 is the faceplate top, the fixture ring is 0 .. POS_FIXTURE_THICK,
the flange lies on it; the top face of the flange is at Z_FLANGE_TOP.
"""
import math

import bpy
import mathutils
import numpy as np

import tools  # noqa: F401  (demo_video on sys.path)
from cell import geom as G, layout as L, materials

PREFIX = "pos_clamp"
ANGLES_DEG = (0.0, 90.0, 180.0, 270.0)      # on the T-slots (every 45 deg), between the flange bolt holes (15 + 30 k)
R_BODY = 0.262                              # clamp axis: outside the fixture ring (r 0.215) and the flange (r 0.2025)
BODY_R, BODY_H = 0.028, 0.050               # cylinder body standing on the faceplate
BASE = (0.074, 0.074, 0.008)                # its mounting foot on the T-slot
ROD_R = 0.010
ARM_LEN, ARM_BACK, ARM_W, ARM_T = 0.100, 0.016, 0.030, 0.014   # arm: ARM_BACK behind the axis, the rest towards the tip
PAD_X, PAD_R, PAD_H = 0.072, 0.011, 0.004   # contact pad under the arm tip: lands at r = R_BODY - PAD_X on the rim
Z_FLANGE_TOP = L.POS_FIXTURE_THICK + L.FLANGE_THK
Z_ARM = Z_FLANGE_TOP + PAD_H                # arm underside when clamped
SWING_DEG = 90.0                            # the open arm is tangential, clear of the flange
LIFT = 0.012                                # the open arm stands this much higher
SWING_PART = 0.65                           # share of the stroke spent swinging; the rest pulls the arm down


def _local(child, parent, loc, rot=(0.0, 0.0, 0.0)):
    """Parent with an explicit local transform (no world-keeping)."""
    child.parent = parent
    child.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    child.matrix_basis = mathutils.Matrix.Translation(loc) @ mathutils.Euler(rot).to_matrix().to_4x4()


def build(pos):
    """Build the four clamps on the faceplate of ``pos``.  Returns {"clamps": [{"swing": empty, "angle": rad}, ...],
    "objects": [...]}; the swing empties are left clamped (arm over the flange) until animate() keys them."""
    rot, col = pos["rot"], pos["collection"]
    steel = materials.get("machined_steel")
    dark = materials.get("dark_metal")
    body_mat = materials.get("painted", color="#2A2D31", roughness=0.45, coat=0.0)
    out, objs = [], []
    for i, deg in enumerate(ANGLES_DEG):
        a = math.radians(deg)
        x, y = R_BODY * math.cos(a), R_BODY * math.sin(a)
        foot = G.box(f"{PREFIX}{i}_foot", BASE, (0, 0, 0), col, bevel=0.002)
        foot.data.materials.append(dark)
        _local(foot, rot, (x, y, BASE[2] / 2), (0, 0, a))
        body = G.cylinder(f"{PREFIX}{i}_body", BODY_R, BODY_H - BASE[2], (0, 0, 0), vertices=32, collection=col)
        body.data.materials.append(body_mat)
        _local(body, rot, (x, y, BASE[2] + (BODY_H - BASE[2]) / 2))
        # moving part: rod, arm, pad and pivot cap under one empty on the clamp axis at the arm underside
        swing = G.empty(f"{PREFIX}{i}_swing", collection=col, size=0.03)
        _local(swing, rot, (x, y, Z_ARM), (0, 0, a + math.pi))          # arm (+X of the empty) towards the axis
        rod = G.cylinder(f"{PREFIX}{i}_rod", ROD_R, 0.040, (0, 0, 0), vertices=20, collection=col)
        rod.data.materials.append(steel)
        _local(rod, swing, (0, 0, -0.020))
        arm = G.box(f"{PREFIX}{i}_arm", (ARM_LEN, ARM_W, ARM_T), (0, 0, 0), col, bevel=0.003)
        arm.data.materials.append(steel)
        _local(arm, swing, (ARM_LEN / 2 - ARM_BACK, 0, ARM_T / 2))
        pad = G.cylinder(f"{PREFIX}{i}_pad", PAD_R, PAD_H, (0, 0, 0), vertices=20, collection=col)
        pad.data.materials.append(dark)
        _local(pad, swing, (PAD_X, 0, -PAD_H / 2))
        cap = G.cylinder(f"{PREFIX}{i}_cap", 0.013, 0.003, (0, 0, 0), vertices=6, collection=col)
        cap.data.materials.append(dark)
        _local(cap, swing, (0, 0, ARM_T + 0.0015))
        out.append({"swing": swing, "angle": a + math.pi})
        objs += [foot, body, swing, rod, arm, pad, cap]
    return {"clamps": out, "objects": objs}


def stroke(s):
    """(swing angle [rad], lift [m]) for the 0..1 clamp channel ``s`` (array): 0 = open, 1 = clamped."""
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    u = np.clip(s / SWING_PART, 0.0, 1.0)
    u = u * u * (3.0 - 2.0 * u)
    v = np.clip((s - SWING_PART) / (1.0 - SWING_PART), 0.0, 1.0)
    return (1.0 - u) * math.radians(SWING_DEG), (1.0 - v) * LIFT


def animate(clamps, s, bake):
    """Key the clamps from the per-frame 0..1 channel ``s`` with ``bake`` (animation2.bake)."""
    theta, lift = stroke(s)
    for c in clamps["clamps"]:
        bake(c["swing"], "rotation_euler", 2, c["angle"] + theta)
        bake(c["swing"], "location", 2, Z_ARM + lift)
