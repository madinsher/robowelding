"""Output roller conveyor (chain-driven live rollers) with carrier pallets for the finished spools.

Everything is placed from layout2 (`CONVEYOR`, `CARRIER`, `CONV_SPOOL_Y`, `CONV_LOAD_X`, `CONV_QC_X`, `CONV_END_X`):

* frame x in [x0, x1] around the centre line y = CONVEYOR["y"], outer width `width`: C-channel side rail on the -Y
  side, a mirrored rail on the drive side (+Y) with the yellow roller-chain guard inside it; roller tops at `top_z`,
  rollers along Y with `roller_pitch` (radius `roller_r`, galvanized tube, shaft stubs in the rails); legs every
  ~1.5 m (levelling feet, anchor bolts, cross ties and stringers); gear motor under the -X end with its drive guard;
* UHMW side guides (carrier width + 2 x 6 mm) on small brackets, interrupted at the photo-eyes; at every station
  (LOAD / QC / END) a retro-reflective photo-eye (-Y side, beam at the carrier plate height) with its reflector (+Y)
  and a pneumatic stop between two rollers under the carrier's -X face (the END stop is up = hard stop at the end of
  the line; LOAD and QC are drawn retracted: the carrier passes over them); RFID reader at the QC station, end plates
  with hazard stripes, ID plate;
* carrier pallet (`CARRIER`): steel plate on the rollers, PU bumpers on its +-X faces (the stops touch them),
  machined seat ring under the flange back face (the raised face and the bolt holes stay free inside the ring) with
  3 centring pins outside the flange rim, V-post under the pipe at spool-local x = 0.86 (steel V + polymer liners,
  45 deg flanks touching the pipe), RFID tag + ID plate on the +Y end.

The carrier root IS the spool frame on it: origin (x, CONV_SPOOL_Y, seat_z), local +X = world +Y (pipe leg),
local +Y = world -X, Z up.  A spool whose root is placed at the carrier root sits on the seat ring / V-post.

Public API (DESIGN.md section 3, conveyor.py):
    build(collection=None, n_carriers=2) -> dict(collection, carriers, rollers, root, objects, stations)
        carriers = [dict(root=<empty 'conv_carrier{i}'>, objects=[...], frame=T)]  root = spool frame (numpy T at build)
                   default x: carrier 0 at CONV_LOAD_X, 1 at CONV_QC_X, 2 at CONV_END_X (n_carriers <= 3)
        rollers  = [roller objects]  (axis = local Y through the origin; channel rotation_euler[1])
        root     = <empty 'conv_root'> parent of the static frame (world placement, identity rotation)
        stations = dict(load=x, qc=x, end=x)
    carrier_frame(x) -> numpy 4x4                   spool frame on a carrier whose flange centre is at x
    set_carrier(conv, i, x, frame=None)             carrier i flange-centre x (keys root location[0] only)
    set_rollers(conv, travel_m, frame=None)         every roller turned by -travel_m / roller_r about its axis (the
                                                    top surface moves toward -X, the conveying direction)
    obstacles() -> list[dict(name, center, size, yaw)]   conservative world AABBs of the static conveyor (no bpy use)
"""
import math

import bpy

import tools  # noqa: F401  (demo_video on sys.path)
from cell import geom as G
from cell import materials
from cell import environment as E
import layout2 as L2
import kin
import cassette as C

NAME = "OutputConveyor"
GAP = C.GAP

CV = L2.CONVEYOR
X0, X1 = CV["x0"], CV["x1"]
YC = CV["y"]
HW = CV["width"] / 2                     # 0.75: outer half width of the frame
TOP = CV["top_z"]                        # roller tops (carrier underside)
PITCH = CV["roller_pitch"]
RR = CV["roller_r"]
AXIS_Z = TOP - RR
N_ROLL = int(round((X1 - X0) / PITCH))
ROLL_X = tuple(X0 + PITCH / 2 + k * PITCH for k in range(N_ROLL))
SEAT_Z = L2.CARRIER["seat_z"]
CAR_SX, CAR_SY = L2.CARRIER["size"]      # (0.70 along world X, 1.45 along world Y)
CAR_T = L2.CARRIER["thick"]
STATIONS = dict(load=L2.CONV_LOAD_X, qc=L2.CONV_QC_X, end=L2.CONV_END_X)

# ------------------------------------------------------------------ frame (world, relative to the centre line)
RAIL = dict(h=0.16, w=0.06, t=0.006, z=(0.55, 0.71))          # C-channel side rails
GUARD = dict(y=(HW - 0.10, HW - 0.06), z=(0.60, 0.712))        # chain guard (inside the +Y rail), yellow
ROLL_Y = (-HW + 0.065, HW - 0.11)                               # roller tube ends (relative y)
SHAFT_R = 0.011
GUIDE = dict(y=0.737, t=0.012, z=(0.714, 0.764))                # UHMW guides at y = +-0.737 (carrier +-0.725)
LEG = 0.08
LEGS_X = (X0 + 0.10, X0 + 1.55, X0 + 3.00, X1 - 0.10)
LEG_Y = HW - LEG / 2
EYE_DX = -0.10                           # photo-eye x offset from a station (beam blocked by a carrier at the station)
EYE_Z = TOP + 0.023                      # beam height (inside the carrier plate 0.72..0.78)
SHELF_TOP = RAIL["z"][1] + 0.006         # top of the photo-eye / reflector shelves on the rail top flange
STOP_HEAD = (0.022, 0.10, 0.05)          # stop head (x, y, z)
STOP_GAP = 0.004                         # stop face .. carrier bumper at the station
MOTOR = dict(x=(X0 + 0.50, X0 + 0.82), y=HW - 0.26, z=0.40)    # gear motor under the -X end (axis along X)

# ------------------------------------------------------------------ carrier (spool-local frame of the carrier root)
CAR_LX = (YC - L2.CONV_SPOOL_Y) - CAR_SY / 2, (YC - L2.CONV_SPOOL_Y) + CAR_SY / 2   # plate local x range
BUMPER_T = 0.02
CAR_LY = CAR_SX / 2 - BUMPER_T           # plate half width (local y); bumpers outside it
PLATE_Z = (TOP - SEAT_Z + GAP, TOP + CAR_T - SEAT_Z)            # (-0.08, -0.02): plate bottom on the roller tops
RING = C.FL_RING                          # seat ring inner / outer radius (as the kit nests)
RING_BASE_R = 0.2245                      # base disc (inside the plate: local x >= -0.225)
RING_BASE_H = 0.006
VPOST_LX = 0.86
V_W = 0.12                                # V-head half width (across the pipe)
V_BOT = 0.04
V_LEN = 0.08                              # V-head length along the pipe
POST = 0.07


def carrier_frame(x):
    """Spool frame (numpy 4x4) on a carrier whose flange centre is at world x."""
    return kin.planar_frame((x, L2.CONV_SPOOL_Y, SEAT_Z), (0.0, 1.0))


def _v_apex():
    """Spool-local z of the V apex: both 45-deg flanks touch the pipe (+GAP)."""
    return L2.PIPE_AXIS_Z - (L2.PIPE_R + GAP) * math.sqrt(2.0)


def _stop_face(station_x):
    return station_x - CAR_SX / 2 - STOP_GAP


def _eye_x(station_x):
    """Photo-eye x: the roller gap nearest to station_x + EYE_DX (brackets clear of the roller shaft nuts)."""
    return X0 + round((station_x + EYE_DX - X0) / PITCH) * PITCH


def _mats():
    return dict(
        blue=materials.get("painted", color="#1F4E8C", roughness=0.42),
        blue2=materials.get("painted", color="#26558F", roughness=0.42),
        frame=materials.get("painted", color="#2B2F36", roughness=0.5),
        charcoal=materials.get("painted", color="#3A3E44", roughness=0.45, coat=0.1),
        motor=materials.get("painted", color="#56606B", roughness=0.4),
        yellow=materials.get("safety_yellow"),
        galv=materials.get("galvanized"),
        steel=materials.get("machined_steel"),
        dark=materials.get("dark_metal"),
        black=materials.get("black_plastic"),
        rubber=materials.get("rubber"),
        cable=materials.get("cable_black"),
        glass=materials.get("glass_dark"),
        uhmw=materials.get("painted", color="#C9C3AF", roughness=0.6, coat=0.0),
        pu=materials.get("painted", color="#D4A017", roughness=0.55, coat=0.0),
        reflector=materials.get("painted", color="#B0101A", roughness=0.15, coat=0.6),
        white=materials.get("painted", color="#E8E8E8", roughness=0.4),
        hazard=E._mat_hazard(),
        led_orange=materials.get("emissive", color="#FF9A1A", strength=8.0),
        led_green=materials.get("emissive", color="#30FF40", strength=6.0),
    )


# ============================================================================ static frame
def _rails(b, M):
    xm, L = (X0 + X1) / 2, X1 - X0
    rh, rw, rt = RAIL["h"], RAIL["w"], RAIL["t"]
    zc = sum(RAIL["z"]) / 2
    poly = E._channel_poly(rh, rw, rt)
    E._prism(b, "conv_rail_m", poly, L, M["blue"], (xm, YC - HW + rw / 2, zc), axis='X')
    E._prism(b, "conv_rail_p", [(-u, v) for u, v in poly][::-1], L, M["blue"], (xm, YC + HW - rw / 2, zc), axis='X')
    # yellow chain guard (inverted U over the roller sprockets) inside the +Y rail
    gy0, gy1 = GUARD["y"]
    gz0, gz1 = GUARD["z"]
    C._bx(b, "conv_guard", (L - 0.02, gy1 - gy0, gz1 - gz0), (xm, YC + (gy0 + gy1) / 2, (gz0 + gz1) / 2), M["yellow"], bev=0.003)
    for k in range(int(L / 0.75)):                      # cover screws
        x = X0 + 0.375 + 0.75 * k
        C._hex(b, f"conv_guard_screw{k}", 0.006, 0.003, M["dark"], (x, YC + (gy0 + gy1) / 2, gz1))
    # end plates with hazard stripes, rubber buffer at the -X end
    for k, (x, s) in enumerate(((X0, -1), (X1, 1))):
        C._bx(b, f"conv_endplate{k}", (0.012, 2 * HW, RAIL["z"][1] - RAIL["z"][0]), (x + s * 0.006, YC, zc), M["hazard"], bev=0.002)
    for k, dy in enumerate((-0.35, 0.35)):                # rubber buffers on plates bolted to the -X end plate
        zb = TOP + 0.005
        C._bx(b, f"conv_end_buffer{k}_plate", (0.012, 0.09, 0.09), (X0 + 0.006, YC + dy, zb), M["frame"], bev=0.002)
        E._rod(b, f"conv_end_buffer{k}", 0.03, 0.05, M["rubber"], (X0 + 0.037, YC + dy, zb), (0, math.pi / 2, 0), 16)
    # rollers
    rollers = []
    ya, yb = ROLL_Y
    prof = [(0.0, -HW + 0.006), (SHAFT_R, -HW + 0.006), (SHAFT_R, ya), (RR - 0.004, ya), (RR, ya + 0.004), (RR, yb - 0.004),
            (RR - 0.004, yb), (SHAFT_R, yb), (SHAFT_R, HW - 0.006), (0.0, HW - 0.006)]
    for k, x in enumerate(ROLL_X):
        ob = E._rev(b, f"conv_roller{k:02d}", prof, M["galv"], (x, YC, AXIS_Z), segs=24, axis='Y')
        rollers.append(ob)
        for j, s in enumerate((-1, 1)):                 # hex shaft nuts on the outer webs
            y = YC + s * HW
            C._hex(b, f"conv_roller{k:02d}_nut{j}", 0.013, 0.008, M["steel"], (x, y, AXIS_Z), (-s * math.pi / 2, 0, 0))
    return rollers


def _legs(b, M):
    z_top = RAIL["z"][0]
    for k, x in enumerate(LEGS_X):
        for j, s in enumerate((-1, 1)):
            y = YC + s * LEG_Y
            C._bx(b, f"conv_leg{k}{j}", (LEG, LEG, z_top - 0.075), (x, y, (z_top + 0.075) / 2), M["blue"], bev=0.003)
            C._bx(b, f"conv_leg{k}{j}_plate", (0.13, 0.13, 0.01), (x, y, 0.07), M["frame"])
            E._rod(b, f"conv_leg{k}{j}_screw", 0.012, 0.05, M["steel"], (x, y, 0.04), segs=10)
            E._rev(b, f"conv_leg{k}{j}_pad", [(0, 0), (0.05, 0), (0.05, 0.012), (0.03, 0.02), (0, 0.02)], M["dark"], (x, y, 0.0), segs=20)
            for n, (dx, dy) in enumerate(((-0.05, -0.05), (0.05, 0.05))):
                C._hex(b, f"conv_leg{k}{j}_bolt{n}", 0.008, 0.008, M["dark"], (x + dx, y + dy, 0.075))
        # cross ties (along Y) under the rails and low
        C._bx(b, f"conv_tie{k}_top", (0.06, 2 * LEG_Y - LEG, 0.05), (x, YC, z_top - 0.025), M["frame"], bev=0.002)
        C._bx(b, f"conv_tie{k}_low", (0.05, 2 * LEG_Y - LEG, 0.05), (x, YC, 0.22), M["frame"], bev=0.002)
    for j, s in enumerate((-1, 1)):                     # stringers along X
        xa, xb = LEGS_X[0] + LEG / 2, LEGS_X[-1] - LEG / 2
        C._bx(b, f"conv_stringer{j}", (xb - xa, 0.05, 0.05), ((xa + xb) / 2, YC + s * LEG_Y, 0.22), M["frame"], bev=0.002)


def _guides(b, M):
    """UHMW guide strips on both sides, interrupted around the photo-eye beams; small angle brackets."""
    eyes = sorted(_eye_x(x) for x in STATIONS.values())
    cuts = [X0 + 0.04] + [v for x in eyes for v in (x - 0.035, x + 0.035)] + [X1 - 0.04]
    segs = [(cuts[i], cuts[i + 1]) for i in range(0, len(cuts), 2)]
    gz0, gz1 = GUIDE["z"]
    for j, s in enumerate((-1, 1)):
        y = YC + s * GUIDE["y"]
        for k, (xa, xb) in enumerate(segs):
            C._bx(b, f"conv_guide{j}_{k}", (xb - xa, GUIDE["t"], gz1 - gz0), ((xa + xb) / 2, y, (gz0 + gz1) / 2), M["uhmw"], bev=0.002)
            n = max(2, int((xb - xa) / 0.6) + 1)
            for m in range(n):
                x = xa + 0.06 + (xb - xa - 0.12) * m / (n - 1)
                yb = YC + s * (GUIDE["y"] + GUIDE["t"] / 2 + 0.003)
                C._bx(b, f"conv_guide{j}_{k}_br{m}", (0.03, 0.006, 0.045), (x, yb, gz0 + 0.0125), M["frame"])
                C._bx(b, f"conv_guide{j}_{k}_bf{m}", (0.03, 0.014, 0.004), (x, yb - s * 0.004, RAIL["z"][1] + 0.002), M["frame"])


def _photo_eye(b, M, name, x):
    """Retro-reflective sensor on an L-bracket over the -Y rail, reflector over the +Y rail; beam at EYE_Z across."""
    zt = RAIL["z"][1]
    ys = YC - HW
    C._bx(b, name + "_bracket", (0.03, 0.005, zt - 0.60), (x, ys - 0.0025, (zt + 0.60) / 2), M["frame"])
    C._bx(b, name + "_shelf", (0.03, 0.045, 0.006), (x, ys - 0.0175, zt + 0.003), M["frame"])
    zb_top = EYE_Z + 0.025                                  # sensor housing sits on the shelf
    C._bx(b, name + "_body", (0.02, 0.032, zb_top - SHELF_TOP), (x, ys - 0.020, (zb_top + SHELF_TOP) / 2), M["charcoal"], bev=0.002)
    C._bx(b, name + "_lens", (0.014, 0.003, 0.02), (x, ys - 0.0025, EYE_Z + 0.006), M["glass"])
    C._bx(b, name + "_led", (0.008, 0.008, 0.003), (x, ys - 0.026, EYE_Z + 0.0265), M["led_orange"])
    E._rod(b, name + "_plug", 0.006, 0.03, M["black"], (x, ys - 0.032, EYE_Z - 0.01), (math.pi / 2, 0, 0), 10)
    E._bar(b, name + "_cable", (x, ys - 0.047, EYE_Z - 0.01), (x, ys - 0.012, 0.58), 0, M["cable"], radius=0.003, segs=6)
    yr = YC + HW
    C._bx(b, name + "_rbracket", (0.03, 0.005, zt - 0.60), (x, yr + 0.0025, (zt + 0.60) / 2), M["frame"])
    C._bx(b, name + "_rshelf", (0.03, 0.03, 0.006), (x, yr + 0.01, zt + 0.003), M["frame"])
    zr_top = EYE_Z + 0.028                                  # reflector holder stands on its shelf
    C._bx(b, name + "_rframe", (0.05, 0.008, zr_top - SHELF_TOP), (x, yr + 0.012, (zr_top + SHELF_TOP) / 2), M["black"])
    E._rod(b, name + "_reflector", 0.02, 0.003, M["reflector"], (x, yr + 0.0065, EYE_Z + 0.003), (math.pi / 2, 0, 0), 20)


def _stopper(b, M, name, station_x, up):
    """Pneumatic stop between two rollers under the carrier's -X face; up = hard stop above the roller tops."""
    face = _stop_face(station_x)
    hx, hy, hz = STOP_HEAD
    xc = face - hx / 2
    z_head_top = TOP + 0.035 if up else TOP - 0.015
    zc = z_head_top - hz / 2
    C._bx(b, name + "_head", (hx - 0.004, hy, hz), (xc - 0.002, YC, zc), M["yellow"], bev=0.002)
    C._bx(b, name + "_pad", (0.004, hy - 0.01, hz - 0.012), (face - 0.002, YC, zc + 0.002), M["black"])
    rod_top = zc - hz / 2
    E._rod(b, name + "_rod", 0.008, rod_top - 0.56, M["steel"], (xc, YC, (rod_top + 0.56) / 2), segs=12)
    E._rev(b, name + "_cyl", [(0, 0.44), (0.026, 0.44), (0.026, 0.56), (0.02, 0.568), (0, 0.568)], M["steel"], (xc, YC, 0.0), segs=20)
    C._bx(b, name + "_mount", (0.07, 0.07, 0.012), (xc, YC, 0.434), M["frame"])
    C._bx(b, name + "_beam", (0.05, 2 * HW - RAIL["w"] + 0.012, 0.03), (xc, YC, 0.413), M["frame"], bev=0.002)
    for k, s in enumerate((-1, 1)):
        C._bx(b, f"{name}_hanger{k}", (0.05, 0.012, RAIL["z"][0] - 0.398), (xc, YC + s * (HW - RAIL["w"] / 2), (RAIL["z"][0] + 0.398) / 2),
              M["frame"])
    E._bar(b, name + "_hose", (xc + 0.02, YC, 0.52), (xc + 0.02, YC - 0.35, 0.40), 0, M["cable"], radius=0.004, segs=6)


def _drive(b, M):
    """Gear motor under the -X end: motor along X, helical-bevel gearbox with its output into the drive guard."""
    (xa, xb), y, z = MOTOR["x"], YC + MOTOR["y"], MOTOR["z"]
    L = xb - xa
    fins = [(0, 0), (0.07, 0)]
    n = 10
    for k in range(n):                                   # cooling fins on the motor housing
        h0, h1 = 0.03 + (L - 0.10) * k / n, 0.03 + (L - 0.10) * (k + 0.5) / n
        fins += [(0.07, h0), (0.078, h0), (0.078, h1), (0.07, h1)]
    fins += [(0.07, L - 0.07), (0.066, L - 0.02), (0.04, L), (0, L)]
    E._rev(b, "conv_motor", fins, M["motor"], (xa, y, z), (0, math.pi / 2, 0), segs=28)
    C._bx(b, "conv_motor_tbox", (0.09, 0.08, 0.06), (xa + L * 0.45, y, z + 0.098), M["motor"], bev=0.004)
    E._rod(b, "conv_motor_gland", 0.009, 0.03, M["black"], (xa + L * 0.45 - 0.035, y, z + 0.13), segs=10)
    E._bar(b, "conv_motor_cable", (xa + L * 0.45 - 0.035, y, z + 0.145), (xa + L * 0.45 - 0.035, YC - HW + 0.03, RAIL["z"][0] - 0.01),
           0, M["cable"], radius=0.006, segs=8)
    gx = xa - 0.09
    C._bx(b, "conv_gearbox", (0.18, 0.17, 0.18), (gx, y + 0.02, z), M["blue2"], bev=0.01)
    # output shaft from the gearbox into the chain case (the sprocket is inside it), bearing cover outside
    gy0, gy1 = GUARD["y"]
    ya_, yb_ = y + 0.02, YC + (gy0 + gy1) / 2
    E._rod(b, "conv_gear_shaft", 0.02, yb_ - ya_, M["steel"], (gx, (ya_ + yb_) / 2, z), (math.pi / 2, 0, 0), 14)
    E._rev(b, "conv_drive_cover", [(0, 0), (0.034, 0), (0.034, 0.005), (0.026, 0.009), (0, 0.009)], M["dark"], (gx, YC + gy1, z),
           (-math.pi / 2, 0, 0), segs=20)
    # drive guard: vertical chain case from the gearbox sprocket up into the chain guard
    C._bx(b, "conv_drive_guard", (0.26, gy1 - gy0, GUARD["z"][0] - z + 0.09), (gx + 0.04, YC + (gy0 + gy1) / 2, (GUARD["z"][0] + z - 0.09) / 2),
          M["yellow"], bev=0.003)
    # torque arm: flat bar from the gearbox to the -X cross tie
    xt = LEGS_X[0]
    C._bx(b, "conv_torque_arm", (gx - 0.08 - xt, 0.012, 0.05), ((gx - 0.08 + xt) / 2, y + 0.02, RAIL["z"][0] - 0.055), M["frame"])


def _rfid_reader(b, M):
    x = STATIONS["qc"] + 0.12
    y = YC + HW + 0.02
    C._bx(b, "conv_rfid_reader", (0.10, 0.03, 0.07), (x, y, TOP + 0.03), M["white"], bev=0.004)
    C._bx(b, "conv_rfid_reader_face", (0.07, 0.002, 0.045), (x, y - 0.0155, TOP + 0.03), M["charcoal"])
    C._bx(b, "conv_rfid_led", (0.008, 0.003, 0.006), (x + 0.035, y - 0.016, TOP + 0.055), M["led_green"])
    C._bx(b, "conv_rfid_bracket", (0.06, 0.006, TOP + 0.03 - RAIL["z"][0]), (x, YC + HW + 0.003, (TOP + 0.03 + RAIL["z"][0]) / 2), M["frame"])


def _labels(b, M, lm):
    # on the +Y rail web, in a roller gap and below the shaft nuts
    C._label(b, "conv_id_plate", lm, 0, (0.16, 0.064), (X1 - 2 * PITCH, YC + HW + 0.0035, RAIL["z"][0] + 0.06),
             (math.pi / 2, 0, math.pi), backing=M["frame"])


# ============================================================================ carrier pallet
def _carrier(col, M, lm, i):
    """Carrier pallet i built in its own (spool) frame, parented to its root empty; returns (root, objects)."""
    b = E._B(col)
    p = f"conv_car{i}"
    lx0, lx1 = CAR_LX
    z0, z1 = PLATE_Z
    C._bx(b, p + "_plate", (lx1 - lx0, 2 * CAR_LY, z1 - z0), ((lx0 + lx1) / 2, 0.0, (z0 + z1) / 2), M["charcoal"], bev=0.004)
    # machined wear strips on the top edges, PU bumpers on the +-Y faces (world -+X): the stops touch the middle one
    for k, s in enumerate((-1, 1)):
        C._bx(b, f"{p}_bumper{k}", (0.20, BUMPER_T, 0.04), ((lx0 + lx1) / 2, s * (CAR_LY + BUMPER_T / 2), (z0 + z1) / 2), M["pu"], bev=0.003)
        for j, dx in enumerate((-0.07, 0.07)):
            C._hex(b, f"{p}_bumper{k}_bolt{j}", 0.006, 0.004, M["dark"], ((lx0 + lx1) / 2 + dx, s * (CAR_LY + BUMPER_T - 0.0045), (z0 + z1) / 2),
                   (-s * math.pi / 2, 0, 0))
        C._bx(b, f"{p}_edge{k}", (lx1 - lx0 - 0.04, 0.03, 0.002), ((lx0 + lx1) / 2, s * (CAR_LY - 0.03), z1 + 0.001), M["hazard"])
    # seat: base flange disc (bolted) + machined ring + 3 centring pins
    ri, ro = RING
    zb = z1
    E._rev(b, p + "_seat_base", [(0, zb), (RING_BASE_R, zb), (RING_BASE_R, zb + RING_BASE_H), (0, zb + RING_BASE_H)], M["blue"], segs=48)
    top = -GAP
    E._rev(b, p + "_seat_ring", [(ri, zb + RING_BASE_H), (ro, zb + RING_BASE_H), (ro, top - 0.002), (ro - 0.002, top), (ri + 0.002, top),
                                 (ri, top - 0.002)], M["steel"], segs=64)
    for k in range(4):                                    # socket-head screws inside the ring (under the bolt-hole annulus)
        a = math.radians(45 + 90 * k)
        C._hex(b, f"{p}_seat_bolt{k}", 0.008, 0.005, M["dark"], (0.184 * math.cos(a), 0.184 * math.sin(a), zb + RING_BASE_H))
    for k, ang in enumerate(C.FL_PIN_ANG):
        a = math.radians(ang)
        h = C.FL_PIN[1] - top
        E._rev(b, f"{p}_pin{k}", [(0, 0), (C.FL_PIN[0], 0), (C.FL_PIN[0], h - 0.006), (C.FL_PIN[0] * 0.55, h), (0, h)], M["steel"],
               (C.FL_PIN_R * math.cos(a), C.FL_PIN_R * math.sin(a), top), segs=12)
    # V-post under the pipe
    za = _v_apex()
    zvb = za - V_BOT
    C._v_block(b, p + "_vhead", zvb, za, V_W, 0.02, math.radians(45.0), V_LEN, (VPOST_LX, 0.0, 0.0), 'X', M["blue"], M["uhmw"],
               span=(0.045, V_W))
    C._bx(b, p + "_vsaddle", (V_LEN + 0.02, 2 * V_W + 0.02, 0.012), (VPOST_LX, 0.0, zvb - 0.006), M["frame"], bev=0.002)
    for k, s in enumerate((-1, 1)):
        C._hex(b, f"{p}_vhead_bolt{k}", 0.008, 0.006, M["dark"], (VPOST_LX, s * (V_W + 0.001), zvb))
    zs = zvb - 0.012
    C._bx(b, p + "_vpost", (POST, POST, zs - z1 - 0.012), (VPOST_LX, 0.0, (zs + z1 + 0.012) / 2), M["blue"], bev=0.003)
    C._bx(b, p + "_vpost_foot", (0.15, 0.15, 0.012), (VPOST_LX, 0.0, z1 + 0.006), M["frame"], bev=0.002)
    for k, (dx, dy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        C._hex(b, f"{p}_vpost_bolt{k}", 0.007, 0.006, M["dark"], (VPOST_LX + dx * 0.055, dy * 0.055, z1 + 0.012))
    for k, s in enumerate((-1, 1)):                      # gussets across the pipe direction
        E._prism(b, f"{p}_vgusset{k}", [(0.0, 0.0), (0.05, 0.0), (0.0, 0.12)], 0.01, M["blue"],
                 (VPOST_LX, s * POST / 2, z1 + 0.012), (0, 0, math.pi / 2 if s > 0 else -math.pi / 2), axis='Y')
    # +Y end (local +x): RFID tag + ID plate; lifting eyes on the top face
    C._bx(b, p + "_rfid", (0.0035, 0.08, 0.04), (lx1 - 0.001, 0.10, (z0 + z1) / 2), M["black"], bev=0.001)
    C._label(b, p + "_idplate", lm, 1 + i, (0.12, 0.045), (lx1 + 0.0006, -0.10, (z0 + z1) / 2), (math.pi / 2, 0, math.pi / 2))
    for k, s in enumerate((-1, 1)):
        E._rev(b, f"{p}_eye{k}", [(0.012, 0.0), (0.02, 0.0), (0.02, 0.004), (0.012, 0.004)], M["steel"], (lx1 - 0.06, s * 0.25, z1), segs=16)
    root = G.empty(f"conv_carrier{i}", collection=col, size=0.25, display='ARROWS')
    C._parent_all(b.objs, root)
    return root, b.objs


# ============================================================================ build / setters
def build(collection=None, n_carriers=2):
    if not 0 < n_carriers <= 3:
        raise ValueError("n_carriers must be 1..3")
    col = collection or bpy.data.collections.new(NAME)
    if collection is None:
        bpy.context.scene.collection.children.link(col)
    M = _mats()
    lm = C._label_mat("conv_label", ["CV-01"] + [f"C-{i + 1:02d}" for i in range(3)])
    b = E._B(col)
    rollers = _rails(b, M)
    _legs(b, M)
    _guides(b, M)
    for name, x in STATIONS.items():
        _photo_eye(b, M, f"conv_eye_{name}", _eye_x(x))
        _stopper(b, M, f"conv_stop_{name}", x, up=(name == "end"))
    _drive(b, M)
    _rfid_reader(b, M)
    _labels(b, M, lm)
    root = G.empty("conv_root", collection=col, size=0.4)
    root.matrix_world = G.M(kin.tr((X0 + X1) / 2, YC, 0.0))
    C._parent_all(b.objs, root)
    carriers = []
    x_init = (STATIONS["load"], STATIONS["qc"], STATIONS["end"])
    for i in range(n_carriers):
        r, objs = _carrier(col, M, lm, i)
        T = carrier_frame(x_init[i])
        r.matrix_world = G.M(T)
        carriers.append(dict(root=r, objects=objs, frame=T))
    return dict(collection=col, carriers=carriers, rollers=rollers, root=root, objects=b.objs, stations=dict(STATIONS))


def set_carrier(conv, i, x, frame=None):
    """Carrier i: flange-centre x (y = CONV_SPOOL_Y, z = seat_z fixed).  Keys location[0] of the carrier root only."""
    r = conv["carriers"][i]["root"]
    r.location[0] = float(x)
    if frame is not None:
        r.keyframe_insert(data_path="location", index=0, frame=frame)


def set_rollers(conv, travel_m, frame=None):
    """Roller angle for a conveyed distance travel_m (m, toward -X): rotation_euler[1] = -travel_m / roller_r."""
    a = -float(travel_m) / RR
    for ob in conv["rollers"]:
        ob.rotation_euler[1] = a
        if frame is not None:
            ob.keyframe_insert(data_path="rotation_euler", index=1, frame=frame)


def obstacles():
    """Conservative world AABBs (dict(name, center, size, yaw=0)) of the static conveyor (carriers excluded: they move)."""
    out = []

    def add(name, lo, hi):
        out.append(dict(name=name, center=tuple((lo[k] + hi[k]) / 2 for k in range(3)),
                        size=tuple(hi[k] - lo[k] for k in range(3)), yaw=0.0))

    # frame, rollers, guides, stops (END stop head up to TOP + 0.035), photo-eyes, reflectors, RFID reader, plates
    add("conv_body", (X0 - 0.02, YC - HW - 0.05, 0.0), (X1 + 0.02, YC + HW + 0.04, TOP + 0.07))
    return out
