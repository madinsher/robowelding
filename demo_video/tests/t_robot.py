"""Robot + positioner look check (Cycles 24 samples, 1280x720, simple test lights).

    python3 tests/t_robot.py            positioner + spool at tilt 90 (2 views), robot at 3 poses (2 views each)
    python3 tests/t_robot.py --pos      positioner stills only          --robot   robot stills only
    ... --only N                        only the N-th still of the selected set
    xvfb-run -a python3 tests/t_robot.py --eevee ...   EEVEE with the build.py cost settings (checks the hose GN/material)
Also checks that the bpy joint chain matches the numpy FK.  Stills go to out/t_robot_*.png.
"""
import bpy, sys, os, time, numpy as np, mathutils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell import robot_build, positioner, spool, geom as G, layout as L, materials

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
DO_POS = "--pos" in sys.argv or "--robot" not in sys.argv
DO_ROBOT = "--robot" in sys.argv or "--pos" not in sys.argv
ONLY = int(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None
SAMPLES = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else 24

# three robot poses: park pose, seam-A welding pose (torch over the top of the flange), seam-B sector pose (wrist pitched)
POSES = [
    ("home", L.ROBOT_Q_HOME, -1.9),
    ("seamA", np.radians([2.1, -7.5, 17.6, -5.9, 35.5, -6.0]), 0.2),
    ("seamB", np.radians([27.0, 33.4, -18.2, 39.1, 100.5, 9.6]), 1.2),
]


def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=20); bpy.context.object.data.materials.append(materials.get("concrete"))
    bpy.ops.object.light_add(type='SUN', location=(3, 2, 6), rotation=(0.5, 0.4, 0)); bpy.context.object.data.energy = 2.0
    bpy.context.object.name = "env_light_key"
    bpy.ops.object.light_add(type='AREA', location=(2, -3, 3)); bpy.context.object.data.energy = 900; bpy.context.object.data.size = 3
    bpy.ops.object.light_add(type='AREA', location=(-1, 3, 3)); bpy.context.object.data.energy = 600; bpy.context.object.data.size = 3
    w = bpy.data.worlds.new("W"); sc.world = w; w.use_nodes = True; w.node_tree.nodes["Background"].inputs[0].default_value = (0.25, 0.27, 0.31, 1)
    sc.render.resolution_x = 1280; sc.render.resolution_y = 720
    if "--eevee" in sys.argv:
        sc.render.engine = 'BLENDER_EEVEE_NEXT'; sc.eevee.taa_render_samples = 8; sc.eevee.use_raytracing = False
        sc.eevee.use_shadows = True; sc.eevee.shadow_resolution_scale = 0.25
        for ob in bpy.data.objects:
            if ob.type == 'LIGHT':
                ob.data.use_shadow = ob.name == "env_light_key"
    else:
        sc.render.engine = 'CYCLES'; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True
    sc.view_settings.view_transform = 'AgX'
    return sc


def shoot(sc, name, loc, tgt, lens):
    bpy.ops.object.camera_add(location=loc); cam = bpy.context.object; cam.data.lens = lens; sc.camera = cam
    d = cam.location - mathutils.Vector(tgt); cam.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(OUT, f"t_robot_{name}{'_eevee' if '--eevee' in sys.argv else ''}.png")
    t0 = time.time(); bpy.ops.render.render(write_still=True)
    print(f"rendered {sc.render.filepath} in {time.time() - t0:.1f}s")


def tris(col):
    return sum(len(o.data.polygons) for o in col.objects if o.type == 'MESH')


if DO_POS:
    sc = setup_scene()
    pos = positioner.build()
    sp = spool.build()
    G.set_parent(sp["root"], pos["mount"], keep_world=False)
    positioner.set_tilt(pos, 90); positioner.set_rot(pos, 0)
    bpy.context.view_layer.update()
    print("positioner tris:", tris(pos["collection"]))
    shots = [
        ("pos_0", (2.6, -2.4, 1.9), (0.1, 0.1, 1.05), 35),      # from the robot side: drive, cables, towers
        ("pos_1", (-2.4, 2.6, 1.6), (0.0, 0.3, 1.1), 35),       # loading side: cover plate, nameplate, faceplate edge-on
        ("pos_2", (0.5, 1.9, 1.75), (0.05, 0.45, 1.35), 50),    # faceplate / fixture close-up (T-slots, bolts)
        ("pos_3", (2.6, -1.9, 1.5), (0.75, 0.0, 0.95), 40),     # tilt drive: gearbox, motor, terminal box, cable, gland
        ("pos_4", (-2.3, 1.6, 1.2), (-0.4, 0.1, 0.8), 40),      # cover plate + nameplate on the loading side
    ]
    for i, (n, loc, tgt, lens) in enumerate(shots):
        if ONLY is None or ONLY == i:
            shoot(sc, n, loc, tgt, lens)

if DO_ROBOT:
    sc = setup_scene()
    rob = robot_build.build()
    arm = rob["arm"]
    print("robot tris:", tris(rob["collection"]))
    k = 0
    for name, q, ty in POSES:
        q = np.array(q, dtype=float)
        robot_build.set_q(rob, q); robot_build.set_track(rob, ty)
        bpy.context.view_layer.update()
        T_np = arm.fk(q, robot_build.base_matrix(ty), robot_build.tool_transform())
        T_bl = G.np4(rob["tcp"].matrix_world)
        err = np.abs(T_np - T_bl).max()
        print(f"{name}: TCP numpy {T_np[:3, 3].round(4)} bpy {T_bl[:3, 3].round(4)} maxdiff {err:.6f}")
        assert err < 1e-4, "bpy chain does not match numpy FK"
        tcp = mathutils.Vector(T_bl[:3, 3]); j4 = rob["joints"][3].matrix_world.translation; j5 = rob["joints"][4].matrix_world.translation
        mid = (j4 + j5) / 2
        # whole arm from the -Y/-X side, and a close-up of wrist + torch + hose
        for i, (loc, tgt, lens) in enumerate([
            (mid + mathutils.Vector((-2.2, -2.6, 1.2)), mid + mathutils.Vector((0, 0, -0.2)), 40),
            (tcp + mathutils.Vector((-0.9, -0.9, 0.55)), (tcp * 0.5 + j5 * 0.5), 50),
        ]):
            if ONLY is None or ONLY == k:
                shoot(sc, f"{name}_{i}", loc, tgt, lens)
            k += 1
        if name == "seamA":     # detail views: laser sensor lens/bracket from the seam side, hose clamps on the forearm
            box = bpy.data.objects["torch_laser_sensor"].matrix_world.translation
            M4 = rob["joints"][3].matrix_world
            for i, (loc, tgt, lens) in enumerate([
                (tcp + (tcp - box) * 0.9 + mathutils.Vector((0.05, -0.25, -0.05)), box, 60),   # from beyond the TCP, up at the lens face
                (M4 @ mathutils.Vector((0.55, -0.55, 0.42)), M4 @ mathutils.Vector((0.55, 0.0, 0.1)), 45),
            ]):
                if ONLY is None or ONLY == k:
                    shoot(sc, f"{name}_detail{i}", loc, tgt, lens)
                k += 1
