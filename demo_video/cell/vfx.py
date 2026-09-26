"""Arc-welding visual effects for EEVEE (BLENDER_EEVEE_NEXT, Blender 4.5).

Public API
----------
build(arc_empty, weld_intervals, collection=None) -> dict
    Arc point light, emissive arc core + soft glow, spark particle systems and a fume plume,
    all following ``arc_empty`` (its world location is the wire tip; its local +Z points from the
    torch into the workpiece).  Everything is switched by keyframes derived from ``weld_intervals``.
setup_compositor(scene) -> None
    Glare (bloom) + subtle vignette on the scene compositor.
laser_line(sensor_empty, intervals, collection=None) -> object
    Red laser line + narrow red spot for the seam-search shot.
bake_particles(scene) -> None
    Fill the spark particle caches (needed before rendering an isolated still; an animation render
    steps the frames itself).

Conventions: metres, Z up, FPS from cell.layout.  All objects go to the "VFX" collection unless
another one is given.  Nothing outside that collection is touched, except the scene's EEVEE
volumetric settings (set once in ``build`` so the fume plume stays cheap).
"""
import math
import random

import bpy
import bmesh
from mathutils import Vector, Matrix

from . import layout as L

# ------------------------------------------------------------------ tunables
ARC_LIGHT_POWER = 500.0          # W; EEVEE point light, flickers +/- ARC_FLICKER
ARC_LIGHT_RANGE = 3.0             # m; EEVEE custom cutoff so the arc does not light the whole hall
ARC_FLICKER = 0.25
ARC_FLICKER_HZ = 10.0
ARC_LIGHT_COLOR = (0.72, 0.82, 1.0)
ARC_CORE_RADIUS = 0.003           # 6 mm core sphere
ARC_GLOW_RADIUS = 0.026           # soft halo sphere
SPARK_RATE = 320.0                # particles / s
SPARK_LIFE = (0.3, 0.8)           # s (min, max)
SPARK_SPEED = 2.2                 # m/s along the reflected torch direction
SMOKE_FADE_IN = 0.4               # s
SMOKE_FADE_OUT = 1.5              # s
LASER_LENGTH = 0.060
LASER_WIDTH = 0.0015


# ------------------------------------------------------------------ helpers
def _collection(name, collection=None):
    """Return the target collection (create + link "name" to the scene if none is given)."""
    if collection is not None:
        return collection
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    if col.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(col)
    return col


def _link(ob, col):
    col.objects.link(ob)
    return ob


def _parent_local(child, parent, location=(0, 0, 0), rotation=None):
    """Parent with identity inverse so the child's local transform is expressed in the parent frame."""
    child.parent = parent
    child.matrix_parent_inverse = Matrix.Identity(4)
    child.location = location
    if rotation is not None:
        child.rotation_euler = rotation


def _mesh_object(name, bm, col, material=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    if material is not None:
        me.materials.append(material)
    return _link(ob, col)


def _uv_sphere(name, radius, col, material=None, segments=24, rings=12, scale=(1, 1, 1)):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius)
    if scale != (1, 1, 1):
        bmesh.ops.scale(bm, vec=Vector(scale), verts=bm.verts)
    return _mesh_object(name, bm, col, material)


def _quad(name, size_x, size_y, col, material=None):
    bm = bmesh.new()
    hx, hy = size_x / 2, size_y / 2
    v = [bm.verts.new((x, y, 0)) for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
    bm.faces.new(v)
    return _mesh_object(name, bm, col, material)


def _set_constant(idblock):
    """Make every keyframe on idblock's action constant-interpolated (hard on/off switching)."""
    ad = idblock.animation_data
    if ad is None or ad.action is None:
        return
    for fc in ad.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'CONSTANT'


def _key(idblock, path, frame, value, index=-1):
    """Set + keyframe a property by data path (supports custom props via '["name"]')."""
    if path.startswith('["'):
        idblock[path[2:-2]] = value
    elif index >= 0:
        v = getattr(idblock, path)
        v[index] = value
        setattr(idblock, path, v)
    else:
        setattr(idblock, path, value)
    idblock.keyframe_insert(path, frame=frame, index=index)


def _switch(idblock, path, intervals, on, off, index=-1, first_frame=1):
    """Keyframe a value that is ``on`` inside the inclusive intervals and ``off`` elsewhere."""
    starts = {s for s, _ in intervals}
    if first_frame not in starts:
        _key(idblock, path, first_frame, off, index)
    for s, e in intervals:
        if s - 1 >= first_frame:
            _key(idblock, path, s - 1, off, index)
        _key(idblock, path, s, on, index)
        _key(idblock, path, e, on, index)
        _key(idblock, path, e + 1, off, index)
    _set_constant(idblock)


def _flicker_series(intervals, seed=7):
    """Yield (frame, factor) with factor in [1-ARC_FLICKER, 1+ARC_FLICKER]: a ~10 Hz sinusoid plus noise."""
    rng = random.Random(seed)
    for s, e in intervals:
        for f in range(s, e + 1):
            t = f / L.FPS
            r = 0.6 * math.sin(2 * math.pi * ARC_FLICKER_HZ * t) + 0.5 * rng.uniform(-1, 1)
            r = max(-1.0, min(1.0, r))
            yield f, 1.0 + ARC_FLICKER * r


def _emission_material(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return m


# ------------------------------------------------------------------ materials
def _glow_material():
    """View-facing soft halo: bright blue-white centre fading to a warm transparent rim."""
    m = bpy.data.materials.new("vfx_arc_glow")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    lw = nt.nodes.new("ShaderNodeLayerWeight")          # Facing: 0 at centre, 1 at the rim
    lw.inputs["Blend"].default_value = 0.5
    inv = nt.nodes.new("ShaderNodeMath"); inv.operation = 'SUBTRACT'
    inv.inputs[0].default_value = 1.0
    nt.links.new(lw.outputs["Facing"], inv.inputs[1])
    pw = nt.nodes.new("ShaderNodeMath"); pw.operation = 'POWER'; pw.inputs[1].default_value = 2.2
    nt.links.new(inv.outputs[0], pw.inputs[0])
    alpha = nt.nodes.new("ShaderNodeMath"); alpha.operation = 'POWER'; alpha.inputs[1].default_value = 2.0
    nt.links.new(inv.outputs[0], alpha.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0; cr.elements[0].color = (1.0, 0.4, 0.1, 1)     # rim: warm orange
    e = cr.elements.new(0.45); e.color = (1.0, 0.75, 0.5, 1)
    cr.elements[-1].position = 1.0; cr.elements[-1].color = (0.7, 0.85, 1.0, 1)   # centre: blue-white
    nt.links.new(pw.outputs[0], ramp.inputs["Fac"])
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    st = nt.nodes.new("ShaderNodeMath"); st.operation = 'MULTIPLY_ADD'
    st.inputs[1].default_value = 40.0; st.inputs[2].default_value = 1.0        # dim warm rim, hot centre
    nt.links.new(pw.outputs[0], st.inputs[0])
    nt.links.new(st.outputs[0], em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(alpha.outputs[0], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.surface_render_method = 'BLENDED'
    m.show_transparent_back = False
    m.use_backface_culling = True
    return m


def _drive_world_location(nt, target):
    """Three Value nodes driven by ``target``'s world location (X, Y, Z) — usable inside EEVEE shaders."""
    outs = []
    for axis in "XYZ":
        v = nt.nodes.new("ShaderNodeValue"); v.name = f"vfx_arc_{axis}"
        drv = v.outputs[0].driver_add("default_value").driver
        drv.type = 'SCRIPTED'
        var = drv.variables.new(); var.name = "loc"; var.type = 'TRANSFORMS'
        var.targets[0].id = target
        var.targets[0].transform_type = f"LOC_{axis}"
        var.targets[0].transform_space = 'WORLD_SPACE'
        drv.expression = "loc"
        outs.append(v.outputs[0])
    return outs


def _spark_material(arc_empty):
    """Hot emissive streak that cools from white-yellow to dark red as it flies away from the arc.

    EEVEE does not evaluate the Particle Info node, so the "age" is approximated by the distance
    from the arc (sparks travel ~2 m/s, so distance ~ age) and per-particle variation comes from
    Object Info -> Random, which EEVEE does provide for instances.
    """
    m = bpy.data.materials.new("vfx_spark")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    ax, ay, az = _drive_world_location(nt, arc_empty)
    arc = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(ax, arc.inputs["X"]); nt.links.new(ay, arc.inputs["Y"]); nt.links.new(az, arc.inputs["Z"])
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    dist = nt.nodes.new("ShaderNodeVectorMath"); dist.operation = 'DISTANCE'
    nt.links.new(geo.outputs["Position"], dist.inputs[0]); nt.links.new(arc.outputs[0], dist.inputs[1])
    age = nt.nodes.new("ShaderNodeMath"); age.operation = 'DIVIDE'; age.use_clamp = True
    age.inputs[1].default_value = 0.75                      # ~fully cooled 0.75 m from the arc
    nt.links.new(dist.outputs["Value"], age.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0; cr.elements[0].color = (1.0, 0.9, 0.6, 1)
    e = cr.elements.new(0.3); e.color = (1.0, 0.55, 0.12, 1)
    e = cr.elements.new(0.7); e.color = (0.85, 0.2, 0.03, 1)
    cr.elements[-1].position = 1.0; cr.elements[-1].color = (0.35, 0.05, 0.0, 1)
    nt.links.new(age.outputs[0], ramp.inputs["Fac"])
    # strength = (1 - age)^1.2 * 3 * (0.6 + 0.8 * random) + 0.3  (kept low: AgX bleaches strong emitters)
    inv = nt.nodes.new("ShaderNodeMath"); inv.operation = 'SUBTRACT'; inv.inputs[0].default_value = 1.0
    nt.links.new(age.outputs[0], inv.inputs[1])
    pw = nt.nodes.new("ShaderNodeMath"); pw.operation = 'POWER'; pw.inputs[1].default_value = 1.2
    nt.links.new(inv.outputs[0], pw.inputs[0])
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    rnd = nt.nodes.new("ShaderNodeMath"); rnd.operation = 'MULTIPLY_ADD'
    rnd.inputs[1].default_value = 0.8; rnd.inputs[2].default_value = 0.6
    nt.links.new(oi.outputs["Random"], rnd.inputs[0])
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = 'MULTIPLY'
    nt.links.new(pw.outputs[0], mul.inputs[0]); nt.links.new(rnd.outputs[0], mul.inputs[1])
    st = nt.nodes.new("ShaderNodeMath"); st.operation = 'MULTIPLY_ADD'
    st.inputs[1].default_value = 3.0; st.inputs[2].default_value = 0.3
    nt.links.new(mul.outputs[0], st.inputs[0])
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    nt.links.new(st.outputs[0], em.inputs["Strength"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return m


def _smoke_material():
    """Principled Volume plume: soft column mask (object space) x drifting world-space noise x
    the object's keyframed "smoke" property (fade in/out)."""
    m = bpy.data.materials.new("vfx_smoke")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (0.18, 0.18, 0.19, 1)
    vol.inputs["Anisotropy"].default_value = 0.35
    nt.links.new(vol.outputs[0], out.inputs["Volume"])

    def math(op, a, b=None, c=None, clamp=False):
        n = nt.nodes.new("ShaderNodeMath"); n.operation = op; n.use_clamp = clamp
        for k, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[k].default_value = v
            else:
                nt.links.new(v, n.inputs[k])
        return n.outputs[0]

    # object-space coordinates of the unit box (mesh spans -0.5..0.5); z=-0.5 is the arc end.
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    h = math('ADD', sep.outputs["Z"], 0.5)                    # 0 at the bottom .. 1 at the top
    # column radius grows with height: r_max = 0.08 + 0.32*h ; radial mask = 1 - (r/r_max)^2
    r2 = math('ADD', math('POWER', sep.outputs["X"], 2.0), math('POWER', sep.outputs["Y"], 2.0))
    rmax = math('MULTIPLY_ADD', h, 0.32, 0.08)
    radial = math('SUBTRACT', 1.0, math('DIVIDE', r2, math('POWER', rmax, 2.0)), clamp=True)
    # vertical envelope: quick rise from the bottom, fade to nothing at the top
    vert = math('MULTIPLY', math('MINIMUM', math('DIVIDE', h, 0.15), 1.0), math('SUBTRACT', 1.0, math('POWER', h, 1.5)), clamp=True)
    mask = math('MULTIPLY', radial, vert)
    # drifting wisps: world-space noise moving upward with time (driver on the Value node)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    t = nt.nodes.new("ShaderNodeValue"); t.name = "vfx_time"
    drv = t.outputs[0].driver_add("default_value").driver
    drv.type = 'SCRIPTED'
    drv.expression = "frame"
    rise = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(math('MULTIPLY', t.outputs[0], -0.3 / L.FPS), rise.inputs["Z"])   # 0.3 m/s upward drift
    wobble = math('MULTIPLY', t.outputs[0], 0.03 / L.FPS)
    nt.links.new(wobble, rise.inputs["X"])
    vm = nt.nodes.new("ShaderNodeVectorMath"); vm.operation = 'ADD'
    nt.links.new(geo.outputs["Position"], vm.inputs[0]); nt.links.new(rise.outputs[0], vm.inputs[1])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 12.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.55
    nt.links.new(vm.outputs[0], noise.inputs["Vector"])
    wisps = math('SUBTRACT', math('MULTIPLY', noise.outputs["Fac"], 2.6), 0.95, clamp=True)
    # keyframed on/off + fade from the box object's "smoke" property
    attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_type = 'OBJECT'; attr.attribute_name = "smoke"
    dens = math('MULTIPLY', math('MULTIPLY', mask, wisps), math('MULTIPLY', attr.outputs["Fac"], 30.0))
    nt.links.new(dens, vol.inputs["Density"])
    return m


# ------------------------------------------------------------------ public: build
def build(arc_empty, weld_intervals, collection=None):
    """Create all arc effects around ``arc_empty``.

    Returns dict(light, core, glow, spark_emitter, spark_instance, spark_systems, smoke, collection).
    """
    col = _collection("VFX", collection)
    scene = bpy.context.scene
    intervals = [(int(s), int(e)) for s, e in weld_intervals if e >= s]

    # --- arc point light (slightly back toward the torch so it never sits inside the surface)
    ld = bpy.data.lights.new("ArcLight", 'POINT')
    ld.color = ARC_LIGHT_COLOR
    ld.energy = 0.0
    ld.shadow_soft_size = 0.004
    ld.use_shadow = True
    ld.volume_factor = 0.004          # a hint of glow in the fume plume, not a white block
    ld.use_custom_distance = True
    ld.cutoff_distance = ARC_LIGHT_RANGE
    light = _link(bpy.data.objects.new("ArcLight", ld), col)
    _parent_local(light, arc_empty, location=(0, 0, -0.004))
    _switch(ld, "energy", intervals, ARC_LIGHT_POWER, 0.0)
    for f, k in _flicker_series(intervals, seed=11):
        _key(ld, "energy", f, ARC_LIGHT_POWER * k)
    _set_constant(ld)

    # --- arc core + soft glow
    core = _uv_sphere("ArcCore", ARC_CORE_RADIUS, col, _emission_material("vfx_arc_core", (0.75, 0.87, 1.0), 120.0))
    _parent_local(core, arc_empty, location=(0, 0, -0.002))
    core.visible_shadow = False
    glow = _uv_sphere("ArcGlow", 1.0, col, _glow_material(), segments=32, rings=16)
    _parent_local(glow, arc_empty, location=(0, 0, -0.004))
    glow.visible_shadow = False
    for ob in (core, glow):
        _switch(ob, "scale", intervals, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    for f, k in _flicker_series(intervals, seed=23):
        s = ARC_GLOW_RADIUS * (0.85 + 0.6 * (k - 1.0 + ARC_FLICKER) / (2 * ARC_FLICKER))
        _key(glow, "scale", f, (s, s, s))
    _set_constant(glow)

    # --- sparks: emitter icosphere + one particle system per weld interval
    spark_mat = _spark_material(arc_empty)
    spark = _uv_sphere("SparkStreak", 1.0, col, spark_mat, segments=8, rings=6, scale=(4.5, 0.16, 0.16))
    spark.location = (0, 0, -50.0)    # the instance source itself stays out of every shot
    spark.visible_shadow = False
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.004)
    emitter = _mesh_object("SparkEmitter", bm, col)
    _parent_local(emitter, arc_empty)
    emitter.show_instancer_for_render = False
    emitter.show_instancer_for_viewport = False
    emitter.visible_shadow = False
    systems = []
    for i, (s, e) in enumerate(intervals):
        emitter.modifiers.new(f"sparks_{i}", 'PARTICLE_SYSTEM')
        psys = emitter.particle_systems[-1]
        psys.seed = 100 + i
        ps = psys.settings
        ps.name = f"vfx_sparks_{i}"
        ps.type = 'EMITTER'
        ps.count = max(1, int(SPARK_RATE * (e - s + 1) / L.FPS))
        ps.frame_start, ps.frame_end = s, e
        ps.lifetime = 0.5 * (SPARK_LIFE[0] + SPARK_LIFE[1]) * L.FPS
        ps.lifetime_random = (SPARK_LIFE[1] - SPARK_LIFE[0]) / (SPARK_LIFE[0] + SPARK_LIFE[1])
        ps.emit_from = 'FACE'
        ps.distribution = 'RAND'
        ps.use_emit_random = True
        # velocity: mostly back along -Z of the torch frame (reflected off the part), spread outward
        ps.normal_factor = 0.7
        ps.object_align_factor = (0.0, 0.0, -SPARK_SPEED)
        ps.factor_random = 1.1
        ps.object_factor = 0.2
        ps.physics_type = 'NEWTON'
        ps.mass = 0.001
        ps.drag_factor = 0.06
        ps.effector_weights.gravity = 1.0
        ps.timestep = 1.0 / L.FPS
        # render as instanced streaks aligned with their velocity
        ps.render_type = 'OBJECT'
        ps.instance_object = spark
        ps.particle_size = 0.006
        ps.size_random = 0.6
        ps.use_rotations = True
        ps.rotation_mode = 'VEL'
        ps.use_dynamic_rotation = True
        ps.use_rotation_instance = False
        ps.show_unborn = False
        ps.use_dead = False
        ps.display_method = 'DOT'
        systems.append(psys)

    # --- fume plume: small volume box hovering above the arc (Copy Location, stays upright)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    smoke = _mesh_object("FumePlume", bm, col)
    smoke.scale = (0.45, 0.45, 0.6)
    smoke.location = (0, 0, 0.26)     # offset added to the arc position (box bottom ~ 0.04 below the arc)
    con = smoke.constraints.new('COPY_LOCATION')
    con.target = arc_empty
    con.use_offset = True
    smoke.visible_shadow = False
    smoke.data.materials.append(_smoke_material())
    smoke["smoke"] = 0.0
    _key(smoke, '["smoke"]', 1, 0.0)
    fi, fo = max(1, int(SMOKE_FADE_IN * L.FPS)), max(1, int(SMOKE_FADE_OUT * L.FPS))
    for s, e in intervals:
        _key(smoke, '["smoke"]', s, 0.0)
        _key(smoke, '["smoke"]', s + fi, 1.0)
        _key(smoke, '["smoke"]', e, 1.0)
        _key(smoke, '["smoke"]', e + fo, 0.0)
    # (linear interpolation here on purpose: the fades)
    # EEVEE volumetrics: coarse froxels are plenty for a soft plume and cost almost nothing on CPU
    scene.eevee.volumetric_tile_size = '16'
    scene.eevee.volumetric_samples = 48
    scene.eevee.volumetric_start = 0.2
    scene.eevee.volumetric_end = 15.0
    scene.eevee.use_volumetric_shadows = False

    return dict(light=light, core=core, glow=glow, spark_emitter=emitter, spark_instance=spark,
                spark_systems=systems, smoke=smoke, collection=col)


# ------------------------------------------------------------------ public: particles
def bake_particles(scene):
    """Fill the spark particle caches so an arbitrary single frame renders correctly.

    Tries the point-cache bake operator, else steps through the frames (cheap, few thousand particles).
    """
    try:
        with bpy.context.temp_override(scene=scene):
            bpy.ops.ptcache.bake_all(bake=True)
    except Exception:
        cur = scene.frame_current
        for f in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(f)
        scene.frame_set(cur)


# ------------------------------------------------------------------ public: compositor
def setup_compositor(scene):
    """Bloom on the arc / hot bead plus a subtle vignette. Cheap: two nodes and a blurred mask."""
    scene.use_nodes = True
    scene.render.use_compositing = True
    nt = scene.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers"); rl.location = (-500, 0)
    glare = nt.nodes.new("CompositorNodeGlare"); glare.location = (-200, 0)
    glare.glare_type = 'BLOOM'
    glare.quality = 'MEDIUM'
    glare.inputs["Threshold"].default_value = 1.5
    glare.inputs["Strength"].default_value = 0.22
    glare.inputs["Size"].default_value = 0.35
    glare.inputs["Saturation"].default_value = 1.0
    nt.links.new(rl.outputs["Image"], glare.inputs["Image"])
    mask = nt.nodes.new("CompositorNodeEllipseMask"); mask.location = (-500, -300)
    mask.mask_width = 1.25
    mask.mask_height = 1.05
    blur = nt.nodes.new("CompositorNodeBlur"); blur.location = (-250, -300)
    blur.filter_type = 'FAST_GAUSS'
    blur.use_relative = True
    blur.factor_x = 30.0
    blur.factor_y = 30.0
    nt.links.new(mask.outputs["Mask"], blur.inputs["Image"])
    mix = nt.nodes.new("CompositorNodeMixRGB"); mix.location = (100, 0)
    mix.blend_type = 'MULTIPLY'
    mix.inputs["Fac"].default_value = 0.45
    nt.links.new(glare.outputs["Image"], mix.inputs[1])
    nt.links.new(blur.outputs["Image"], mix.inputs[2])
    comp = nt.nodes.new("CompositorNodeComposite"); comp.location = (350, 0)
    nt.links.new(mix.outputs["Image"], comp.inputs["Image"])


# ------------------------------------------------------------------ public: laser line
def laser_line(sensor_empty, intervals, collection=None):
    """Red laser line (60 x 1.5 mm quad, 20 mm along the sensor's +Z) and a narrow red spot light.

    Both are on only inside ``intervals``.  Returns the quad; the spot light "LaserSpot" is parented
    to ``sensor_empty`` next to it.
    """
    col = _collection("VFX", collection)
    intervals = [(int(s), int(e)) for s, e in intervals if e >= s]
    quad = _quad("LaserLine", LASER_LENGTH, LASER_WIDTH, col, _emission_material("vfx_laser", (1.0, 0.02, 0.01), 40.0))
    _parent_local(quad, sensor_empty, location=(0, 0, 0.02))
    quad.visible_shadow = False
    _switch(quad, "scale", intervals, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    ld = bpy.data.lights.new("LaserSpot", 'SPOT')
    ld.color = (1.0, 0.03, 0.01)
    ld.energy = 0.0
    ld.spot_size = math.radians(6.0)
    ld.spot_blend = 0.4
    ld.shadow_soft_size = 0.002
    ld.use_shadow = False
    spot = _link(bpy.data.objects.new("LaserSpot", ld), col)
    # spot lights shine along their local -Z: flip so it shines along the sensor's +Z (into the part)
    _parent_local(spot, sensor_empty, location=(0, 0, 0.0), rotation=(math.pi, 0, 0))
    _switch(ld, "energy", intervals, 12.0, 0.0)
    return quad
