"""Procedural materials (EEVEE-friendly). Use materials.get(name) — created once, cached."""
import math
import bpy

_BUILDERS = {}


def register(name):
    def deco(fn):
        _BUILDERS[name] = fn
        return fn
    return deco


def get(name, **kw):
    key = name if not kw else name + "|" + "|".join(f"{k}={v}" for k, v in sorted(kw.items()))
    m = bpy.data.materials.get(key)
    if m is None:
        m = _BUILDERS[name](**kw)
        m.name = key
    return m


def _base(name, color=(0.8, 0.8, 0.8), metallic=0.0, roughness=0.5, coat=0.0, spec=0.5, emission=None, emission_strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Specular IOR Level"].default_value = spec
    if emission is not None:
        b.inputs["Emission Color"].default_value = (*emission, 1.0)
        b.inputs["Emission Strength"].default_value = emission_strength
    return m


def _srgb(hexstr):
    """'#FF6600' -> linear rgb tuple."""
    h = hexstr.lstrip('#')
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return tuple(out)


def _noise_variation(m, scale=8.0, strength=0.12, rough_var=0.15, coord='OBJECT'):
    """Multiply base colour by a noise (dirt/dust) and vary roughness."""
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.6
    nt.links.new(tc.outputs["Object" if coord == 'OBJECT' else "Generated"], noise.inputs["Vector"])
    base_col = b.inputs["Base Color"].default_value[:]
    rgb = nt.nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = base_col
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = 'RGBA'
    mix.blend_type = 'MULTIPLY'
    mix.inputs["Factor"].default_value = strength
    nt.links.new(rgb.outputs[0], mix.inputs[6])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (0.55, 0.5, 0.45, 1)
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix.inputs[7])
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    if rough_var > 0:
        r0 = b.inputs["Roughness"].default_value
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = 0.0
        mr.inputs["From Max"].default_value = 1.0
        mr.inputs["To Min"].default_value = max(0.0, r0 - rough_var)
        mr.inputs["To Max"].default_value = min(1.0, r0 + rough_var)
        nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
    return m


# ------------------------------------------------------------------ robot / machines
@register("abb_orange")
def _abb_orange():
    m = _base("abb_orange", _srgb("#FF6A13"), metallic=0.0, roughness=0.32, coat=0.35, spec=0.6)
    return _noise_variation(m, scale=6, strength=0.08, rough_var=0.08)


@register("abb_grey")
def _abb_grey():
    return _noise_variation(_base("abb_grey", _srgb("#8C8F93"), metallic=0.1, roughness=0.45), strength=0.1)


@register("kuka_orange")
def _kuka_orange():
    return _base("kuka_orange", _srgb("#FF5800"), roughness=0.35, coat=0.3)


@register("dark_metal")
def _dark_metal():
    return _base("dark_metal", (0.03, 0.03, 0.033), metallic=0.85, roughness=0.45)


@register("black_plastic")
def _black_plastic():
    return _base("black_plastic", (0.015, 0.015, 0.017), metallic=0.0, roughness=0.55)


@register("brass")
def _brass():
    return _base("brass", (0.83, 0.62, 0.28), metallic=1.0, roughness=0.3)


@register("copper")
def _copper():
    return _base("copper", (0.85, 0.45, 0.3), metallic=1.0, roughness=0.35)


@register("machined_steel")
def _machined_steel():
    return _base("machined_steel", (0.62, 0.62, 0.64), metallic=1.0, roughness=0.28)


@register("galvanized")
def _galvanized():
    return _noise_variation(_base("galvanized", (0.55, 0.57, 0.6), metallic=0.9, roughness=0.5), scale=25, strength=0.25, rough_var=0.15)


@register("painted")
def _painted(color="#3A5A8C", roughness=0.4, coat=0.1):
    m = _base("painted", _srgb(color), metallic=0.0, roughness=roughness, coat=coat)
    return _noise_variation(m, scale=5, strength=0.12, rough_var=0.1)


@register("safety_yellow")
def _safety_yellow():
    return _noise_variation(_base("safety_yellow", _srgb("#F2B400"), roughness=0.45), strength=0.1)


@register("safety_red")
def _safety_red():
    return _base("safety_red", _srgb("#C8102E"), roughness=0.45)


@register("rubber")
def _rubber():
    return _base("rubber", (0.02, 0.02, 0.02), roughness=0.75)


@register("cable_black")
def _cable_black():
    return _base("cable_black", (0.02, 0.02, 0.022), roughness=0.6, spec=0.3)


@register("glass_dark")
def _glass_dark():
    m = _base("glass_dark", (0.02, 0.02, 0.03), roughness=0.05)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Transmission Weight"].default_value = 0.6
    return m


@register("emissive")
def _emissive(color="#FFFFFF", strength=5.0):
    m = _base("emissive", (0, 0, 0), emission=_srgb(color), emission_strength=strength)
    return m


# ------------------------------------------------------------------ pipe steel
@register("steel_pipe")
def _steel_pipe():
    """Hot-rolled carbon steel with mill scale and rust patches (as in the reference photos)."""
    m = _base("steel_pipe", (0.075, 0.062, 0.055), metallic=0.5, roughness=0.62)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise"); n1.inputs["Scale"].default_value = 3.0; n1.inputs["Detail"].default_value = 6
    n2 = nt.nodes.new("ShaderNodeTexNoise"); n2.inputs["Scale"].default_value = 40.0; n2.inputs["Detail"].default_value = 3
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.42; cr.elements[0].color = (0.11, 0.095, 0.085, 1)     # dark mill scale
    e = cr.elements.new(0.55); e.color = (0.2, 0.17, 0.14, 1)                             # grey-brown
    cr.elements[-1].position = 0.72; cr.elements[-1].color = (0.36, 0.15, 0.06, 1)        # rust
    nt.links.new(n1.outputs["Fac"], ramp.inputs["Fac"])
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'; mix.inputs["Factor"].default_value = 0.6
    nt.links.new(ramp.outputs["Color"], mix.inputs[6])
    r2 = nt.nodes.new("ShaderNodeValToRGB"); r2.color_ramp.elements[0].position = 0.3; r2.color_ramp.elements[0].color = (0.45, 0.45, 0.45, 1)
    nt.links.new(n2.outputs["Fac"], r2.inputs["Fac"])
    nt.links.new(r2.outputs["Color"], mix.inputs[7])
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs["To Min"].default_value = 0.5; mr.inputs["To Max"].default_value = 0.8
    nt.links.new(n1.outputs["Fac"], mr.inputs["Value"]); nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.15; bump.inputs["Distance"].default_value = 0.002
    nt.links.new(n2.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


@register("bevel_steel")
def _bevel_steel():
    """Freshly machined weld bevel: bright metal."""
    return _base("bevel_steel", (0.6, 0.6, 0.62), metallic=1.0, roughness=0.35)


@register("concrete")
def _concrete():
    m = _base("concrete", (0.36, 0.35, 0.33), metallic=0.0, roughness=0.75, spec=0.35)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise"); n1.inputs["Scale"].default_value = 0.6; n1.inputs["Detail"].default_value = 8; n1.inputs["Roughness"].default_value = 0.7
    n2 = nt.nodes.new("ShaderNodeTexNoise"); n2.inputs["Scale"].default_value = 30; n2.inputs["Detail"].default_value = 4
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"]); nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.3; ramp.color_ramp.elements[0].color = (0.22, 0.21, 0.2, 1)
    ramp.color_ramp.elements[1].position = 0.75; ramp.color_ramp.elements[1].color = (0.5, 0.48, 0.45, 1)
    nt.links.new(n1.outputs["Fac"], ramp.inputs["Fac"])
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'; mix.inputs["Factor"].default_value = 0.25
    nt.links.new(ramp.outputs["Color"], mix.inputs[6])
    r2 = nt.nodes.new("ShaderNodeValToRGB"); r2.color_ramp.elements[0].color = (0.7, 0.7, 0.7, 1)
    nt.links.new(n2.outputs["Fac"], r2.inputs["Fac"]); nt.links.new(r2.outputs["Color"], mix.inputs[7])
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs["To Min"].default_value = 0.55; mr.inputs["To Max"].default_value = 0.85
    nt.links.new(n1.outputs["Fac"], mr.inputs["Value"]); nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.12; bump.inputs["Distance"].default_value = 0.01
    nt.links.new(n2.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


@register("wall_panel")
def _wall_panel():
    return _noise_variation(_base("wall_panel", (0.62, 0.63, 0.62), roughness=0.6), scale=0.4, strength=0.25, rough_var=0.1)


@register("fence_mesh")
def _fence_mesh(color="#F2B400", pitch=0.05, wire=0.004):
    """Welded-wire fence panel: brick texture with transparent 'bricks' and opaque 'mortar' (the wire)."""
    m = bpy.data.materials.new("fence_mesh")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*_srgb(color), 1)
    bsdf.inputs["Roughness"].default_value = 0.45
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mixs = nt.nodes.new("ShaderNodeMixShader")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.0
    brick.inputs["Scale"].default_value = 1.0 / pitch
    brick.inputs["Mortar Size"].default_value = wire / pitch * 0.5
    brick.inputs["Brick Width"].default_value = 1.0
    brick.inputs["Row Height"].default_value = 1.0
    brick.inputs["Color1"].default_value = (0, 0, 0, 1)
    brick.inputs["Color2"].default_value = (0, 0, 0, 1)
    brick.inputs["Mortar"].default_value = (1, 1, 1, 1)
    nt.links.new(tc.outputs["Object"], brick.inputs["Vector"])
    nt.links.new(brick.outputs["Fac"], mixs.inputs["Fac"])
    nt.links.new(tr.outputs[0], mixs.inputs[1])
    nt.links.new(bsdf.outputs[0], mixs.inputs[2])
    nt.links.new(mixs.outputs[0], out.inputs["Surface"])
    m.surface_render_method = 'DITHERED'
    m.use_backface_culling = False
    return m


# ------------------------------------------------------------------ weld bead with shader-driven progress
N_ARCS = 4


def _arc_mask(nt, u, i):
    """Nodes for one weld arc slot i: reads object props w{i}_start, w{i}_prog, w{i}_dir, w{i}_hot.

    Returns (welded_mask [0/1], heat [0..1]).  rel = floored_mod((u - start) * dir, 1)
    welded = rel < prog ; heat = clamp(1 - (prog - rel) / 0.07) * hot  (only when welded)
    """
    def attr(name):
        a = nt.nodes.new("ShaderNodeAttribute")
        a.attribute_type = 'OBJECT'
        a.attribute_name = name
        return a.outputs["Fac"]

    def math(op, a, b=None, c=None):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = False
        for k, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[k].default_value = v
            else:
                nt.links.new(v, n.inputs[k])
        return n.outputs[0]

    start, prog, dr, hot = attr(f"w{i}_start"), attr(f"w{i}_prog"), attr(f"w{i}_dir"), attr(f"w{i}_hot")
    d = math('SUBTRACT', u, start)
    d = math('MULTIPLY', d, dr)
    rel = math('FLOORED_MODULO', d, 1.0)
    welded = math('LESS_THAN', rel, prog)
    # heat behind the arc
    gap = math('SUBTRACT', prog, rel)
    heat = math('DIVIDE', gap, 0.07)
    heat = math('SUBTRACT', 1.0, heat)
    heat = math('MAXIMUM', heat, 0.0)
    heat = math('MINIMUM', heat, 1.0)
    heat = math('MULTIPLY', heat, welded)
    heat = math('MULTIPLY', heat, hot)
    return welded, heat


@register("weld_bead")
def _weld_bead():
    m = bpy.data.materials.new("weld_bead")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    out = nt.nodes["Material Output"]
    b.inputs["Base Color"].default_value = (0.5, 0.5, 0.52, 1)
    b.inputs["Metallic"].default_value = 0.9
    b.inputs["Roughness"].default_value = 0.38
    # angle around the object's local Z -> u in [0,1)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    at = nt.nodes.new("ShaderNodeMath"); at.operation = 'ARCTAN2'
    nt.links.new(sep.outputs["Y"], at.inputs[0]); nt.links.new(sep.outputs["X"], at.inputs[1])
    div = nt.nodes.new("ShaderNodeMath"); div.operation = 'DIVIDE'; div.inputs[1].default_value = 2 * math.pi
    nt.links.new(at.outputs[0], div.inputs[0])
    u = nt.nodes.new("ShaderNodeMath"); u.operation = 'FLOORED_MODULO'; u.inputs[1].default_value = 1.0
    nt.links.new(div.outputs[0], u.inputs[0])
    u = u.outputs[0]
    welded_any, heat_any = None, None
    for i in range(N_ARCS):
        w, h = _arc_mask(nt, u, i)
        if welded_any is None:
            welded_any, heat_any = w, h
        else:
            mx = nt.nodes.new("ShaderNodeMath"); mx.operation = 'MAXIMUM'
            nt.links.new(welded_any, mx.inputs[0]); nt.links.new(w, mx.inputs[1]); welded_any = mx.outputs[0]
            mh = nt.nodes.new("ShaderNodeMath"); mh.operation = 'MAXIMUM'
            nt.links.new(heat_any, mh.inputs[0]); nt.links.new(h, mh.inputs[1]); heat_any = mh.outputs[0]
    # alpha = welded
    nt.links.new(welded_any, b.inputs["Alpha"])
    # emission = heat^2 * hot colour
    p2 = nt.nodes.new("ShaderNodeMath"); p2.operation = 'POWER'; p2.inputs[1].default_value = 2.2
    nt.links.new(heat_any, p2.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0; cr.elements[0].color = (0, 0, 0, 1)
    e1 = cr.elements.new(0.35); e1.color = (0.6, 0.05, 0.0, 1)
    e2 = cr.elements.new(0.7); e2.color = (1.0, 0.35, 0.05, 1)
    cr.elements[-1].position = 1.0; cr.elements[-1].color = (1.0, 0.85, 0.5, 1)
    nt.links.new(p2.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Emission Color"])
    es = nt.nodes.new("ShaderNodeMath"); es.operation = 'MULTIPLY'; es.inputs[1].default_value = 25.0
    nt.links.new(p2.outputs[0], es.inputs[0]); nt.links.new(es.outputs[0], b.inputs["Emission Strength"])
    # ripples: wave texture along the angle
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = 'BANDS'; wave.bands_direction = 'X'
    wave.inputs["Scale"].default_value = 1.0
    wave.inputs["Distortion"].default_value = 1.5; wave.inputs["Detail"].default_value = 2
    sc = nt.nodes.new("ShaderNodeMath"); sc.operation = 'MULTIPLY'; sc.inputs[1].default_value = 140.0
    nt.links.new(u, sc.inputs[0])
    comb = nt.nodes.new("ShaderNodeCombineXYZ"); nt.links.new(sc.outputs[0], comb.inputs["X"]); nt.links.new(sep.outputs["Z"], comb.inputs["Y"])
    nt.links.new(comb.outputs[0], wave.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.45; bump.inputs["Distance"].default_value = 0.002
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    # colour: slightly darker/bluer heat tint near the ripples
    m.surface_render_method = 'DITHERED'
    m.use_transparent_shadow = True
    return m


def init_bead_props(ob):
    for i in range(N_ARCS):
        ob[f"w{i}_start"] = 0.0
        ob[f"w{i}_prog"] = 0.0
        ob[f"w{i}_dir"] = 1.0
        ob[f"w{i}_hot"] = 0.0
