"""Logos of the editions (editions.py "brand"): the montage overlays (title, corner, end card) and the plates / screen
headers inside the 3D scene.

    import branding2 as BR
    BR.set_brand("atomix")                      # build2.build_scene does this from the edition
    img = BR.logo_rgba(width=400)               # PIL RGBA, transparent background, brand colour
    img = BR.logo_plate(height=120)             # the logo on a white rounded plate (reads on any background)
    BR.paste_logo(pil_img, (x0, y0, x1, y1), plate=True)    # fit into a box of an HMI texture
    mat = BR.logo_material("qc_logo")           # bpy material with the plate texture (bpy imported lazily)
    ob = BR.logo_plate_object("stn_logo", width=0.30, matrix=M, parent=root)

The logos are cleaned, upscaled copies of the files the customer sent (assets/logos/*_source.jpg, cleaned by
assets/logos/clean_logo.py): flat brand colour on a transparent background.  Nothing here recolours or redraws them.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOGOS = os.path.join(HERE, "assets", "logos")

BRANDS = {
    "pigrupp": dict(name="ПИГРУПП", logo=os.path.join(LOGOS, "pigrupp.png"), color=(0xDA, 0x05, 0x06),
                    source=os.path.join(LOGOS, "pigrupp_source.jpg")),
    "atomix": dict(name="ATOMIX", logo=os.path.join(LOGOS, "atomix.png"), color=(0xE8, 0x1F, 0x63),
                   source=os.path.join(LOGOS, "atomix_source.jpg")),
}
PLATE_BG = (255, 255, 255, 255)
_BRAND = "pigrupp"
_CACHE = {}


def set_brand(brand):
    global _BRAND
    if brand not in BRANDS:
        raise ValueError(f"brand {brand!r} not in {list(BRANDS)}")
    _BRAND = brand


def get_brand():
    return _BRAND


def info(brand=None):
    return dict(BRANDS[brand or _BRAND], key=brand or _BRAND)


def _master(brand):
    from PIL import Image
    if brand not in _CACHE:
        _CACHE[brand] = Image.open(BRANDS[brand]["logo"]).convert("RGBA")
    return _CACHE[brand]


def logo_rgba(brand=None, width=None, height=None):
    """The logo (RGBA, transparent background) scaled to fit inside width x height (either may be None)."""
    from PIL import Image
    im = _master(brand or _BRAND)
    ar = im.width / im.height
    if width is None and height is None:
        return im.copy()
    if width is None or (height is not None and width / height > ar):
        w, h = max(1, round(height * ar)), max(1, round(height))
    else:
        w, h = max(1, round(width)), max(1, round(width / ar))
    return im.resize((w, h), Image.LANCZOS)


def logo_plate(brand=None, width=None, height=None, pad=0.16, radius=0.22, bg=PLATE_BG, outline=None):
    """The logo on a rounded plate.  width / height = plate size (either may be None, the other follows the logo
    aspect); pad = margin around the logo as a fraction of the plate height, radius = corner radius (same unit)."""
    from PIL import Image, ImageDraw
    im = _master(brand or _BRAND)
    ar = im.width / im.height
    k = ar * (1 - 2 * pad) + 2 * pad           # plate aspect: logo height = ph (1 - 2 pad), margins pad * ph
    if width is None and height is None:
        height = 120
    if height is None:
        height = width / k
    pw = int(round(width if width is not None else height * k))
    ph = int(round(height))
    m = pad * ph
    # transparent corners carry the plate colour: a texture filter (bpy) blends the rim toward white, not black
    plate = Image.new("RGBA", (pw, ph), (*bg[:3], 0))
    d = ImageDraw.Draw(plate)
    d.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=int(radius * ph), fill=bg,
                        outline=outline, width=max(1, ph // 60) if outline else 0)
    lg = logo_rgba(brand, width=pw - 2 * m, height=ph - 2 * m)
    plate.alpha_composite(lg, (int((pw - lg.width) / 2), int((ph - lg.height) / 2)))
    return plate


def paste_logo(img, box, brand=None, plate=False, **plate_kw):
    """Fit the logo (or the logo plate) into box = (x0, y0, x1, y1) of a PIL image, centred; returns the pasted size."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    bw, bh = x1 - x0, y1 - y0
    if plate:
        im = _master(brand or _BRAND)
        pad = plate_kw.get("pad", 0.16)
        ar_plate = (im.width / im.height) * (1 - 2 * pad) + 2 * pad
        h = min(bh, bw / ar_plate)
        lg = logo_plate(brand, height=h, **plate_kw)
    else:
        lg = logo_rgba(brand, width=bw, height=bh)
    ox, oy = x0 + (bw - lg.width) // 2, y0 + (bh - lg.height) // 2
    if img.mode == "RGBA":
        img.alpha_composite(lg, (ox, oy))
    else:
        img.paste(lg, (ox, oy), lg)
    return lg.size


# ----------------------------------------------------------------------------- Blender (lazy bpy)
def logo_image(name, brand=None, height_px=512, plate=True):
    """bpy image (packed) of the logo plate (or the bare logo on transparency)."""
    import bpy
    import numpy as np
    bi = bpy.data.images.get(name)
    if bi is not None:
        return bi
    im = logo_plate(brand, height=height_px) if plate else logo_rgba(brand, height=height_px)
    arr = np.asarray(im, dtype=np.float32)[::-1] / 255.0
    bi = bpy.data.images.new(name, im.width, im.height, alpha=True)
    bi.colorspace_settings.name = 'sRGB'
    bi.pixels.foreach_set(arr.ravel())
    bi.pack()
    return bi


def logo_material(name, brand=None, height_px=512, plate=True, roughness=0.35, emission=0.0):
    """Material with the logo texture, alpha-blended (the plate's rounded corners are transparent as well as the bare
    logo's background).  emission > 0 for screens."""
    import bpy
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    bi = logo_image(name + "_img", brand, height_px, plate)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bs = nt.nodes["Principled BSDF"]
    bs.inputs["Roughness"].default_value = roughness
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bi
    tex.interpolation = 'Cubic'
    tex.extension = 'CLIP'
    nt.links.new(tex.outputs["Color"], bs.inputs["Base Color"])
    # alpha for the plate too: without it the transparent rounded corners rendered black
    nt.links.new(tex.outputs["Alpha"], bs.inputs["Alpha"])
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = 'DITHERED'
    if emission > 0:
        nt.links.new(tex.outputs["Color"], bs.inputs["Emission Color"])
        bs.inputs["Emission Strength"].default_value = emission
    return m


def logo_aspect(brand=None, plate=True, pad=0.16):
    im = _master(brand or _BRAND)
    ar = im.width / im.height
    return ar * (1 - 2 * pad) + 2 * pad if plate else ar


def logo_plate_object(name, width, matrix, parent=None, brand=None, collection=None, plate=True, thickness=0.0):
    """A quad (width x width/aspect, in its local XY plane, facing local +Z) with the logo, placed by the 4x4
    ``matrix`` (world, or parent space when ``parent`` is given).  thickness > 0 adds a thin backing box."""
    import bpy
    import bmesh
    from mathutils import Matrix
    h = width / logo_aspect(brand, plate)
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    vs = [bm.verts.new((x, y, 0.0)) for x, y in ((-width / 2, -h / 2), (width / 2, -h / 2), (width / 2, h / 2), (-width / 2, h / 2))]
    f = bm.faces.new(vs)
    uv = bm.loops.layers.uv.new("UVMap")
    for loop, (u, v) in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
        loop[uv].uv = (u, v)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(logo_material(f"logo_{brand or _BRAND}_{'plate' if plate else 'bare'}", brand, plate=plate))
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if parent is not None:
        ob.parent = parent
    ob.matrix_basis = Matrix([list(r) for r in matrix]) if not isinstance(matrix, Matrix) else matrix
    if thickness > 0:
        bk = bpy.data.meshes.new(name + "_back")
        bm = bmesh.new()
        import bmesh.ops as bo
        bo.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co.x *= width * 1.02
            v.co.y *= h * 1.02
            v.co.z = (v.co.z - 0.5) * thickness - 0.0005
        bm.to_mesh(bk)
        bm.free()
        mat = bpy.data.materials.get("logo_backing") or bpy.data.materials.new("logo_backing")
        mat.diffuse_color = (0.9, 0.9, 0.9, 1.0)
        bk.materials.append(mat)
        back = bpy.data.objects.new(name + "_back", bk)
        (collection or bpy.context.scene.collection).objects.link(back)
        back.parent = ob
    return ob
