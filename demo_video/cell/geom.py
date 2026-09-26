"""Mesh-building helpers (numpy -> bpy mesh) for axisymmetric and swept parts."""
import math
import bpy
import bmesh
import numpy as np
import mathutils


def new_object(name, verts, faces, collection=None, smooth=True, auto_smooth_angle=math.radians(35)):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(map(int, f)) for f in faces])
    me.update()
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
        # Blender 4.1+: smooth by angle via modifier-free attribute is gone; use the operator-less approach
        try:
            bpy.ops.object.select_all(action='DESELECT')
            ob.select_set(True)
            bpy.context.view_layer.objects.active = ob
            bpy.ops.object.shade_smooth_by_angle(angle=auto_smooth_angle)
        except Exception:
            pass
    return ob


def revolve(name, profile, segments=96, axis='Z', close=True, collection=None, smooth=True, cap=None):
    """Revolve a 2D profile [(r, h), ...] around the given axis.

    The profile is a polyline in the (radius, height) plane.  If close=True the
    polyline is treated as a closed loop (solid of revolution with wall).
    axis: 'Z' (default), 'X' or 'Y' — the height coordinate runs along this axis.
    """
    prof = np.array(profile, dtype=float)
    n = len(prof)
    verts = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        c, s = math.cos(a), math.sin(a)
        for r, h in prof:
            if axis == 'Z':
                verts.append((r * c, r * s, h))
            elif axis == 'X':
                verts.append((h, r * c, r * s))
            else:
                verts.append((r * s, h, r * c))
    faces = []
    m = n if close else n - 1
    for i in range(segments):
        i2 = (i + 1) % segments
        for k in range(m):
            k2 = (k + 1) % n
            a = i * n + k
            b = i2 * n + k
            c_ = i2 * n + k2
            d = i * n + k2
            # skip degenerate (r==0) quads by still emitting; Blender handles zero-area tris poorly, so filter
            if prof[k][0] < 1e-9 and prof[k2][0] < 1e-9:
                continue
            if prof[k][0] < 1e-9:
                faces.append((a, c_, d))
            elif prof[k2][0] < 1e-9:
                faces.append((a, b, d))
            else:
                faces.append((a, b, c_, d))
    return new_object(name, verts, faces, collection, smooth)


def torus_sweep(name, bend_radius, bend_angle, tube_radius_fn, seg_bend=48, seg_tube=64,
                collection=None, smooth=True):
    """Sweep circle(s) along a circular arc (elbow).

    Local frame: arc lies in the XZ plane, starts at origin heading +Z, bending
    toward +X, centre of bend at (bend_radius, 0, 0).  tube_radius_fn(t) returns a
    list of radii (multi-wall: e.g. [r_outer, r_inner]) for t in [0,1].
    Returns object whose mesh is a set of tubes (outer & inner) — end caps are
    added by connecting rings so the wall thickness is visible at the ends.
    """
    verts = []
    rings = []  # rings[i] = list of vertex index start per radius
    for i in range(seg_bend + 1):
        t = i / seg_bend
        th = bend_angle * t
        # centre point on arc and local frame
        cx = bend_radius * (1 - math.cos(th))
        cz = bend_radius * math.sin(th)
        # tangent direction
        tx, tz = math.sin(th), math.cos(th)
        # normal in XZ plane pointing away from bend centre: (-cos th, 0, sin th) points... compute
        nx, nz = -math.cos(th), math.sin(th)
        radii = tube_radius_fn(t)
        starts = []
        for r in radii:
            starts.append(len(verts))
            for k in range(seg_tube):
                a = 2 * math.pi * k / seg_tube
                # circle basis: n (in-plane) and y
                px = cx + r * (math.cos(a) * nx)
                py = r * math.sin(a)
                pz = cz + r * (math.cos(a) * nz)
                verts.append((px, py, pz))
        rings.append(starts)
    faces = []
    nr = len(rings[0])
    for i in range(seg_bend):
        for ri in range(nr):
            s0, s1 = rings[i][ri], rings[i + 1][ri]
            for k in range(seg_tube):
                k2 = (k + 1) % seg_tube
                if ri == 0:
                    faces.append((s0 + k, s0 + k2, s1 + k2, s1 + k))
                else:  # inner wall: flip winding
                    faces.append((s0 + k, s1 + k, s1 + k2, s0 + k2))
    # end caps between outer and inner rings
    if nr >= 2:
        for i in (0, seg_bend):
            so, si = rings[i][0], rings[i][1]
            for k in range(seg_tube):
                k2 = (k + 1) % seg_tube
                if i == 0:
                    faces.append((so + k, si + k, si + k2, so + k2))
                else:
                    faces.append((so + k, so + k2, si + k2, si + k))
    return new_object(name, verts, faces, collection, smooth)


def box(name, size, location=(0, 0, 0), collection=None, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    ob = bpy.context.object
    ob.name = name
    ob.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if collection is not None:
        for c in ob.users_collection:
            c.objects.unlink(ob)
        collection.objects.link(ob)
    if bevel > 0:
        m = ob.modifiers.new("bevel", 'BEVEL')
        m.width = bevel
        m.segments = 3
        m.limit_method = 'ANGLE'
    return ob


def cylinder(name, radius, depth, location=(0, 0, 0), rotation=(0, 0, 0), vertices=48, collection=None, smooth=True):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location, rotation=rotation, vertices=vertices)
    ob = bpy.context.object
    ob.name = name
    if smooth:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))
    if collection is not None:
        for c in ob.users_collection:
            c.objects.unlink(ob)
        collection.objects.link(ob)
    return ob


def set_parent(child, parent, keep_world=True):
    bpy.context.view_layer.update()
    if keep_world:
        mw = child.matrix_world.copy()
        child.parent = parent
        child.matrix_parent_inverse = parent.matrix_world.inverted()
        child.matrix_world = mw
    else:
        child.parent = parent
        child.matrix_parent_inverse = mathutils.Matrix.Identity(4)


def empty(name, matrix=None, collection=None, size=0.1, display='PLAIN_AXES'):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = display
    ob.empty_display_size = size
    (collection or bpy.context.scene.collection).objects.link(ob)
    if matrix is not None:
        ob.matrix_world = matrix if isinstance(matrix, mathutils.Matrix) else mathutils.Matrix(np.asarray(matrix).tolist())
    return ob


def M(np4):
    return mathutils.Matrix(np.asarray(np4).tolist())


def np4(mat):
    return np.array([[mat[i][j] for j in range(4)] for i in range(4)])
