# Robotic pipe-spool welding cell — demo video (Blender/bpy)

Goal: a ~42 s, 24 fps demo video (RoboDK-style walkthrough, but photoreal-ish) of a
6-axis ABB IRB 4600 on a linear track welding a DN250 pipe spool (weld-neck flange
+ 90° LR elbow + straight pipe) held by a 2-axis tilt/rotate positioner.

Run everything with the system `python3` (Blender 4.5 is installed as the `bpy` module):

    cd /home/user/robowelding/demo_video
    python3 build.py --still 300 --res 960 540         # one preview still at frame 300
    python3 build.py --preview                          # 480x270 every 8th frame -> preview.mp4
    xvfb-run -a python3 build.py --render                # final EEVEE render (needs xvfb for EEVEE)

Package layout (all modules import `cell.layout` for the shared constants — never hard-code
positions that exist there):

    cell/layout.py       constants (positions, sizes, fps)            <- single source of truth
    cell/geom.py         mesh helpers (revolve, torus_sweep, box, cylinder, empty, set_parent, M)
    cell/materials.py    procedural materials (function per material, cached by name)
    cell/spool.py        spool geometry + weld-bead rings (shader-driven progress)
    cell/positioner.py   2-axis positioner: returns dict with joint empties `tilt`, `rot`, `faceplate`
    cell/robot_urdf.py   URDF FK/IK (numpy)               cell/robot_build.py  bpy assembly + track + torch
    cell/environment.py  hall, fence, equipment (static)  cell/vfx.py  arc light, sparks, smoke, glare
    cell/animation.py    storyboard: keyframes robot/positioner/track/cameras, weld intervals
    build.py             entry point

Conventions:
* Units metres, Z up. FPS 24. Scene frame 1..1008.
* Every builder puts its objects in its own collection (`bpy.data.collections.new(name)` linked to
  the scene collection) and returns the objects it created; nothing touches other collections.
* Materials come from `cell.materials` — call `materials.get("abb_orange")` etc.; add new ones there.
* Keep geometry moderate (< 1.5 M triangles total); EEVEE render, 1080p, ~10 s/frame budget.
* Blender 4.5 API: `bpy.ops.wm.stl_import`, `shade_smooth_by_angle`, EEVEE is `BLENDER_EEVEE_NEXT`,
  no bloom setting (use compositor Glare node), lights use `energy`, `shadow_soft_size`.
* Test scripts go in `tests/` and are run with `python3 tests/<name>.py` (they render stills to
  `out/`); never leave debugging prints in modules.
