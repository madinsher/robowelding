import bpy, sys, os, math, numpy as np, mathutils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell import robot_build, geom as G, layout as L, materials
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
rob = robot_build.build()
arm = rob["arm"]
q = np.array(L.ROBOT_Q_HOME)
robot_build.set_q(rob, q)
robot_build.set_track(rob, -1.2)
bpy.context.view_layer.update()
# check that the bpy chain matches numpy FK
T_np = arm.fk(q, robot_build.base_matrix(-1.2), robot_build.tool_transform())
T_bl = G.np4(rob["tcp"].matrix_world)
print("TCP numpy:", T_np[:3,3].round(4), " bpy:", T_bl[:3,3].round(4), " maxdiff:", np.abs(T_np-T_bl).max().round(6))
bpy.ops.mesh.primitive_plane_add(size=20); bpy.context.object.data.materials.append(materials.get("concrete"))
bpy.ops.object.light_add(type='SUN', location=(3,2,6), rotation=(0.5,0.4,0)); bpy.context.object.data.energy=3
bpy.ops.object.light_add(type='AREA', location=(2,-3,3)); bpy.context.object.data.energy=2000; bpy.context.object.data.size=3
w = bpy.data.worlds.new("W"); sc.world=w; w.use_nodes=True; w.node_tree.nodes["Background"].inputs[0].default_value=(0.3,0.33,0.38,1)
for i,(loc,tgt,lens) in enumerate([((4.5,-4.5,2.5),(2.0,-1.0,1.2),35), ((0.5,-2.6,1.6),(2.0,-1.2,1.3),50)]):
    bpy.ops.object.camera_add(location=loc); cam=bpy.context.object; cam.data.lens=lens; sc.camera=cam
    d = cam.location - mathutils.Vector(tgt); cam.rotation_euler = d.to_track_quat('Z','Y').to_euler()
    sc.render.resolution_x=1280; sc.render.resolution_y=720; sc.render.engine='CYCLES'; sc.cycles.samples=24; sc.cycles.use_denoising=True
    sc.render.filepath=f"/home/user/robowelding/demo_video/out/t_robot_{i}.png"; bpy.ops.render.render(write_still=True)
