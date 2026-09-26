import bpy, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell import spool, positioner, geom as G, layout as L
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
pos = positioner.build()
sp = spool.build()
G.set_parent(sp["root"], pos["mount"], keep_world=False)
positioner.set_tilt(pos, 90)
positioner.set_rot(pos, 0)
# show half of bead A welded and hot
b = sp["beads"]["A"]; b["w0_start"] = 0.5; b["w0_prog"] = 0.55; b["w0_dir"] = -1.0; b["w0_hot"] = 1.0
bpy.ops.mesh.primitive_plane_add(size=20)
bpy.context.object.data.materials.append(__import__('cell.materials', fromlist=['x']).get("concrete"))
bpy.ops.object.light_add(type='SUN', location=(3,2,6), rotation=(0.5,0.4,0)); bpy.context.object.data.energy=3
bpy.ops.object.light_add(type='AREA', location=(2,-3,3)); bpy.context.object.data.energy=1500; bpy.context.object.data.size=3
w = bpy.data.worlds.new("W"); sc.world=w; w.use_nodes=True; w.node_tree.nodes["Background"].inputs[0].default_value=(0.3,0.33,0.38,1)
bpy.context.view_layer.update()
for i,(loc,tgt) in enumerate([((2.4,-2.2,1.9),(0.4,0,1.0)), ((1.1,-0.7,1.35),(0.4,0.05,1.1))]):
    bpy.ops.object.camera_add(location=loc); cam=bpy.context.object; cam.data.lens=35 if i==0 else 50; sc.camera=cam
    d = cam.location - __import__('mathutils').Vector(tgt); cam.rotation_euler = d.to_track_quat('Z','Y').to_euler()
    sc.render.resolution_x=1280; sc.render.resolution_y=720; sc.render.engine='CYCLES'; sc.cycles.samples=24; sc.cycles.use_denoising=True
    sc.render.filepath=f"/home/user/robowelding/demo_video/out/t_spool_pos_{i}.png"; bpy.ops.render.render(write_still=True)
print("tris:", sum(len(o.data.polygons) for o in bpy.data.objects if o.type=='MESH'))
