import bpy
import random

add_seed = 1123
location_y = 0
step = 2.

area_size = 20.

for i in range(30):
    # Create bezier Curve and enter edit mode
    bpy.ops.curve.primitive_bezier_curve_add(radius=2.0,
                                                    location=(0.0, 0.0, 0.0),
                                                    enter_editmode=True)
                                            
    ## Set resolution of curve
    #bpy.ops.curve.subdivide(number_cuts=1)

    # Randomize curve
    bpy.ops.transform.vertex_random(offset=1.0, uniform=0.1, normal=0.0, seed=i+add_seed)
    bpy.ops.object.editmode_toggle()
    
    # Rotate object for z alignment
    bpy.ops.transform.rotate(value=-1.5708, orient_axis='Y', orient_type='GLOBAL')
    
    # Move object for better view
    bpy.ops.transform.translate(value=(random.uniform(0, area_size), random.uniform(0, area_size), 0), orient_axis_ortho='X', orient_type='GLOBAL')
#    location_y += step
    
    obj = bpy.context.active_object
    
    obj.name = f'Curve_{i}'
    
    # Create geo node with name
    geo_nodes = obj.modifiers.new('create_flower', "NODES")
    # Apply geo node just created
    geo_nodes.node_group = bpy.data.node_groups['PROC_Flower']
    
    # Set parameters {first name of geo node, the Input_nr}
    bpy.data.objects[obj.name].modifiers["create_flower"]["Input_2"] = random.randint(0,100)
    bpy.data.objects[obj.name].modifiers["create_flower"]["Input_3"] = random.randint(0,100)
    