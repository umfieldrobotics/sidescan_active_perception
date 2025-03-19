#write a script for blender py ot generate rocks
#
#

import bpy
import numpy as np
import os
import glob
import random




bpy.ops.mesh.add_mesh_rock()

#export the rock mesh as .obj 
bpy.ops.export_scene.obj(filepath="/home/advaith/Downloads/object_meshes/train_files/rocks/rock_{}.obj".format(0))