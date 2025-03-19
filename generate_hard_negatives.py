import trimesh 
import numpy as np 
import glob 
import os
from generate_scenes import scale_to_unit_cube



def generate_hard_negative(mesh_path, iters=3):
    #open the mesh in trimesh 
    mesh = trimesh.load(mesh_path)

    #subdivide the mesh
    for i in range(iters):
        mesh.vertices, mesh.faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    
    #scale to fit within 1x1x1 cube
    mesh = scale_to_unit_cube(mesh)


    #apply random noise to the vertices of the mesh
    #choose 10 percent of vertices 
    num_verts = mesh.vertices.shape[0]
    num_verts_to_perturb = int(0.1*num_verts)
    vert_indices = np.random.choice(num_verts, num_verts_to_perturb, replace=False)
    mesh.vertices[vert_indices] += np.random.normal(0, 0.01, size=(num_verts_to_perturb, 3))

    return mesh     






def main():
    search_dir = '/home/advaith/Downloads/object_meshes/train_files/mines'
    export_dir = '/home/advaith/Downloads/object_meshes/train_files/rocks'
    meshes = glob.glob(os.path.join(search_dir, '*.obj'))
    for mesh in meshes: 
        mesh = trimesh.load(mesh)
        dims = mesh.bounding_box_oriented.primitive.extents
        


if __name__ == "__main__":
    main()