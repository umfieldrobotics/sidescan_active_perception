import pyrender 
import trimesh
from tqdm import tqdm 
import time
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import imageio
import glob 
import math
import pyvista as pv
from scipy.spatial import Delaunay
from PIL import Image
import os
from PIL import Image, ImageEnhance
import open3d as o3d
import noise 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

def scale_to_unit_cube(mesh):
    #get the bounding box of the mesh 
    bbox = mesh.bounds
    #translate to originusing centroid 
    old_centroid = mesh.centroid
    mesh.vertices -= mesh.centroid
    #scale all dimensions to 1x1x1 
    # mesh.vertices[:,0] /= (bbox[1,0] - bbox[0,0])
    # mesh.vertices[:,1] /= (bbox[1,1] - bbox[0,1])
    # mesh.vertices[:,2] /= (bbox[1,2] - bbox[0,2])
    #scale the largest dim to 1
    max_dim = np.max(bbox[1,:] - bbox[0,:])
    mesh.vertices /= max_dim
    #translate back to old centroid
    mesh.vertices += old_centroid
    return mesh

def create_map(shape, scale):
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    base = np.random.randint(0, 10000)
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(i/scale, 
                                        j/scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=1024, 
                                        repeaty=1024, 
                                        base=base)
    return world

def generate_terrain_mesh(width, height, resolution, elevation_map):
    # Generate a grid of vertices
    x = np.linspace(-width/2, width/2, resolution)
    y = np.linspace(-height/2, height/2, resolution)
    X, Y = np.meshgrid(x, y)

    # Generate terrain elevations
    Z = elevation_map

    # Create vertices from the grid
    vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    # Create faces (triangles) from the vertices
    tri = Delaunay(vertices[:, :2])

    # Create the mesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=tri.simplices[:,::-1])

    #generate uv coordinates for each vertex 
    uv = np.zeros((vertices.shape[0], 2))
    uv[:,0] = (vertices[:,0] - vertices[:,0].min())/width
    uv[:,1] = (vertices[:,1] - vertices[:,1].min())/height
    mesh.uv = uv

    #swap the y and z coordinates of the mesh
    mesh.vertices[:,[1,2]] = mesh.vertices[:,[2,1]]

    #calculate normals 
    

    return mesh

def create_single_object_scene(mesh_path, background_path, eulers, acoustic_reflectance, save_path, burial_amount, size=[1,1,1], rotate=True):

    # # os.environ['PYOPENGL_PLATFORM'] = 'egl'
    width = 40
    height = 40
    # z = 0
    # # Define the vertices of the plane
    # vertices = np.array([
    #     [-width/2, z, -height/2],
    #     [width/2,z,  -height/2],
    #     [width/2,z, height/2],
    #     [-width/2,z, height/2]
    # ])

    # # Define the face indices of the plane
    # faces = np.array([[2,1,0], [3,2,0]])
    shape = (500, 500)
    scale = np.random.uniform(low=1, high=1)
    elev_scale = np.random.uniform(low=150, high=200)
    elev_map = create_map(shape,elev_scale)
    # elev_map = np.zeros(shape)
    plane_mesh = generate_terrain_mesh(width, height, shape[0], scale*elev_map)

    # Create a Trimesh object representing the plane
    img = Image.open(background_path)

    #randomly crop this image along the vertical axi
    img_width = img.size[0]

    #crop the image at a random location 
    crop_start = np.random.randint(width, img.size[1] - 2*width)
    img = img.crop((0, crop_start, img_width, crop_start + img_width))

    #brighten this image using IMageEncahane
    #choose random value between 0.1 and 0.23 
    b_decrease = np.random.uniform(low=1.3, high=1.7)
    img = ImageEnhance.Brightness(img).enhance(b_decrease)

    #terrain hijacking, make an image completely green 
    img = Image.new('RGB', (500, 500), color = (0, 255, 0))

    #zero out the r channel of img 
    img = np.array(img)
    # img[:,:,0] = 0
    img = Image.fromarray(img)

    #generate random noise in a grayscale image from a beta distribution 
    # img = Image.fromarray(np.random.beta(0.5, 0.5, size=(10000,10000))*40).convert('L')

    # Create a Trimesh object representing the plane
    # plane_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    plane_mesh.visual = trimesh.visual.TextureVisuals(uv=plane_mesh.uv, image=img)
    #set the color of plane mesh to red 
    # plane_mesh.visual.vertex_colors = np.tile([0.1,0,0], (plane_mesh.vertices.shape[0], 1))

    #set all the normals for plane_mesh 
    # plane_mesh.face_normals = np.array([[0,1, 0], [0, 1, 0]])
    #change the color of the mesh

    fuze_trimesh = trimesh.load(mesh_path)
    fuze_trimesh = trimesh.Trimesh(vertices=fuze_trimesh.vertices, faces=fuze_trimesh.faces)
    fuze_trimesh = scale_to_unit_cube(fuze_trimesh)

    #translate so that the lowest part of the mesh is above 0 
    

    #center centroid at 0 
    fuze_trimesh.vertices -= fuze_trimesh.centroid

    #scale the vertice by size 
    fuze_trimesh.vertices *= size
    #make 4x4 matrix from euler angles 
    rot_mat = np.eye(4)
    rot_mat[0:3,0:3] = Rotation.from_euler('xyz', eulers).as_matrix()
    #apply the rotation to can_trimesh
    fuze_trimesh.apply_transform(rot_mat)
    # fuze_trimesh.vertices[:,1] -= np.min(fuze_trimesh.vertices[:,1])
    #translate so that burial amount percentage of the mesh is under 0 level using the aabb in world frame 
    #get the aabb of the mesh in world frame
    aabb = fuze_trimesh.bounding_box.bounds
    #get the height of the aabb
    height = aabb[1,1] - aabb[0,1]
    #translate the mesh by -burial_amount*height in y
    #get the distance to ground directly below the object using raytrace 
    #create a ray from the centroid of the mesh to the ground
    ray_start = np.array([0, 100, 0])
    ray_end = - np.array([0,1,0])
    #raytrace
    intersections, index_ray, index_tri =  plane_mesh.ray.intersects_location(
        ray_origins=[ray_start],
        ray_directions=[ray_end],
        multiple_hits=False
    )
    #get the distance to the ground
    dist_to_ground = np.min(fuze_trimesh.vertices[:,1]) - intersections[0][1]

    fuze_trimesh.vertices[:,1] -=  dist_to_ground + burial_amount*height
    # fuze_trimesh.vertices[:,1] *= 2 
    color = [acoustic_reflectance, 0, 0]  # RGB values
    vertex_colors = np.tile(color, (fuze_trimesh.vertices.shape[0], 1))
    fuze_trimesh.visual.vertex_colors = vertex_colors

    #apply a random rotation 
    #generate a random rotation matrix as 4x4
    # random_rot = np.random.rand(4,4)
    # random_rot[0:3,0:3] = Rotation.random().as_matrix()
    # #set the last row to 0,0,0,1
    # random_rot[-1,:] = np.array([0,0,0,1])
    # # apply the random rotation to can_trimesh
    # fuze_trimesh.apply_transform(random_rot)
    #rotate it 90 degrees in x
    #apply rotation given by eulers 
    


    #merge plane_mesh and fuze_trimesh into one mesh 
    # box = trimesh.creation.box(extents=[10, 0.01, 10])
    # box2 = trimesh.creation.box(extents=[20, 0.01, 10])
    
    #create a trimesh scene 
    m = trimesh.Scene([plane_mesh, fuze_trimesh])

    #save mas gltf 
    m.export(os.path.join(save_path, "scene.gltf"))

def generate_scenes(out_dir, num_each_angle=10, r_min=0.7, r_max=1.0, num_r=10, 
                    z_size_min = 0.5, z_size_max=2.0, num_z_size=10, 
                    burial_amount_min=0.5, burial_amount_max=0.9, num_burial_amount=10):
    #make a meshgrid of all combinations of x,y,z euler angles from 0 to 360 
    cube_root = num_each_angle
    x = np.linspace(-np.pi/2, np.pi/2, cube_root)
    y = np.linspace(-np.pi/2, np.pi/2, cube_root)
    z = np.linspace(-np.pi/2, np.pi/2, cube_root)
    #create a meshgrid of all combinations of x,y,z euler angles from 0 to 360
    eulers = np.array(np.meshgrid(x,y,z)).T.reshape(-1,3)

    acoustic_reflectances = np.linspace(r_min, r_max, num_r)

    z_sizes = np.linspace(z_size_min, z_size_max, num_z_size)

    burial_amounts = np.linspace(burial_amount_min, burial_amount_max, num_burial_amount)

    background_imgs = glob.glob("/home/advaith/Downloads/jpg_exports/*.JPG")
    meshes = glob.glob("/home/advaith/Downloads/object_meshes/train_files/rocks/*.obj")
    filtered = []
    filter = 1992
    for mesh in meshes: 
        num = int(os.path.basename(mesh).replace(".obj", "").split("_")[-1])
        if num >= filter: 
            filtered.append(mesh)
    meshes = filtered
    total_scenes = len(meshes)*len(eulers)*len(acoustic_reflectances)*len(z_sizes)*len(burial_amounts)
    print("Generating {} scenes".format(total_scenes))
    for mesh_file in tqdm(meshes, desc="mesh"):
        for ei in range(len(eulers)): 
            for acoustic_reflectance in acoustic_reflectances:
                for z_size in z_sizes:
                    for burial_amount in burial_amounts:
                        #randomly choose a background img
                        background_path = np.random.choice(background_imgs)

                        save_path = os.path.join(out_dir, "N_{}_R_{}_Z_{}_B_{}_E_{}".format(os.path.basename(mesh_file).split(".")[0],
                                                                                            acoustic_reflectance, 
                                                                                            z_size, 
                                                                                            np.random.uniform(burial_amount_min, burial_amount_max), 
                                                                                            ei))
                        os.makedirs(save_path, exist_ok=True)
                        try:
                            if num_each_angle == 1: 
                                rand_eul = np.random.rand(3)*np.pi - np.pi/2
                                create_single_object_scene(mesh_file, background_path, rand_eul, acoustic_reflectance, save_path, burial_amount, size=[z_size, z_size, z_size])
                            else:
                                create_single_object_scene(mesh_file, background_path, eulers[ei], acoustic_reflectance, save_path, burial_amount, size=[z_size, z_size, z_size])
                        except: 
                            print("Failed to generate scene at {}".format(save_path))
                            continue
                    

def main():
    out_dir = "/home/advaith/Documents/harder_scenes2"
    num_each_angle = 1 #focus more on orientation randomization
    r_min = 0.95
    r_max = 0.95
    num_r = 1 #reflectances dont get randomized anyway -> deal with this in post
    z_size_min = 0.5
    z_size_max = 0.5
    num_z_size = 1 #size randomization
    burial_amount_min = 0.1
    burial_amount_max = 0.3
    num_burial_amount = 1 #anything above 0.1 burial is unrealistic for this project
    generate_scenes(out_dir, num_each_angle, r_min, r_max, num_r, z_size_min, z_size_max, num_z_size, burial_amount_min, burial_amount_max, num_burial_amount)

if __name__ == "__main__":
    main()