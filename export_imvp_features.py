import glob 
import trimesh 
from tqdm import tqdm 



#find all the .gltf files ina directory 
search_dir = '/home/advaith/Documents/harder_scenes2'
files = glob.glob('{}/**/*.gltf'.format(search_dir), recursive=True)
files.sort()
volumes = []
for file in tqdm(files): 
    #open file with trimesh 
    mesh = trimesh.load(file)
    #find all the submeshes 
    min_volume = float('inf')
    min_volume_mesh = None
    for key in mesh.geometry.keys():
        submesh = mesh.geometry[key]
        if submesh.volume < min_volume and submesh.volume > 0:
            min_volume = submesh.volume
            min_volume_mesh = submesh
    #find the submesh with smallest volume 
    volumes.append(min_volume)
    #write a .txt file in the directory with the volume 
    with open(file.replace("scene.gltf", "volume.txt"), "w") as f:
        f.write(str(min_volume))

#make a historgram of volumes 
import matplotlib.pyplot as plt
plt.hist(volumes, bins=100)
plt.savefig('volumes.png')
