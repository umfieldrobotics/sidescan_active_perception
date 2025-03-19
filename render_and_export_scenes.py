import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
# %matplotlib tk
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import glob 
from tqdm import tqdm 
from scipy.ndimage import rotate
import sys 
import PIL

SM_SIGMA = 0.03
SA_SIGMA = 0.03
FILTER_SIZE=1.2

def export_exr_sidescan(file_path, terrain_img, in_ppm, out_ppm, max_range, altitude, 
                        num_theta, crop_size, crop_center):
    num_bins = int(out_ppm*max_range)
    #print all input vars 
    # print('file_path: ', file_path)
    # print('in_ppm: ', in_ppm)
    # print('out_ppm: ', out_ppm)
    # print('max_range: ', max_range)
    # Usage
    img = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    #read .tif image 
   
    #convert to bgr 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rasters = []
    terrain_masks = []
    from fast_histogram import histogram1d
    for i in range(img.shape[0]):
        counts = histogram1d(img[i,:,0], range=(1.0, max_range), bins=num_bins)
        values = histogram1d(img[i,:,0], range=(1.0, max_range), weights=img[i,:,1], bins=num_bins)
        raster = np.zeros((num_bins,))
        non_zero_mask = counts > 0
        raster[non_zero_mask] = values[non_zero_mask]/counts[non_zero_mask]

        terrain_mask = non_zero_mask & (raster == 0) #terrain mask is where there is no signal but there is terrain/a hit
        terrain_masks.append(terrain_mask)
        rasters.append(raster)
    sss_img = np.stack(rasters, axis=0)
    terrain_masks = np.stack(terrain_masks, axis=0)
    new_img = sss_img
    #gaussian blur this image 
    #resize the new_img 
    new_img = cv2.resize(new_img, (new_img.shape[1], int(new_img.shape[0]*out_ppm/in_ppm)), interpolation=cv2.INTER_NEAREST)
    terrain_masks = cv2.resize(terrain_masks.astype(np.uint8), (terrain_masks.shape[1], int(terrain_masks.shape[0]*out_ppm/in_ppm)), interpolation=cv2.INTER_NEAREST)
    zero_mask = new_img == 0

    #perform a crop centered at crop_center
    crop_center = (int(new_img.shape[0]//2), crop_center[1])
    big_crop_size = int(crop_size*1.25)
    new_img = new_img[crop_center[0]-big_crop_size:crop_center[0]+big_crop_size, crop_center[1]-big_crop_size:crop_center[1]+big_crop_size]
    terrain_masks = terrain_masks[crop_center[0]-big_crop_size:crop_center[0]+big_crop_size, crop_center[1]-big_crop_size:crop_center[1]+big_crop_size]

    #find the rotation angle 
    theta = int(file_path.split("_")[-1].split(".")[0])*360/num_theta + 180

    new_img = rotate(new_img, theta, reshape=False, mode='reflect')
    terrain_masks = rotate(terrain_masks, theta, reshape=False, mode='reflect')
    terrain_img = rotate(terrain_img, theta, reshape=False, mode='reflect')

    #crop again from center
    new_img = new_img[new_img.shape[0]//2-crop_size:new_img.shape[0]//2+crop_size, new_img.shape[1]//2-crop_size:new_img.shape[1]//2+crop_size]
    terrain_masks = terrain_masks[terrain_masks.shape[0]//2-crop_size:terrain_masks.shape[0]//2+crop_size, terrain_masks.shape[1]//2-crop_size:terrain_masks.shape[1]//2+crop_size]

    #make new_img brighter 
    new_img = new_img/new_img.max()*np.random.uniform(0.85, 1.0)
    new_img = new_img*(1 + np.random.normal(0, SM_SIGMA, new_img.shape)) + np.random.rayleigh(SA_SIGMA, new_img.shape)
    # new_img = gaussian_filter1d(new_img, sigma=FILTER_SIZE, axis=0)

    new_img = (terrain_img/255)*terrain_masks + new_img*(1-terrain_masks)

    #perform 2d gaussina blur 
    # new_img = gaussian_filter(new_img, sigma=FILTER_SIZE)
    im = PIL.Image.open('/home/advaith/Documents/triadelphia_dataset/cropped_regions/mine1_1/crop_1.png')
    #make subplots 
    # fig, ax = plt.subplots(1,2)

    # ax[0].imshow(np.array(im)/255, cmap='gray', vmin=0, vmax=1)
   
    # #gaussian blur in 1d 
    # # new_img = gaussian_filter1d(new_img, sigma=FILTER_SIZE, axis=0)


    # #additive normal noise
    # # new_img = new_img*(1 + np.random.normal(0, SM_SIGMA, new_img.shape)) + np.random.rayleigh(SA_SIGMA, new_img.shape)
    # # new_img = gaussian_filter1d(new_img, sigma=FILTER_SIZE, axis=0)
    # ax[1].imshow(new_img, cmap='gray', vmin=0, vmax=1)
    # ax[1].set_title("{}".format(theta-180))
    # plt.show()
    # new_img = (~zero_mask)*(new_img*(1 + np.random.normal(0, SM_SIGMA, new_img.shape)) + np.random.rayleigh(SA_SIGMA, new_img.shape))
    return new_img
    # plt.imshow(new_img, cmap='afmhot', vmin=0,vmax=1)


def main():
    #recursively glob for all .gltf files 
    #read worker number from command line args 
    worker_num = int(sys.argv[1])
    total_workers = int(sys.argv[2])
    #set the gpu to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(worker_num)%4) #mildly suspicious but ok 
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("I AM WORKER: {}".format(worker_num))
    search_dir = "/home/advaith/Documents/harder_scenes2"
    terrain_search_dir = '/home/advaith/Downloads/terrain_imgs'
    terrain_imgs = glob.glob(os.path.join(terrain_search_dir, '*.TIF'))
    pass_len = 8
    num_theta = 6
    altitude = 3 #based on field work on 7/7
    pass_radius = 12
    in_ppm = 300
    crop_size = 3 #in meters
    
    out_ppm = 20
    crop_size_pp = (crop_size*out_ppm)
    max_range = 20 #field work 7/7
    files = glob.glob('{}/**/*.gltf'.format(search_dir), recursive=True)
    files.sort()
    print("======================================")
    print("GLOBAL STATE")
    print("TOTAL FILES: {}".format(len(files)))
    #glob all the .png files 
    png_files = glob.glob('{}/**/*.png'.format(search_dir), recursive=True)
    print("TOTAL PNG FILES: {}".format(len(png_files)))
    print("======================================")
    

    open_files = []
    for file in tqdm(files): 
        #first call the renderer 
        #check if there are pngs in the folder already 
        if len(glob.glob(file.replace("scene.gltf", "render*"))) == num_theta and len(glob.glob(file.replace("scene.gltf", "raster*"))) == num_theta:
            print("Skipping {}".format(file))
            continue
        open_files.append(file)
    
    print("OPEN_FILES: ", len(open_files))
    my_block_size = len(open_files)//int(total_workers)
    open_files = open_files[my_block_size*int(worker_num):my_block_size*(int(worker_num)+1)]
    print("BLOCK START: {}, BLOCK END: {}".format(my_block_size*int(worker_num), my_block_size*(int(worker_num)+1)-1))
    print("BLOCK SIZE: {}".format(len(open_files)))

    for file in tqdm(open_files): 
        #first call the renderer 
        #check if there are pngs in the folder already 
        if len(glob.glob(file.replace("scene.gltf", "render*"))) == num_theta and len(glob.glob(file.replace("scene.gltf", "raster*"))) == num_theta:
            print("Skipping {}".format(file))
            continue
        os.system("/home/advaith/Documents/optix_sidescan/SDK/build/bin/optix_sidescan --max_range {} --pass_len {} --num_theta {} --altitude {} --pass_radius {} -m {} -f {} > /dev/null 2>&1".format(max_range, pass_len, num_theta, altitude, pass_radius, file, file.replace("scene.gltf", "render")))
        #find all files wiht output_number 
        output_files = glob.glob(file.replace("scene.gltf", "render*"))
        #randomly choose a terrain image
        terrain_path = np.random.choice(terrain_imgs)
        terrain_img = cv2.imread(terrain_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        #convert to grayscale
        terrain_img = cv2.cvtColor(terrain_img, cv2.COLOR_BGR2GRAY)

        #crop on either left side or right side of image 
        if np.random.random() > 0.5:
            terrain_img = terrain_img[:,:terrain_img.shape[1]//2 - crop_size_pp]
        else:
            terrain_img = terrain_img[:,terrain_img.shape[1]//2+crop_size_pp:]

        #perform random crop of the image of size crop_size_pp
        crop_center = (np.random.randint(crop_size_pp, terrain_img.shape[0]-crop_size_pp), np.random.randint(crop_size_pp, terrain_img.shape[1]-crop_size_pp))
        terrain_img = terrain_img[crop_center[0]-crop_size_pp:crop_center[0]+crop_size_pp, crop_center[1]-crop_size_pp:crop_center[1]+crop_size_pp]
        
        for out_file in output_files:
            #export exr sidescan 
            number = int(out_file.split("_")[-1].split(".")[0])
            dir_name = os.path.dirname(out_file)

            crop_center = (pass_len*out_ppm, pass_radius*out_ppm)
            new_img = export_exr_sidescan(out_file, terrain_img, in_ppm, out_ppm, max_range, altitude, num_theta, crop_size_pp, crop_center)
            plt.imsave(os.path.join(dir_name, 'raster_{}.png'.format(number)), new_img, cmap='gray', vmin=0, vmax=1)
            # #save as png 
            # cv2.imwrite(out_file.replace("render", "sss"), new_img*255)
            # #delete the exr file 
            # os.remove(out_file)

        
if __name__ == "__main__":
    main()