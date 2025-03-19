import numpy as np 
import cv2
import scipy 
import glob 
import rasterio
import folium
import utm 
import os
from rasterio.windows import Window
from shapely.geometry import box
from matplotlib import pyplot as plt
from rasterio.windows import from_bounds

def main():
    search_dir = "/home/advaith/Documents/7_7_2m_processed"
    out_path = "/home/advaith/Documents/triadelphia_dataset/cropped_regions/rock6"
    #read all the .tif files in a dir 
    files = glob.glob(search_dir + '**/*.tif', recursive=True)
    # map_center = [39.1940094458, -77.0108323376]  # anchor point 
    # map_center = [39.1939757018, -77.0108810239] #rock point
    # map_center = [39.1937538871, -77.0110966079] #cylinder point 7/11 2m survey 
    map_center=[39.1939787827, -77.0109936632]
    # map_center = [39.1937760765, -77.0111007854] #cylinder point 7/11 4m survey
    zoom_level = 100  # Adjust the zoom level as per your preference
    map = folium.Map(location=map_center, zoom_start=zoom_level)
    
    #add marker for map center with a different icon 
    marker = folium.Marker(location=map_center, icon=folium.Icon(color='red', icon='info-sign'))
    marker.add_to(map)

    CROP_BOX_SIZE = 3 #in meters
    count = 0
    for file in files:
        #read geotiff image 
        dataset = rasterio.open(file)

        # heading = np.rad2deg(np.arccos(dataset.transform[0]))

        #crop the image around the map_center coordinate
        #get the utm coordinates of the map center
        utm_center = utm.from_latlon(map_center[0], map_center[1])[0:2]

        
        #get the utm coordinates of the top right corner
        utm_top_right = utm_center + np.array([CROP_BOX_SIZE, CROP_BOX_SIZE])
        #get the utm coordinates of the bottom left corner
        utm_bottom_left = utm_center + np.array([-CROP_BOX_SIZE, -CROP_BOX_SIZE])
        bbox = (utm_bottom_left[0], utm_bottom_left[1], utm_top_right[0], utm_top_right[1])

        #check if utm_center falls within bounds 
        if utm_center[0] < dataset.bounds.left or utm_center[0] > dataset.bounds.right or utm_center[1] < dataset.bounds.bottom or utm_center[1] > dataset.bounds.top:
            #skip this image 
            print("Skipping {}".format(file))
        else: 
            #adjust the bbox to be within bounds 
            bbox = (max(bbox[0], dataset.bounds.left), max(bbox[1], dataset.bounds.bottom), min(bbox[2], dataset.bounds.right), min(bbox[3], dataset.bounds.top))
            # Get the window coordinates for the bounding box
            with rasterio.open(file) as src:
                ul_col, ul_row = src.index(bbox[0], bbox[3])
                lr_col, lr_row = src.index(bbox[2], bbox[1])
                window = Window.from_slices((ul_row, lr_row + 1), (ul_col, lr_col + 1))
                # Read the windowed data from the input GeoTIFF
                data = src.read(window=from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], src.transform))
                # data = data.mean(axis=0)
                data = data.transpose(1,2,0)

                if data.shape[2] == 3: 
                    #convert to grayscale 
                    data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

                #turn to opencv bgr 
                

                #save the image 
                #apply a color map using plt to image 
                # data = np.uint8(data)
                # data = cv2.applyColorMap(data, cv2.COLORMAP_JET)
                #get the file path 
                #write the numpy array as image 


                cv2.imwrite(os.path.join(out_path, "crop_{}.png".format(count)), data)
                count += 1



        #get the utm coordinates of the top left corner
        top_left = dataset.transform * (0, 0)
        #get the utm coordinates of the bottom right corner
        bottom_right = dataset.transform * (dataset.width, dataset.height)
        #get the utm coordinates of the top right corner
        top_right = dataset.transform * (dataset.width, 0)
        #get the utm coordinates of the bottom left corner
        bottom_left = dataset.transform * (0, dataset.height)

        # #make sure all the utm_top_left coordinates are within the bounds of top_left etc
        # utm_top_left[0] = max(utm_top_left[0], top_left[0])


        

        # #convert the utm coordinates to lat long
        top_left_lat_long = utm.to_latlon(top_left[0], top_left[1], 18, 'T')
        bottom_right_lat_long = utm.to_latlon(bottom_right[0], bottom_right[1], 18, 'T')
        top_right_lat_long = utm.to_latlon(top_right[0], top_right[1], 18, 'T')
        bottom_left_lat_long = utm.to_latlon(bottom_left[0], bottom_left[1], 18, 'T')

        # # print("Minimum X:", min_x)
        # # print("Minimum Y:", min_y)
        # # print("Maximum X:", max_x)
        # # print("Maximum Y:", max_y)
        bbox = [top_left_lat_long, top_right_lat_long, bottom_right_lat_long, bottom_left_lat_long]
        #create a bbox with 4 corners 
        polygon = folium.Polygon(bbox, color="blue", weight=2, fill_opacity=0.0)
        polygon.add_to(map)

    map.save("map.html")

    
if __name__ == '__main__':
    main()