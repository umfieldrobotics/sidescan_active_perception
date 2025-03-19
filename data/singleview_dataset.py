import torch 
#import dataset from pytorch 
from torch.utils.data import Dataset
import os 
import glob
import numpy as np
import PIL 

import torchvision
name_to_lbl = {
    "mine1": 1, #cylinder
    "mine2": 2, #block 
    "mine3": 3, #cone 
    "mine4": 4, #pyramid
}
lbl_to_name = {
    0: "not mine", 
    1: "mine1", 
    2: "mine2",
    3: "mine3",
    4: "mine4"
}
volume_cutoffs = [[0.0, 0.020], [0.020, 0.035], [0.035, 0.05], [0.05, 100]]

def decide_volume_label(volume):
    lbl = 0 
    for i, cutoff in enumerate(volume_cutoffs):
        if volume >= cutoff[0] and volume < cutoff[1]:
            lbl = i 
    return lbl 

def decide_label(folder):
    for key in name_to_lbl.keys():
        if key in folder:
            label = name_to_lbl[key]
            return label 
    return 0 #not mine class

#write a template dataset class 
class SingleviewDataset(Dataset):
    def __init__(self, folders, data_path, input_tf, group=False):
        self.data_path = data_path
        self.folders = folders
        # self.folders.sort()
        self.folders = np.array(self.folders)
        self.imgs = []
        self.group = group
        if not group:
            for folder in self.folders: 
                #only add images if they are greater than 0 mb
                if len(glob.glob(os.path.join(data_path, folder, '*.png'))) > 0:
                    self.imgs += glob.glob(os.path.join(data_path, folder, '*.png'))
        else: 
            for folder in self.folders: 
                if len(glob.glob(os.path.join(data_path, folder, '*.png'))) > 0:
                    self.imgs.append(glob.glob(os.path.join(data_path, folder, '*.png')))
        self.input_tf = input_tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        file = self.imgs[idx]
        if len(file) > 0:
            if self.group:
                label = decide_label(file[0])
                imgs = []
                for f in file:
                    img = PIL.Image.open(f).convert('RGB')
                    img = self.input_tf(img)
                    imgs.append(img)
                img = torch.stack(imgs)
                return img, label, file

            else:
                #read the label with filename sss_label.txt 
                label = decide_label(file)
                # label = np.loadtxt(os.path.join(self.data_path, folder, 'sss_label.txt'))
                #read images using torchvision.io 
                # img = torchvision.io.read_image(file, torchvision.io.image.ImageReadMode.RGB)
                img = PIL.Image.open(file).convert('RGB')
                img = self.input_tf(img)

                return img, label, file 
        else: 
            print("No images found in folder: ", file)
            return None, None, None
    
#write a template dataset class 
class IMVPDataset(Dataset):
    def __init__(self, folders, data_path, input_tf):
        self.data_path = data_path
        self.folders = folders
        # self.folders.sort()
        self.folders = np.array(self.folders)
        self.imgs = []
        for folder in self.folders: 
            #only add images if they are greater than 0 mb
            if len(glob.glob(os.path.join(data_path, folder, '*.png'))) > 0:
                self.imgs += glob.glob(os.path.join(data_path, folder, '*.png'))
        self.input_tf = input_tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        file = self.imgs[idx]
        if len(file) > 0:
            #read the label with filename sss_label.txt 
            label = decide_label(file)
            # label = np.loadtxt(os.path.join(self.data_path, folder, 'sss_label.txt'))
            #read images using torchvision.io 
            # img = torchvision.io.read_image(file, torchvision.io.image.ImageReadMode.RGB)
            img = PIL.Image.open(file).convert('RGB')

            #read the .txt file in the same directory 
            dir_name = os.path.dirname(file)
            txt_file = os.path.join(dir_name, 'volume.txt')
            with open(txt_file, 'r') as f:
                volume = float(f.read())
            volume_lbl = decide_volume_label(volume)
            # volume_lbl = volume 
            img = self.input_tf(img)

            return img, label, volume_lbl, file 
       