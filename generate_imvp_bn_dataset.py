import pandas as pd 
import bnlearn 
import torch 
import numpy as np 
import pytorch_lightning as pl
import torchvision
from data.singleview_dataset import *
from models.imvp_classifier import imvp_classifier
from models.singleview_classifier import *
import gin 
import sys
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import random
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import SGDClassifier
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pytorch_lightning.loggers import TensorBoardLogger
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = '/home/advaith/Documents/harder_scenes2'
num_angles = 6
# folders = os.listdir(data_dir)
seed = 14
# seed = 1917
# seed = 8125
#read txt file one line 

train_folders = open("train_files_{}.txt".format(seed), "r").read().splitlines()
val_folders = open("val_files_{}.txt".format(seed), "r").read().splitlines()
test_folders = open("test_files_{}.txt".format(seed), "r").read().splitlines()

train_folders = [folder for folder in train_folders if len(glob.glob(os.path.join(data_dir, folder, '*.png')))==num_angles]
val_folders = [folder for folder in val_folders if len(glob.glob(os.path.join(data_dir, folder, '*.png')))==num_angles]
test_folders = [folder for folder in test_folders if len(glob.glob(os.path.join(data_dir, folder, '*.png')))==num_angles]

# #filter the folders with .pngs in them 
# folders = [folder for folder in folders if len(glob.glob(os.path.join(data_dir, folder, '*.png')))>0]
# # random.shuffle(folders)
# train_val_mask = np.zeros(len(folders), dtype=np.bool)
# #randomly choose 80% to be true 
#set np random seed 
np.random.seed(seed)
torch.manual_seed(seed)

val_tf = torchvision.transforms.Compose([
        #translate image 
        torchvision.transforms.Resize((120, 120)),
        #normalize to resnet constants 
        #make grayscale 
        torchvision.transforms.Grayscale(num_output_channels=3),
        # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                                     std=[0.5, 0.5, 0.5]),
        torchvision.transforms.ToTensor(),
        #perform a center crop of the image 
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
        
        
    ])

net = imvp_classifier(None, None, 16, 256, 1e-3, 5, num_angles, 0)
net.to(device)
net.eval()

# net2 = sv_sss_classifier(0, 0, 0, 0, 0, 5, num_angles, 0)
# net2.cuda()
# net2.eval()

checkpoint = torch.load('/home/advaith/Documents/optix_sidescan/tb_logs/imvp_14/version_30/best-epoch=49-val_accuracy=0.89.ckpt')
net.load_state_dict(checkpoint["state_dict"])
print("Loaded frm checkpoint")

# checkpoint = torch.load('/home/advaith/Documents/optix_sidescan/tb_logs/my_model_sv_14/version_7/checkpoints/epoch=159-step=3520.ckpt')
# net2.load_state_dict(checkpoint["state_dict"])
# print("Loaded frm checkpoint")

dataset = IMVPDataset(val_folders,  data_dir, val_tf)

exported_data = []
exported_data_cols = ['CLASS', 'SHAPE', 'VOL', 'SHAPE_PRED', 'VOL_PRED', 'CLASS_PRED', 'ASPECT']

for data in tqdm(dataset): 
    imgs, lbl, volume_lbl, file = data

    aspect = int(file.split('/')[-1].split('_')[1].replace(".png", ""))
    class_pred, volume_pred = net(imgs.unsqueeze(0).to(device))

    class_pred = class_pred.argmax().item()
    volume_pred = volume_pred.argmax().item()

    row = [lbl, lbl, volume_lbl, class_pred, volume_pred, class_pred, aspect]
    exported_data.append(row)

#make a pd dataframe using the exported data
df = pd.DataFrame(exported_data, columns=exported_data_cols)
print(df.head())
#save to csv
df.to_csv('imvp_predictions_{}.csv'.format(seed), index=False)
