import torch 
import numpy as np 
import pytorch_lightning as pl
import torchvision
from data.singleview_dataset import *
from models.singleview_classifier import sv_sss_classifier
import gin 
import sys
import os

@gin.configurable
def main(data_dir='./', 
         val_dir='./',
        num_epochs=1000, 
        num_gpus=4, 
        weight_save_dir='./',
        num_workers=16, 
        batch_size=16, 
        lr=1e-4, 
        l1_lambda=0.001
        ):
    
    num_classes = np.unique(list(name_to_lbl.values()))[0]+1
    num_angles = 1
    input_tf = torchvision.transforms.Compose([
            #translate image 
            # lambda x: torchvision.transforms.functional.affine(x, 0.0, (-30, 0), 1.0, 0.0),
            # torchvision.transforms.RandomAffine(0.0, translate=(10/820, 0)),
            # torchvision.transforms.CenterCrop(120),
            torchvision.transforms.Resize((120,120)),
            torchvision.transforms.RandAugment(3, 9),
            #normalize to resnet constants 
            #make grayscale 
            torchvision.transforms.Grayscale(num_output_channels=3),
            #add randaugment
            # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                                     std=[0.5, 0.5, 0.5]),
            torchvision.transforms.ToTensor(),
            #normalize using resnet constants 
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
            #perform a center crop of the image
            
        ])

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

    folders = os.listdir(data_dir)

    #filter the folders with .pngs in them 
    folders = [folder for folder in folders if len(glob.glob(os.path.join(data_dir, folder, '*.png')))>0]
    train_val_mask = np.zeros(len(folders), dtype=np.bool)
    #randomly choose 80% to be true 
    train_val_mask[np.random.choice(len(train_val_mask), int(0.8*len(train_val_mask)), replace=False)] = True
    dataset = SingleviewDataset(folders, train_val_mask, data_dir, input_tf)
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


    val_dataset = SingleviewDataset(folders, ~train_val_mask, data_dir, input_tf)
    # 
    real_dataset = SingleviewDataset(os.listdir(val_dir), np.ones(len(os.listdir(val_dir))).astype(bool), val_dir, val_tf)

    