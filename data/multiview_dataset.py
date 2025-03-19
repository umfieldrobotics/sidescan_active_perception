import torch 
#import dataset from pytorch 
from torch_geometric.data import Data
from torch.utils.data import Dataset
import os 
import glob
import PIL 
import torchvision.models as models
import numpy as np
from itertools import combinations
import numpy as np 
import torchvision
from tqdm import tqdm 
from models.multiview_classifier import create_fc_spatial_graph

# def create_fc_spatial_graph(features, node_inds, target, discretization=6): 
#     #features is NxC embedding 
#     #node_inds is N list with index of the nodes, this is used to create the spatial edges 
#     # a = time.time()
#     #create the node features
#     x = features

#     # b = time.time()
    
#     #create the edge features
#     edge_index = torch.tensor([[i, j] for i in node_inds for j in node_inds], dtype=torch.long).t().contiguous().to(features.device)

#     # c = time.time()

#     #assign the rotation transformation between nodes 
#     phase_vect = torch.exp(torch.tensor([1j*2*np.pi*i/discretization for i in range(discretization)])) #create a vector of phasors
#     #populate phase_mat with phase_vect*complex conjugate of phase_vect
#     phase_mat = torch.outer(np.conj(phase_vect), phase_vect)

#     # d = time.time()

#     #calculate the phase of each complex number 
#     phase_mat = torch.angle(phase_mat).to(features.device)

#     # e = time.time()
    
#     #make edge attrs the phase diff 
#     edge_attrs = phase_mat[edge_index[0], edge_index[1]]

#     # f = time.time()

#     edge_attrs = edge_attrs.reshape(-1, 1).to(features.device)  
#     #replace the unique labels in edge_index with arange nubmers 
#     unique_nodes = torch.unique(edge_index)

#     # g = time.time()

#     new_nodes = torch.arange(len(unique_nodes))
#     for i in range(len(unique_nodes)):
#         edge_index[edge_index == unique_nodes[i]] = new_nodes[i]
    
#     # h = time.time()
#     data = Data(x=x, edge_index=edge_index, y=torch.tensor([target]), edge_attr=edge_attrs)

#     # i = time.time()
#     # print("Time to create node features: ", b-a)
#     # print("Time to create edge index: ", c-b)
#     # print("Time to create edge features: ", d-c)
#     # print("Time to create phase mat: ", e-d)
#     # print("Time to calculate phase mat: ", f-e)
#     # print("Time to reshape edge attrs: ", g-f)
#     # print("Time to replace unique nodes: ", h-g)
#     # print("Time to create data: ", i-h)
#     # print("Total time: ", i-a)

#     return data

name_to_lbl = {
    "mine1": 1,
    "mine2": 2,
    "mine3": 3,
    "mine4": 4,
}
lbl_to_name = {
    0: "not mine", 
    1: "mine1", 
    2: "mine2",
    3: "mine3",
    4: "mine4"
}

def decide_label(folder):
    for key in name_to_lbl.keys():
        if key in folder:
            label = name_to_lbl[key]
            return label 
    return 0


#write a template dataset class 
class MultiviewDataset(Dataset):
    def __init__(self, folders, data_path, input_tf, num_angles, shuffle=True, train=True):
        self.data_path = data_path
        self.folders = folders
        # self.folders = os.listdir(data_path)
        # self.folders.sort()
        self.folders = np.array(self.folders)

        #go through each folder and check the number of .png images 
        
        self.input_tf = input_tf
        self.num_angles = num_angles
        self.shuffle = shuffle
        self.rand_aug = torchvision.transforms.RandAugment(3, 9)
        self.train = train
        self.norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        

        # num_augs = 10 if train else 1
        # arr = np.arange(6)
        # Cs = []
        # for i in range(1,7):
        #     Cs += list(combinations(arr, i))

        # self.traj_lens = np.random.randint(low=1, high=self.num_angles+1, size=(len(self.folders)))
        # self.valid_inds = []
        # for traj_len in self.traj_lens:
        #     self.valid_inds.append(np.random.choice(self.num_angles, int(traj_len), replace=False))
        # self.data = []
        # for folder in tqdm(self.folders): 
        #      #glob all the png files in a folder 
        #     files = glob.glob(os.path.join(self.data_path, folder, '*.png'))
        #     if len(files) == 0: 
        #         print("No files found in folder: ", folder)
        #     files.sort()

        #     angles = []
        #     for file in files:
        #         #get the angle from the filename 
        #         angle = int(os.path.basename(file).split('_')[-1].split('.')[0])
        #         angles.append(angle)

        #     #read the data
        #     img_data = []
        #     for file in files:
        #         #read images using torchvision.io 
        #         img = PIL.Image.open(file).convert('RGB')
        #         img_data.append(torch.tensor(np.array(self.input_tf(img))))

        #     img_data = torch.stack(img_data).permute(0,3,1,2)
            
        #     for _ in range(num_augs):
        #         if self.train:
        #             img_data = self.rand_aug(img_data)
        #         data = (img_data/255.0).float()
        #         data = self.norm(data)
        #         label = torch.tensor(decide_label(folder)) #single target for group of images

        #         with torch.no_grad():
        #             features = self.image_encoder(data.cuda()).cpu()
        #         for comb in Cs: 
        #             cc = np.array(comb)
        #             valid_feats = features[cc]
        #             graph = create_fc_spatial_graph(valid_feats, cc, label, discretization=self.num_angles)
        #             self.data.append(graph)
        
        # self.C_mid = 20.0

    def __len__(self):
        return len(self.folders)


    def __getitem__(self, idx):
        folder = self.folders[idx]

        # if self.shuffle and self.train:
        #     traj_len = np.random.randint(low=1, high=self.num_angles+1)
        #     valid_inds = np.random.choice(self.num_angles, int(traj_len), replace=False)
        #     valid_mask = np.zeros(self.num_angles, dtype=np.bool)
        #     valid_mask[valid_inds] = True
        # elif self.shuffle: #in validation 
        #     valid_inds = self.valid_inds[idx]
        #     valid_mask = np.zeros(self.num_angles, dtype=np.bool)
        #     valid_mask[valid_inds] = True
        # else: 
        #     valid_mask = np.ones(self.num_angles, dtype=np.bool)
        files = glob.glob(os.path.join(self.data_path, folder, '*.png'))
        if len(files) == 0: 
            print("No files found in folder: ", folder)
        files.sort()

        angles = []
        for file in files:
            #get the angle from the filename 
            angle = int(os.path.basename(file).split('_')[-1].split('.')[0])
            angles.append(angle)

        #read the data
        img_data = []
        for file in files:
            #read images using torchvision.io 
            img = PIL.Image.open(file).convert('RGB')
            img_data.append(torch.tensor(np.array(self.input_tf(img))))

        img_data = torch.stack(img_data)
        label = torch.tensor(decide_label(folder))

       
        # data = self.data[idx]
        return img_data, label 
