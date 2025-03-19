import torch
import pytorch_lightning as pl
# import wandb
#import dataloader 
from torch_geometric.loader import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from data.multiview_dataset import *
import math 
import random 
from models.multiview_classifier import *
import torch.nn as nn 
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import PIL
import torchvision.models as models
from torch_geometric.nn import global_mean_pool
from torchmetrics.classification import BinaryF1Score
from sklearn.metrics import confusion_matrix

np.random.seed(0)
torch.manual_seed(0)

def load_image_paths(seed, split, data_dir): 
    # folders = os.listdir(data_dir)
    
    # num_subset = 4000
    # #filter the folders with .pngs in them 
    # folders = [folder for folder in folders if len(glob.glob(os.path.join(data_dir, folder, '*.png')))==num_angles]

    # #collect num_subset rock folders 
    # new_folders_rock = []
    # new_folders_mine = []
    # total_rock = 0
    # total_mine = 0
    # for folder in folders:
    #     if "rock" in folder: 
    #         total_rock += 1
    #         if len(new_folders_rock) < num_subset:
    #             new_folders_rock.append(folder)
    #     elif "mine" in folder: 
    #         total_mine += 1
    #         if len(new_folders_mine) < num_subset:
    #             new_folders_mine.append(folder)
    # print("Total rock folders: ", total_rock)
    # print("Total mine folders: ", total_mine)
    # folders = new_folders_rock + new_folders_mine
    # # folders = np.random.choice(folders, num_subset, replace=False)
    # random.shuffle(folders)

    # train_size = int(0.7 * len(folders))
    # val_size = int(0.1 * len(folders))
    # test_size = len(folders) - train_size - val_size

    # train_mask = np.zeros(len(folders), dtype=np.bool)
    # train_mask[:train_size] = True

    # val_mask = np.zeros(len(folders), dtype=np.bool)
    # val_mask[train_size:train_size+val_size] = True

    # test_mask = np.zeros(len(folders), dtype=np.bool)
    # test_mask[train_size+val_size:] = True

    # ret_mask = val_mask if split == 'val' else test_mask if split == 'test' else train_mask
    folders = open("{}_files_{}.txt".format(split, seed), "r").read().splitlines()
    filter_func = lambda x: len(glob.glob(os.path.join(data_dir, x, '*.png'))) > 0 
    folders = list(filter(filter_func, folders))
    return folders


class MultiSurveyEnv(): 
    def __init__(self, weights_path, num_angles, data_dir, seed, target_id, gnn_layers=0, gnn_hidden_dim=128, p=0.5, 
                 split='val'): 
        self.test_tf = torchvision.transforms.Compose([
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
        self.val_tf = torchvision.transforms.Compose([
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
        # #normalize using resnet constants 
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
        #perform a center crop of the image
        
    ])
        self.tf = self.test_tf if split == 'test' else self.val_tf
        self.step_penalty  = -1
        self.classification_reward = 10
        self.num_angles = num_angles
        self.folders  = load_image_paths(seed, split, data_dir)
        self.val_dataset = MultiviewDataset(self.folders, data_dir, self.tf, num_angles, shuffle=True, train=False)

        self.target_id = target_id #this is the index into the dataset of the target we care about
        self.net = mv_sss_classifier(None, None, 1, 1, 1e-3, 5, num_angles, gnn_layers, gnn_hidden_dim, p)
        

        checkpoint = torch.load(weights_path, map_location='cuda:0')
        self.net.load_state_dict(checkpoint["state_dict"])
        self.state = None
        self.timeout = 6
        self.net.eval()
        self.FCC = -1
        self.steps = 0
        print("Loaded frm checkpoint")

    def update_target_id(self, target_id):
        self.target_id = target_id
        
    def get_init_state(self, roll_amt, compressor=None, randomize_target=False):
        #initially, we have not seen the target before so we must randomly take a view
        random_view=0 #first view you get is 0, everything is an offset
        if randomize_target: 
            self.target_id = np.random.choice(len(self.val_dataset))
        data, lbl = self.val_dataset[self.target_id]
        with torch.no_grad():
            self.features = self.net.image_encoder(data)
        #roll the features by a random amount 
        self.features = torch.roll(self.features, roll_amt, dims=0)
        self.lbl = lbl
        self.steps = 1 
        self.FCC = -1
        #now we create a graph for the chosen view 
        graph = create_fc_spatial_graph(self.features[np.array([random_view])], [random_view], lbl, discretization=self.num_angles)
        if compressor: 
            compressed_features = compressor.transform(graph.x.cpu().detach().numpy())
            #transmitting tehe ...
            reconstructed_features = compressor.inverse_transform(compressed_features)
            graph.x[0] = torch.tensor(reconstructed_features, dtype=torch.float32, device=graph.x.device)[0]
        with torch.no_grad():
            pred = self.net.GCN(graph)
            pred = torch.sigmoid(pred).argmax(dim=1)
            pred = pred.item()
        if pred == self.lbl and self.FCC < 0: #this means this is the first correct classification
            self.FCC = self.steps
        self.state = [random_view]
        return graph

        print('gotem')
    
    def step(self, action, compressor=None):
        #P(s, a) -> s', r 
        # if self.steps == 1: 
        #     #remove the first view from the state
        #     self.state = []
        #action is a number between 0 5, or 6 which stops the simulation
        if action == 6 or len(list(set(self.state))) == 6 or len(self.state) > self.timeout: 
            #calculate the reward by inference
            chosen_feats = list(set(self.state))
            graph = create_fc_spatial_graph(self.features[np.array(chosen_feats)], chosen_feats, self.lbl, discretization=self.num_angles)
            if compressor: 
                compressed_features = compressor.transform(graph.x.cpu().detach().numpy())
                #transmitting tehe ...
                reconstructed_features = compressor.inverse_transform(compressed_features)
                graph.x[0] = torch.tensor(reconstructed_features, dtype=torch.float32, device=graph.x.device)[0] #only replace the first features
            with torch.no_grad():
                pred = self.net.GCN(graph)
                pred = torch.sigmoid(pred).argmax(dim=1)
                pred = pred.item()
            reward = self.classification_reward if (pred == self.lbl) else -self.classification_reward
                #otherwise it means that when STOP is the FCC, good job, keep all the reward
            return None, reward, pred #None signifies end of round 
        else: 
            self.state.append(action)
            chosen_feats = list(set(self.state))
            graph = create_fc_spatial_graph(self.features[np.array(chosen_feats)], chosen_feats, self.lbl, discretization=self.num_angles)
            if compressor: 
                compressed_features = compressor.transform(graph.x.cpu().detach().numpy())
                #transmitting tehe ...
                reconstructed_features = compressor.inverse_transform(compressed_features)
                graph.x[0] = torch.tensor(reconstructed_features, dtype=torch.float32, device=graph.x.device)[0] #only replace the first features
            with torch.no_grad():
                pred = self.net.GCN(graph)
                pred = torch.sigmoid(pred).argmax(dim=1)
                pred = pred.item()
            if pred == self.lbl and self.FCC < 0: #this means this is the first correct classification
                self.FCC = self.steps
            reward = 0 
            if self.steps > self.FCC: 
                reward -= 1
            
        self.steps += 1

        return graph, reward, None 

        
def main(gnn_layers=0, gnn_hidden_dim=128, p=0.5):
    weights_path = '/home/advaith/Documents/optix_sidescan/tb_logs/my_model_mv/version_260/best-epoch=129-val_accuracy=0.98.ckpt'
    num_angles = 6
    data_dir = '/home/advaith/Documents/harder_scenes2'
    big_rewards = []
    env = MultiSurveyEnv(weights_path, num_angles, data_dir, 0, gnn_layers, gnn_hidden_dim, p)
    for i in range(1, 100):
        target_id = i 
        env.update_target_id(target_id)

        state = env.get_init_state()

        rewards = []
        num_views = []
        for j in tqdm(range(30)): 
            #randomly shuffle an array from 0-6 
            ep_reward = 0
            while state:
                action = np.random.choice(7)
                state, reward = env.step(action)
                ep_reward += reward
            num_views.append(len(env.state))
            state = env.get_init_state()
            rewards.append(ep_reward)
        
        big_rewards.append(np.mean(rewards))
    
    plt.plot(big_rewards)



if __name__ == '__main__':
    main()
