import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
from tqdm import tqdm 
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import numpy as np
from torch_geometric.nn import global_mean_pool
from rl_env import MultiSurveyEnv
import joblib 


compressor_model = joblib.load('/mnt/syn/advaiths/NRL_GMVATR/optix_sidescan/compressor_weights/pca_18.pkl')

test_feature = np.random.uniform(size=(1, 1000))

compressed_features = compressor_model.transform(test_feature)
                #transmitting here
reconstructed_features = compressor_model.inverse_transform(compressed_features)

#compare the two
print(np.linalg.norm(test_feature - reconstructed_features))
