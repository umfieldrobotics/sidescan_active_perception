import torch
import pytorch_lightning as pl
# import wandb
#import dataloader 
from torch_geometric.loader import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from data.multiview_dataset import *
import math 
from torch_geometric.data import Batch
import torch.nn as nn 
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import PIL
import torchvision.models as models
from torch_geometric.nn import global_mean_pool
from torchmetrics.classification import BinaryF1Score
from sklearn.metrics import confusion_matrix
import time
from torch.profiler import profile, record_function, ProfilerActivity

def freeze_layers_resnet(model, layer):
    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx < layer:
            param.requires_grad = False
        else:
            break
    for idx, (name, param) in enumerate(model.named_parameters()):
        print(f'Layer {idx+1}: {name}, Requires Grad: {param.requires_grad}')

def create_fc_spatial_graph(features, node_inds, target, discretization=6): 
    #features is NxC embedding 
    #node_inds is N list with index of the nodes, this is used to create the spatial edges 
    # a = time.time()
    #create the node features
    x = features

    # b = time.time()
    
    #create the edge features
    edge_index = torch.tensor([[i, j] for i in node_inds for j in node_inds], dtype=torch.long).t().contiguous().to(features.device)

    #take edges that are only 1 away from each other THIS IS ONLY FOR VIEW-GCN
    # edge_index = edge_index[:, (edge_index[0] - edge_index[1]).abs() <= 1]
    # # c = time.time()

    #assign the rotation transformation between nodes 
    phase_vect = torch.exp(torch.tensor([1j*2*np.pi*i/discretization for i in range(discretization)])) #create a vector of phasors
    #populate phase_mat with phase_vect*complex conjugate of phase_vect
    phase_mat = torch.outer(np.conj(phase_vect), phase_vect)

    # # d = time.time()

    #calculate the phase of each complex number 
    phase_mat = torch.angle(phase_mat).to(features.device)

    # # e = time.time()
    
    #make edge attrs the phase diff 
    edge_attrs = phase_mat[edge_index[0], edge_index[1]]

    # # f = time.time()

    edge_attrs = edge_attrs.reshape(-1, 1).to(features.device) 

    # edge_attrs = torch.ones_like(edge_attrs) #NEED TO REMOVE THIS AFTER ABLATIONS
    #replace the unique labels in edge_index with arange nubmers 
    unique_nodes = torch.unique(edge_index)

    # # g = time.time()

    new_nodes = torch.arange(len(unique_nodes))
    for i in range(len(unique_nodes)):
        edge_index[edge_index == unique_nodes[i]] = new_nodes[i]
    
    # # h = time.time()
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([target]), edge_attr=edge_attrs)

    # data = Data(x=x, edge_index=edge_index, y=torch.tensor([target]))

    # # i = time.time()
    # print("Time to create node features: ", b-a)
    # print("Time to create edge index: ", c-b)
    # print("Time to create edge features: ", d-c)
    # print("Time to create phase mat: ", e-d)
    # print("Time to calculate phase mat: ", f-e)
    # print("Time to reshape edge attrs: ", g-f)
    # print("Time to replace unique nodes: ", h-g)
    # print("Time to create data: ", i-h)
    # print("Total time: ", i-a)

    return data

# class resGBlock(torch.nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.layer = GCNConv(in_dim, out_dim)

#     def forward(self, x, edge_index, edge_weight):
#         identity = x 
#         out = self.layer(x, edge_index, edge_weight=edge_weight)
#         out = F.relu(out)
#         out = identity + out
#         return out 
    
class GCN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, p, num_layers=1):
        super().__init__()
        self.conv1 = GCNConv(feature_dim, hidden_dim)
        stem = []
        for i in range(num_layers):
            stem.append(GCNConv(hidden_dim, hidden_dim))
        print("NUM_LAYERS: ", len(stem))
        self.stem = nn.ModuleList(stem)
        self.p = p 
        self.dropout = nn.Dropout(p=p)
        self.lin1 = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
        self.edge_embedding = nn.Sequential(
            nn.Linear(3,1), 
            nn.ReLU(),
            nn.Linear(1, 1), 
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        assert torch.any(torch.isnan(x)) == False
        assert torch.any(torch.isnan(edge_index)) == False
        assert torch.any(torch.isnan(edge_weight)) == False
        edge_weight = self.edge_embedding(torch.cat([edge_index, edge_weight.T], dim=0).T)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        for layer in self.stem:
            x = layer(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.dropout(x)
        x = self.lin1(x)
        
        return x
    
class mv_sss_classifier(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, num_workers, 
                 batch_size, lr, num_classes, num_angles, 
                 gnn_layers, gnn_hidden_dim, p): 
        super(mv_sss_classifier, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.num_angles = num_angles
        self.num_classes = num_classes
        self.gnn_layers=gnn_layers
        # self.loss = nn.MSELoss()

        # self.loss = vair_loss
        self.GCN = GCN(1000, gnn_hidden_dim, num_classes, p, num_layers=self.gnn_layers)

        

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.image_encoder = models.resnet50(pretrained=True)
        # freeze_layers_resnet(self.image_encoder, 72)
        
        # self.image_encoder.eval()
        # layers = list(self.image_encoder.children())[:-1]
        # self.feature_extractor = nn.Sequential(*layers)

        # self.test_linear = nn.Linear(1000, 2)
        # self.rnn = nn.LSTM(1000, 1024, 2, batch_first=True)
        # self.fc = nn.Linear(1024, num_classes)

        # self.run = wandb.init(
        #     project="nrl_classifier",
        #     config={"batch_size": self.batch_size, "lr": self.lr, }
        # )
        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.training_losses = []
        self.val_preds = []
        self.val_gt = []
        self.val_weight_vect = []
        self.train_preds = []
        self.train_gts = []
        self.C_classes = []
        self.cms = np.zeros((self.num_angles, self.num_classes, self.num_classes))
        self.num_val_iters = 0
        # self.automatic_optimization = False

        arr = np.arange(self.num_angles)
        self.Cs = []
        self.class_cardinality = [1]
        for i in range(1,self.num_angles+1):
            all_combs = list(combinations(arr, i))
            self.Cs += all_combs #add the actual indices 
            self.class_cardinality.append(len(all_combs)) #add the number of combinations
        #calculate self.num_angles choose self.num_angles//2
        self.C_mid = math.factorial(self.num_angles)//(math.factorial(self.num_angles//2)*math.factorial(self.num_angles//2))


    def create_graph_batch_from_images(self, data, lbls):
        #takes in a batch of images and creates a list of graphs
        #data: NxLxCxHxW, lbls = N, valid_mask = NxL
        graphs = []
        #features is length LxD
        N, L, C, H, W = data.shape
        
        features = self.image_encoder(data.reshape(N*L, C, H, W)).reshape(N, L, -1)
        data_lens = []
        for i in range(data.shape[0]):
            for comb in self.Cs: 
                cc = np.array(comb)
                valid_feats = features[i, cc]
                graph = create_fc_spatial_graph(valid_feats, cc, lbls[i], discretization=self.num_angles)
                graphs.append(graph)

                C_len = len(cc)
                class_cardinality = self.class_cardinality[C_len]
                data_lens.append(class_cardinality)
        
        weight_vect = self.C_mid/np.array(data_lens)
        graphs = Batch.from_data_list(graphs)
        return graphs, weight_vect #graphs will be N*63 long 

    def forward(self, graphs):
        # preds = []
        # for graph in graphs: 
        #     pred = self.GCN(graph)
        #     preds.append(pred.squeeze())
        #     # in_feat = graph.x.mean(dim=0, keepdim=True)
        #     # pred = self.test_linear(in_feat)
        #     # preds.append(pred.squeeze())
        # preds = torch.stack(preds, dim=0)
        preds = self.GCN(graphs)
        # preds = self.GCN(graphs)
        
        return preds

    def train_dataloader(self):
        persistent = True if self.num_workers > 0 else False
        print("Persistent: ", persistent)
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=persistent, pin_memory=True
        )

    def val_dataloader(self):
        persistent = True if self.num_workers > 0 else False
        print("Persistent: ", persistent)
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=persistent, pin_memory=True
        )

    def training_step(self, batch, batch_idx):
        data, lbls = batch
        lbls = lbls.repeat_interleave(len(self.Cs))
        # #create a graph from the images
        graphs, weight_vect = self.create_graph_batch_from_images(data, lbls)
        # # opt = self.optimizers()
        # opt.zero_grad()

        # if coords.shape[0] == 1:
        #     coords = coords.repeat(2,1,1)
        #     pcd = pcd.repeat(2,1,1)
        #     apc = apc.repeat(2,1,1)
        #     batch['gt_udf'] = batch['gt_udf'].repeat(2,1,1)
        model_out = self(graphs) #model_out is Bx1 now
        # losses = self.loss(model_out, batch, grad=True)
        train_loss = (torch.tensor(weight_vect).to(model_out.device)*self.loss(model_out.squeeze(), lbls.long())).mean()
        self.training_losses.append(train_loss.item())

        self.train_preds.append(model_out)
        self.train_gts.append(lbls)
        # self.manual_backward(train_loss)
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(name)
        # opt.step()
        return train_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=self.lr, params=self.parameters())
        # return optimizer
        # sched1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-12, end_factor=1.0, total_iters=self.warmup_epochs)
        # sched2 = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer,warmup_start_lr=1e-4, 
        #                                                                                 warmup_epochs=self.warmup_epochs,
        #                                                                                   max_epochs=self.cosine_start)
        # fused_scheduler = torch.optim.lr_scheduler.ChainedScheduler([sched2, sched1])
        # scheduler = {
        #     # "scheduler0": ReduceLROnPlateau(optimizer, patience=200, factor=0.5),
        #     "scheduler": sched1,
        #     "monitor": "train_loss",
        # }
        return {"optimizer": optimizer}

    def on_train_epoch_end(self):
        self.log('iter_loss', np.mean(self.training_losses)) 
        self.log('train_loss', np.mean(self.training_losses))

        #calculate accuracy
        self.train_preds = torch.cat(self.train_preds)
        self.train_gts = torch.cat(self.train_gts)
        accuracy = (torch.softmax(self.train_preds, dim=1).argmax(dim=1) == self.train_gts).float().mean()
        self.log('train_full_accuracy', accuracy)
    
        # for key in self.loss_dict:
        #     self.run.log({key: np.mean(self.loss_dict[key])})
        # self.loss_dict = {}
        self.training_losses = []
        self.train_preds = []
        self.train_gts = []
    
    def on_validation_epoch_end(self) -> None:
        self.val_preds = torch.cat(self.val_preds)
        self.val_gt = torch.cat(self.val_gt)
        self.val_weight_vect = np.concatenate(self.val_weight_vect)
        self.C_classes  = torch.tensor(np.array(self.C_classes)).to(self.val_gt.device)
        #calculate accuracy
        accuracy = (torch.softmax(self.val_preds, dim=1).argmax(dim=1) == self.val_gt).float().mean()
        self.log('val_accuracy', accuracy, sync_dist=True)

        #calculate val loss 
        val_loss = (torch.tensor(self.val_weight_vect).to(self.val_preds.device)*self.loss(torch.tensor(self.val_preds), torch.tensor(self.val_gt).long())).mean()
        self.log('val_loss', val_loss)
        

        # accuracies = []
        # for i in range(1, self.num_angles+1):
        #     #calculate the accuracy for each C_class 
        #     accuracy = (torch.softmax(self.val_preds[self.C_classes == i], dim=1).argmax(dim=1) == self.val_gt[self.C_classes == i]).float().mean()
        #     accuracies.append(accuracy.item())
        #     self.log(f'val_accuracy_{i}', accuracy, sync_dist=True)

        # #make a graph 
        # plt.figure()
        # plt.plot(range(1, self.num_angles+1), accuracies)
        # plt.xlabel('C class')
        # plt.ylabel('Accuracy')
        
        # #save fig 
        # plt.savefig("C_class_vs_accuracy.png")
        # plt.close()


        self.val_preds = []
        self.val_gt = []
        self.val_weight_vect = []
        self.C_classes = []
        # for num in range(self.cms.shape[0]):
        #     #normalize the cm 
        #     cm = self.cms[num]/self.cms[num].sum(axis=1)[:, np.newaxis]
        #     cm = np.round(cm, 2)
        #     fig, ax = plt.subplots(figsize=(7.5, 7.5))
        #     ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        #     for i in range(cm.shape[0]):
        #         for j in range(cm.shape[1]):
        #             ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
        #     # tick_marks = np.arange(len(lbl_to_name.keys()))
        #     # plt.xticks(tick_marks, list(name_to_lbl.keys()) + ["not mine"], rotation=45)
        #     # plt.yticks(tick_marks, list(name_to_lbl.keys()) + ["not mine"])
        #     plt.xlabel('Predictions', fontsize=18)
        #     plt.ylabel('Actuals', fontsize=18)
        #     plt.title('Confusion Matrix', fontsize=18)
        #     #save figure 
        #     # plt.savefig("cm_{}.png".format(str(num+1)))
        #     # Draw figure on canvas
        #     fig.canvas.draw()

        #     # Convert the figure to numpy array, read the pixel values and reshape the array
        #     img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        #     img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        #     # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
        #     img = img / 255.0
        #     # im = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        #     self.logger.experiment.add_image('confusion_matrix_{}'.format(num+1), img, self.current_epoch, dataformats='HWC')
        #     plt.close(fig)
        # self.cms = np.zeros((self.num_angles, self.num_classes, self.num_classes))
        # self.num_val_iters = 0 
        # freeze_layers_resnet(self.image_encoder, 72) #just in case it gets unfrozen 

    def validation_step(self, batch, batch_idx):
        data, lbls = batch
        lbls = lbls.repeat_interleave(len(self.Cs))
        # #create a graph from the images
        graphs, weight_vect = self.create_graph_batch_from_images(data, lbls)
        # self.C_classes += [d.x.shape[0]  for d in graphs]
        # if coords.shape[0] == 1:
        #     coords = coords.repeat(2,1,1)
        #     pcd = pcd.repeat(2,1,1)
        #     apc = apc.repeat(2,1,1)
        #     batch['gt_udf'] = batch['gt_udf'].repeat(2,1,1)
        model_out = self(graphs) #model_out is Bx1 now
        # val_loss = self.loss(model_out.squeeze(), lbls.long())
        # self.log('val_loss', val_loss)

        # #calculate f1 score for classiier 
        # pred = torch.sigmoid(model_out)
        # pred = pred > 0.5
        # pred = pred.int()
        # metric = BinaryF1Score().to(pred.device)
        # f1 = metric(pred.squeeze(), lbls.int())
       
        #make cm into a plt plot
        
        self.num_val_iters += 1
        # pred = torch.softmax(model_out, dim=1)
        # pred = torch.argmax(pred, dim=1)
        #calculate accuracy 
        # accuracy = (pred.squeeze() == lbls.int()).float().mean()
        # self.log('full_accuracy', accuracy)
        self.val_preds.append(model_out)
        self.val_weight_vect.append(weight_vect)
        self.val_gt.append(lbls)

        # cm = confusion_matrix(lbls.int().cpu().numpy(), pred.squeeze().cpu().numpy(), labels=np.arange(0, self.num_classes))
        # self.cms[-1] += cm
        # self.log('full_f1', f1)

        # for num in range(self.num_angles):
        #     partial_views = data[:, :num+1]
        #     partial_gt = lbls[:, 0] #label is constant per sequence
        #     model_out = self(partial_views)[:,:,-1] #take the last prediction
        #     pred = torch.softmax(model_out, dim=1)
        #     pred = torch.argmax(pred, dim=1)

        #     #create confusion matrix using sklearn and output a png
        #     cm = confusion_matrix(partial_gt.int().cpu().numpy(), pred.squeeze().cpu().numpy(), labels=np.arange(0, self.num_classes))
        #     self.cms[num] += cm

        #     accuracy = (pred.squeeze() == partial_gt.int()).float().mean()
        #     self.log(f'accuracy_{num+1}', accuracy)
