import torch
import pytorch_lightning as pl
# import wandb
#import dataloader 
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from data.singleview_dataset import *
import torch.nn as nn 
import PIL
from torchmetrics.classification import BinaryF1Score
from sklearn.metrics import confusion_matrix

def freeze_layers_resnet(model, layer):
    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx < layer:
            param.requires_grad = False
        else:
            break
    for idx, (name, param) in enumerate(model.named_parameters()):
        print(f'Layer {idx+1}: {name}, Requires Grad: {param.requires_grad}')

class sv_sss_classifier(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, num_workers, 
                 batch_size, lr, num_classes, num_angles, l1_lambda): 
        super(sv_sss_classifier, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.num_angles = num_angles
        self.num_classes = num_classes
        # self.loss = nn.MSELoss()

        # self.loss = vair_loss

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
       
        self.image_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        # freeze_layers_resnet(self.image_encoder, 30)
        # self.image_encoder.eval()

        # self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Sequential(
        #     nn.Linear(1000, 64), 
        #     nn.ReLU(),
        #     nn.Linear(64, 64), 
        #     nn.ReLU(),
        #     nn.Linear(64, self.num_classes)
        # )
        self.fc = nn.Linear(1000, self.num_classes)
        self.l1_lambda = l1_lambda

        # self.run = wandb.init(
        #     project="nrl_classifier",
        #     config={"batch_size": self.batch_size, "lr": self.lr, }
        # )
        self.loss = nn.CrossEntropyLoss()
        self.training_losses = []
        self.cms = np.zeros((self.num_angles, self.num_classes, self.num_classes))
        self.num_val_iters = 0
        self.misclassifications = []
        self.val_logits = []
        self.val_lbls = []
        self.train_logits = []
        self.train_lbls = []
   
    def forward(self, images, feat=False):
        # print(images.shape)
        # N, C, H, W = images.shape #images will be of size N, C, H, W
        features = self.image_encoder(images)
        # features = self.dropout(features)
        prediction = self.fc(features) #only use the last output of the rnn
        if feat: 
            return prediction, features
        return prediction

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
            self.val_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers, persistent_workers=persistent, pin_memory=True
        )

    def training_step(self, batch, batch_idx):
        try:
            data, lbls, _ = batch
            # if coords.shape[0] == 1:
            #     coords = coords.repeat(2,1,1)
            #     pcd = pcd.repeat(2,1,1)
            #     apc = apc.repeat(2,1,1)
            #     batch['gt_udf'] = batch['gt_udf'].repeat(2,1,1)
            #visualize the data in grid 
            # grid = torchvision.utils.make_grid(data, nrow=int(np.sqrt(self.batch_size)))
            # #display using plt 
            # plt.imshow(grid.permute(1,2,0).cpu().numpy())
            model_out = self(data) #model_out is Bx1 now
            # model_out = model_out.reshape(N, L, -1)

            #repeat lbls
            # lbls = lbls.repeat_interleave(L, dim=0)
            # losses = self.loss(model_out, batch, grad=True)
            train_loss = self.loss(model_out.squeeze(), lbls.long())
            reg_term = 0.0
            # for name, param in self.named_parameters():
            #     if 'bias' not in name:
            #         reg_term += torch.norm(param, p=1) #l1 reg
            self.training_losses.append(train_loss.item())
            # self.log('train_loss', train_loss, sync_dist=True)
            self.train_logits.append(model_out)
            self.train_lbls.append(lbls)
        except: 
            print("Error in training step")
            train_loss = torch.tensor(0.0)
        
        return train_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(lr=self.lr, params=self.parameters())
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

        self.train_logits = torch.cat(self.train_logits, dim=0)
        self.train_lbls = torch.cat(self.train_lbls, dim=0)
        accuracy = (torch.argmax(self.train_logits, dim=1).squeeze() == self.train_lbls.int()).float().mean()
        self.log('train_accuracy', accuracy, sync_dist=True)
    
        # for key in self.loss_dict:
        #     self.run.log({key: np.mean(self.loss_dict[key])})
        # self.loss_dict = {}
        self.train_logits = []
        self.train_lbls = []
        self.training_losses = []

    def on_validation_epoch_start(self):
        #remove the misclassifications folder
        # os.system("rm -rf /home/advaith/Documents/optix_sidescan/misclassifications")
        # os.system("mkdir /home/advaith/Documents/optix_sidescan/misclassifications")
        pass
    
    def on_validation_epoch_end(self) -> None:
        #concate the features and lbls
        self.val_logits = torch.cat(self.val_logits, dim=0)
        self.val_lbls = torch.cat(self.val_lbls, dim=0)

        preds = torch.softmax(self.val_logits, dim=1)
        preds = torch.argmax(preds, dim=1)

        #calculate accuracy 
        accuracy = (preds.squeeze() == self.val_lbls.int()).float().mean()
        self.log('val_accuracy', accuracy, sync_dist=True)

        #calculate val loss 
        val_loss = self.loss(self.val_logits, self.val_lbls.long())
        self.log('val_loss', val_loss, sync_dist=True)
        # print(preds.shape)
        # print(self.val_lbls.shape)
        if preds.shape[0] > 1:
            cm = confusion_matrix(self.val_lbls.int().cpu().numpy(), preds.squeeze().cpu().numpy(), labels=np.arange(0, self.num_classes))
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
            tick_marks = np.arange(len(lbl_to_name.keys()))
            # plt.xticks(tick_marks, list(name_to_lbl.keys()) + ["not mine"], rotation=45)
            # plt.yticks(tick_marks, list(name_to_lbl.keys()) + ["not mine"])
            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actuals', fontsize=18)
            plt.title('Confusion Matrix', fontsize=18)
            #save figure 
            plt.savefig("cm_sv.png")
            # Draw figure on canvas
            fig.canvas.draw()

            # Convert the figure to numpy array, read the pixel values and reshape the array
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
            img = img / 255.0
            # im = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            self.logger.experiment.add_image('confusion_matrix_', img, self.current_epoch, dataformats='HWC')
            plt.close(fig)

        #calculate
        # #visualize using tsne 
        # from sklearn.manifold import TSNE
        # tsne = TSNE(n_components=2, verbose=1, perplexity = self.val_features.shape[0]-1, n_iter=300)
        # tsne_results = tsne.fit_transform(self.val_features.cpu().numpy())
        # plt.scatter(tsne_results[:,0], tsne_results[:,1], c=self.val_lbls.cpu().numpy())
        # #add a legend
        # plt.legend(["not mine", "mine"])
        # plt.savefig("tsne.png")
        self.val_logits = []
        self.val_lbls = []
        # for num in range(self.cms.shape[0]):
        #     #normalize the cm 
        #     cm = self.cms[num]
        #     cm = np.round(cm, 2)
        #     fig, ax = plt.subplots(figsize=(7.5, 7.5))
        #     ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        #     for i in range(cm.shape[0]):
        #         for j in range(cm.shape[1]):
        #             ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
        #     tick_marks = np.arange(len(lbl_to_name.keys()))
        #     plt.xticks(tick_marks, list(name_to_lbl.keys()) + ["not mine"], rotation=45)
        #     plt.yticks(tick_marks, list(name_to_lbl.keys()) + ["not mine"])
        #     plt.xlabel('Predictions', fontsize=18)
        #     plt.ylabel('Actuals', fontsize=18)
        #     plt.title('Confusion Matrix', fontsize=18)
        #     #save figure 
        #     plt.savefig("cm_{}.png".format(str(num+1)))
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
        freeze_layers_resnet(self.image_encoder, 30)

    def validation_step(self, batch, batch_idx):
        data, lbls, files = batch
        # if coords.shape[0] == 1:
        #     coords = coords.repeat(2,1,1)
        #     pcd = pcd.repeat(2,1,1)
        #     apc = apc.repeat(2,1,1)
        #     batch['gt_udf'] = batch['gt_udf'].repeat(2,1,1)

        #TODO: calculate f1 scores for 1, 3, 6 image sequences. Sequences must be randomly shuffled 
        model_out, feats = self(data, feat=True)
        self.val_logits.append(model_out) #ensemble
        #repeat the lbls 
        # lbls = lbls.repeat_interleave(data.shape[1], dim=0)
        self.val_lbls.append(lbls)
        # val_loss = self.loss(model_out.squeeze(), lbls.long())
        # self.log('val_loss', val_loss, sync_dist=True)

        # #calculate f1 score for classiier 
        # pred = torch.sigmoid(model_out)
        # pred = pred > 0.5
        # pred = pred.int()
        # metric = BinaryF1Score().to(pred.device)
        # f1 = metric(pred.squeeze(), lbls.int())
       
        #make cm into a plt plot
        

        # self.num_val_iters += 1
        # pred = torch.softmax(model_out, dim=1)
        # pred = torch.argmax(pred, dim=1)
        # #calculate accuracy 
        # accuracy = (pred.squeeze() == lbls.int()).float().mean()
        # # self.log('full_accuracy', accuracy, sync_dist=True)
        # # self.log('full_f1', f1)

        # for num in range(self.num_angles):
        #     partial_views = data
        #     partial_gt = lbls
        #     model_out = self(partial_views)
        #     pred = torch.softmax(model_out, dim=1)
        #     pred = torch.argmax(pred, dim=1)

        #     #find all the images that are misclassified 
        #     misclassifications = (pred.squeeze() != partial_gt.int()).nonzero(as_tuple=True)[0]
        #     #go through the images and write text on the image of the gt class and predicted 
        #     #delete all files in folder 
            

        #     # for misclassification in misclassifications:
        #     #     img = partial_views[misclassification]
        #     #     img = img.squeeze().permute(1,2,0).cpu().numpy()

        #     #     #unnormalize the img using resnet constants
        #     #     # img = img*torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(0).repeat(224,224,1).numpy()
        #     #     # img = img + torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(0).repeat(224,224,1).numpy()
        #     #     img = (img*255).astype(np.uint8)
        #     #     img = PIL.Image.fromarray(img)

        #     #     #draw text on image
        #     #     draw = PIL.ImageDraw.Draw(img)
        #     #     font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 10)
        #     #     draw.text((0, 0), "GT: {}".format(lbl_to_name[partial_gt[misclassification].int().item()]), (255, 255, 255), font=font)
        #     #     draw.text((0, 20), "Pred: {}".format(lbl_to_name[pred[misclassification].int().item()]), (255, 255, 255), font=font)
        #     #     draw.text((0, 40), "File: {}".format(files[misclassification].split("/")[-2]), (255, 255, 255), font=font)

        #     #     #save it to dir 
        #     #     img.save(os.path.join("/home/advaith/Documents/optix_sidescan/misclassifications", "misclassification_{}_{}.png".format(files[misclassification].split("/")[-1], files[misclassification].split("/")[-2])))


        #     #create confusion matrix using sklearn and output a png
        #     cm = confusion_matrix(partial_gt.int().cpu().numpy(), pred.squeeze().cpu().numpy(), labels=np.arange(0, self.num_classes))
        #     self.cms[num] += cm

        #     accuracy = (pred.squeeze() == partial_gt.int()).float().mean()
        #     self.log(f'accuracy_{num+1}', accuracy, sync_dist=True)
