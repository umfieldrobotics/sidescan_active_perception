import torch 
import numpy as np 
import pytorch_lightning as pl
import torchvision
import joblib
from data.multiview_dataset import *
from models.graph_compressor import GraphAutoencoder
from models.multiview_classifier import create_fc_spatial_graph, mv_sss_classifier
import gin 
from sklearn.decomposition import PCA
import numpy as np
import random
import torchvision
import sys
import os
import matplotlib 
# matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger


def validate(net, dataloader):
    net.eval()
    with torch.no_grad():
        v_loss = []
        v_preds = []
        v_gt = []
        C_classes = []
        for data, lbls in tqdm(dataloader):
            lbls = lbls.repeat_interleave(len(net.Cs))
            # #create a graph from the images
            graphs, weight_vect = net.create_graph_batch_from_images(data.cuda(), lbls.cuda())
            C_classes += [d.x.shape[0]  for d in graphs]
            out = net(graphs)
            pred = torch.softmax(out, dim=1)
            v_preds.append(pred[:,1].cpu().numpy())
            v_gt.append(lbls.cpu().numpy())
            # print("Accuracy", (out.argmax(dim=1) == lbls.cuda()).float().mean())
            # logger.log_metrics({"accuracy": (out.argmax(dim=1) == lbls.cuda()).float().mean()}, step=i)
        v_preds = np.concatenate(v_preds)
        v_gt = np.concatenate(v_gt)
    net.train()
    


    dset_type = 'multiview'
    #find confusion matrix from the predictions
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    # sns.set()
    # plt.figure(figsize=(16,10))
    # cm = confusion_matrix(real_labels, real_preds)
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Real Rock', 'Real Mine'], yticklabels=['Real Rock', 'Real Mine'])
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.savefig("real_cm.png")

    #calculate roc curve 
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(v_gt, v_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(16,10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title("ROC Curve for {} data".format(dset_type))
    plt.savefig("{}_roc.png".format(dset_type))

    #get optimal threshold 
    from sklearn.metrics import f1_score
    f1_scores = []
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        f1_scores.append(f1_score(v_gt, v_preds>threshold))
    f1_scores = np.array(f1_scores)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print("Best F1 Score: ", np.max(f1_scores))
    sns.set()
    plt.figure(figsize=(16,10))
    cm = confusion_matrix(v_gt, v_preds > optimal_threshold, normalize='true')
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Real Rock', 'Real Mine'], yticklabels=['Real Rock', 'Real Mine'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CM for on {} data".format(dset_type))
    plt.savefig("{}_cm.png".format(dset_type))

    print("Val accuracy: ", ((v_preds > optimal_threshold) == v_gt).mean())

    #plot the precision recall curve 
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(v_gt, v_preds)
    precision, recall, _ = precision_recall_curve(v_gt, v_preds)
    plt.figure(figsize=(16,10))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall curve:on {} data".format(dset_type))
    plt.savefig("{}_pr.png".format(dset_type))




    
@gin.configurable
def main(data_dir='./', 
         val_dir='./',
        num_epochs=1000, 
        num_gpus=4, 
        weight_save_dir='./',
        num_workers=16, 
        batch_size=16, 
        lr=1e-4, 
        gnn_hidden_dim=256, 
        gnn_layers=2, 
        p=0.2
        ):
    
    num_classes = 5
    num_angles = 6
    #creat tf to resize the height to 224 
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
        # #normalize using resnet constants 
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
    seed = 14
    # seed = 1917
    # seed = 8125
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_folders = open("train_files_{}.txt".format(seed), "r", encoding='utf-8').read().splitlines()
    val_folders = open("val_files_{}.txt".format(seed), "r", encoding='utf-8').read().splitlines()
    test_folders = open("test_files_{}.txt".format(seed), "r", encoding='utf-8').read().splitlines()
    # seed = 14 

    # folders = os.listdir(data_dir)
    # 
    # num_subset = 4000
    # #filter the folders with .pngs in them 
    train_folders = [folder for folder in train_folders if len(glob.glob(os.path.join(data_dir, folder, '*.png')))==num_angles]
    val_folders = [folder for folder in val_folders if len(glob.glob(os.path.join(data_dir, folder, '*.png')))==num_angles]
    test_folders = [folder for folder in test_folders if len(glob.glob(os.path.join(data_dir, folder, '*.png')))==num_angles]

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


    # im_vis = train_val_mask.reshape()
    # plt.imshow(im_vis)
    # train_val_mask[:] = True
    dataset = MultiviewDataset(train_folders, data_dir, input_tf, num_angles, shuffle=True)
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_dataset = MultiviewDataset(test_folders, data_dir, val_tf, num_angles, shuffle=True, train=False)

    test_dataset = MultiviewDataset(test_folders, data_dir, val_tf, num_angles, shuffle=False, train=False)


    print("-"*50)
    print("Train dataset size: ", len(dataset))
    print("Val dataset size: ", len(val_dataset))

    train_imgs = set(train_folders)
    val_imgs = set(val_folders)
    print("Train and val imgs overlap: ", len(train_imgs.intersection(val_imgs)))
    print("-"*50)

    net = GraphAutoencoder(1000, 32, 1000)
    gmvatr = mv_sss_classifier(None, None, 1, 1, 1e-3, 5, num_angles, gnn_layers, gnn_hidden_dim, p)

    checkpoint = torch.load('/mnt/syn/advaiths/DoD SAFE-2twjN5BZ5TQnffsY/version_9/best-epoch=19-val_accuracy=0.98.ckpt', map_location='cuda:0')
    gmvatr.load_state_dict(checkpoint["state_dict"])
    gmvatr.cuda()
    net.to('cuda')

    dset_features = []
    with torch.no_grad():
        for i in tqdm(range(100)):
            data, lbl = dataset[i]
            L, C, H, W = data.shape
            features = gmvatr.image_encoder(data.reshape(L, C, H, W).cuda()).reshape(1, L, -1)

            dset_features.append(features.squeeze())

    
    dset_features = torch.cat(dset_features, dim=0).cpu().squeeze()

    n_components = np.arange(2, 256, 16)
    mses = []
    explained_variance = []

    for n in tqdm(n_components):
        pca = PCA(n_components=n)
        reduced_data = pca.fit_transform(dset_features)
        data_reconstructed = pca.inverse_transform(reduced_data)

        explained_variance_ratio = pca.explained_variance_ratio_
        total_explained_variance = np.sum(explained_variance_ratio)
        explained_variance.append(total_explained_variance)

        mse = torch.sum((dset_features - data_reconstructed)**2)/(dset_features.shape[0]*dset_features.shape[1])
        mses.append(mse.item())
       
        #save the pca weights 
        joblib.dump(pca, "compressor_weights/pca_{}.pkl".format(n))

    #plot subplots 
    plt.figure(figsize=(16,10))
    plt.subplot(1,2,1)
    plt.plot(n_components, explained_variance)
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.title("Explained Variance vs Number of Components")

    plt.subplot(1,2,2)
    plt.plot(n_components, mses)
    plt.xlabel("Number of Components")
    plt.ylabel("MSE")
    plt.title("MSE vs Number of Components")


   







if __name__ == "__main__":
    gin.parse_config_file(sys.argv[1])
    main()