import torch 
import numpy as np 
import pytorch_lightning as pl
import torchvision
from data.singleview_dataset import *
from models.imvp_classifier import imvp_classifier
import gin 
import sys
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from models.singleview_classifier import *
import random
import bnlearn as bn
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
from sklearn.linear_model import SGDClassifier
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pytorch_lightning.loggers import TensorBoardLogger
random.seed(0)

def bayes_update_imvp_fast(model, data, prior, likelihoods):
    '''prior is an array of size 5 (num classes), w categorical distribution
        E_i is the new evidence that has been captured - {SHAPE_PRED, VOL_PRED, ASPECT}, 
        data is 
        '''
    # evidence = 0.0
    # # for yi in range(1, 6):
    # #     edf_i = bn.inference.fit(model, variables=['SHAPE_PRED', 'VOL_PRED', 'CLASS_PRED', 'ASPECT'], evidence={}, verbose=0).df
    # evidence = np.array(edf[(edf['SHAPE_PRED'] == data[0]) & (edf['VOL_PRED'] == data[1]) & (edf['CLASS_PRED'] == data[2]) & (edf['ASPECT'] == data[3])]['p'])[0]
    # print(evidence)
    post = np.zeros_like(prior)
    likelihood = np.zeros_like(prior)
    for i in range(len(likelihood)):
        likelihood_i = likelihoods[i]
        likelihood_i = likelihood_i[(likelihood_i['SHAPE_PRED'] == data[0]) & (likelihood_i['VOL_PRED'] == data[1]) & (likelihood_i['CLASS_PRED'] == data[2]) & (likelihood_i['ASPECT'] == data[3])]['p'].iloc[0]
        likelihood[i] = likelihood_i
    # print(likelihoods)
    # joint = np.array(joints[(joints['SHAPE_PRED'] == data[0]) & (joints['VOL_PRED'] == data[1]) & (joints['CLASS_PRED'] == data[2]) & (joints['ASPECT'] == data[3])]['p'])
    post = likelihood*prior/(likelihood*prior).sum()
    return post 

def bayes_update_imvp(model, data, prior):
    '''prior is an array of size 5 (num classes), w categorical distribution
        E_i is the new evidence that has been captured - {SHAPE_PRED, VOL_PRED, ASPECT}, 
        data is 
        '''
    # evidence = 0.0
    # # for yi in range(1, 6):
    # #     edf_i = bn.inference.fit(model, variables=['SHAPE_PRED', 'VOL_PRED', 'CLASS_PRED', 'ASPECT'], evidence={}, verbose=0).df
    # evidence = np.array(edf[(edf['SHAPE_PRED'] == data[0]) & (edf['VOL_PRED'] == data[1]) & (edf['CLASS_PRED'] == data[2]) & (edf['ASPECT'] == data[3])]['p'])[0]
    # print(evidence)
    post = np.zeros_like(prior)
    likelihood = np.zeros_like(prior)
    for i in range(len(likelihood)):
        likelihood_i = bn.inference.fit(model, variables=['SHAPE_PRED', 'VOL_PRED', 'CLASS_PRED', 'ASPECT'], evidence={'CLASS':i+1}, verbose=0).df
        likelihood_i = likelihood_i[(likelihood_i['SHAPE_PRED'] == data[0]) & (likelihood_i['VOL_PRED'] == data[1]) & (likelihood_i['CLASS_PRED'] == data[2]) & (likelihood_i['ASPECT'] == data[3])]['p'].iloc[0]
        likelihood[i] = likelihood_i
    # print(likelihoods)
    # joint = np.array(joints[(joints['SHAPE_PRED'] == data[0]) & (joints['VOL_PRED'] == data[1]) & (joints['CLASS_PRED'] == data[2]) & (joints['ASPECT'] == data[3])]['p'])
    post = likelihood*prior/(likelihood*prior).sum()
    return post 

def pZ_given_Evidence_Aspect(model, pY_given_Mi_k, Aspect_k_plus1):
    #takes in the belief using data up to this timestep k 
    #takes in an arbitrary aspect value, calculates the probability of the features
    Ys = np.arange(1, 6)
    pZ_given_Evidence_Aspect = np.zeros(100)
    for y_i in Ys:
        edf = np.array(bn.inference.fit(model, variables=['SHAPE_PRED', 'VOL_PRED', 'CLASS_PRED'], evidence={'CLASS':y_i, 'ASPECT':Aspect_k_plus1}, verbose=0).df['p'])
        #edf is a 100 len np array representing all the Zi_k+1s
        #now we multiply by p(Y_i | Mi_k)
        prior = pY_given_Mi_k[y_i-1]
        pZ_given_Evidence_Aspect += edf*prior
    return pZ_given_Evidence_Aspect
    
def ecl(model, pY_given_Mi_k, Aspect_i_kplus1):
    #returns a (100) array of probabilities for each Ei(k+1)
    #go through each Ei - shape pred, vol pred, class pred 
    a = time.time()
    shapes = np.arange(1,6)
    vols = np.arange(1,5)
    classes = np.arange(1,6)
    pZ_given_evidence_aspect = pZ_given_Evidence_Aspect(model, pY_given_Mi_k, Aspect_i_kplus1)
    cls = []

    likelihoods = [bn.inference.fit(model, variables=['SHAPE_PRED', 'VOL_PRED', 'CLASS_PRED', 'ASPECT'], evidence={'CLASS':i}, verbose=0).df for i in classes]

    joints = []
    for y_i in range(5):
        #go through each possible y_i 
        l_i = likelihoods[y_i][likelihoods[y_i]['ASPECT']==Aspect_i_kplus1]['p']*pY_given_Mi_k[y_i] #calculate the joint here 
        joints.append(l_i)
    joints = np.stack(joints)
    post = joints/(joints.sum(axis=0))
    cl = post.max(axis=0)
    # for s in shapes: 
    #     for v in vols: 
    #         for cl in classes:
    #             a = time.time()
    #             CL = bayes_update_imvp_fast(model, [s, v, cl, Aspect_i_kplus1], pY_given_Mi_k, likelihoods).max()
    #             b = time.time()
    #             # print(b-a)
    #             #CL is a 5 len array of a distribution over Yi that is then maxxed 
    #             cls.append(CL)

    # cls = np.array(cls)
    ecl = (cl*pZ_given_evidence_aspect).sum()
    b = time.time()
    return ecl

def create_model():
    #create a DAG in bnlearn 
    edges = [('CLASS', 'SHAPE'),
            ('CLASS', 'VOL'), 
            ('SHAPE', 'SHAPE_PRED'),
            ('VOL', 'SHAPE_PRED'),
            ('SHAPE', 'VOL_PRED'),
            ('VOL', 'VOL_PRED'),
            ('SHAPE', 'CLASS_PRED'), 
                ('VOL', 'CLASS_PRED'),
            ('ASPECT', 'SHAPE_PRED'), 
            ('ASPECT', 'VOL_PRED'), 
            ('ASPECT', 'CLASS_PRED')]
    DAG = bn.make_DAG(edges)
    df_raw = pd.read_csv('/home/advaith/Documents/optix_sidescan/imvp_predictions_14.csv')
    dfhot, dfnum = bn.df2onehot(df_raw)
    model = bn.parameter_learning.fit(DAG, dfnum)
    return model

def eval_survey(dataset, net, num_surveys, num_targets, seed):
    model = create_model()

    '''
    1. We randomly sample a target from the dataset
    2. then we randomly sample a first view 
    3. calculate the ECL for all viewing aspects 
    4. Choose the aspect with the maximal ECL 
    5. Update the trajectory'''
    net.eval()
    num_angles=6

    #genenerate random indices of num_surveysx num_targets 
    trajectory_views = []
    accuracies = []
    recalls = []
    trajectory = []
    for i in tqdm(range(num_surveys)):
        indices = np.random.choice(len(dataset), num_targets, replace=False)
        # indices = np.arange(len(dataset))
        imvp_dec = []
        gt = []
        for idx in tqdm(indices, desc='Target'):
            imgs, lbl, file = dataset[idx]
            threshold = 0.85

            prior = np.array(bn.inference.fit(model, variables=['CLASS'], evidence={}, verbose=0).df['p'])
            current_cl = 0
            num_views = 0
            chosen_views = []
            while ((num_views == 0) or (current_cl/num_views < threshold)):
                # rand_views = np.random.choice(num_angles, num_rand, replace=False)
                if len(chosen_views) >= num_angles:
                    break
                view_ecls = [ecl(model, prior, asp) for asp in range(1, num_angles+1)]
                
                #take the argmax of views that arent already chosen 
                for view in chosen_views:
                    view_ecls[view] = 0.0
                chosen_view = np.argmax(view_ecls)
                current_cl += view_ecls[chosen_view]
                chosen_views.append(chosen_view)
                img = imgs[chosen_view]
                class_pred, vol_pred = net(img.unsqueeze(0).cuda())
                data = [class_pred.argmax().item()+1, vol_pred.argmax().item()+1, class_pred.argmax().item()+1, chosen_view+1]
                post = bayes_update_imvp(model, data, prior)
                prior = post 
                num_views += 1
            # for i in range(num_angles):
            #     img = imgs[i]
            #     class_pred, vol_pred = net(img.unsqueeze(0).cuda())
            #     data = [class_pred.argmax().item()+1, vol_pred.argmax().item()+1, class_pred.argmax().item()+1, i+1]
            #     post = bayes_update_imvp(model, data, prior)
            #     prior = post 
            trajectory.append(chosen_views)
            gt.append(lbl)
            imvp_dec.append(post.argmax())
        
        #calculate accuracy and recall for run 
        imvp_dec = np.array(imvp_dec)
        gt = np.array(gt)
        #calculate the accuracy using sklearn
        
        acc = accuracy_score(gt, imvp_dec)
        recall = recall_score(gt, imvp_dec, average='macro')
        accuracies.append(acc)
        recalls.append(recall)

    
    print("Average accuracy: ", np.mean(accuracies))
    print("Average recall: ", np.mean(recalls))

    print("std accuracy", np.std(accuracies))
    print("std recall", np.std(recalls))
    
    

    #save trajectory views as a list 
    np.save("trajectory_views_IMVP{}.npy".format(seed), np.array(trajectory, dtype=object))





    
def visualize_val_set(dataset, net, dset_type="val"):
    net.eval()
    features = []
    labels = []
    C_class = []
    OR_preds  = []
    ensemble_preds = []

    Cs = []
    arr = np.arange(net.num_angles)
    for i in range(1,net.num_angles+1):
        all_combs = list(combinations(arr, i))
        Cs += all_combs #add the actual indices 


    for i in tqdm(range(len(dataset)), desc="{} dataset".format(dset_type)):
        data, lbl, _ = dataset[i]
        pred, feat = net(data.cuda(), feat=True)
        pred = torch.softmax(pred, dim=1)
        for c in Cs: 
            valid_preds = pred[np.array(c)]
            ensemble_decision = torch.mean(valid_preds, dim=0, keepdim=True).argmax(dim=1)
            OR_decision = torch.max(valid_preds, dim=0, keepdim=True).values.argmax(dim=1)
            ensemble_preds.append(ensemble_decision.cpu().detach().numpy())
            OR_preds.append(OR_decision.cpu().detach().numpy())
            C_class.append(len(c))
            labels.append(lbl)

    #calculate the accuracy for each C_class 
    C_class = np.array(C_class)
    OR_preds = np.array(OR_preds).squeeze()
    ensemble_preds = np.array(ensemble_preds).squeeze()
    labels = np.array(labels)
    en_accuracies = []
    OR_accuracies = []
    for c in np.unique(C_class):
        en_accuracies.append(np.sum(ensemble_preds[C_class==c]==labels[C_class==c])/len(labels[C_class==c]))
        OR_accuracies.append(np.sum(OR_preds[C_class==c]==labels[C_class==c])/len(labels[C_class==c]))
    
    #pretty print them 
    print("Ensemble accuracies: ", en_accuracies)
    print("OR accuracies: ", OR_accuracies)

    #plot the accuracies
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16,10))
    plt.plot(np.unique(C_class), en_accuracies, "-o", label="Ensemble")
    plt.plot(np.unique(C_class), OR_accuracies, "-o", label="OR")
    plt.xlabel("Number of views")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("C_class_accuracy_sv.png")



        # features.append(torch.softmax(pred, dim=1).cpu().detach().numpy())
        # labels.append(lbl)
        # pred = torch.softmax(pred, dim=1)
        # ensemble_decision = torch.mean(pred[:,1]) #take average of all the outputs
        # OR_decision = torch.max(pred[:,1]) #OR all outputs
        # pred = torch.argmax(pred, dim=1)
        # OR_preds.append(OR_decision.cpu().detach().numpy())
        # ensemble_preds.append(ensemble_decision.cpu().detach().numpy())

    
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    OR_preds = np.array(OR_preds)
    ensemble_preds = np.array(ensemble_preds)

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
    fpr, tpr, _ = roc_curve(labels, OR_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(16,10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title("ROC Curve for OR on {} data".format(dset_type))
    plt.savefig("{}_roc_or.png".format(dset_type))

    #get optimal threshold 
    from sklearn.metrics import f1_score
    f1_scores = []
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        f1_scores.append(f1_score(labels, OR_preds>threshold))
    f1_scores = np.array(f1_scores)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print("Best F1 Score for OR: ", np.max(f1_scores))
    sns.set()
    plt.figure(figsize=(16,10))
    cm = confusion_matrix(labels, OR_preds > optimal_threshold, normalize='true')
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Real Rock', 'Real Mine'], yticklabels=['Real Rock', 'Real Mine'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CM for OR on {} data".format(dset_type))
    plt.savefig("{}_cm_or.png".format(dset_type))

    #plot the precision recall curve 
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(labels, OR_preds)
    precision, recall, _ = precision_recall_curve(labels, OR_preds)
    plt.figure(figsize=(16,10))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall curve:on {} data".format(dset_type))
    plt.savefig("{}_pr_or.png".format(dset_type))

    #do the same thing for ensemble preds
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, ensemble_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(16,10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title("ROC Curve for Ensemble on {} data".format(dset_type))
    plt.savefig("{}_roc_ensemble.png".format(dset_type))

    #get optimal threshold 
    from sklearn.metrics import f1_score
    f1_scores = []
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        f1_scores.append(f1_score(labels, ensemble_preds>threshold))
    f1_scores = np.array(f1_scores)
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    sns.set()
    plt.figure(figsize=(16,10))
    cm = confusion_matrix(labels, ensemble_preds > optimal_threshold, normalize='true')
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Real Rock', 'Real Mine'], yticklabels=['Real Rock', 'Real Mine'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CM for Ensemble on {} data".format(dset_type))
    plt.savefig("{}_cm_ensemble.png".format(dset_type))

    #plot the precision recall curve
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(labels, ensemble_preds)
    precision, recall, _ = precision_recall_curve(labels, ensemble_preds)
    plt.figure(figsize=(16,10))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall curve: on {} data".format(average_precision, dset_type))
    plt.savefig("{}_pr_ensemble.png".format(dset_type))
    
    print("Best F1 Score for Ensemble: ", np.max(f1_scores))

    #visualize tsne plot of all features with labels for synth and real
    # from sklearn.manifold import TSNE
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set()
    # plt.figure(figsize=(16,10))
    # tsne_results = np.concatenate([sfeatures, real_features], axis=0)
    # sns.scatterplot(
    #     x=tsne_results[:,0], y=tsne_results[:,1],
    #     hue=np.concatenate([synth_labels, real_labels]),
    #     palette=sns.color_palette("tab10", 4),
    #     legend="full",
    #     alpha=1.0
    # )
    # plt.xlabel("Rockness")
    # plt.ylabel("Mineness")
    # #plot the predicted labels as a contour map 

    # #change the legend labels
    # # plt.legend(['Synthetic Rock ', 'Synth Mine', 'Real Rock', 'Real Mine'])
    # plt.savefig("tsne.png")


def train_bn():
    data_path = '/home/advaith/Documents/optix_sidescan/bn.csv'


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
    
    num_classes = np.unique(list(name_to_lbl.values())).size+1
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

    #go through all the folders and read the volume.txt files 
    train_folder_filtered = []
    for folder in train_folders:
        #read the volume.txt file
        with open(os.path.join(data_dir, folder, 'volume.txt'), 'r') as f:
            vol = float(f.read())
        if not np.isinf(vol):
            train_folder_filtered.append(folder)
    
    #do for val and test
    val_folder_filtered = []
    for folder in val_folders:
        #read the volume.txt file
        with open(os.path.join(data_dir, folder, 'volume.txt'), 'r') as f:
            vol = float(f.read())
        if not np.isinf(vol):
            val_folder_filtered.append(folder)
    
    test_folder_filtered = []
    for folder in test_folders:
        #read the volume.txt file
        with open(os.path.join(data_dir, folder, 'volume.txt'), 'r') as f:
            vol = float(f.read())
        if not np.isinf(vol):
            test_folder_filtered.append(folder)

    print("Train folders: ", len(train_folders), len(train_folder_filtered))
    print("Val folders: ", len(val_folders), len(val_folder_filtered))
    print("Test folders: ", len(test_folders), len(test_folder_filtered))

    # #filter the folders with .pngs in them 
    # folders = [folder for folder in folders if len(glob.glob(os.path.join(data_dir, folder, '*.png')))>0]
    # # random.shuffle(folders)
    # train_val_mask = np.zeros(len(folders), dtype=np.bool)
    # #randomly choose 80% to be true 
    #set np random seed 
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # train_val_mask[np.random.choice(len(train_val_mask), int(0.8*len(train_val_mask)), replace=False)] = True
    dataset = IMVPDataset(train_folder_filtered,  data_dir, input_tf)
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # np.random.seed(0)
    # torch.manual_seed(0)
    val_dataset = IMVPDataset(val_folder_filtered, data_dir, val_tf)

    test_dataset = SingleviewDataset(test_folder_filtered, data_dir, input_tf, group=True)

    #print line of - 
    print("-"*50)
    print("Train dataset size: ", len(dataset))
    print("Val dataset size: ", len(val_dataset))

    train_imgs = set(train_folders)
    val_imgs = set(val_folders)
    test_imgs = set(test_folders)
    print("Train and val imgs overlap: ", len(train_imgs.intersection(val_imgs)))
    print("Train and test imgs overlap: ", len(train_imgs.intersection(test_imgs)))
    print("Val and test imgs overlap: ", len(val_imgs.intersection(test_imgs)))
    print("-"*50)

    #create net here
    net = imvp_classifier(dataset, val_dataset, num_workers, batch_size, lr, num_classes, num_angles, l1_lambda)
    net.cuda()


    # net2 = sv_sss_classifier(0, 0, 0, 0, 0, 2, num_angles, 0)
    # net2.cuda()
    # net2.eval()
    # checkpoint = torch.load('/home/advaith/Documents/optix_sidescan/tb_logs/my_model_sv/version_44/checkpoints/epoch=149-step=16800.ckpt')
    # net2.load_state_dict(checkpoint["state_dict"])
    # print("Loaded frm checkpoint")

    # #svm experiment 
    # X = []
    # Y = []
    # for i in tqdm(range(len(dataset))):
    #     data, lbl, _ = dataset[i]
    #     pred, feat = net(data.cuda().unsqueeze(0), feat=True)
    #     X.append(feat.cpu().detach().numpy())
    #     Y.append(lbl)
    # X = np.concatenate(X, axis=0)
    # Y = np.array(Y)

    # #do the same for val dataset 
    # X_val = []
    # Y_val = []
    # for i in tqdm(range(len(val_dataset))):
    #     data, lbl, _ = val_dataset[i]
    #     pred, feat = net(data.cuda(), feat=True)
    #     X_val.append(feat.cpu().detach().numpy())
    #     Y_val.append(lbl)
    # X_val = np.concatenate(X_val, axis=0)
    # Y_val = np.array(Y_val)

    # #create a pipeline
    # clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, verbose=1))
    # clf.fit(X, Y)

    # #predict on val set
    # val_preds = clf.predict(X_val)
    # print("Val accuracy: ", np.sum(val_preds==Y_val)/len(Y_val))



    # # data, lbl, angles = dataset[0]
    # # pred = net(data.cuda().unsqueeze(0))

    
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    checkpoint = torch.load('/home/advaith/Documents/optix_sidescan/tb_logs/imvp_14/version_30/best-epoch=49-val_accuracy=0.89.ckpt')
    net.load_state_dict(checkpoint["state_dict"])
    print("Loaded frm checkpoint")
    #load weights if resume is true with pytorch lightning checkpoint 
    logger = TensorBoardLogger("tb_logs", name="imvp_{}".format(seed))
    from pytorch_lightning.strategies.ddp import DDPStrategy
    checkpoint_callback = ModelCheckpoint(monitor='val_volume_accuracy', dirpath=logger.log_dir, filename='best-{epoch:02d}-{val_accuracy:.2f}', save_top_k=3, mode='max')
    trainer = pl.Trainer(logger=logger, max_epochs=num_epochs, accelerator="gpu", devices=num_gpus, default_root_dir=weight_save_dir, 
                         check_val_every_n_epoch=10, num_sanity_val_steps=1, 
                         precision=16, 
                         callbacks=[checkpoint_callback])
    #start training

    # volumes = []
    # for data in val_dataset:
    #     img, lbl, vol_lbl, _ = data 
    #     volumes.append(vol_lbl)
    # for data in dataset:
    #     img, lbl, vol_lbl, _ = data 
    #     volumes.append(vol_lbl)
    # for data in test_dataset:
    #     img, lbl, vol_lbl, _ = data 
    #     volumes.append(vol_lbl)
    # print(np.max(volumes))

    # trainer.fit(net)

    #perform one validation loop 
    # trainer.validate(net)
    # visualize_val_set(val_dataset, net, dset_type="synthetic")
    eval_survey(test_dataset, net,  100, 16, seed)
    
if __name__ == "__main__":
    gin.parse_config_file(sys.argv[1])
    main()