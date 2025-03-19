import torch 
import numpy as np 
import pytorch_lightning as pl
import torchvision
from data.singleview_dataset import *
from models.singleview_classifier import sv_sss_classifier
import gin 
import sys
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import SGDClassifier
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pytorch_lightning.loggers import TensorBoardLogger


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


def main(): 
     net = sv_sss_classifier(dataset, val_dataset, num_workers, batch_size, lr, num_classes, num_angles, l1_lambda)
    net.cuda()