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
import random
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import SGDClassifier
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pytorch_lightning.loggers import TensorBoardLogger
random.seed(0)


def eval_survey(dataset, net, num_surveys, num_targets, seed, rand=False, single_view=False): 
    net.eval()

    #genenerate random indices of num_surveysx num_targets 
    en_accuracies = []
    en_CE = []
    OR_CE = []
    OR_accuracies = []
    trajectory_views = []
    for i in tqdm(range(num_surveys)):
        indices = np.random.choice(len(dataset), num_targets, replace=False)
        # indices = np.arange(len(dataset))
        en_survey_dec = []
        or_survey_dec = []
        gt = []
        for idx in indices: 
            data, lbl, _ = dataset[idx]
            pred = net(data.cuda())
            if rand: 
                num_views = np.random.randint(1, net.num_angles+1)
                valid_views = np.random.choice(net.num_angles, num_views, replace=False)
                trajectory_views.append(valid_views)
                pred = pred[valid_views]
            elif single_view: 
                num_views = 1
                valid_views = np.random.choice(net.num_angles, num_views, replace=False)
                trajectory_views.append(valid_views)
                pred = pred[valid_views]
                
            ensemble_decision = torch.mean(pred, dim=0, keepdim=True).argmax(dim=1)
            OR_decision = torch.max(pred, dim=0, keepdim=True).values.argmax(dim=1)
            en_survey_dec.append(ensemble_decision.cpu().detach().numpy())
            or_survey_dec.append(OR_decision.cpu().detach().numpy())
            gt.append(lbl)
        en_survey_dec = np.array(en_survey_dec).reshape(-1)
        or_survey_dec = np.array(or_survey_dec).reshape(-1)
        gt = np.array(gt)
        en_accuracies.append((en_survey_dec==gt).mean())
        OR_accuracies.append((or_survey_dec==gt).mean())

        #calculate the recall using sklearn 
        from sklearn.metrics import recall_score
        en_recall = recall_score(gt, en_survey_dec, average='macro')
        OR_recall = recall_score(gt, or_survey_dec, average='macro')
        en_CE.append(en_recall)
        OR_CE.append(OR_recall)
    
    print("Mean en accuracy: ", np.mean(en_accuracies))
    print("Std en accuracy: ", np.std(en_accuracies))

    print("Mean OR accuracy: ", np.mean(OR_accuracies))
    print("Std OR accuracy: ", np.std(OR_accuracies))

    print("Mean en CE: ", np.mean(en_CE))
    print("Std en CE: ", np.std(en_CE))

    print("Mean OR CE: ", np.mean(OR_CE))
    print("Std OR CE: ", np.std(OR_CE))

    #save all lists as np arrays 
    np.save("en_accuracies_lm_{}_rand_{}_exh_{}.npy".format(single_view, rand, not rand), np.array(en_accuracies))
    np.save("OR_accuracies_lm_{}_rand_{}_exh_{}.npy".format(single_view, rand, not rand), np.array(OR_accuracies))
    np.save("en_CE_lm_{}_rand_{}_exh_{}.npy".format(single_view, rand, not rand), np.array(en_CE))
    np.save("OR_CE_lm_{}_rand_{}_exh_{}.npy".format(single_view, rand, not rand), np.array(OR_CE))

    #save trajectory views as a list 
    np.save("trajectory_views_lm_{}_rand_{}_exh_{}.npy".format(single_view, rand, not rand), np.array(trajectory_views, dtype=object))

    #save the accuracies 


    #calculate the confusion matrices 


    
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

    # #filter the folders with .pngs in them 
    # folders = [folder for folder in folders if len(glob.glob(os.path.join(data_dir, folder, '*.png')))>0]
    # # random.shuffle(folders)
    # train_val_mask = np.zeros(len(folders), dtype=np.bool)
    # #randomly choose 80% to be true 
    #set np random seed 
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # train_val_mask[np.random.choice(len(train_val_mask), int(0.8*len(train_val_mask)), replace=False)] = True
    dataset = SingleviewDataset(train_folders,  data_dir, input_tf, group=False)
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # np.random.seed(0)
    # torch.manual_seed(0)
    val_dataset = SingleviewDataset(val_folders, data_dir, input_tf, group=False)

    test_dataset = SingleviewDataset(test_folders, data_dir, input_tf, group=True)

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
    net = sv_sss_classifier(dataset, val_dataset, num_workers, batch_size, lr, num_classes, num_angles, l1_lambda)
    net.cuda()


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
    checkpoint = torch.load('/home/advaith/Documents/optix_sidescan/tb_logs/my_model_sv_14/version_8/checkpoints/epoch=999-step=22000.ckpt')
    net.load_state_dict(checkpoint["state_dict"])
    print("Loaded frm checkpoint")
    #load weights if resume is true with pytorch lightning checkpoint 
    logger = TensorBoardLogger("tb_logs", name="my_model_sv_{}".format(seed))
    from pytorch_lightning.strategies.ddp import DDPStrategy
    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', dirpath=logger.log_dir, filename='best-{epoch:02d}-{val_accuracy:.2f}', save_top_k=3, mode='max')
    trainer = pl.Trainer(logger=logger, max_epochs=num_epochs, accelerator="gpu", devices=num_gpus, default_root_dir=weight_save_dir, 
                         check_val_every_n_epoch=10, num_sanity_val_steps=1, 
                         precision=16,  
                         callbacks=[checkpoint_callback])
    #start training
    # trainer.fit(net)

    #perform one validation loop 
    # trainer.validate(net)
    # visualize_val_set(val_dataset, net, dset_type="synthetic")
    eval_survey(test_dataset, net, 100, 16, seed, rand=True, single_view=False)
    
if __name__ == "__main__":
    gin.parse_config_file(sys.argv[1])
    main()