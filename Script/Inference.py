#!/usr/bin/env python
# coding: utf-8

# In[47]:


import cv2, random, os, glob, math, pathlib, csv, PIL
import zipfile

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pylab as pylab
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pydicom

from os import listdir
from PIL import Image
from tqdm import tqdm 
from pathlib import Path
from skimage import io


# In[2]:


import torch
import torchvision
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt


# In[48]:


train_df = pd.read_csv('/home/Sean/[Report]LungUS/LungUS/DATA/Data_label/[Insternal]_train_LUS_SNUH.csv')
valid_df = pd.read_csv('/home/Sean/[Report]LungUS/LungUS/DATA/Data_label/[Insternal]_valid_LUS_SNUH.csv')
test_df = pd.read_csv('/home/Sean/[Report]LungUS/LungUS/DATA/Data_label/[Insternal]_test_LUS_SNUH.csv')
temp_df = pd.read_csv('/home/Sean/[Report]LungUS/LungUS/DATA/Data_label/[External]Temporally_separated.csv')


# In[4]:


labels = ['A', 
          'B', 
          'Effusion', 
          'Consolidation']

columns = test_df.keys()
columns = list(columns)
print(columns)

# Remove unnecesary elements
columns.remove('Patient_id')
columns.remove('age')
columns.remove('sex')
columns.remove('Observation')
columns.remove('filename')

class_names = columns

print(f"There are {len(columns)} columns of labels for these conditions: {columns} \n")

for column in columns:
    print(f"The class {column} has {test_df[column].sum()} samples")


# In[5]:


test_df = test_df.drop(list(test_df.iloc[:,:4].columns.values),axis = 1)
print("Size of test set : {}".format(len(test_df)))


# In[6]:


test_df


# # Load Model

# In[17]:


from tqdm import tqdm
from PIL import Image
from os import listdir
from operator import itemgetter
from collections import OrderedDict
from torchvision import*

import timm
import torch
import torchvision
from torch import optim,nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"'{device}' is avilable.")
model = timm.create_model('tf_efficientnet_b0', pretrained = True)
out_dim    = 4

# adjust classifier
model.classifier = nn.Linear(model.classifier.in_features, out_dim)
model.sigmoid = nn.Sigmoid()
#model = nn.DataParallel(model, device_ids = [0,1])   # 2개의 GPU를 이용할 경우
model.to(device);


# In[18]:


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)

import sys  
sys.path.insert(0, '/home/Sean/pytorchtools/')

from pytorchtools import EarlyStopping


# In[19]:


model.load_state_dict(torch.load('/home/Sean/[Report]LungUS/LungUS/Results/Model_weights/model.pt'))
optimizer.load_state_dict(torch.load('/home/Sean/[Report]LungUS/LungUS/Results/Model_weights/optimizer.pt'))


# In[20]:


class US_Dataset(torch.utils.data.Dataset):

    def __init__(self, data , img_dir, transform):
        self.data = data
        self.img_dir = img_dir 
        self.transform = transform 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file = self.img_dir + self.data.iloc[:,0][idx]
        img = Image.open(img_file).convert('RGB')
        label = np.array(self.data.iloc[:,1:].iloc[idx])

        if self.transform:
            img = self.transform(img)

        return img,label
    
def create_datasets(batch_size):

    # percentage of training set to use as validation
    #valid_size = 0.2

    # convert data to torch.FloatTensor
    #transform = transforms.ToTensor()
    train_transforms  = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(10),
                                            transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                                            transforms.ToTensor(),  # Convert numpy array to tensor
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225]),  # Use mean and std from preprocessing notebook
                                           ])

    val_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),  # Convert numpy array to tensor
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])  # Use mean and std from preprocessing notebook
                                        ])
    
    # choose the training and test datasets
    basepath = '/home/Sean/[Report]LungUS/LungUS' # %pwd
    img_directory = basepath + '/DATA/Preprocessed_PNG/SNUH/'
    
    train_dataset = US_Dataset(train_df,img_dir = img_directory, transform = train_transforms)
    valid_dataset = US_Dataset(valid_df, img_dir = img_directory, transform = val_transforms)
    test_dataset = US_Dataset(test_df, img_dir = img_directory, transform = val_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, test_loader, valid_loader


# In[21]:


batch_size = 128
n_epochs = 1000
num_workers = 4

train_loader, test_loader, valid_loader = create_datasets(batch_size)


# In[22]:


preds = []
labels = []

# eval mode
model.eval()

with torch.no_grad():
    try :
        for data, label in tqdm(test_loader.dataset):
            data = data.to(device).float().unsqueeze(0)
            pred = torch.sigmoid(model(data)[0].cpu()).numpy()
            preds.append(pred)
            labels.append(label)
            
        preds = torch.tensor(preds)
        labels = torch.tensor(labels).int()
    
    except (RuntimeError, TypeError, NameError, ValueError, KeyError):
        pass


# # Load Data

# # Test Eval

# In[27]:


test_results = pd.DataFrame(preds, columns = ['A_pred', 'B_pred', 'Effusion_pred', 'Consolidation_pred'])
test_results.to_csv('/home/Sean/[Report]LungUS/LungUS/Results/Validation_results/Classification_reports/SNUH_internal_test.csv',index=False)
test_results = pd.read_csv("/home/Sean/[Report]LungUS/LungUS/Results/Validation_results/Classification_reports/SNUH_internal_test.csv")
# 여기 수정 필수
test_results = pd.concat([test_df, test_results], axis = 1) 

# the labels in our dataset
class_labels = ['A', 'B', 'Effusion', 'Consolidation']
# the labels for prediction values in our dataset
pred_labels = [l + "_pred" for l in class_labels]

y = test_results[class_labels].values
pred = test_results[pred_labels].values
test_results[np.concatenate([class_labels, pred_labels])].head()


# In[34]:


Youden_Threshold = [[0.2074], [0.2328], 
                    [0.6411],[0.2357]]

Youden_Threshold


# In[35]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# In[36]:



def get_true_pos(y, pred, th=Youden_Threshold):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 1))


def get_true_neg(y, pred, th=Youden_Threshold):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 0))


def get_false_neg(y, pred, th=Youden_Threshold):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 1))


def get_false_pos(y, pred, th=Youden_Threshold):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 0))


def get_performance_metrics(y, pred, class_labels, tp=get_true_pos,
                            tn=get_true_neg, fp=get_false_pos,
                            fn=get_false_neg,
                            acc=None, prevalence=None, spec=None,
                            sens=None, ppv=None, npv=None, auc=None, f1=None,
                            thresholds=[]):
    if len(thresholds) != len(class_labels):
        thresholds = [0.2074, 0.2328, 0.6411, 0.2357] * len(class_labels)

    columns = ["", "TP", "TN", "FP", "FN", "Accuracy", "Prevalence",
               "Sensitivity",
               "Specificity", "PPV", "NPV", "AUC", "F1", "Threshold"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(class_labels)):
        df.loc[i] = [""] + [0] * (len(columns) - 1)
        df.loc[i][0] = class_labels[i]
        df.loc[i][1] = round((tp(y[:, i], pred[:, i])//4),
                             3) if tp != None else "Not Defined"
        df.loc[i][2] = round((tn(y[:, i], pred[:, i])//4),
                             3) if tn != None else "Not Defined"
        df.loc[i][3] = round((fp(y[:, i], pred[:, i])//4),
                             3) if fp != None else "Not Defined"
        df.loc[i][4] = round((fn(y[:, i], pred[:, i])//4),
                             3) if fn != None else "Not Defined"
        df.loc[i][5] = round(acc(y[:, i], pred[:, i], thresholds[i]),
                             3) if acc != None else "Not Defined"
        df.loc[i][6] = round(prevalence(y[:, i]),
                             3) if prevalence != None else "Not Defined"
        df.loc[i][7] = round(sens(y[:, i], pred[:, i], thresholds[i]),
                             3) if sens != None else "Not Defined"
        df.loc[i][8] = round(spec(y[:, i], pred[:, i], thresholds[i]),
                             3) if spec != None else "Not Defined"
        df.loc[i][9] = round(ppv(y[:, i], pred[:, i], thresholds[i]),
                             3) if ppv != None else "Not Defined"
        df.loc[i][10] = round(npv(y[:, i], pred[:, i], thresholds[i]),
                              3) if npv != None else "Not Defined"
        df.loc[i][11] = round(auc(y[:, i], pred[:, i]),
                              3) if auc != None else "Not Defined"
        df.loc[i][12] = round(f1(y[:, i], pred[:, i] > thresholds[i]),
                              3) if f1 != None else "Not Defined"
        df.loc[i][13] = round(thresholds[i], 3)

    df = df.set_index("")
    return df



def print_confidence_intervals(class_labels, statistics):
    df = pd.DataFrame(columns=["Mean AUC (CI 5%-95%)"])
    for i in range(len(class_labels)):
        mean = statistics.mean(axis=1)[i]
        max_ = np.quantile(statistics, .95, axis=1)[i]
        min_ = np.quantile(statistics, .05, axis=1)[i]
        df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
    return df


def get_curve(gt, pred, target_names, curve='roc'):
    for i in range(len(target_names)):
        if curve == 'roc':
            curve_function = roc_curve
            auc_roc = roc_auc_score(gt[:, i], pred[:, i])
            label = target_names[i] + " AUC: %.3f " % auc_roc
            xlabel = "False positive rate"
            ylabel = "True positive rate"
            a, b, _ = curve_function(gt[:, i], pred[:, i])
            plt.figure(1, figsize=(7, 7))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(a, b, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)
        elif curve == 'prc':
            precision, recall, _ = precision_recall_curve(gt[:, i], pred[:, i])
            average_precision = average_precision_score(gt[:, i], pred[:, i])
            label = target_names[i] + " Avg.: %.3f " % average_precision
            plt.figure(1, figsize=(7, 7))
            plt.step(recall, precision, where='post', label=label)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)

            
def true_positives(y, pred, th=Youden_Threshold):
    TP = 0    
    # get thresholded predictions
    thresholded_preds = pred >= th
    # compute TP
    TP = np.sum((y == 1) & (thresholded_preds == 1))
    return TP

def true_negatives(y, pred, th=Youden_Threshold):
    TN = 0
    # get thresholded predictions
    thresholded_preds = pred >= th

    # compute TN
    TN = np.sum((y == 0 ) & (thresholded_preds == 0 ))
    return TN

def false_positives(y, pred, th=Youden_Threshold):
    FP = 0
    
    # get thresholded predictions
    thresholded_preds = pred >= th
    # compute FP
    FP = np.sum((y == 0) & (thresholded_preds == 1))
    return FP

def false_negatives(y, pred, th=Youden_Threshold):
    FN = 0
    
    # get thresholded predictions
    thresholded_preds = pred >= th
    # compute FN
    FN = np.sum((y == 1) & (thresholded_preds == 0))
    return FN


def get_accuracy(y, pred, th=Youden_Threshold):
    accuracy = 0.0
    TP = true_positives(y, pred, th)   
    FP = false_positives(y, pred, th)
    TN = true_negatives(y, pred, th)
    FN = false_negatives(y,pred, th)

    # Compute accuracy using TP, FP, TN, FN
    accuracy = (TP + TN) / ( TP + FP + TN + FN)
    
    return accuracy

def get_prevalence(y):
    prevalence = 0.0
    prevalence = np.mean(y)
    return prevalence


def get_sensitivity(y, pred, th=Youden_Threshold):

    sensitivity = 0.0
    # get TP and FN using our previously defined functions
    TP = true_positives(y,pred, th)
    FN = false_negatives(y, pred, th)

    # use TP and FN to compute sensitivity
    sensitivity = TP / (TP + FN)
    return sensitivity

def get_specificity(y, pred, th=Youden_Threshold):
    specificity = 0.0
    
    # get TN and FP using our previously defined functions
    TN = true_negatives(y,pred, th)
    FP = false_positives(y, pred, th)
    
    # use TN and FP to compute specificity 
    specificity = TN / (TN + FP)
    return specificity


def get_ppv(y, pred, th=Youden_Threshold):
    PPV = 0.0
    
    # get TP and FP using our previously defined functions
    TP = true_positives(y,pred,th)
    FP = false_positives(y,pred,th)

    # use TP and FP to compute PPV
    PPV = TP / (TP + FP)
    return PPV

def get_npv(y, pred, th=Youden_Threshold):
    NPV = 0.0
    # get TN and FN using our previously defined functions
    TN = true_negatives(y,pred,th)
    FN = false_negatives(y,pred,th)

    # use TN and FN to compute NPV
    NPV = TN / (TN + FN)
    return NPV


# In[31]:


get_curve(y, pred, class_labels)


# In[32]:


from sklearn.metrics import f1_score
get_performance_metrics(y, pred, class_labels, acc=get_accuracy, 
                        prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, 
                        ppv=get_ppv, npv=get_npv, 
                        auc=roc_auc_score,f1=f1_score)


# In[33]:


test_dataset = test_loader.dataset

test_resize_transform = transforms.Compose([ transforms.Resize((224, 224))])
test_dataset_vis = US_Dataset(test_df, 
                              img_dir = '/home/Sean/[Report]LungUS/LungUS/DATA/Preprocessed_PNG/',
                              transform = test_resize_transform)

print(len(test_loader.dataset))


# In[38]:


rand_val = range(len(test_loader.dataset))
start_idx, end_idx =  0, len(test_loader.dataset)
print(start_idx, end_idx)


# In[42]:


print(model.bn2)


# In[43]:


from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

LABEL_INT2STR = {0:'A', 1:'B',
                 2:'Effusion', 3:'Consolidation'}

target_layer = [model.bn2] 
cam = GradCAMPlusPlus(model=model, target_layers=target_layer, use_cuda=True)


# In[41]:


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]
    


# In[47]:


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

for i in range(start_idx, end_idx):
    feature, label = test_dataset[i]
    img, _ = test_dataset_vis[i] 
    #print(img, _)
    feature = torch.unsqueeze(feature, 0)
    img = np.asanyarray(img)
    img = img.astype(np.float)/255 
    
    A_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(0)],aug_smooth=False,eigen_smooth=False)
    A_cam = A_cam[0, :] 
    A_cam_image = show_cam_on_image(img, A_cam, use_rgb=True)
    

    B_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(1)],aug_smooth=False,eigen_smooth=False)
    B_cam = B_cam[0, :]
    B_cam_image = show_cam_on_image(img, B_cam, use_rgb=True)
    
    Effusion_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(2)],aug_smooth=False,eigen_smooth=False)
    Effusion_cam = Effusion_cam[0, :]
    Effusion_cam_image = show_cam_on_image(img, Effusion_cam, use_rgb=True)
    
    Consolidation_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(3)],aug_smooth=False,eigen_smooth=False)
    Consolidation_cam = Consolidation_cam[0, :]
    Consolidation_cam_image = show_cam_on_image(img, Consolidation_cam, use_rgb=True)
    
    fig,ax = plt.subplots(1, 5, figsize=(20, 7))
    ax[0].imshow(img)
    ax[0].set_title(f'original image({labels[i]})')
    
    if pred[i][0] > 0.2074 :
        A_cam_image = show_cam_on_image(img, A_cam, use_rgb=True)
        ax[1].imshow(A_cam_image)
        ax[1].set_title(f'Normal-line heatmap (Conf:{pred[i][0]:.2f})')
    else:
        A_cam = A_cam*0 
        A_cam_image = show_cam_on_image(img, A_cam, use_rgb=True)
        ax[1].imshow(A_cam_image)
        ax[1].set_title(f'Normal-line heatmap (Conf:{pred[i][0]:.2f})')
    
    
    if pred[i][1] > 0.2328 :
        B_cam_image = show_cam_on_image(img, B_cam, use_rgb=True)
        ax[2].imshow(B_cam_image)
        ax[2].set_title(f'B-line heatmap (Conf:{pred[i][1]:.2f})')
    else:
        B_cam = B_cam*0 
        B_cam_image = show_cam_on_image(img, B_cam, use_rgb=True)
        ax[2].imshow(B_cam_image)
        ax[2].set_title(f'B-line heatmap (Conf:{pred[i][1]:.2f})')
    
    
    if pred[i][2] > 0.6411 :
        Effusion_cam_image = show_cam_on_image(img, Effusion_cam, use_rgb=True)
        ax[3].imshow(Effusion_cam_image)
        ax[3].set_title(f'Effusion heatmap (Conf:{pred[i][2]:.2f})')
    else:
        Effusion_cam = Effusion_cam*0 
        Effusion_cam_image = show_cam_on_image(img, Effusion_cam, use_rgb=True)
        ax[3].imshow(Effusion_cam_image)
        ax[3].set_title(f'Effusion heatmap (Conf:{pred[i][2]:.2f})')
    
    
    if pred[i][3] > 0.2357 :
        Consolidation_cam_image = show_cam_on_image(img, Consolidation_cam, use_rgb=True)
        ax[4].imshow(Consolidation_cam_image)
        ax[4].set_title(f'Consolidation heatmap (Conf:{ pred[i][3]:.2f})')
    else:
        Consolidation_cam = Consolidation_cam*0 
        Consolidation_cam_image = show_cam_on_image(img, Consolidation_cam, use_rgb=True)
        ax[4].imshow(Consolidation_cam_image)
        ax[4].set_title(f'Consolidation heatmap (Conf:{ pred[i][3]:.2f})')
    
    plt.suptitle(f'File Name: {test_dataset.data.iloc[:,0][i]} \n Ground Truth: {label}\ Probability : {pred[i][0]:.2f}')
    plt.savefig('/home/Sean/[Report]LungUS/LungUS/Results/Validation_results/GradCAM/SNUH_internal_test/SNUH_internal_test/GradCAM_{}.png'.format(i), bbox_inches='tight')


# # External-중앙대

# In[48]:


cauh_df = pd.read_csv('/home/Sean/[Report]LungUS/LungUS/DATA/Data_label/[External]CAUH.csv')
cauh_df


# In[49]:


cauh_path = '/home/Sean/[Report]LungUS/LungUS/DATA/Preprocessed_PNG/CAUH/'


# In[58]:


val_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),  # Convert numpy array to tensor
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])  # Use mean and std from preprocessing notebook
                                    ])

cauh_dataset = US_Dataset(cauh_df, 
                          img_dir = cauh_path, 
                          transform = val_transforms)

cauh_loader = torch.utils.data.DataLoader(cauh_dataset, 
                                          batch_size=batch_size, 
                                          num_workers=num_workers, 
                                          shuffle=False)
len(cauh_dataset)


# In[66]:


test_resize_transform = transforms.Compose([ transforms.Resize((224, 224))])


# In[67]:


cauh_dataset_vis = US_Dataset(cauh_df, 
                              img_dir = cauh_path,
                              transform = test_resize_transform)


# In[59]:


preds = []
labels = []


with torch.no_grad():
    try :
        for data, label in tqdm(cauh_loader.dataset):
            data = data.to(device).float().unsqueeze(0)
            pred = torch.sigmoid(model(data)[0].cpu()).numpy()
            preds.append(pred)
            labels.append(label)
            
        preds = torch.tensor(preds)
        labels = torch.tensor(labels).int()
    
    except (RuntimeError, TypeError, NameError, ValueError, KeyError):
        pass


# In[60]:


test_results = pd.DataFrame(preds, columns = ['A_pred', 'B_pred', 'Effusion_pred', 'Consolidation_pred'])
test_results.to_csv('/home/Sean/[Report]LungUS/LungUS/Results/Validation_results/Classification_reports/CAUH_external.csv',index=False)
test_results = pd.read_csv("/home/Sean/[Report]LungUS/LungUS/Results/Validation_results/Classification_reports/CAUH_external.csv")
# 여기 수정 필수
test_results = pd.concat([cauh_df, test_results], axis = 1) 

# the labels in our dataset
class_labels = ['A', 'B', 'Effusion', 'Consolidation']
# the labels for prediction values in our dataset
pred_labels = [l + "_pred" for l in class_labels]

y = test_results[class_labels].values
pred = test_results[pred_labels].values
test_results[np.concatenate([class_labels, pred_labels])].head()


# In[61]:


get_curve(y, pred, class_labels)


# In[62]:


get_performance_metrics(y, pred, class_labels, acc=get_accuracy, 
                        prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, 
                        ppv=get_ppv, npv=get_npv, 
                        auc=roc_auc_score,f1=f1_score)


# In[63]:


rand_val = range(len(cauh_loader.dataset))
start_idx, end_idx =  0, len(cauh_loader.dataset)
print(start_idx, end_idx)


# In[64]:


LABEL_INT2STR = {0:'A', 1:'B',
                 2:'Effusion', 3:'Consolidation'}

target_layer = [model.bn2] 
cam = GradCAMPlusPlus(model=model, target_layers=target_layer, use_cuda=True)


# In[68]:



for i in range(start_idx, end_idx):
    feature, label = cauh_dataset[i]
    img, _ = cauh_dataset_vis[i] 
    #print(img, _)
    feature = torch.unsqueeze(feature, 0)
    img = np.asanyarray(img)
    img = img.astype(np.float)/255 
    
    A_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(0)],aug_smooth=False,eigen_smooth=False)
    A_cam = A_cam[0, :] 
    A_cam_image = show_cam_on_image(img, A_cam, use_rgb=True)
    

    B_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(1)],aug_smooth=False,eigen_smooth=False)
    B_cam = B_cam[0, :]
    B_cam_image = show_cam_on_image(img, B_cam, use_rgb=True)
    
    Effusion_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(2)],aug_smooth=False,eigen_smooth=False)
    Effusion_cam = Effusion_cam[0, :]
    Effusion_cam_image = show_cam_on_image(img, Effusion_cam, use_rgb=True)
    
    Consolidation_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(3)],aug_smooth=False,eigen_smooth=False)
    Consolidation_cam = Consolidation_cam[0, :]
    Consolidation_cam_image = show_cam_on_image(img, Consolidation_cam, use_rgb=True)
    
    fig,ax = plt.subplots(1, 5, figsize=(20, 7))
    ax[0].imshow(img)
    ax[0].set_title(f'original image({labels[i]})')
    
    if pred[i][0] > 0.2074 :
        A_cam_image = show_cam_on_image(img, A_cam, use_rgb=True)
        ax[1].imshow(A_cam_image)
        ax[1].set_title(f'Normal-line heatmap (Conf:{pred[i][0]:.2f})')
    else:
        A_cam = A_cam*0 
        A_cam_image = show_cam_on_image(img, A_cam, use_rgb=True)
        ax[1].imshow(A_cam_image)
        ax[1].set_title(f'Normal-line heatmap (Conf:{pred[i][0]:.2f})')
    
    
    if pred[i][1] > 0.2328 :
        B_cam_image = show_cam_on_image(img, B_cam, use_rgb=True)
        ax[2].imshow(B_cam_image)
        ax[2].set_title(f'B-line heatmap (Conf:{pred[i][1]:.2f})')
    else:
        B_cam = B_cam*0 
        B_cam_image = show_cam_on_image(img, B_cam, use_rgb=True)
        ax[2].imshow(B_cam_image)
        ax[2].set_title(f'B-line heatmap (Conf:{pred[i][1]:.2f})')
    
    
    if pred[i][2] > 0.6411 :
        Effusion_cam_image = show_cam_on_image(img, Effusion_cam, use_rgb=True)
        ax[3].imshow(Effusion_cam_image)
        ax[3].set_title(f'Effusion heatmap (Conf:{pred[i][2]:.2f})')
    else:
        Effusion_cam = Effusion_cam*0 
        Effusion_cam_image = show_cam_on_image(img, Effusion_cam, use_rgb=True)
        ax[3].imshow(Effusion_cam_image)
        ax[3].set_title(f'Effusion heatmap (Conf:{pred[i][2]:.2f})')
    
    
    if pred[i][3] > 0.2357 :
        Consolidation_cam_image = show_cam_on_image(img, Consolidation_cam, use_rgb=True)
        ax[4].imshow(Consolidation_cam_image)
        ax[4].set_title(f'Consolidation heatmap (Conf:{ pred[i][3]:.2f})')
    else:
        Consolidation_cam = Consolidation_cam*0 
        Consolidation_cam_image = show_cam_on_image(img, Consolidation_cam, use_rgb=True)
        ax[4].imshow(Consolidation_cam_image)
        ax[4].set_title(f'Consolidation heatmap (Conf:{ pred[i][3]:.2f})')
    
    plt.suptitle(f'File Name: {cauh_dataset.data.iloc[:,0][i]} \n Ground Truth: {label}\ Probability : {pred[i][0]:.2f}')


# # Temp-Separated

# In[49]:


for column in columns:
    print(f"The class {column} has {temp_df[column].sum()} samples")


# In[50]:


temp_df = temp_df[['filename', 'A', 'B', 'Effusion', 'Consolidation']]
temp_df


# In[51]:


temp_path = '/home/Sean/[Report]LungUS/LungUS/DATA/Preprocessed_PNG/Temporally_separated/'

val_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),  # Convert numpy array to tensor
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])  # Use mean and std from preprocessing notebook
                                    ])

temp_dataset = US_Dataset(temp_df, 
                          img_dir = temp_path, 
                          transform = val_transforms)

temp_loader = torch.utils.data.DataLoader(temp_dataset, 
                                          batch_size=batch_size, 
                                          num_workers=num_workers, 
                                          shuffle=False)
len(temp_dataset)

test_resize_transform = transforms.Compose([ transforms.Resize((224, 224))])

temp_dataset_vis = US_Dataset(temp_df, 
                              img_dir = temp_path,
                              transform = test_resize_transform)


# In[52]:


preds = []
labels = []

# eval mode
#model.eval()

with torch.no_grad():
    try :
        for data, label in tqdm(temp_loader.dataset):
            data = data.to(device).float().unsqueeze(0)
            pred = torch.sigmoid(model(data)[0].cpu()).numpy()
            preds.append(pred)
            labels.append(label)
            
        preds = torch.tensor(preds)
        labels = torch.tensor(labels).int()
    
    except (RuntimeError, TypeError, NameError, ValueError, KeyError):
        pass


# In[54]:


test_results = pd.DataFrame(preds, columns = ['A_pred', 'B_pred', 'Effusion_pred', 'Consolidation_pred'])
test_results.to_csv('/home/Sean/[Report]LungUS/LungUS/Results/Validation_results/Classification_reports/SNUH_temporally_seperated.csv',index=False)
test_results = pd.read_csv("/home/Sean/[Report]LungUS/LungUS/Results/Validation_results/Classification_reports/SNUH_temporally_seperated.csv")
# 여기 수정 필수
test_results = pd.concat([temp_df, test_results], axis = 1) 

# the labels in our dataset
class_labels = ['A', 'B', 'Effusion', 'Consolidation']
# the labels for prediction values in our dataset
pred_labels = [l + "_pred" for l in class_labels]

y = test_results[class_labels].values
pred = test_results[pred_labels].values
test_results[np.concatenate([class_labels, pred_labels])].head()


# In[55]:


get_curve(y, pred, class_labels)
plt.savefig('/home/Sean/[Report]LungUS/LungUS/Results/Validation_results/Classification_reports/ROC_curves_Temporarily_Separated.png', bbox_inches='tight')


# In[56]:


from sklearn.metrics import f1_score

get_performance_metrics(y, pred, class_labels, acc=get_accuracy, 
                        prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, 
                        ppv=get_ppv, npv=get_npv, 
                        auc=roc_auc_score,f1=f1_score)


# In[57]:


get_performance_metrics(y, pred, class_labels, acc=get_accuracy, 
                        prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, 
                        ppv=get_ppv, npv=get_npv, 
                        auc=roc_auc_score,f1=f1_score).to_csv('/home/Sean/[Report]LungUS/LungUS/Results/Validation_results/Classification_reports/Confussion_matrics_Temporarily_Separated_external.csv',index=False)


# In[58]:


rand_val = range(len(temp_loader.dataset))
start_idx, end_idx =  0, len(temp_loader.dataset)
print(start_idx, end_idx)


# In[59]:


from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

LABEL_INT2STR = {0:'A', 1:'B',
                 2:'Effusion', 3:'Consolidation'}

target_layer = [model.bn2] 
cam = GradCAMPlusPlus(model=model, target_layers=target_layer, use_cuda=True)


# In[60]:


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

for i in range(start_idx, end_idx):
    feature, label = temp_dataset[i]
    img, _ = temp_dataset_vis[i] 
    #print(img, _)
    feature = torch.unsqueeze(feature, 0)
    img = np.asanyarray(img)
    img = img.astype(np.float)/255 
    
    A_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(0)],aug_smooth=False,eigen_smooth=False)
    A_cam = A_cam[0, :] 
    A_cam_image = show_cam_on_image(img, A_cam, use_rgb=True)
    

    B_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(1)],aug_smooth=False,eigen_smooth=False)
    B_cam = B_cam[0, :]
    B_cam_image = show_cam_on_image(img, B_cam, use_rgb=True)
    
    Effusion_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(2)],aug_smooth=False,eigen_smooth=False)
    Effusion_cam = Effusion_cam[0, :]
    Effusion_cam_image = show_cam_on_image(img, Effusion_cam, use_rgb=True)
    
    Consolidation_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(3)],aug_smooth=False,eigen_smooth=False)
    Consolidation_cam = Consolidation_cam[0, :]
    Consolidation_cam_image = show_cam_on_image(img, Consolidation_cam, use_rgb=True)
    
    fig,ax = plt.subplots(1, 5, figsize=(20, 7))
    ax[0].imshow(img)
    ax[0].set_title(f'original image({labels[i]})')
    
    if pred[i][0] > 0.2074 :
        A_cam_image = show_cam_on_image(img, A_cam, use_rgb=True)
        ax[1].imshow(A_cam_image)
        ax[1].set_title(f'Normal-line heatmap (Conf:{pred[i][0]:.2f})')
    else:
        A_cam = A_cam*0 
        A_cam_image = show_cam_on_image(img, A_cam, use_rgb=True)
        ax[1].imshow(A_cam_image)
        ax[1].set_title(f'Normal-line heatmap (Conf:{pred[i][0]:.2f})')
    
    
    if pred[i][1] > 0.2328 :
        B_cam_image = show_cam_on_image(img, B_cam, use_rgb=True)
        ax[2].imshow(B_cam_image)
        ax[2].set_title(f'B-line heatmap (Conf:{pred[i][1]:.2f})')
    else:
        B_cam = B_cam*0 
        B_cam_image = show_cam_on_image(img, B_cam, use_rgb=True)
        ax[2].imshow(B_cam_image)
        ax[2].set_title(f'B-line heatmap (Conf:{pred[i][1]:.2f})')
    
    
    if pred[i][2] > 0.6411 :
        Effusion_cam_image = show_cam_on_image(img, Effusion_cam, use_rgb=True)
        ax[3].imshow(Effusion_cam_image)
        ax[3].set_title(f'Effusion heatmap (Conf:{pred[i][2]:.2f})')
    else:
        Effusion_cam = Effusion_cam*0 
        Effusion_cam_image = show_cam_on_image(img, Effusion_cam, use_rgb=True)
        ax[3].imshow(Effusion_cam_image)
        ax[3].set_title(f'Effusion heatmap (Conf:{pred[i][2]:.2f})')
    
    
    if pred[i][3] > 0.2357 :
        Consolidation_cam_image = show_cam_on_image(img, Consolidation_cam, use_rgb=True)
        ax[4].imshow(Consolidation_cam_image)
        ax[4].set_title(f'Consolidation heatmap (Conf:{ pred[i][3]:.2f})')
    else:
        Consolidation_cam = Consolidation_cam*0 
        Consolidation_cam_image = show_cam_on_image(img, Consolidation_cam, use_rgb=True)
        ax[4].imshow(Consolidation_cam_image)
        ax[4].set_title(f'Consolidation heatmap (Conf:{ pred[i][3]:.2f})')
    
    plt.suptitle(f'File Name: {temp_dataset.data.iloc[:,0][i]} \n Ground Truth: {label}\ Probability : {pred[i][0]:.2f}')
    plt.savefig('/home/Sean/[Report]LungUS/LungUS/Results/Validation_results/GradCAM/SNUH_temporarily_separated/GradCAM_{}.png'.format(i), bbox_inches='tight')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




