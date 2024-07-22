#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


train_df = pd.read_csv('/home/Sean/[Report]LungUS/LungUS/DATA/Data_label/[Insternal]_train_LUS_SNUH.csv')
valid_df = pd.read_csv('/home/Sean/[Report]LungUS/LungUS/DATA/Data_label/[Insternal]_valid_LUS_SNUH.csv')
test_df = pd.read_csv('/home/Sean/[Report]LungUS/LungUS/DATA/Data_label/[Insternal]_test_LUS_SNUH.csv')

train_df.head()


# In[3]:


train_df.columns

labels = ['A', 
          'B', 
          'Effusion', 
          'Consolidation']

columns = train_df.keys()
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
    print(f"The class {column} has {train_df[column].sum()} samples")


# In[4]:


train_df.iloc[:,:4]


# In[5]:


def check_for_leakage(df1, df2, patient_col):

    df1_patients_unique = set(df1[patient_col])
    df2_patients_unique = set(df2[patient_col])
    
    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)

    if patients_in_both_groups:
        leakage = True # boolean (true if there is at least 1 patient in both groups)
    else:
        leakage = False
    
    return leakage

print("leakage between train and valid: {}".format(check_for_leakage(train_df, valid_df, 'Patient_id')))
print("leakage between train and test: {}".format(check_for_leakage(train_df, test_df, 'Patient_id')))
print("leakage between valid and test: {}".format(check_for_leakage(valid_df, test_df, 'Patient_id')))

train_df = train_df.drop(list(train_df.iloc[:,:4].columns.values),axis = 1)
test_df = test_df.drop(list(test_df.iloc[:,:4].columns.values),axis = 1)
valid_df = valid_df.drop(list(valid_df.iloc[:,:4].columns.values),axis = 1)

print("Size of train set : {}".format(len(train_df)))
print("Size of valid set : {}".format(len(valid_df)))
print("Size of test set : {}".format(len(test_df)))


# # Model Developing

# In[6]:


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


# In[7]:


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


# In[8]:


get_ipython().run_line_magic('pwd', '')
# /home/Sean/[Report]LungUS/LungUS/DATA/Preprocessed_PNG/SNUH


# In[9]:


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


# In[10]:


batch_size = 128
n_epochs = 2000
num_workers = 4

train_loader, test_loader, valid_loader = create_datasets(batch_size)

# early stopping patience; how long to wait after last time validation loss improved.
patience = 100


# In[11]:


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

seed = 77
torch.manual_seed(seed)
np.random.seed(seed)

torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"'{device}' is avilable.")


# # Hyper- Setting

# In[12]:


model = timm.create_model('tf_efficientnet_b0', pretrained = True)
out_dim    = 4

# adjust classifier
model.classifier = nn.Linear(model.classifier.in_features, out_dim)
model.sigmoid = nn.Sigmoid()
#model = nn.DataParallel(model, device_ids = [0,1])   # 2개의 GPU를 이용할 경우
model.to(device);


# In[13]:


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)

import sys  
sys.path.insert(0, '/home/Sean/pytorchtools/')

from pytorchtools import EarlyStopping


# In[14]:


def train_model(model, batch_size, patience, n_epochs):
    
    train_losses = [] # to track the training loss as the model trains
    valid_losses = [] # to track the validation loss as the model trains
    avg_train_losses = [] # to track the average training loss per epoch as the model trains
    avg_valid_losses = [] # to track the average validation loss per epoch as the model trains
    
    early_stopping = EarlyStopping(patience=patience, verbose=True) # initialize the early_stopping object
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for batch, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad() # clear the gradients of all optimized variables
            data = data.to(device)
            target = target.to(device)
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            loss = criterion(output, target.float()) # calculate the loss
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            optimizer.step() # perform a single optimization step (parameter update)
            train_losses.append(loss.item()) # record training loss

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target.float()) # calculate the loss
            valid_losses.append(loss.item()) # record validation loss

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return  model, avg_train_losses, avg_valid_losses


# In[16]:


model, train_loss, valid_loss = train_model(model, batch_size, patience, n_epochs)


# # Visualizing the Loss and the Early Stopping Checkpoint
# 

# In[ ]:


# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('EfficientNet_B0_loss_plot.png', bbox_inches='tight')


# # Save Train Model

# In[ ]:


torch.save(model.state_dict(), '/home/Sean/[Report]LungUS/LungUS/Results/Model_weights/model.pt')
torch.save(optimizer.state_dict(), '/home/Sean/[Report]LungUS/LungUS/Results/Model_weights/optimizer.pt')
#torch.save(scheduler.state_dict(), 'saved_data/scheduler.pt')


# # Validation w/ Youden's J'Statics

# In[ ]:


preds = []
labels = []

# eval mode
model.eval()

with torch.no_grad():
    try :
        for data, label in tqdm(valid_loader.dataset):
            data = data.to(device).float().unsqueeze(0)
            pred = torch.sigmoid(model(data)[0].cpu()).numpy()
            preds.append(pred)
            labels.append(label)
            
        preds = torch.tensor(preds)
        labels = torch.tensor(labels).int()
    
    except (RuntimeError, TypeError, NameError, ValueError, KeyError):
        pass


# In[ ]:


valid_results = pd.DataFrame(preds, columns = ['A_pred', 'B_pred', 'Effusion_pred', 'Consolidation_pred'])
valid_results.to_csv('SNUH_internal_validation.csv',index=False)
valid_results = pd.read_csv("/home/Sean/[Report]LungUS/LungUS/Results/Validation_results/Classification_reports/SNUH_internal_validation.csv")
valid_results = pd.concat([valid_df, valid_results], axis = 1)

# the labels in our dataset
class_labels = ['A', 'B', 'Effusion', 'Consolidation']
# the labels for prediction values in our dataset
pred_labels = [l + "_pred" for l in class_labels]

y = valid_results[class_labels].values
pred = valid_results[pred_labels].values
valid_results[np.concatenate([class_labels, pred_labels])].head()


# In[ ]:


true_list = valid_df.iloc[:,1:]
y_list = true_list.to_numpy()


# In[ ]:


def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

threshold_B = Find_Optimal_Cutoff(y_list[:,1], pred[:,1])


# In[ ]:


threshold_A = Find_Optimal_Cutoff(y_list[:,0], pred[:,0])
threshold_B = Find_Optimal_Cutoff(y_list[:,1], pred[:,1])
threshold_Eff = Find_Optimal_Cutoff(y_list[:,2], pred[:,2])
threshold_Con = Find_Optimal_Cutoff(y_list[:,3], pred[:,3])

print(threshold_A, threshold_B, threshold_Eff, threshold_Con)


# In[ ]:


Youden_Threshold =  [threshold_A, threshold_B, threshold_Eff, threshold_Con]
Youden_Threshold


# In[ ]:


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


# In[ ]:


def get_performance_metrics(y, pred, class_labels, tp=get_true_pos,
                            tn=get_true_neg, fp=get_false_pos,
                            fn=get_false_neg,
                            acc=None, prevalence=None, spec=None,
                            sens=None, ppv=None, npv=None, auc=None, f1=None,
                            thresholds=[]):
    if len(thresholds) != len(class_labels):

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


# In[ ]:


from sklearn.metrics import f1_score

get_performance_metrics(y, pred, class_labels, acc=get_accuracy, 
                        prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, 
                        ppv=get_ppv, npv=get_npv, 
                        auc=roc_auc_score,f1=f1_score)


# In[ ]:


get_curve(y, pred, class_labels)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




