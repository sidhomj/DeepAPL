from DeepAPL.DeepAPL import DeepAPL_WF
import os
import numpy as np
import pickle
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
warnings.filterwarnings('ignore')
import cv2
from sklearn.metrics import roc_auc_score

data = 'load_data'
name = 'discovery_blasts_2'
blasts = True

name = 'discovery_all'
blasts = False

#open model
gpu = 2
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
DAPL = DeepAPL_WF(data,gpu)
DAPL.Import_Data(directory=None, Load_Prev_Data=True)

#load metadata & select data in discovery cohort for training
df_meta = pd.read_csv('../../Data/master.csv')
df_meta['Date of Diagnosis'] = df_meta['Date of Diagnosis'].astype('datetime64[ns]')
df_meta.sort_values(by='Date of Diagnosis',inplace=True)
df_meta = df_meta[df_meta['Cohort']=='Discovery']

#select samples in discovery and cell types for training
idx_samples_keep = np.isin(DAPL.patients,df_meta['JH Number'])
if blasts:
    cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
    cell_type_keep = np.isin(DAPL.cell_type,cell_types)
    idx_keep = idx_samples_keep*cell_type_keep
else:
    idx_keep = idx_samples_keep
label_dict = dict(zip(df_meta['JH Number'],df_meta['Diagnosis']))

DAPL_train = DeepAPL_WF(name,gpu)
DAPL_train.imgs = DAPL.imgs[idx_keep]
DAPL_train.patients = DAPL.patients[idx_keep]
DAPL_train.cell_type = DAPL.cell_type[idx_keep]
DAPL_train.files = DAPL.files[idx_keep]
DAPL_train.smears = DAPL.smears[idx_keep]

#add blurred images as a third outgroup to encourage learning of morphological features
img_w = []
pt_w = []
cell_type_w = []
files_w = []
smears_w = []
for pt in np.unique(DAPL_train.patients):
    img_add = DAPL_train.imgs[DAPL_train.patients==pt]
    img_add = np.stack([cv2.GaussianBlur(x,(101,101),100) for x in img_add])
    img_w.append(img_add)
    pt = pt + '_'
    label_dict[pt] = 'out'
    pt_w.append(np.array([pt]*len(img_add)))
    cell_type_w.append(np.array(['None']*len(img_add)))
    files_w.append(np.array(['None']*len(img_add)))
    smears_w.append(np.array(['None']*len(img_add)))

img_w = np.vstack(img_w)
pt_w = np.hstack(pt_w)
cell_type_w = np.hstack(cell_type_w)
files_w = np.hstack(files_w)
smears_w = np.hstack(smears_w)

DAPL_train.imgs = np.concatenate([DAPL_train.imgs,img_w],axis=0)
DAPL_train.patients = np.concatenate([DAPL_train.patients,pt_w])
DAPL_train.cell_type = np.concatenate([DAPL_train.cell_type,cell_type_w])
DAPL_train.files = np.concatenate([DAPL_train.files,files_w])
DAPL_train.smears = np.concatenate([DAPL_train.smears,smears_w])

DAPL_train.labels = np.array([label_dict[x] for x in DAPL_train.patients])
DAPL_train.lb = LabelEncoder().fit(DAPL_train.labels)
DAPL_train.Y = DAPL_train.lb.transform(DAPL_train.labels)
DAPL_train.Y = OneHotEncoder(sparse=False).fit_transform(DAPL_train.Y.reshape(-1,1))
DAPL_train.predicted = np.zeros((len(DAPL_train.Y), len(DAPL_train.lb.classes_)))

#Set training parameters and train in MC fashion
folds = 100
seeds = np.array(range(folds))
epochs_min = 10
graph_seed = 0
subsample = 25
DAPL_train.Monte_Carlo_CrossVal(folds=folds,seeds=seeds,epochs_min=epochs_min,
                          stop_criterion=0.25,test_size=0.25,graph_seed=graph_seed,
                          weight_by_class=True,subsample=subsample,combine_train_valid=True,
                                train_loss_min=0.25,learning_rate=0.001)

DAPL_train.Get_Cell_Predicted()
with open(name+'.pkl', 'wb') as f:
    pickle.dump([DAPL_train.Cell_Pred,DAPL_train.DFs_pred,
                 DAPL_train.imgs,
                DAPL_train.patients,DAPL_train.cell_type,DAPL_train.files,DAPL_train.smears,
                DAPL_train.labels,DAPL_train.Y,DAPL_train.predicted,DAPL_train.lb],f,protocol=4)

import copy
df_pred = copy.deepcopy(DAPL_train.DFs_pred)
df_pred_apl = df_pred['APL']
df_pred_apl = df_pred_apl[~df_pred_apl['Samples'].str.endswith('_')]
roc_auc_score(df_pred_apl['y_test'],df_pred_apl['y_pred'])

agg = df_pred['APL'].groupby(['Samples']).agg({'y_test':'first','y_pred':'mean'}).reset_index()
agg = agg[~agg['Samples'].str.endswith('_')]
roc_auc_score(agg['y_test'],agg['y_pred'])

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
plt.figure()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=24)
plt.ylabel('True Positive Rate',fontsize=24)
y_test = agg['y_test']
y_pred = agg['y_pred']
roc_score = roc_auc_score(y_test,y_pred)
fpr, tpr, th = roc_curve(y_test, y_pred)
id = 'APL'
plt.plot(fpr, tpr, lw=2, label='%s (%0.3f)' % (id, roc_score),c='grey')
plt.legend(loc="upper left",prop={'size':16})
plt.tight_layout()

