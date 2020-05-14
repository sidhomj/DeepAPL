from DeepAPL.DeepAPL import DeepAPL_SC
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
name = 'all_blast_post2018_rmrbc'
name = 'ig_test_discovery'

gpu = 1
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
DAPL = DeepAPL_SC(data,gpu)
DAPL.Import_Data(directory=None, Load_Prev_Data=True)

df_meta = pd.read_csv('../Data/master.csv')
df_meta['Date of Diagnosis'] = df_meta['Date of Diagnosis'].astype('datetime64[ns]')
df_meta.sort_values(by='Date of Diagnosis',inplace=True)
# df_meta =df_meta[df_meta['Date of Diagnosis']>= '2018-01-01']
df_meta = df_meta[df_meta['Cohort']=='Discovery']

idx_samples_keep = np.isin(DAPL.patients,df_meta['JH Number'])
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
cell_type_keep = np.isin(DAPL.cell_type,cell_types)
idx_keep = idx_samples_keep*cell_type_keep
label_dict = dict(zip(df_meta['JH Number'],df_meta['Diagnosis']))

DAPL_train = DeepAPL_SC(name,gpu)
DAPL_train.imgs = DAPL.imgs[idx_keep]
DAPL_train.patients = DAPL.patients[idx_keep]
DAPL_train.cell_type = DAPL.cell_type[idx_keep]
DAPL_train.files = DAPL.files[idx_keep]
DAPL_train.smears = DAPL.smears[idx_keep]
#
img_w = []
pt_w = []
cell_type_w = []
files_w = []
smears_w = []
num_w = 10
for pt in np.unique(DAPL_train.patients):
    # img_w_temp = np.ones(shape=[num_w,DAPL_train.imgs.shape[1],DAPL_train.imgs.shape[2],DAPL_train.imgs.shape[3]])
    # img_b_temp = np.zeros(shape=[num_w,DAPL_train.imgs.shape[1],DAPL_train.imgs.shape[2],DAPL_train.imgs.shape[3]])
    # img_add = np.concatenate([img_w_temp,img_b_temp])
    # img_add = DAPL_train.imgs[DAPL_train.patients==pt][0]
    img_add = DAPL_train.imgs[DAPL_train.patients==pt]
    img_add = np.stack([cv2.GaussianBlur(x,(101,101),100) for x in img_add])
    # img_add = cv2.GaussianBlur(img_add, (101, 101), 100)
    # img_add = np.stack([img_add] * num_w)
    img_w.append(img_add)
    pt = pt + '_'
    label_dict[pt] = 'out'
    pt_w.append(np.array([pt]*num_w))
    cell_type_w.append(np.array(['None']*num_w))
    files_w.append(np.array(['None']*num_w))
    smears_w.append(np.array(['None']*num_w))

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

folds = 100
seeds = np.array(range(folds))
# seeds = np.array([1])
epochs_min = 5
graph_seed = 0
DAPL_train.Monte_Carlo_CrossVal(folds=folds,seeds=seeds,epochs_min=epochs_min,
                          stop_criterion=0.25,test_size=0.25,graph_seed=graph_seed,
                          weight_by_class=True)
DAPL_train.Get_Cell_Predicted()
with open(name+'.pkl', 'wb') as f:
    pickle.dump([DAPL_train.Cell_Pred,DAPL_train.w,DAPL_train.imgs,
                DAPL_train.patients,DAPL_train.cell_type,DAPL_train.files,DAPL_train.smears,
                DAPL_train.labels,DAPL_train.Y,DAPL_train.predicted,DAPL_train.lb],f,protocol=4)

df = DAPL_train.Cell_Pred[DAPL_train.Cell_Pred['Counts']>=1]
df = df[df['Label']!='out']
df['y_test'] = df['Label']=='APL'
roc_auc_score(df['y_test'],df['APL'])





