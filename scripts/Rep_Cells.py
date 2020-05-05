from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
import pandas as pd

gpu = 1

classes = ['AML','APL']
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
device = '/device:GPU:'+ str(gpu)
DAPL = DeepAPL_SC('load_data_norm_nmf',device=device)
DAPL.Import_Data(directory='Data/All', Load_Prev_Data=True, classes=classes,
                 include_cell_types=cell_types)

pred_file = 'Cell_Preds_blast_norm_nmf_2018plus.pkl'
mask_file = 'Cell_Masks_blast_norm_nmf_2018plus.pkl'

with open(pred_file,'rb') as f:
    DAPL.Cell_Pred = pickle.load(f)
with open(mask_file,'rb') as f:
    DAPL.w = pickle.load(f)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
df_meta = pd.read_csv('../Data/master.csv')
df_meta['Date of Diagnosis'] = df_meta['Date of Diagnosis'].astype('datetime64[ns]')
df_meta.sort_values(by='Date of Diagnosis',inplace=True)
df_meta =df_meta[df_meta['Date of Diagnosis']>= '2018-01-01']

idx_keep = np.where(DAPL.Cell_Pred['Patient'].isin(df_meta['JH Number']))[0]
DAPL.w = DAPL.w[idx_keep]
DAPL.Cell_Pred = DAPL.Cell_Pred[DAPL.Cell_Pred['Patient'].isin(df_meta['JH Number'])].reset_index(drop=True)
idx_samples_keep = np.isin(DAPL.patients,df_meta['JH Number'])
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
cell_type_keep = np.isin(DAPL.cell_type,cell_types)
idx_keep = idx_samples_keep*cell_type_keep
label_dict = dict(zip(df_meta['JH Number'],df_meta['Diagnosis']))

DAPL.imgs = DAPL.imgs[idx_keep]
DAPL.patients = DAPL.patients[idx_keep]
DAPL.cell_type = DAPL.cell_type[idx_keep]
DAPL.files = DAPL.files[idx_keep]
DAPL.smears = DAPL.smears[idx_keep]
DAPL.labels = np.array([label_dict[x] for x in DAPL.patients])
DAPL.lb = LabelEncoder().fit(DAPL.labels)
DAPL.Y = DAPL.lb.transform(DAPL.labels)
DAPL.Y = OneHotEncoder(sparse=False).fit_transform(DAPL.Y.reshape(-1,1))
DAPL.predicted = np.zeros((len(DAPL.Y), len(DAPL.lb.classes_)))
DAPL.lb.fit(DAPL.Cell_Pred['Label'])

DAPL.Representative_Cells('AML',9)
plt.tight_layout()
DAPL.Representative_Cells('APL',12,Load_Prev_Data=True,prob_show=False,show_title=False)
plt.tight_layout()

cell_types = ['Blast, no lineage spec', 'Promonocyte', 'Promyelocyte', 'Myelocyte', 'Metamyelocyte', ]
DAPL.Representative_Cells('APL',12,Load_Prev_Data=True,cell_type = cell_types[4])
DAPL.Representative_Cells('AML',12,Load_Prev_Data=True,cell_type=cell_types[4])
