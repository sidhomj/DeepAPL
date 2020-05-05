from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
import matplotlib
import pandas as pd
matplotlib.rc('font', family='Times New Roman')
gpu = 1

DAPL = DeepAPL_SC('temp')
file = 'Discovery_nmf2.pkl'
file = 'Validation_Inference_nmf2.pkl'
with open(file,'rb') as f:
    DAPL.Cell_Pred,DAPL.w,DAPL.imgs,\
    DAPL.patients,DAPL.cell_type,DAPL.files,\
    DAPL.smears,DAPL.labels,DAPL.Y,DAPL.predicted,DAPL.lb = pickle.load(f)

# df_meta = pd.read_csv('../Data/master.csv')
# df_meta['Date of Diagnosis'] = df_meta['Date of Diagnosis'].astype('datetime64[ns]')
# df_meta.sort_values(by='Date of Diagnosis',inplace=True)
# df_meta =df_meta[df_meta['Date of Diagnosis']>= '2018-01-01']
# idx_keep = np.where(DAPL.Cell_Pred['Patient'].isin(df_meta['JH Number']))[0]
# DAPL.Cell_Pred = DAPL.Cell_Pred.iloc[idx_keep].reset_index(drop=True)
# DAPL.imgs = DAPL.imgs[idx_keep]
# DAPL.predicted = DAPL.predicted[idx_keep]
# DAPL.w = DAPL.w[idx_keep]
DAPL.imgs[:,:,:,0:2]
a = DAPL.imgs[:,:,:,0:2]
b =np.expand_dims(DAPL.imgs[:,:,:,0],-1)
DAPL.imgs = np.concatenate((a,b),axis=-1)

DAPL.Representative_Cells('APL',9)
plt.tight_layout()
DAPL.Representative_Cells('APL',12,Load_Prev_Data=True,prob_show=False,show_title=False)
plt.tight_layout()

DAPL.Representative_Cells('AML',200)
plt.tight_layout()