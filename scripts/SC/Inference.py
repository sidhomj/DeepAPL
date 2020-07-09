"""
This script is used to apply a model trained on the discovery cohort to the validation cohort in ensemble.
"""
from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
warnings.filterwarnings('ignore')

data = 'load_data'
name = 'discovery_blasts'
name_out = 'validation_blasts'
blasts = True

# name = 'discovery_all'
# name_out = 'validation_all'
# blasts = False

#Load Trained Model
gpu = 1
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
DAPL = DeepAPL_SC(data,gpu)
DAPL.Import_Data(directory=None, Load_Prev_Data=True)

#Get data from Validation Cohort
df_meta = pd.read_csv('../../Data/master.csv')
df_meta = df_meta[df_meta['Cohort']=='Validation']

idx_samples_keep = np.isin(DAPL.patients,df_meta['Patient_ID'])
if blasts:
    cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
    cell_type_keep = np.isin(DAPL.cell_type,cell_types)
    idx_keep = idx_samples_keep*cell_type_keep
else:
    idx_keep = idx_samples_keep
label_dict = dict(zip(df_meta['Patient_ID'],df_meta['Diagnosis']))

DAPL_train = DeepAPL_SC(name,gpu)
DAPL_train.imgs = DAPL.imgs[idx_keep]
DAPL_train.patients = DAPL.patients[idx_keep]
DAPL_train.cell_type = DAPL.cell_type[idx_keep]
DAPL_train.files = DAPL.files[idx_keep]
DAPL_train.smears = DAPL.smears[idx_keep]
DAPL_train.labels = np.array([label_dict[x] for x in DAPL_train.patients])
DAPL_train.lb = LabelEncoder().fit(['AML','APL','out'])
DAPL_train.Y = DAPL_train.lb.transform(DAPL_train.labels)
DAPL_train.Y = OneHotEncoder(sparse=False).fit_transform(DAPL_train.Y.reshape(-1,1))
DAPL_train.predicted = np.zeros((len(DAPL_train.Y), len(DAPL_train.lb.classes_)))

#Conduct Inference over ensemble of trained models on discovery cohort
DAPL_train.Ensemble_Inference()
DAPL_train.Get_Cell_Predicted()

with open(name_out+'.pkl','wb') as f:
    pickle.dump([DAPL_train.Cell_Pred,DAPL_train.imgs,
                DAPL_train.patients,DAPL_train.cell_type,DAPL_train.files,DAPL_train.smears,
                DAPL_train.labels,DAPL_train.Y,DAPL_train.predicted,DAPL_train.lb],f,protocol=4)