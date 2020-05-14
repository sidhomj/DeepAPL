from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

data = 'load_data'
name = 'discovery_blast_2018'
name = 'ig_test_discovery'
name_out = 'validation_blast_2018'
name_out = 'ig_test_validation'
num_mc = 100

gpu = 1
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
DAPL = DeepAPL_SC(data,gpu)
DAPL.Import_Data(directory=None, Load_Prev_Data=True)

df_meta = pd.read_csv('../Data/master.csv')
df_meta['Date of Diagnosis'] = df_meta['Date of Diagnosis'].astype('datetime64[ns]')
df_meta.sort_values(by='Date of Diagnosis',inplace=True)
# df_meta =df_meta[df_meta['Date of Diagnosis']>= '2018-01-01']
# df_meta = df_meta[df_meta['Cohort']=='Discovery']
df_meta = df_meta[df_meta['Cohort']=='Validation']

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
DAPL_train.labels = np.array([label_dict[x] for x in DAPL_train.patients])
DAPL_train.lb = LabelEncoder().fit(DAPL_train.labels)
DAPL_train.Y = DAPL_train.lb.transform(DAPL_train.labels)
DAPL_train.Y = OneHotEncoder(sparse=False).fit_transform(DAPL_train.Y.reshape(-1,1))
DAPL_train.predicted = np.zeros((len(DAPL_train.Y), len(DAPL_train.lb.classes_)))
DAPL_train.Ensemble_Inference()
DAPL_train.counts = np.ones_like(DAPL_train.predicted)*num_mc
DAPL_train.Get_Cell_Predicted()

with open(name_out+'.pkl','wb') as f:
    pickle.dump([DAPL_train.Cell_Pred,DAPL_train.w,DAPL_train.imgs,
                DAPL_train.patients,DAPL_train.cell_type,DAPL_train.files,DAPL_train.smears,
                DAPL_train.labels,DAPL_train.Y,DAPL_train.predicted,DAPL_train.lb],f,protocol=4)
check=1