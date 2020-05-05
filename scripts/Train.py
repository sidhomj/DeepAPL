from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
warnings.filterwarnings('ignore')

data = 'load_data'
name = 'discovery_blast'

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

folds = 10
seeds = np.array(range(folds))
epochs_min = 25
graph_seed = 0
DAPL.Monte_Carlo_CrossVal(folds=folds,seeds=seeds,epochs_min=epochs_min,
                          stop_criterion=0.25,test_size=0.25,graph_seed=graph_seed,
                          weight_by_class=True)
DAPL.Get_Cell_Predicted()
with open(name+'.pkl', 'wb') as f:
    pickle.dump([DAPL.Cell_Pred,DAPL.w,DAPL.imgs,
                DAPL.patients,DAPL.cell_type,DAPL.files,DAPL.smears,
                DAPL.labels,DAPL.Y,DAPL.predicted,DAPL.lb],f,protocol=4)





