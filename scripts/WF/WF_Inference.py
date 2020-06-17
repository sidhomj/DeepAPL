from DeepAPL.DeepAPL import DeepAPL_SC, DeepAPL_WF
import os
import numpy as np
import pickle
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

data = 'load_data'
name = 'discovery_blasts'
name_out = 'validation_blasts'
blasts = True

name = 'discovery_all'
name_out = 'validation_all'
blasts = False

#Load Trained Model
gpu = 4
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
DAPL = DeepAPL_WF(data,gpu)
DAPL.Import_Data(directory=None, Load_Prev_Data=True)

#Get data from Validation Cohort
df_meta = pd.read_csv('../../Data/master.csv')
df_meta['Date of Diagnosis'] = df_meta['Date of Diagnosis'].astype('datetime64[ns]')
df_meta.sort_values(by='Date of Diagnosis',inplace=True)
df_meta = df_meta[df_meta['Cohort']=='Validation']


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
DAPL_train.labels = np.array([label_dict[x] for x in DAPL_train.patients])
DAPL_train.lb = LabelEncoder().fit(['AML','APL','out'])
DAPL_train.Y = DAPL_train.lb.transform(DAPL_train.labels)
DAPL_train.Y = OneHotEncoder(sparse=False).fit_transform(DAPL_train.Y.reshape(-1,1))
DAPL_train.predicted = np.zeros((len(DAPL_train.Y), len(DAPL_train.lb.classes_)))

#Conduct Inference over ensemble of trained models on discovery cohort
predicted,sample_list = DAPL_train.Ensemble_Inference()
DAPL_train.Get_Cell_Predicted()

with open(name_out+'.pkl','wb') as f:
    pickle.dump([DAPL_train.Cell_Pred,DAPL_train.DFs_pred,DAPL_train.imgs,
                DAPL_train.patients,DAPL_train.cell_type,DAPL_train.files,DAPL_train.smears,
                DAPL_train.labels,DAPL_train.Y,DAPL_train.predicted,DAPL_train.lb],f,protocol=4)

# df_test = pd.DataFrame()
# df_test['samples'] = DAPL_train.patients
# df_test['apl'] = DAPL_train.Y[:,1]
# df_test = df_test.groupby(['samples']).agg({'apl':'first'}).reset_index()
# label_dict = dict(zip(df_test['samples'],df_test['apl']))
#
# df_out = pd.DataFrame()
# df_out['samples'] = sample_list
# df_out['y_pred'] = predicted[:,1]
# df_out['y_test'] = df_out['samples'].map(label_dict)
# from sklearn.metrics import roc_auc_score, roc_curve
# roc_auc_score(df_out['y_test'],df_out['y_pred'])
#
# plt.figure()
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate',fontsize=24)
# plt.ylabel('True Positive Rate',fontsize=24)
# y_test = df_out['y_test']
# y_pred = df_out['y_pred']
# roc_score = roc_auc_score(y_test,y_pred)
# fpr, tpr, th = roc_curve(y_test, y_pred)
# id = 'APL'
# plt.plot(fpr, tpr, lw=2, label='%s (%0.3f)' % (id, roc_score),c='grey')
# plt.legend(loc="upper left",prop={'size':16})
# plt.tight_layout()


with open(name_out+'.pkl','wb') as f:
    pickle.dump([DAPL_train.Cell_Pred,DAPL_train.imgs,
                DAPL_train.patients,DAPL_train.cell_type,DAPL_train.files,DAPL_train.smears,
                DAPL_train.labels,DAPL_train.Y,DAPL_train.predicted,DAPL_train.lb],f,protocol=4)