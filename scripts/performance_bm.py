import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import confusion_matrix
from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
import matplotlib
import pandas as pd
import copy
matplotlib.rc('font', family='Times New Roman')
gpu = 1

files = glob.glob('../Data/BM_Results/*')
key = pd.read_csv('../Data/key.csv')
key_dict = dict(zip(key['ID'],key['Diagnosis']))
tpr_list = []
fpr_list = []
for file in files:
    df = pd.read_csv(file)
    df['call'] = df['call'].str.upper()
    df['label'] = df['patient'].map(key_dict)
    df['y_pred'] = (df['call']=='APL').astype(int)
    df['y_test'] = (df['label']=='APL').astype(int)
    (tn, fp, fn, tp) = confusion_matrix(df['y_test'],df['y_pred']).ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    fpr = fp/(fp+tn)
    tpr_list.append(sens)
    fpr_list.append(fpr)


name = 'validation_all'
file = 'WF/validation_all.pkl'

DAPL = DeepAPL_SC('temp')
with open(file,'rb') as f:
    DAPL.Cell_Pred,DAPL.DFs_pred,DAPL.imgs,\
    DAPL.patients,DAPL.cell_type,DAPL.files,\
    DAPL.smears,DAPL.labels,DAPL.Y,DAPL.predicted,DAPL.lb = pickle.load(f)

#remove cells that do not have training data or are in the blurred out group
DAPL.Cell_Pred = DAPL.Cell_Pred[DAPL.Cell_Pred['Counts']>=1]
DAPL.Cell_Pred = DAPL.Cell_Pred[DAPL.Cell_Pred['Label']!='out']

#map patients to label
label_dict = pd.DataFrame()
label_dict['Patient'] = DAPL.patients
label_dict['Label'] = DAPL.labels
label_dict.drop_duplicates(inplace=True)
label_dict = dict(zip(label_dict['Patient'],label_dict['Label']))


#Sample Level Performance MIL
df_agg = DAPL.DFs_pred['APL'].groupby(['Samples']).agg({'y_pred':'mean'}).reset_index()
df_agg = df_agg[~df_agg['Samples'].str.endswith('_')]
df_agg['Label'] = df_agg['Samples'].map(label_dict)
df_agg.rename(columns={'y_pred':'APL'},inplace=True)
df_agg.set_index('Samples',inplace=True)
sample_summary = copy.deepcopy(df_agg)


plt.figure()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=24)
plt.ylabel('True Positive Rate',fontsize=24)
y_test = np.asarray(sample_summary['Label']) == 'APL'
y_pred = np.asarray(sample_summary['APL'])
roc_score = roc_auc_score(y_test,y_pred)
fpr, tpr, th = roc_curve(y_test, y_pred)
id = 'CNN'
plt.plot(fpr, tpr, lw=2, label='%s (%0.3f)' % (id, roc_score),c='grey')
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = th[optimal_idx]

df_promy = pd.DataFrame()
df_promy['Patient'] = DAPL.Cell_Pred['Patient']
df_promy['Cell Type'] = DAPL.Cell_Pred['Cell_Type']
df_promy['Cell Type'].value_counts()
df_promy['Pro'] =  df_promy['Cell Type'] == 'Promyelocyte'
df_promy_agg = df_promy.groupby(['Patient']).agg({'Pro':'sum'})

df_promy_tc = df_promy['Patient'].value_counts().to_frame()
df_pro = pd.concat([df_promy_agg,df_promy_tc],axis=1)

df_pro['Label'] = df_pro.index.map(label_dict)
bin_dict = {'AML':0,'APL':1}
df_pro['Label_Bin'] = df_pro['Label'].map(bin_dict)
df_pro['Pro_Prop'] = df_pro['Pro']/df_pro['Patient']
pro_dict = dict(zip(df_pro.index,df_pro['Pro_Prop']))

y_test = np.array(df_pro['Label_Bin'])
y_pred = np.array(df_pro['Pro_Prop'])
roc_score = roc_auc_score(y_test,y_pred)
fpr, tpr, th = roc_curve(y_test, y_pred)
id = 'Proportion of Promyelocytes'
plt.plot(fpr, tpr, lw=2, label='%s (%0.3f)' % (id, roc_score),c='blue')

plt.legend(loc="lower right",prop={'size':14},frameon=False)
plt.tight_layout()
ax = plt.gca()
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.scatter(fpr_list,tpr_list,c='r',marker='+',s=500,zorder=10)

