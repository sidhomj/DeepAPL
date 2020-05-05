from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

gpu = 1
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
#Train Classifier on Discovery Cohort
classes = ['AML','APL']

#Select for only Immature cells
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
device = '/device:GPU:'+ str(gpu)
DAPL = DeepAPL_SC('all_class',device=device)
DAPL.Import_Data(directory='../Data/Final/Validation', Load_Prev_Data=False, classes=classes,color_norm=True,save_data=False)
DAPL.Ensemble_Inference()
DAPL.counts = np.ones_like(DAPL.predicted)*25
DAPL.Get_Cell_Predicted()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
plt.figure()
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=24)
plt.ylabel('True Positive Rate',fontsize=24)
y_test = np.asarray(DAPL.Cell_Pred['Label']) == 'APL'
y_pred = np.asarray(DAPL.Cell_Pred['APL'])
roc_score = roc_auc_score(y_test,y_pred)
fpr, tpr, th = roc_curve(y_test, y_pred)
id = 'APL'
plt.plot(fpr, tpr, lw=2, label='%s (%0.3f)' % (id, roc_score),c='grey')
plt.legend(loc="upper left",prop={'size':16})
plt.tight_layout()
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = th[optimal_idx]
ax = plt.gca()
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis='y', labelsize=16)

#Sample Level Performance
DAPL.Sample_Summary()
plt.figure()
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=24)
plt.ylabel('True Positive Rate',fontsize=24)
y_test = np.asarray(DAPL.sample_summary['Label']) == 'APL'
y_pred = np.asarray(DAPL.sample_summary['APL'])
roc_score = roc_auc_score(y_test,y_pred)
fpr, tpr, th = roc_curve(y_test, y_pred)
id = 'All Pts'
plt.plot(fpr, tpr, lw=2, label='%s (%0.3f)' % (id, roc_score),c='grey')
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = th[optimal_idx]

#Sample Level Performance with samples >= 10 cells
DAPL.Cell_Pred['n'] = 1
agg = DAPL.Cell_Pred.groupby(['Patient']).agg({'Label':'first','n':'sum'})

DAPL.Sample_Summary()
keep = np.array(list(agg[agg['n']>=10].index))
DAPL.sample_summary = DAPL.sample_summary[DAPL.sample_summary.index.isin(keep)]
# plt.figure()
# # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate',fontsize=16)
# plt.ylabel('True Positive Rate',fontsize=16)
y_test = np.asarray(DAPL.sample_summary['Label']) == 'APL'
y_pred = np.asarray(DAPL.sample_summary['APL'])
roc_score = roc_auc_score(y_test,y_pred)
fpr, tpr, th = roc_curve(y_test, y_pred)
id = 'Pts >= 10 cells'
plt.plot(fpr, tpr, lw=2, label='%s (%0.3f)' % (id, roc_score),c='green')

df_promy = pd.DataFrame()
df_promy['Patient'] = DAPL.patients
df_promy['Cell Type'] = DAPL.cell_type
df_promy['Cell Type'].value_counts()
df_promy['Pro'] =  df_promy['Cell Type'] == 'Promyelocyte'
df_promy_agg = df_promy.groupby(['Patient']).agg({'Pro':'sum'})

df_promy_tc = df_promy['Patient'].value_counts().to_frame()
df_pro = pd.concat([df_promy_agg,df_promy_tc],axis=1)

label_dict = pd.DataFrame()
label_dict['Patient'] = DAPL.patients
label_dict['Label'] = DAPL.labels
label_dict.drop_duplicates(inplace=True)
label_dict = dict(zip(label_dict['Patient'],label_dict['Label']))
df_pro['Label'] = df_pro.index.map(label_dict)
bin_dict = {'AML':0,'APL':1}
df_pro['Label_Bin'] = df_pro['Label'].map(bin_dict)
df_pro['Pro_Prop'] = df_pro['Pro']/df_pro['Patient']

y_test = np.array(df_pro['Label_Bin'])
y_pred = np.array(df_pro['Pro_Prop'])
roc_score = roc_auc_score(y_test,y_pred)
fpr, tpr, th = roc_curve(y_test, y_pred)
id = 'Proportion of Promyelocytes'
plt.plot(fpr, tpr, lw=2, label='%s (%0.3f)' % (id, roc_score),c='blue')

plt.legend(loc="lower right",prop={'size':12},frameon=False)
plt.tight_layout()
ax = plt.gca()
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis='y', labelsize=16)
