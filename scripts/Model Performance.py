from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
import matplotlib
matplotlib.rc('font', family='Times New Roman')
gpu = 1

classes = ['AML','APL']
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
device = '/device:GPU:'+ str(gpu)
DAPL = DeepAPL_SC('blast_class',device=device)
DAPL.Import_Data(directory='Data/All', Load_Prev_Data=True, classes=classes,
                 include_cell_types=cell_types)
pred_file = 'Cell_Preds.pkl'
mask_file = 'Cell_Masks.pkl'

with open(pred_file,'rb') as f:
    DAPL.Cell_Pred = pickle.load(f)
with open(mask_file,'rb') as f:
    DAPL.w = pickle.load(f)

#Cell Performance
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

#Cell Predictions by Cell Type
order = ['Blast, no lineage spec', 'Promonocyte', 'Promyelocyte', 'Myelocyte', 'Metamyelocyte', ]
fig,ax = plt.subplots(figsize=(5,5))
sns.violinplot(data=DAPL.Cell_Pred,x='Cell_Type',y='APL',order=order,cut=0,ax=ax)
plt.xlabel('Cellavision Cell Type',fontsize=24)
plt.ylabel('Probability of APL',fontsize=24)
ax.xaxis.set_ticks_position('top')
plt.xticks(rotation=-45,fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='x', which=u'both',length=0)


# sns.violinplot(data=DAPL.Cell_Pred,x='Label',y='APL',hue='Cell_Type',hue_order=order,cut=0)

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
plt.legend(loc="upper left",prop={'size':12},frameon=False)
plt.tight_layout()
ax = plt.gca()
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis='y', labelsize=16)

