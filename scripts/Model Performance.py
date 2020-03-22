from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
gpu = 1

classes = ['AML','APL']
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
device = '/device:GPU:'+ str(gpu)
DAPL = DeepAPL_SC('Blast_S_'+str(gpu),device=device)
DAPL.Import_Data(directory='Data/All', Load_Prev_Data=True, classes=classes,
                 include_cell_types=cell_types)
with open('Cell_Preds.pkl','rb') as f:
    DAPL.Cell_Pred = pickle.load(f)
with open('Cell_Masks.pkl','rb') as f:
    DAPL.w = pickle.load(f)

DAPL.Cell_Pred['n'] = 1
agg = DAPL.Cell_Pred.groupby(['Patient']).agg({'Label':'first','n':'sum'})
sns.swarmplot(data=agg,x='Label',y='n')
plt.ylabel('Number of Cells')

#Cell Performance
plt.figure()
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
y_test = np.asarray(DAPL.Cell_Pred['Label']) == 'APL'
y_pred = np.asarray(DAPL.Cell_Pred['APL'])
roc_score = roc_auc_score(y_test,y_pred)
fpr, tpr, th = roc_curve(y_test, y_pred)
id = 'APL'
plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (id, roc_score))
plt.legend(loc="lower right")

#Cell Predictions by Cell Type
order = ['Blast, no lineage spec', 'Promonocyte', 'Promyelocyte', 'Myelocyte', 'Metamyelocyte', ]
sns.violinplot(data=DAPL.Cell_Pred,x='Cell_Type',y='APL',order=order,cut=0)
sns.violinplot(data=DAPL.Cell_Pred,x='Label',y='APL',hue='Cell_Type',hue_order=order,cut=0)

#Sample Level Performance
DAPL.Sample_Summary(Load_Prev_Data=True)
plt.figure()
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
y_test = np.asarray(DAPL.sample_summary['Label']) == 'APL'
y_pred = np.asarray(DAPL.sample_summary['APL'])
roc_score = roc_auc_score(y_test,y_pred)
fpr, tpr, th = roc_curve(y_test, y_pred)
id = 'APL'
plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (id, roc_score))
plt.legend(loc="lower right")

#Sample Level Performance with samples >= 10 cells
DAPL.Sample_Summary(Load_Prev_Data=True)
keep = np.array(list(agg[agg['n']>=10].index))
DAPL.sample_summary = DAPL.sample_summary[DAPL.sample_summary.index.isin(keep)]
plt.figure()
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
y_test = np.asarray(DAPL.sample_summary['Label']) == 'APL'
y_pred = np.asarray(DAPL.sample_summary['APL'])
roc_score = roc_auc_score(y_test,y_pred)
fpr, tpr, th = roc_curve(y_test, y_pred)
id = 'APL'
plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (id, roc_score))
plt.legend(loc="lower right")

