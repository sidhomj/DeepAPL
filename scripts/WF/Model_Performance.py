"""
This script assesses performance metrics for a given model applied to either discovery or validation.
"""
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
matplotlib.rc('font', family='sans-serif')
gpu = 1

blasts = True
name = 'discovery_blasts'
file = 'discovery_blasts.pkl'

name = 'validation_blasts'
file = 'validation_blasts.pkl'
# # # #
blasts = False
name = 'discovery_all'
file = 'discovery_all.pkl'
# # # #
name = 'validation_all'
file = 'validation_all.pkl'

class graph_object(object):
    def __init__(self):
        self.init=0
    def Sample_Summary(self):
        if hasattr(self,'predicted_dist'):
            group_dict = {'Label':'first'}
            for ii in self.lb.classes_:
                group_dict[ii] = 'mean'
                group_dict[ii+'_ci'] = 'mean'
        else:
            group_dict = {'Label':'first'}
            for ii in self.lb.classes_:
                group_dict[ii] = 'mean'
        self.sample_summary = self.Cell_Pred.groupby(['Patient']).agg(group_dict)
DAPL = graph_object()
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

#Cell Performance
plt.figure()
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
plt.savefig(name+'_sc_auc.eps',transparent=True)

# #Cell Predictions by Cell Type
if blasts:
    fig, ax = plt.subplots(figsize=(5, 5))
    order = ['Blast, no lineage spec', 'Promonocyte', 'Promyelocyte', 'Myelocyte', 'Metamyelocyte']
else:
    fig, ax = plt.subplots(figsize=(15, 8))
    order = DAPL.Cell_Pred.groupby(['Cell_Type']).agg({'APL':'mean'}).sort_values(by='APL').index
sns.violinplot(data=DAPL.Cell_Pred,x='Cell_Type',y='APL',cut=0,ax=ax,order=order)
plt.xlabel('Cellavision Cell Type',fontsize=24)
plt.ylabel('P(APL)',fontsize=24)
ax.xaxis.set_ticks_position('top')
plt.xticks(rotation=-45,fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([0,1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='x', which=u'both',length=0)
plt.tight_layout()
plt.savefig(name+'_celltype.eps',transparent=True)

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

plt.legend(loc="lower right",prop={'size':12},frameon=False)
plt.tight_layout()
ax = plt.gca()
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.savefig(name+'_sample_auc.eps',transparent=True)
check=1

