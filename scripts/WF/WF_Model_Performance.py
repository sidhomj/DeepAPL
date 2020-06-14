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

name = 'discovery_blasts_2'
file = 'discovery_blasts_2.pkl'

name = 'validation_blasts'
file = 'validation_blasts.pkl'
#
name = 'discovery_all'
file = 'discovery_all.pkl'
#
name = 'validation_all'
file = 'validation_all.pkl'

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
plt.savefig(name+'_sc_auc.eps')

# #Cell Predictions by Cell Type
order = ['Blast, no lineage spec', 'Promonocyte', 'Promyelocyte', 'Myelocyte', 'Metamyelocyte', ]
fig,ax = plt.subplots(figsize=(5,5))
# sns.violinplot(data=DAPL.Cell_Pred,x='Cell_Type',y='APL',cut=0,ax=ax,order=order)
sns.violinplot(data=DAPL.Cell_Pred,x='Cell_Type',y='APL',cut=0,ax=ax)
plt.xlabel('Cellavision Cell Type',fontsize=24)
plt.ylabel('Probability of APL',fontsize=24)
ax.xaxis.set_ticks_position('top')
plt.xticks(rotation=-45,fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='x', which=u'both',length=0)
plt.savefig(name+'_celltype.eps')

#Sample Level Performance MIL
df_agg = DAPL.DFs_pred['APL'].groupby(['Samples']).agg({'y_pred':'mean'}).reset_index()
df_agg = df_agg[~df_agg['Samples'].str.endswith('_')]
df_agg['Label'] = df_agg['Samples'].map(label_dict)
df_agg.rename(columns={'y_pred':'APL'},inplace=True)
df_agg.set_index('Samples',inplace=True)
sample_summary = copy.deepcopy(df_agg)

# #Sample Level with cell pred
# DAPL.Sample_Summary()
# sample_summary = copy.deepcopy(DAPL.sample_summary)

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

plt.legend(loc="lower right",prop={'size':16},frameon=False)
plt.tight_layout()
ax = plt.gca()
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.savefig(name+'_sample_auc.eps')

#Assess performance over number of cells per sample
DAPL.Cell_Pred['n'] = 1
cell_counts = DAPL.Cell_Pred.groupby(['Patient']).agg({'Label':'first','n':'sum'})

#MIL method
df_agg = DAPL.DFs_pred['APL'].groupby(['Samples']).agg({'y_pred':'mean'}).reset_index()
df_agg = df_agg[~df_agg['Samples'].str.endswith('_')]
df_agg['Label'] = df_agg['Samples'].map(label_dict)
df_agg.rename(columns={'y_pred':'APL'},inplace=True)
df_agg.set_index('Samples',inplace=True)
sample_summary = df_agg

#Cell Pred method
# DAPL.Sample_Summary()
# sample_summary = DAPL.sample_summary

n_list = []
auc_list = []
auc_pro = []
number_pos = []
number_neg = []
for n in range(0,np.max(cell_counts['n'])):
    try:
        keep = np.array(list(cell_counts[cell_counts['n']>=n].index))
        sample_summary_temp = sample_summary[sample_summary.index.isin(keep)]
        sample_summary_temp['pro'] = sample_summary_temp.index.map(pro_dict)
        y_test = np.asarray(sample_summary_temp['Label']) == 'APL'
        y_pred = np.asarray(sample_summary_temp['APL'])
        roc_score = roc_auc_score(y_test,y_pred)
        auc_list.append(roc_score)
        n_list.append(n)
        auc_pro.append(roc_auc_score(y_test,sample_summary_temp['pro']))
        number_pos.append(np.sum(y_test))
        number_neg.append(np.sum(y_test!=True))
    except:
        continue

df_auc = pd.DataFrame()
df_auc['num_cells_per_sample'] = n_list
df_auc['auc'] = auc_list
df_auc['auc_pro'] = auc_pro
df_auc['number_pos'] = number_pos
df_auc['number_neg'] = number_neg
plt.figure()
sns.lineplot(data=df_auc,x='num_cells_per_sample',y='auc',label='CNN')
sns.lineplot(data=df_auc,x='num_cells_per_sample',y='auc_pro',label='Proportion of Promyelocytes')
plt.ylim([0,1.1])
plt.xlabel('Num Cells Per Sample',fontsize=24)
plt.ylabel('AUC',fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc="lower right",prop={'size':16},frameon=False)
plt.tight_layout()
plt.savefig(name+'_auc_v_numcells.eps')
