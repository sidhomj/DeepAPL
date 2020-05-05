from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
import matplotlib
import pandas as pd
matplotlib.rc('font', family='Times New Roman')
gpu = 1

# classes = ['AML','APL']
# cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
# device = '/device:GPU:'+ str(gpu)
# DAPL = DeepAPL_SC('temp')
# DAPL.Import_Data(directory='Data/All', Load_Prev_Data=True, classes=classes,
#                  include_cell_types=cell_types)
# pred_file = 'Cell_Preds_blast_norbc.pkl'
# mask_file = 'Cell_Masks_blast_norbc.pkl'
#
# with open(pred_file,'rb') as f:
#     DAPL.Cell_Pred = pickle.load(f)
# with open(mask_file,'rb') as f:
#     DAPL.w = pickle.load(f)
file = 'Discovery_nmf2.pkl'
file = 'Validation_Inference_nmf2.pkl'
file = 'Discovery_blast_norbc.pkl'
file = 'Validation_Inference_norbc.pkl'
DAPL = DeepAPL_SC('temp')
with open(file,'rb') as f:
    DAPL.Cell_Pred,DAPL.w,DAPL.imgs,\
    DAPL.patients,DAPL.cell_type,DAPL.files,\
    DAPL.smears,DAPL.labels,DAPL.Y,DAPL.predicted,DAPL.lb = pickle.load(f)
DAPL.Representative_Cells('AML',50)


# df_meta = pd.read_csv('../Data/master.csv')
# df_meta['Date of Diagnosis'] = df_meta['Date of Diagnosis'].astype('datetime64[ns]')
# df_meta.sort_values(by='Date of Diagnosis',inplace=True)
# df_meta =df_meta[df_meta['Date of Diagnosis']>= '2018-01-01']
# DAPL.Cell_Pred = DAPL.Cell_Pred[DAPL.Cell_Pred['Patient'].isin(df_meta['JH Number'])]

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

df_promy = pd.DataFrame()
df_promy['Patient'] = DAPL.Cell_Pred['Patient']
df_promy['Cell Type'] = DAPL.Cell_Pred['Cell_Type']
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

#Assess performance over min number of cells per sample

# #Sample Level Performance with samples >= 10 cells
DAPL.Cell_Pred['n'] = 1
agg = DAPL.Cell_Pred.groupby(['Patient']).agg({'Label':'first','n':'sum'})
#
DAPL.Sample_Summary()
n_list = []
auc_list = []
number_pos = []
number_neg = []
for n in range(0,np.max(agg['n'])):
    try:
        keep = np.array(list(agg[agg['n']>=n].index))
        sample_summary_temp = DAPL.sample_summary[DAPL.sample_summary.index.isin(keep)]
        y_test = np.asarray(sample_summary_temp['Label']) == 'APL'
        y_pred = np.asarray(sample_summary_temp['APL'])
        roc_score = roc_auc_score(y_test,y_pred)
        auc_list.append(roc_score)
        n_list.append(n)
        number_pos.append(np.sum(y_test))
        number_neg.append(np.sum(y_test!=True))
    except:
        continue

df_auc = pd.DataFrame()
df_auc['num_cells_per_sample'] = n_list
df_auc['auc'] = auc_list
df_auc['number_pos'] = number_pos
df_auc['number_neg'] = number_neg
sns.lineplot(data=df_auc,x='num_cells_per_sample',y='auc')
plt.ylim([0,1.1])
plt.xlabel('Num Cells Per Sample',fontsize=24)
plt.ylabel('AUC',fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

sns.lineplot(data=df_auc,x='number_pos',y='auc',label='APL')
sns.lineplot(data=df_auc,x='number_neg',y='auc',label='AML')
plt.ylim([0,1.1])
plt.xlabel('Number of Samples',fontsize=24)
plt.ylabel('AUC',fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.legend()
# # plt.figure()
# # # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# # plt.xlabel('False Positive Rate',fontsize=16)
# # plt.ylabel('True Positive Rate',fontsize=16)
# y_test = np.asarray(DAPL.sample_summary['Label']) == 'APL'
# y_pred = np.asarray(DAPL.sample_summary['APL'])
# roc_score = roc_auc_score(y_test,y_pred)
# fpr, tpr, th = roc_curve(y_test, y_pred)
# id = 'Pts >= 10 cells'
# plt.plot(fpr, tpr, lw=2, label='%s (%0.3f)' % (id, roc_score),c='green')






pred_file = 'Cell_Preds_blast_nonorm_2018plus.pkl'
with open(pred_file,'rb') as f:
    Cell_Pred_nonorm = pickle.load(f)

pred_file = 'Cell_Preds_blast_norm_2018plus.pkl'
with open(pred_file,'rb') as f:
    Cell_Pred_norm = pickle.load(f)

plt.scatter(Cell_Pred_norm['APL'],Cell_Pred_nonorm['APL'])
plt.xlabel('norm_preds')
plt.ylabel('no_norm_preds')

sns.violinplot(data=Cell_Pred_norm,x='Label',y='APL')
plt.title('Norm')

df1 = Cell_Pred_norm[['Label','APL']]
df1['type'] = 'norm'
df2 = Cell_Pred_nonorm[['Label','APL']]
df2['type'] = 'nonorm'
df_comp = pd.concat([df1,df2],axis=0)

sns.violinplot(data=df_comp,x='Label',y='APL',hue='type',cut=0)
