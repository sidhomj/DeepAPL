from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib
matplotlib.rc('font', family='Times New Roman')

gpu = 1


classes = ['AML','APL']
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
device = '/device:GPU:'+ str(gpu)
DAPL = DeepAPL_SC('blast_class',device=device)
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
count_dict = dict(zip(agg.index,agg['n']))

DAPL.Sample_Summary()
bin_dict = {'AML':0,'APL':1}
DAPL.sample_summary['Label_Bin'] = DAPL.sample_summary['Label'].map(bin_dict)
DAPL.sample_summary['n'] = DAPL.sample_summary.index.map(count_dict)

def bce(s1,t1):
    return -t1*np.log(s1) - (1-t1)*np.log(1-s1)
DAPL.sample_summary['BCE'] = bce(np.array(DAPL.sample_summary['APL']),
                                 np.array(DAPL.sample_summary['Label_Bin']))

sns.scatterplot(data=DAPL.sample_summary,x='n',y='BCE',hue='Label',palette=sns.color_palette(['red', 'blue']))
x_min,x_max = plt.xlim()
# plt.plot([x_min,x_max],[bce(0.43,1),bce(0.43,1)],c='r')
plt.xlabel('Number Of Cells',fontsize=24)
plt.ylabel('Cross-Entropy Loss',fontsize=24)
plt.tight_layout()
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:],prop={'size': 16})
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis='y', labelsize=16)

from scipy.stats import spearmanr
spearmanr(DAPL.sample_summary['n'],DAPL.sample_summary['BCE'])
idx = DAPL.sample_summary['Label']=='APL'
spearmanr(DAPL.sample_summary['n'][idx],DAPL.sample_summary['BCE'][idx])
idx = DAPL.sample_summary['Label']=='AML'
spearmanr(DAPL.sample_summary['n'][idx],DAPL.sample_summary['BCE'][idx])


# DAPL.sample_summary['Call_Correct'] = DAPL.sample_summary['BCE'] < bce(0.43,1)
# sel = 'APL'
# sample_summary_model = DAPL.sample_summary#[DAPL.sample_summary['Label']==sel]
# LR = LogisticRegression()
# x = np.array(sample_summary_model['n']).reshape(-1,1)
# y = np.array(sample_summary_model['Call_Correct'])
# LR.fit(x,y)
# pred = LR.predict_proba(x)
# df_plot = pd.DataFrame()
# df_plot['Number Of Cells'] = np.squeeze(x)
# df_plot['Pred'] = pred[:,1]
# df_plot['APL'] = np.array(sample_summary_model['APL'])
# df_plot['Label'] = np.array(sample_summary_model['Label'])
# df_plot['Call'] = np.array(sample_summary_model['Call_Correct'])
# sns.scatterplot(data=df_plot,x='Number Of Cells',y='Pred',label=sel)
# plt.xlabel('Number Of Cells')
# plt.ylabel('Probability of Model Being Correct')
