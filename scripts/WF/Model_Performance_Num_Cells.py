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
from matplotlib.ticker import MaxNLocator
matplotlib.rc('font', family='sans-serif')
gpu = 1

blasts = True
name = 'discovery_blasts'
file = 'discovery_blasts.pkl'

name = 'validation_blasts'
file = 'validation_blasts.pkl'
# # # # # #
blasts = False
name = 'discovery_all'
file = 'discovery_all.pkl'
# # # # # #
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

#Sample Level Performance MIL
df_agg = DAPL.DFs_pred['APL'].groupby(['Samples']).agg({'y_pred':'mean'}).reset_index()
df_agg = df_agg[~df_agg['Samples'].str.endswith('_')]
df_agg['Label'] = df_agg['Samples'].map(label_dict)
df_agg.rename(columns={'y_pred':'APL'},inplace=True)
df_agg.set_index('Samples',inplace=True)
sample_summary = copy.deepcopy(df_agg)

#Sample Level Performance
counts = pd.DataFrame(DAPL.Cell_Pred['Patient'].value_counts()).reset_index()
auc_list = []
count_list = []
num_samples = []
for c in np.array(range(np.min(counts['Patient']),np.max(counts['Patient'])+1)):
    try:
        sel_idx = np.array(counts['index'][counts['Patient']>=c])
        df_sel = sample_summary[sample_summary.index.isin(sel_idx)]
        y_test = np.asarray(df_sel['Label']) == 'APL'
        y_pred = np.asarray(df_sel['APL'])
        roc_score = roc_auc_score(y_test, y_pred)
        auc_list.append(roc_score)
        count_list.append(c)
        num_samples.append(len(df_sel))
    except:
        continue



fig,ax = plt.subplots()
ax.plot(count_list,auc_list,color='b')
ax.set_xlabel('Number of Cells per Sample',fontsize=18)
ax.set_ylabel('AUC',fontsize=18,color='b')
ax.tick_params(axis='y', labelcolor='b')
ax2 = ax.twinx()
ax2.plot(count_list,num_samples,color='r')
ax2.set_ylabel('Number of Samples', color='r',fontsize=18)  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor='r')
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(name+'_numcells.eps')
