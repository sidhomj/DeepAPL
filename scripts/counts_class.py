from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib
matplotlib.rc('font', family='Times New Roman')
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

gpu = 1
classes = ['AML','APL']
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
device = '/device:GPU:'+ str(gpu)
DAPL = DeepAPL_SC('blast_class',device=device)
DAPL.Import_Data(directory='../Data/All', Load_Prev_Data=False, classes=classes,
                 include_cell_types=cell_types)

df = pd.DataFrame()
df['Patient'] = DAPL.patients
df['Label'] = DAPL.labels
df['n'] = 1
agg = df.groupby(['Patient']).agg({'Label':'first','n':'sum'})

df_add = pd.DataFrame()
df_add['Patient'] = DAPL.pts_exclude
df_add['Label'] = DAPL.pts_exclude_label
df_add['n'] = 0
df_add.set_index('Patient',inplace=True)

df_samples = pd.concat([agg,df_add])
bin_dict = {'AML':0,'APL':1}
df_samples['Label_Bin'] = df_samples['Label'].map(bin_dict)
color_dict = {'AML':'b','APL':'r'}
df_samples['color'] = df_samples['Label'].map(color_dict)

fig,ax = plt.subplots()
sns.boxplot(data=df_samples,x='Label',y='n',ax=ax,palette=sns.color_palette(['blue', 'red']),order=['AML','APL'])
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.0))
sns.swarmplot(data=df_samples,x='Label',y='n',ax=ax,palette=sns.color_palette(['blue', 'red']),order=['AML','APL'])
plt.ylabel('Number of Cells per Sample',fontsize=24)
plt.xlabel('')
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis='y', labelsize=16)
plt.tight_layout()
plt.xticks()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


#stat test
np.mean(df_samples['n'][df_samples['Label']=='APL'])
mannwhitneyu(df_samples['n'][df_samples['Label']=='APL'],
          df_samples['n'][df_samples['Label']=='AML'])



