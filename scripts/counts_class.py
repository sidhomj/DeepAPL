from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
gpu = 1

classes = ['AML','APL']
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
device = '/device:GPU:'+ str(gpu)
DAPL = DeepAPL_SC('Blast_S_'+str(gpu),device=device)
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

fig,ax = plt.subplots()
sns.swarmplot(data=df_samples,x='Label',y='n',ax=ax)
plt.ylabel('Number of Cells per Sample',fontsize=16)
plt.xlabel('')
ax.tick_params(axis="x", labelsize=16)
plt.tight_layout()
plt.xticks()

np.mean(df_samples['n'][df_samples['Label']=='APL'])
from scipy.stats import ttest_ind
ttest_ind(df_samples['n'][df_samples['Label']=='APL'],
          df_samples['n'][df_samples['Label']=='AML'])

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
x = np.array(df_samples['n']).reshape(-1,1)
y = np.array(df_samples['Label_Bin'])
LR.fit(x,y)
pred = LR.predict_proba(np.array(df_samples['n']).reshape(-1,1))
plt.scatter(x,pred[:,1])
34/85

