import pandas as pd
import numpy as np
from DeepAPL.DeepAPL import DeepAPL_WF
import os
import seaborn as sns
import matplotlib.pyplot as plt

data = 'WF/load_data'
DAPL = DeepAPL_WF(data)
DAPL.Import_Data(directory=None, Load_Prev_Data=True)
df_meta = pd.read_csv('../Data/master.csv')
cell_dict = {}
for pt in np.unique(DAPL.patients):
    cell_dict[pt] = np.sum(DAPL.patients==pt)
df_meta['num_cells'] = df_meta['Patient_ID'].map(cell_dict)

fig,ax = plt.subplots(2,1,figsize=(5,10))
sns.violinplot(data=df_meta,x='Cohort',y='num_cells',cut=0,ax=ax[0])
ax[0].set_ylabel('Number of Cells',fontsize=18)
sns.violinplot(data=df_meta,x='Diagnosis',y='num_cells',cut=0,ax=ax[1])
ax[1].set_ylabel('Number of Cells',fontsize=18)
plt.tight_layout()
plt.savefig('num_cells_dist.eps')

df_meta.to_csv('meta_table.csv',index=False)






