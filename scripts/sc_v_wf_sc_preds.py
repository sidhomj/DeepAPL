import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
def GKDE(x,y,z=None):
    xy = np.vstack([x, y])
    kernel = gaussian_kde(xy,weights=z)
    z = kernel(xy)
    r = np.argsort(z)
    x ,y, z = x[r], y[r], z[r]
    return x,y,z,kernel,r


file = 'SC/validation_blasts.pkl'
file = 'SC/validation_all.pkl'
# file = 'SC/discovery_all.pkl'
with open(file, 'rb') as f:
    Cell_Pred_sc, _, \
    _, _, _, \
    _, _, _, _, _= pickle.load(f)
Cell_Pred_sc = Cell_Pred_sc[Cell_Pred_sc['Label']!='out']


file = 'WF/validation_blasts.pkl'
file = 'WF/validation_all.pkl'
# file = 'WF/discovery_all.pkl'
with open(file, 'rb') as f:
    Cell_Pred_wf, _,_, \
    _, _, _, \
    _, _, _, _, _= pickle.load(f)
Cell_Pred_wf = Cell_Pred_wf[Cell_Pred_wf['Label']!='out']

data_plot = pd.DataFrame()
data_plot['sc'] = Cell_Pred_sc['APL']
data_plot['wf'] = Cell_Pred_wf['APL']
data_plot['cell_type'] = Cell_Pred_sc['Cell_Type']
data_plot['Label'] = Cell_Pred_sc['Label']

fig,ax = plt.subplots(1,2,figsize=(12,5))
idx = data_plot['Label']=='AML'
x = np.array(data_plot['sc'][idx])
y = np.array(data_plot['wf'][idx])
x,y,z,_,_ = GKDE(x,y)
ax[0].scatter(x,y,c=z,s=5,cmap='jet')
ax[0].set_xlabel('SC')
ax[0].set_ylabel('WF')
ax[0].set_title('AML')

idx = data_plot['Label']=='APL'
x = np.array(data_plot['sc'][idx])
y = np.array(data_plot['wf'][idx])
x,y,z,_,_ = GKDE(x,y)
ax[1].scatter(x,y,c=z,s=5,cmap='jet')
ax[1].set_xlabel('SC')
ax[1].set_ylabel('WF')
ax[1].set_title('APL')

long = pd.melt(data_plot,id_vars=['Label'],value_vars=['sc','wf'])
sns.violinplot(data=long,x='variable',y='value',hue='Label',cut=0)
long['y_test'] = (long['Label']=='APL').astype(int)

from sklearn.metrics import roc_auc_score
idx = long['variable']=='wf'
roc_auc_score(long['y_test'][idx],long['value'][idx])
