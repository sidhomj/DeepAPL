from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import copy
import matplotlib
matplotlib.rc('font', family='Times New Roman')
from matplotlib.colors import ListedColormap


name = 'discovery_model'

file = 'discovery_model.pkl'
write = 'discovery'

file = 'validation_model_all.pkl'
write = 'validation_all'

gpu = 1
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

DAPL = DeepAPL_SC(name,gpu)
with open(file,'rb') as f:
    DAPL.Cell_Pred,DAPL.w,DAPL.imgs,\
    DAPL.patients,DAPL.cell_type,DAPL.files,\
    DAPL.smears,DAPL.labels,DAPL.Y,DAPL.predicted,DAPL.lb = pickle.load(f)

df = copy.deepcopy(DAPL.Cell_Pred)
sel = 'APL'
if sel == 'AML':
    a = 0
    b = 1
else:
    a = 1
    b = 0
df.sort_values(by=sel, inplace=True, ascending=False)
df = df[df['Label'] == sel]
img_idx = np.array(df.index)[0:9]
np.random.seed(0)
models = np.random.choice(range(100), 25, replace=False)
models = ['model_' + str(x) for x in models]

#Plot Representative Sequences
fig,ax = plt.subplots(3,3,figsize=(9,9))
ax = np.ndarray.flatten(ax)
for ii,ax_ in zip(img_idx,ax):
    img = DAPL.imgs[ii]
    ax_.imshow(img)
    ax_.set_xticks([])
    ax_.set_yticks([])
    # ax_.set_title(str(np.round(df.loc[ii][sel],3)))
plt.tight_layout()

cmap = plt.cm.jet
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
my_cmap = ListedColormap(my_cmap)
#Plot IG maps
fig,ax = plt.subplots(3,3,figsize=(9,9))
ax = np.ndarray.flatten(ax)
for ii,ax_ in zip(img_idx,ax):
    img = DAPL.imgs[ii]
    att = DAPL.IG(img=img,a=a,b=b,models=models)
    vmax,vmin = np.percentile(att,99), np.percentile(att,0)
    ax_.imshow(img)
    ax_.imshow(att,cmap=my_cmap,alpha=0.5,vmax=vmax,vmin=vmin)
    ax_.set_xticks([])
    ax_.set_yticks([])
    # ax_.set_title(str(np.round(df.loc[ii][sel],3)))
plt.tight_layout()

