"""
This script is used to apply integrated gradients method to select cells that carry the most predictive value for APL vs non-APL.
"""
from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import copy
import matplotlib
matplotlib.rc('font', family='sans-serif')
from matplotlib.colors import ListedColormap


name = 'discovery_blasts'
file = 'discovery_blasts.pkl'
write = 'discovery_blasts'

file = 'validation_blasts.pkl'
write = 'validation_blasts'

# name = 'discovery_all'
# file = 'discovery_all.pkl'
# write = 'discovery_all'
#
# file = 'validation_all.pkl'
# write = 'validation_all'

gpu = 2
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

DAPL = DeepAPL_SC(name,gpu)
with open(file,'rb') as f:
    DAPL.Cell_Pred,DAPL.imgs,\
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
fig,ax = plt.subplots(3,3,figsize=(7,7))
ax = np.ndarray.flatten(ax)
for ii,ax_ in zip(img_idx,ax):
    img = DAPL.imgs[ii]
    ax_.imshow(img)
    ax_.set_xticks([])
    ax_.set_yticks([])
    ax_.set_axis_off()
plt.tight_layout()
plt.savefig(write+'_'+sel+'.tif',dpi=1200,transparent=True)


cvals = [0, 1]
colors = ['lime', 'lime']
norm = plt.Normalize(min(cvals), max(cvals))
tuples = list(zip(map(norm, cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

my_cmap = cmap(np.arange(cmap.N))
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
my_cmap = ListedColormap(my_cmap)
#Plot IG maps
fig,ax = plt.subplots(3,3,figsize=(7,7))
ax = np.ndarray.flatten(ax)
n = 1
for ii,ax_ in zip(img_idx,ax):
    print(n)
    img = DAPL.imgs[ii]
    att = DAPL.IG(img=img,a=a,b=b,models=models)
    vmax,vmin = np.percentile(att,99), np.percentile(att,0)
    ax_.imshow(img)
    ax_.imshow(att,cmap=my_cmap,alpha=0.5,vmax=vmax,vmin=vmin)
    ax_.set_xticks([])
    ax_.set_yticks([])
    ax_.set_axis_off()
    n+=1
plt.tight_layout()
plt.savefig(write+'_'+sel+'_att.tif',dpi=1200,transparent=True)


