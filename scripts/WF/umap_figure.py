"""
This script is used to create the umap representations of the learned feature space for all cells in the validation cohort.
"""
import pickle
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering,KMeans
import matplotlib.patheffects as path_effects
import colorcet as cc
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib
matplotlib.rc('font', family='sans-serif')
def GKDE(x,y,z=None):
    xy = np.vstack([x, y])
    kernel = gaussian_kde(xy,weights=z)
    z = kernel(xy)
    r = np.argsort(z)
    x ,y, z = x[r], y[r], z[r]
    return x,y,z,kernel,r

name_out = 'validation_all'
with open(name_out+'_features.pkl','rb') as f:
    Cell_Pred, DFs_pred, imgs, features,\
    patients, cell_type, files, smears,\
    labels, Y, predicted, lb = pickle.load(f)

pca = PCA()
X_pca = pca.fit_transform(features)
X_pca = X_pca[:,:np.where(np.cumsum(pca.explained_variance_ratio_)>0.99)[0][0]]
X_2 = umap.UMAP().fit_transform(X_pca)
data = pd.DataFrame(X_2)
data['ct'] = cell_type
np.random.seed(0)
num_clusters = 3
cluster_obj = KMeans(n_clusters=num_clusters,random_state=0)
c_idx = cluster_obj.fit_predict(X_2)
data['cluster'] = c_idx
cmap = cc.glasbey_bw[0:num_clusters]

#density plots
fig,ax = plt.subplots(1,2,figsize=(8,4))
idx = labels == 'AML'
x ,y = X_2[idx,0], X_2[idx,1]
x,y,z,_,_ = GKDE(x,y)
ax[0].scatter(x,y,c=z,cmap='jet',s=5)
# ax[0].set_title('non-APL')
ax[0].set_xticks([])
ax[0].set_yticks([])
idx = labels == 'APL'
x ,y = X_2[idx,0], X_2[idx,1]
x,y,z,_,_ = GKDE(x,y)
ax[1].scatter(x,y,c=z,cmap='jet',s=5)
# ax[1].set_title('APL')
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.tight_layout()
plt.savefig('density_umap.eps',transparent=True)

#pred plots
fig,ax = plt.subplots(1,2,figsize=(8,4))
x ,y = X_2[:,0], X_2[:,1]
ax[0].scatter(x,y,c=predicted[:,0],cmap='jet',s=5)
# ax[0].set_title('non-APL')
ax[0].set_xticks([])
ax[0].set_yticks([])
x ,y = X_2[:,0], X_2[:,1]
ax[1].scatter(x,y,c=predicted[:,1],cmap='jet',s=5)
# ax[1].set_title('APL')
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.tight_layout()
plt.savefig('pred_umap.eps',transparent=True)


#celltype plot
fig,ax = plt.subplots(figsize=(10,10))
cmap = cc.glasbey_bw[0:len(np.unique(cell_type))]
palette = cmap
sns.scatterplot(data=data,x=0,y=1,hue='ct',linewidth=0,palette=palette,ax=ax)
# plt.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:],frameon=False,prop={'size': 12})
plt.savefig('ct_umap.eps',transparent=True)


#cluster plot to identify myeloid cluster
fig,ax = plt.subplots()
np.random.seed(0)
cmap = cc.glasbey_bw[0:num_clusters]
palette = cmap
sns.scatterplot(data=data,x=0,y=1,hue='cluster',linewidth=0,palette=palette,ax=ax)
plt.legend(bbox_to_anchor=(1.10,1))
plt.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
for _ in np.unique(c_idx):
    loc = np.mean(X_2[np.where(c_idx==_)[0]],0)
    text = plt.text(loc[0],loc[1],str(_),fontdict={'size':24,'color':'white'})
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
                           path_effects.Normal()])

#myeloid plot
zoom_idx = np.where(c_idx==2)[0]
X_2_sel = X_2[zoom_idx]
img_sel = imgs[zoom_idx]
pred_sel = predicted[zoom_idx,1]
fig,ax = plt.subplots(figsize=(10,10))
ax.scatter(X_2_sel[:,0],X_2_sel[:,1],c='k',s=5,alpha=0.0)
ax.set_xticks([])
ax.set_yticks([])
for coord,img_plot,pred in zip(X_2_sel,img_sel,pred_sel):
    ab = AnnotationBbox(OffsetImage(img_plot,zoom=0.1),coord, frameon=True,pad=0.0,
                        bboxprops =dict(edgecolor=plt.cm.jet(pred),linewidth=5.0))
    ax.add_artist(ab)
plt.tight_layout()
plt.savefig('zoom.tif',transprent=True,dpi=1200)