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

name = 'discovery_model'
file = 'validation_model.pkl'
DAPL = DeepAPL_SC(name)
with open(file,'rb') as f:
    DAPL.Cell_Pred,DAPL.w,DAPL.imgs,\
    DAPL.patients,DAPL.cell_type,DAPL.files,\
    DAPL.smears,DAPL.labels,DAPL.Y,DAPL.predicted,DAPL.lb = pickle.load(f)

DAPL.Cell_Pred.sort_values(by='APL',inplace=True,ascending=False)
img_sel = DAPL.Cell_Pred.index[1]
img = DAPL.imgs[img_sel][np.newaxis,:,:,:]
Y = DAPL.Y[img_sel]
DAPL = DeepAPL_SC(name)
DAPL.imgs = img
DAPL.Y = Y
DAPL.Inference()

def save_img(img,path,cmap=None):
    fig,ax = plt.subplots(figsize=(5,5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.imshow(img,cmap=cmap)
    fig.savefig(path)
    plt.close()

dir = 'layers'
if not os.path.exists(dir):
    os.makedirs(dir)

# save_img(DAPL.imgs[img_sel],'../results/arch/input.png')

subdir = 'l1'
if not os.path.exists(os.path.join(dir,subdir)):
    os.makedirs(os.path.join(dir,subdir))
for f in range(DAPL.l1.shape[-1]):
    save_img(DAPL.l1[0][:,:,f],os.path.join(dir,subdir,str(f)+'.png'),cmap='jet')

subdir = 'l2'
if not os.path.exists(os.path.join(dir,subdir)):
    os.makedirs(os.path.join(dir,subdir))
for f in range(DAPL.l2.shape[-1]):
    save_img(DAPL.l2[0][:,:,f],os.path.join(dir,subdir,str(f)+'.png'),cmap='jet')

subdir = 'l3'
if not os.path.exists(os.path.join(dir,subdir)):
    os.makedirs(os.path.join(dir,subdir))
for f in range(DAPL.l3.shape[-1]):
    save_img(DAPL.l3[0][:,:,f],os.path.join(dir,subdir,str(f)+'.png'),cmap='jet')

subdir = 'l4'
if not os.path.exists(os.path.join(dir,subdir)):
    os.makedirs(os.path.join(dir,subdir))
for f in range(DAPL.w.shape[-1]):
    save_img(DAPL.l4[0][:,:,f],os.path.join(dir,subdir,str(f)+'.png'),cmap='jet')

