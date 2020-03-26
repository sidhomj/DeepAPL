from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

gpu = 1
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
#Train Classifier on Discovery Cohort
classes = ['AML','APL']

#Select for only Immature cells
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
device = '/device:GPU:'+ str(gpu)
DAPL = DeepAPL_SC('blast_class',device=device)
DAPL.Import_Data(directory='../Data/All', Load_Prev_Data=True, classes=classes,
                 include_cell_types=cell_types)

DAPL.Inference()
pred_file = 'Cell_Preds.pkl'
mask_file = 'Cell_Masks.pkl'

with open(pred_file,'rb') as f:
    DAPL.Cell_Pred = pickle.load(f)
with open(mask_file,'rb') as f:
    DAPL.w = pickle.load(f)

DAPL.Cell_Pred.sort_values(by='AML',inplace=True,ascending=False)
img_sel = DAPL.Cell_Pred.index[0]

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

save_img(DAPL.imgs[img_sel],'../results/arch/input.png')

for f in range(DAPL.l1.shape[-1]):
    save_img(DAPL.l1[img_sel][:,:,f],'../results/arch/l1/'+str(f)+'.png',cmap='jet')

for f in range(DAPL.l2.shape[-1]):
    save_img(DAPL.l2[img_sel][:,:,f],'../results/arch/l2/'+str(f)+'.png',cmap='jet')

for f in range(DAPL.l3.shape[-1]):
    save_img(DAPL.l3[img_sel][:,:,f],'../results/arch/l3/'+str(f)+'.png',cmap='jet')

for f in range(DAPL.w.shape[-1]):
    save_img(DAPL.w[img_sel][:,:,f],'../results/arch/output/'+str(f)+'.png',cmap='jet')

