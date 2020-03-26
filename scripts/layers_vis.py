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
plt.imshow(DAPL.imgs[img_sel])
plt.xticks([])
plt.yticks([])
plt.savefig('../results/arch/input.png')

for f in range(DAPL.l1.shape[-1]):
    plt.figure()
    plt.imshow(DAPL.l1[img_sel][:,:,f],cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('../results/arch/l1/'+str(f)+'.png')
    plt.close()

for f in range(DAPL.l2.shape[-1]):
    plt.figure()
    plt.imshow(DAPL.l2[img_sel][:,:,f],cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('../results/arch/l2/'+str(f)+'.png')
    plt.close()

for f in range(DAPL.l3.shape[-1]):
    plt.figure()
    plt.imshow(DAPL.l3[img_sel][:,:,f],cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('../results/arch/l3/'+str(f)+'.png')
    plt.close()

for f in range(DAPL.w.shape[-1]):
    plt.figure()
    plt.imshow(DAPL.w[img_sel][:,:,f],cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('../results/arch/output/'+str(f)+'.png')
    plt.close()
