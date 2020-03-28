from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
gpu = 1

classes = ['AML','APL']
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
device = '/device:GPU:'+ str(gpu)
DAPL = DeepAPL_SC('blast_class',device=device)
DAPL.Import_Data(directory='Data/All', Load_Prev_Data=True, classes=classes,
                 include_cell_types=cell_types)
pred_file = 'Cell_Preds.pkl'
mask_file = 'Cell_Masks.pkl'
with open(pred_file,'rb') as f:
    DAPL.Cell_Pred = pickle.load(f)
with open(mask_file,'rb') as f:
    DAPL.w = pickle.load(f)

DAPL.Representative_Cells('APL',9,Load_Prev_Data=True,prob_show=False,show_title=False,figsize=(8,8))
plt.tight_layout()
DAPL.Representative_Cells('APL',12,Load_Prev_Data=True,prob_show=False,show_title=False)
plt.tight_layout()

cell_types = ['Blast, no lineage spec', 'Promonocyte', 'Promyelocyte', 'Myelocyte', 'Metamyelocyte', ]
DAPL.Representative_Cells('APL',12,Load_Prev_Data=True,cell_type = cell_types[4])
DAPL.Representative_Cells('AML',12,Load_Prev_Data=True,cell_type=cell_types[4])
