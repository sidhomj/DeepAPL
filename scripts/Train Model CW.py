from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import warnings
warnings.filterwarnings('ignore')

gpu = 1
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

#Train Classifier on Discovery Cohort
classes = ['AML','APL']
#Select for only Immature cells
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
device = '/device:GPU:'+ str(gpu)
DAPL = DeepAPL_SC('Blast_S_CW'+str(gpu),device=device)
DAPL.Import_Data(directory='../Data/All', Load_Prev_Data=True, classes=classes,
                 include_cell_types=cell_types)

folds = 100
seeds = np.array(range(folds))
epochs_min = 25
graph_seed = 0
DAPL.Monte_Carlo_CrossVal(folds=folds,seeds=seeds,epochs_min=epochs_min,
                          stop_criterion=0.25,test_size=0.25,graph_seed=graph_seed,weight_by_class=True)
DAPL.Get_Cell_Predicted()
with open('Cell_Preds_CW.pkl','wb') as f:
    pickle.dump(DAPL.Cell_Pred,f,protocol=4)
with open('Cell_Masks_CW.pkl','wb') as f:
    pickle.dump(DAPL.w,f,protocol=4)