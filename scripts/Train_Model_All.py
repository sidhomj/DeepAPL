from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

gpu = 2
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
#Train Classifier on Discovery Cohort
classes = ['AML','APL']

device = '/device:GPU:'+ str(gpu)
DAPL = DeepAPL_SC('all_class',device=device)
DAPL.Import_Data(directory='../Data/Final/Discovery', Load_Prev_Data=False, classes=classes,color_norm=True)

folds = 25
seeds = np.array(range(folds))
epochs_min = 25
graph_seed = 0
DAPL.Monte_Carlo_CrossVal(folds=folds,seeds=seeds,epochs_min=epochs_min,
                          stop_criterion=0.25,test_size=0.25,graph_seed=graph_seed,
                          weight_by_class=True)
DAPL.Get_Cell_Predicted()
with open('Cell_Preds_all.pkl','wb') as f:
    pickle.dump(DAPL.Cell_Pred,f,protocol=4)
with open('Cell_Masks_all.pkl','wb') as f:
    pickle.dump(DAPL.w,f,protocol=4)