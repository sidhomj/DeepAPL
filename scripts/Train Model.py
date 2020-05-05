from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

gpu = 1
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

DAPL = DeepAPL_SC('load_data',device=gpu)
DAPL.Import_Data(directory=None, Load_Prev_Data=True)

folds = 100
seeds = np.array(range(folds))
epochs_min = 25
graph_seed = 0
DAPL.Monte_Carlo_CrossVal(folds=folds,seeds=seeds,epochs_min=epochs_min,
                          stop_criterion=0.25,test_size=0.25,graph_seed=graph_seed,
                          weight_by_class=True)
DAPL.Get_Cell_Predicted()
with open('Cell_Preds_blast_norm.pkl','wb') as f:
    pickle.dump(DAPL.Cell_Pred,f,protocol=4)
with open('Cell_Masks_blast_norm.pkl','wb') as f:
    pickle.dump(DAPL.w,f,protocol=4)