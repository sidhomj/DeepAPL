from DeepAPL.DeepAPL import DeepAPL_SC
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#Train Classifier on Discovery Cohort

classes = ['AML','APL']

#Select for only Immature cells
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
DAPL = DeepAPL_SC('Blast_S',device='/device:GPU:1')
DAPL.Import_Data(directory='Data/Discovery', Load_Prev_Data=True, classes=classes,
                 include_cell_types=cell_types,sample=None)
DAPL.Monte_Carlo_CrossVal(folds=10,epochs_min=10,stop_criterion=0.25,test_size=0.25,
                          dropout_rate=0.5,multisample_dropout_rate=0.0)
DAPL.Representative_Cells('AML')

import pickle
# with open('pred_cell_class.pkl','wb') as f:
#     pickle.dump([DAPL.y_pred,DAPL.y_test,DAPL.predicted,DAPL.w],f,protocol=4)

with open('pred_cell_class.pkl','rb') as f:
    DAPL.y_pred,DAPL.y_test,DAPL.predicted = pickle.load(f)

#Cell Classification Accuracy
DAPL.AUC_Curve()

#Sample Classification Accuracy
DAPL.Get_Cell_Predicted()
DAPL.Sample_Summary()
DAPL.Sample_AUC_Curve()