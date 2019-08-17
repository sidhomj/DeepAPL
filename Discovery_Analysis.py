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
                 include_cell_types=cell_types)
#
DAPL.Monte_Carlo_CrossVal(folds=100,epochs_min=25,stop_criterion=0.25,test_size=0.25,
                          drop_out_rate=0.5,combine_train_valid=False,weight_by_class=False)

#Cell Classification Accuracy
DAPL.AUC_Curve()

#Sample Classification Accuracy
DAPL.Get_Cell_Predicted()
DAPL.Sample_Summary()
DAPL.Sample_AUC_Curve()

DAPL.Get_Kernels()