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

DAPL.Import_Data(directory='Data/Grand Rounds', Load_Prev_Data=False, classes=classes,save_data=False,
                 include_cell_types=cell_types)
DAPL.Ensemble_Inference()
DAPL.Get_Cell_Predicted()
DAPL.Sample_Summary()

#Representative Cells
DAPL.Representative_Cells('APL')
DAPL.Representative_Cells('AML')




