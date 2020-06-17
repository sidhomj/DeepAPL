"""
This script is used to load images.
"""

from DeepAPL.DeepAPL import DeepAPL_WF
import warnings
warnings.filterwarnings('ignore')
import pickle
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

DAPL = DeepAPL_WF('cc')
DAPL.Import_Data(directory='../../Data/Clinical_Case', Load_Prev_Data=False)

name = 'discovery_all'
gpu=1
DAPL_train = DeepAPL_WF(name,gpu)
DAPL_train.imgs = DAPL.imgs
DAPL_train.patients = DAPL.patients
DAPL_train.cell_type = DAPL.cell_type
DAPL_train.files = DAPL.files
DAPL_train.smears = DAPL.smears
label_dict = {'JH57644706':'AML'}
DAPL_train.labels = np.array([label_dict[x] for x in DAPL_train.patients])
DAPL_train.lb = LabelEncoder().fit(['AML','APL','out'])
DAPL_train.Y = DAPL_train.lb.transform(DAPL_train.labels)
DAPL_train.Y = OneHotEncoder(sparse=False).fit_transform(DAPL_train.Y.reshape(-1,1))
DAPL_train.predicted = np.zeros((len(DAPL_train.Y), len(DAPL_train.lb.classes_)))
predicted,sample_list = DAPL_train.Ensemble_Inference()
DAPL_train.Get_Cell_Predicted()

name_out = 'clinical_case'
with open(name_out+'.pkl','wb') as f:
    pickle.dump([DAPL_train.Cell_Pred,DAPL_train.DFs_pred,DAPL_train.imgs,
                DAPL_train.patients,DAPL_train.cell_type,DAPL_train.files,DAPL_train.smears,
                DAPL_train.labels,DAPL_train.Y,DAPL_train.predicted,DAPL_train.lb],f,protocol=4)





