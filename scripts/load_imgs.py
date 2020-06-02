"""
This script is used to load images.
"""

from DeepAPL.DeepAPL import DeepAPL_SC
import warnings
import h5py
warnings.filterwarnings('ignore')

DAPL = DeepAPL_SC('load_data')
DAPL.Import_Data(directory='../Data/All', Load_Prev_Data=False)

import numpy as np
data = np.random.normal(0,1,(1000,10))

with h5py.File('h5py_test.h5','w') as f:
    f.create_dataset('imgs',data=data)

sel_idx = np.random.choice(len(data),25,replace=False)
with h5py.File('h5py_test.h5','r') as f:
    order = np.argsort(sel_idx)
    data_out = f['imgs'][np.sort(sel_idx),:]
    data_out[order] = data_out




