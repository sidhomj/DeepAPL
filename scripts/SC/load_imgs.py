"""
This script is used to load images.
"""

from DeepAPL.DeepAPL import DeepAPL_SC
import warnings
warnings.filterwarnings('ignore')

DAPL = DeepAPL_SC('load_data')
DAPL.Import_Data(directory='../../Data/All', Load_Prev_Data=False)




