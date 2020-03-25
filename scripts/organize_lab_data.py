import pandas as pd
import numpy as np
import glob
from DeepAPL.DeepAPL import DeepAPL_SC

key_df = pd.read_csv('../Data/key.csv',dtype=object)
key_dict = dict(zip(key_df['Test_ID'],key_df['JHH_ID']))

samples_apl = glob.glob('../Data/All/APL/*')
samples_apl = np.array([x.split('/')[-1] for x in samples_apl])
df_apl = pd.DataFrame(samples_apl)
df_apl[1] = 'APL'

samples_aml = glob.glob('../Data/All/AML/*')
samples_aml = np.array([x.split('/')[-1] for x in samples_aml])
df_aml = pd.DataFrame(samples_aml)
df_aml[1] = 'AML'
df_amlapl = pd.concat([df_apl,df_aml])

gpu=1
#Train Classifier on Discovery Cohort
classes = ['AML','APL']
#Select for only Immature cells
cell_types = ['Blast, no lineage spec','Myelocyte','Promyelocyte','Metamyelocyte','Promonocyte']
device = '/device:GPU:'+ str(gpu)
DAPL = DeepAPL_SC('Blast_S_'+str(gpu),device=device)
DAPL.Import_Data(directory='../Data/All', Load_Prev_Data=False, classes=classes,
                 include_cell_types=cell_types)

df_smears = pd.DataFrame()
df_smears['Patient'] = DAPL.patients
df_smears['Label'] = DAPL.labels
df_smears['Smear'] = DAPL.smears
df_smears = df_smears.drop_duplicates()

lab_cols = pd.read_csv('../Data/lab_columns.csv')
df_labs = pd.DataFrame()
for c in lab_cols.columns:
    df_labs[c] = [None]*len(df_smears)

df_out = pd.concat([df_smears.reset_index(drop=True),df_labs],axis=1)
df_out.to_csv('clinical_fill.csv',index=False)