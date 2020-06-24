import os
import glob
import pandas as pd

df_meta = pd.read_csv('../Data/master.csv')
df_meta.reset_index(inplace=True)
df_meta.rename(columns={'index':'Patient_ID'},inplace=True)
df_meta['Patient_ID'] = 'Patient_'+df_meta['Patient_ID'].astype(str)
label_dict = dict(zip(df_meta['JH Number'],df_meta['Patient_ID']))
dirs = os.listdir('../Data_Pub/All/')
for dir in dirs:
    os.rename('../Data_Pub/All/'+dir,'../Data_Pub/All/'+label_dict[dir])

df_meta.to_csv('../Data/master_key.csv',index=False)
# df_meta.drop(columns=['JH Number'],inplace=True)
df_meta =df_meta[['Patient_ID', 'Diagnosis', 'Cohort','Age at Diagnosis', 'Gender']]
df_meta.to_csv('../Data_Pub/master.csv',index=False)