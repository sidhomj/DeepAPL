from DeepAPL.DeepAPL import DeepAPL_WF
import os
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import cv2
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

data = 'load_data'
#open model
gpu = 2
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
DAPL = DeepAPL_WF(data,gpu)
DAPL.Import_Data(directory=None, Load_Prev_Data=True)

#load metadata & select data in discovery cohort for training
df_meta = pd.read_csv('../Data/master.csv')
df_meta['Date of Diagnosis'] = df_meta['Date of Diagnosis'].astype('datetime64[ns]')
df_meta.sort_values(by='Date of Diagnosis',inplace=True)
df_meta = df_meta[df_meta['Cohort']=='Validation']
df_meta['ID'] = range(len(df_meta))
df_meta['ID'] = 'Patient_'+df_meta['ID'].astype(str)

main_dir = '../Data/Validation_BM'
for id,jhn in zip(df_meta['ID'],df_meta['JH Number']):
    if not os.path.exists(os.path.join(main_dir,id)):
        os.makedirs(os.path.join(main_dir,id))

    idx = DAPL.patients==jhn
    cell_type = DAPL.cell_type[idx]
    imgs = DAPL.imgs[idx]
    for ct in np.unique(cell_type):
        idx_ct = cell_type==ct
        img_ct = imgs[idx_ct]
        dim = int(np.ceil(np.sqrt(img_ct.shape[0])))
        num_empty = dim*dim-len(img_ct)
        if num_empty != 0:
            add = np.array([np.ones_like(img_ct[0])]*num_empty)
            if len(add.shape)<4:
                add = add[np.newaxis,:,:,:]
            img_ct = np.concatenate([img_ct,add])
        img_ct = np.reshape(img_ct,[dim,-1,img_ct.shape[1],img_ct.shape[2],img_ct.shape[3]])
        img_ct = np.concatenate(np.concatenate(img_ct, 1), 1)
        cv2.imwrite(img=cv2.cvtColor(img_ct*255, cv2.COLOR_RGB2BGR),filename=os.path.join(main_dir,id,ct+'.jpg'))


df_write = pd.DataFrame()
df_write['patient'] = df_meta['ID']
df_write['call'] = None
df_write.to_csv('call_sheet.csv',index=False)