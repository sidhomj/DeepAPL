import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import glob
import matplotlib.pyplot as plt
import pickle

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
label_dict = dict(zip(df_amlapl[0],df_amlapl[1]))

bin_dict = {'AML':0,'APL':1}

samples = glob.glob('../Data/clinical_test/*')
tpr_list = []
fpr_list = []
for s in samples:
    df_temp = pd.read_csv(s,dtype=object)
    df_temp['JH'] = df_temp['Test_ID'].map(key_dict)
    df_temp = df_temp[df_temp['JH'] != 'JH97316053']
    df_temp['GT'] = df_temp['JH'].map(label_dict)
    df_temp['Call_Bin'] = df_temp['Call'].map(bin_dict)
    df_temp['GT_Bin'] = df_temp['GT'].map(bin_dict)
    tn, fp, fn, tp = confusion_matrix(df_temp['GT_Bin'],df_temp['Call_Bin']).ravel()
    tpr = tp/np.sum(df_temp['GT_Bin'])
    fpr = fp/(len(df_temp)-np.sum(df_temp['GT_Bin']))
    tpr_list.append(tpr)
    fpr_list.append(fpr)

with open('Cell_Preds.pkl','rb') as f:
    cell_preds = pickle.load(f)

keep = np.array(df_temp['JH'])
cell_preds = cell_preds[cell_preds['Patient'].isin(keep)]
group_dict = {'Label': 'first'}
for ii in ['AML','APL']:
    group_dict[ii] = 'mean'
sample_preds = cell_preds.groupby(['Patient']).agg(group_dict)

df_add = pd.DataFrame()
df_add['Patient'] = ['JH97316053']
df_add['Label'] = 'APL'
df_add['AML'] = 1.0
df_add['APL'] = 0.0
df_add.set_index('Patient',inplace=True)
# sample_preds = pd.concat([sample_preds,df_add])

plt.figure()
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=16)
plt.ylabel('True Positive Rate',fontsize=16)
y_pred = np.asarray(sample_preds['APL'])
y_test = np.asarray(sample_preds['Label']) == 'APL'
roc_score = roc_auc_score(y_test, y_pred)
fpr, tpr, th = roc_curve(y_test, y_pred)
ii = 'CNN'
plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (ii, roc_score),zorder=1,c='grey')
plt.scatter(fpr_list,tpr_list,marker='*',s=100,c='r',linewidths=5,zorder=2,label='Clinicians')
plt.legend(loc="lower right")
plt.tight_layout()