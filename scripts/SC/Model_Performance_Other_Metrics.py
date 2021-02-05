"""
This script assesses other performance metrics for a given model applied to either discovery or validation.
"""
from DeepAPL.DeepAPL import DeepAPL_SC
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_fscore_support, f1_score,precision_score,recall_score
import seaborn as sns
import matplotlib
import pandas as pd
matplotlib.rc('font', family='sans-serif')

blasts = True
name = 'discovery_blasts'
file = 'discovery_blasts.pkl'
#
# name = 'validation_blasts'
# file = 'validation_blasts.pkl'
#
# blasts = False
# name = 'discovery_all'
# file = 'discovery_all.pkl'
# # # # # # # #
# name = 'validation_all'
# file = 'validation_all.pkl'

class graph_object(object):
    def __init__(self):
        self.init=0
    def Sample_Summary(self):
        if hasattr(self,'predicted_dist'):
            group_dict = {'Label':'first'}
            for ii in self.lb.classes_:
                group_dict[ii] = 'mean'
                group_dict[ii+'_ci'] = 'mean'
        else:
            group_dict = {'Label':'first'}
            for ii in self.lb.classes_:
                group_dict[ii] = 'mean'
        self.sample_summary = self.Cell_Pred.groupby(['Patient']).agg(group_dict)
DAPL = graph_object()
with open(file,'rb') as f:
    DAPL.Cell_Pred,DAPL.imgs,\
    DAPL.patients,DAPL.cell_type,DAPL.files,\
    DAPL.smears,DAPL.labels,DAPL.Y,DAPL.predicted,DAPL.lb = pickle.load(f)

#remove cells that do not have training data or are in the blurred out group
DAPL.Cell_Pred = DAPL.Cell_Pred[DAPL.Cell_Pred['Counts']>=1]
DAPL.Cell_Pred = DAPL.Cell_Pred[DAPL.Cell_Pred['Label']!='out']

y_true = np.array((DAPL.Cell_Pred['Label']=='APL').astype(int))
y_pred = np.array(DAPL.Cell_Pred['APL'])
precision_recall_fscore_support(y_true,y_pred)
f1_score(y_true,y_pred)
recall_score(y_true,y_pred)
