# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:46:48 2017

Scipt to rank (weight) average the 2 prediction files
produced from stacknet

It creates a file to be submitted on kaggle
"""
import numpy as np
from scipy.stats import rankdata
#name of the prediction files produced by the 2 StackNets
linear_model="amazon_linear_pred.csv"
count_moddel="amazon_count_pred.csv"
#blend weights
weight_linear_model=0.3
weight_count_model=0.7
##load data
linear_preds=np.loadtxt(linear_model, delimiter=",",usecols=[1])
count_preds=np.loadtxt(count_moddel, delimiter=",",usecols=[1])
#create ranks
linear_preds=rankdata(linear_preds, method='min') 
count_preds=rankdata(count_preds, method='min') 
#divide with length to make certain all values are between [0,1]
linear_preds=linear_preds/float(len(linear_preds))
count_preds=count_preds/float(len(count_preds))
# rank average them based on the pre-defined weights
preds=linear_preds*weight_linear_model + count_preds*weight_count_model
#Create the submission file
submission_file=open("sub_70_30.csv","w")
submission_file.write("id,action\n")
for i in range (0,len(preds)):
    submission_file.write("%d,%f\n" %(i+1,preds[i]))
submission_file.close()
print("Done!")



