# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 01:34:03 2017

@author: mimar
"""

base="output_dataset2.csv" # stacknet preds
prediction="sub20170821_221910.csv"#script preds
output="output_dataset2_merged.csv"# file to generate submissions to

ff=open(base, "r")
ff_pred=open(prediction, "r")
fs=open(output,"w")
fs.write(ff.readline())
ff_pred.readline()

s=0
for line in ff:
    splits=line.replace("\n","").split(",")
    ids=splits[0]
    preds=[]
    for j in range (1,7):
        preds.append(float(splits[j]))
        
        
    pre_line_splits=ff_pred.readline().replace("\n","").split(",")
    for j in range (1,7):
        preds[j-1]=(preds[j-1]*0.25 + float(pre_line_splits[j])*0.75)
        
    fs.write(ids)
    for j in range(6):
        fs.write( "," +str(preds[j] ))
    fs.write("\n")
    s+=1
    
ff.close() 
ff_pred.close()
fs.close()    
   
print ("done")
