# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:01:29 2017

@author: mariosm
"""

file_to_read="querry_pred.csv" # the scatcknet file
second_file="xgb_seed12357_n315.csv" # if you want you can replace for better score with "" if you want only stackent
outputfile="stacknet_70_seed12357_30.csv" #




 


if second_file!="": # we open the second file if specified
    tenp_preds=[]
    #we load the 2nd file
    file2nd=open(second_file, "r")
    file2nd.readline()# header
    for line in file2nd:
        splits=line.replace("\n","").split(",")  
        tenp_preds.append(float(splits[1]))
    file2nd.close()
    
file_pred=open(file_to_read, "r") # the stacknet prediction file
file_to_print=open(outputfile,"w")# the file to print
file_to_print.write("test_id,is_duplicate\n") # write the header
counter=0
for line in file_pred:
    splits=line.replace("\n","").split(",") 
    #print to the new file
    if second_file!="":
        file_to_print.write("%d,%f\n" %(counter,float(splits[1])*0.7 +tenp_preds[counter]*0.3 )) #70% stacknet, 30% the other
    else :
        file_to_print.write("%d,%f\n" %(counter,float(splits[1])))
    counter+=1
    if counter%100000==0:
        print( " printing row %d " % (counter))
    
file_pred.close()
file_to_print.close()