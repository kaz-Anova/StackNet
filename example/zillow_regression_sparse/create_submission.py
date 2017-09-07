# -*- coding: utf-8 -*-

#generates submission based on 1-column prediction csv

sample="input/sample_submission.csv" # name of sample sybmission
prediction="pred2.csv"# prediction file
output="output_dataset2.csv"# output submission

#the predictions are copied 6 times

ff=open(sample, "r")
ff_pred=open(prediction, "r")
fs=open(output,"w")
fs.write(ff.readline())
s=0
for line in ff: #read sample submission file
    splits=line.split(",")
    ids=splits[0] # get id
    pre_line=ff_pred.readline().replace("\n","") # parse prediction file and get prediction for the row
    fs.write(ids) # write id
    for j in range(6): # copy the prediction 6 times
        fs.write( "," +pre_line )
    fs.write("\n")
    s+=1
ff.close() 
ff_pred.close()
fs.close()       
print ("done")
