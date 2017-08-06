# -*- coding: utf-8 -*-
"""

Copyright (c) 2017 Marios Michailidis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Created on Wed Aug 02 16:49:28 2017

@author: Marios Michailidis


Example python script to perform Logistic Regression, to be run in conjuction with 
StackNet's PythonGenericClassifier.java class

The purpose of this class is for the user to put whatever he/she wants.
The only hyper parameters from StackNet's point of view is the index of at the end of the script 
(in this case zero(0))




"""

from sklearn.linear_model import LogisticRegression
import sys
import os
from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_file
import numpy as np



"""
conf_name: name with parameters
params : python dictionary with parameters and types
return: several filenames and a new parameters's file with the actual values
"""

def read_file_end_return_parameters(conf_name):
    columns=0
    task=""
    model_name=""
    data_name=""
    prediction_name=""
    n_jobs=1
    random_state=1
    

    if not os.path.isfile(conf_name):
        raise Exception(" %s config file does not exist... " % (conf_name)) 

    f_file=open(conf_name, "r")
    for line in  f_file:
        line=line.replace("\n","").replace("\r","")
        splits=line.split("=")
        if len(splits)!=2:
            raise Exception(" this (%s) line in %s config file has the wrong format..the corerct format should be: parameter=value " % (line,conf_name))             
        parameter=splits[0]
        value=splits[1]
        if parameter=="task":
            task=value
        elif parameter=="columns": 
            try:
                columns=int(value)  
            except:
                raise Exception(" Parameter %s is expecting a int value but the current could nto be converted: %s " % (parameter,value))                                       
        
        elif parameter=="model": 
            model_name=value               
        elif parameter=="data": 
            data_name=value  
        elif parameter=="prediction": 
            prediction_name=value  
        elif parameter=="random_state": 
            random_state=int(value)    
        elif parameter=="n_jobs": 
            n_jobs=int(value)   
            

    f_file.close()
                
    return  task,model_name,data_name,prediction_name,columns,random_state,n_jobs               

"""
Loads svmlight data
fname: filename to load
returns X, y
"""                            

def get_data(fname,cols):
    data = load_svmlight_file(fname,n_features =cols)
    return data[0], data[1]


def preprocess_data(X):
    # dp some preocessing here
    return X
            
#main method to get executed when calling python
def main():
    

    config_file=""   

    
    arguments=sys.argv
    print ("arguments: ",arguments )
    if len(arguments)!=2:
        raise Exception(" was expecting only one argument pointing to the config file... process will terminate")

    else :
        config_file=arguments[1] 
        task_type,model_file,data_file,prediction_file,column,random_state,n_jobs=read_file_end_return_parameters(config_file)  
        #YOU could use random_state as seed and n_jobs for threads if you want.
        #sanity checks
        if task_type not in ["train","predict"]:
            raise Exception("task needs to be either train or predict, here it was %s ... " % (task_type))   
        if model_file=="":
            raise Exception("model file cannot be empty")       
        if data_file=="":
            raise Exception("data file file cannot be empty")    
        if not os.path.isfile(data_file):
            raise Exception(" %s data file does not exist... " % (data_file))           
        if task_type=="predict" and  prediction_file=="":
            raise Exception("prediction file  cannot be empty when task=predict")  
        if column<1:
            raise Exception("columns cannot be less than 1...")             
            
        ################### Model training ###############
        if  task_type =="train":
            X,y=get_data(data_file, column) #load data in sparse (svmlight) format
            
            ######preprocess data if you want here
            #preprocess_data(X)
            
            #modelling
            model=LogisticRegression(C=1.0) # set model parameters
            model.fit(X,y) #fitting model
            
            #dumping model
            joblib.dump((model) , model_file)
            if not os.path.isfile(model_file):
                raise Exception(" %s model file could not be exported - check permissions ... " % (model_file))             
                
            sys.exit(-1)# exit script
        ################### predicting ###############            
        else :
            if not os.path.isfile(model_file):
                raise Exception(" %s model file could not be imported " % (model_file))    
                
            X,y=get_data(data_file, column) #load data
            
            ######preprocess data if you want here
            #preprocess_data(X)
            
            model=joblib.load(model_file) # load model
            preds=model.predict_proba(X)  # make predictions
            
            np.savetxt(prediction_file, preds, delimiter=",", fmt='%.9f')
            if not os.path.isfile(prediction_file):
                raise Exception(" %s prediction file could not be exported - check permissions ... " % (prediction_file))             
            sys.exit(-1)# exit script        
                     
if __name__=="__main__":
  main()
  


