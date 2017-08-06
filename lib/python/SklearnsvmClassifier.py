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


Supplementart python script to perform sklearn's SVC, to berun in conjuction with 
StackNet's SklearnsvmClassifier.java class

Parameters:
    
usedense=true
shrinking=false
kernel=rbf
degree=3
C=1.0
tol =0.0001
coef0 =0.0
gamma =0.0
max_iter=-1
n_jobs=1
random_state=1
use_scale=true
"""

from sklearn.svm import SVC

import sys
import os
from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.preprocessing import MaxAbsScaler


"""
conf_name: name with parameters
params : python dictionary with parameters and types
return: several filenames and a new parameters's file with the actual values
"""

def read_file_end_return_parameters(conf_name, params):
    new_params={}
    Use_dense=False
    use_scale=False
    columns=0
    task=""
    model_name=""
    data_name=""
    prediction_name=""

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
        elif parameter=="usedense":
            if value.lower()=="true":
                Use_dense=True
            else :
                Use_dense=False
                
        elif parameter=="use_scale":
            if value.lower()=="true":
                use_scale=True
            else :
                use_scale=False 
                
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
        else : # it must be a model parameter
            #search if parameter is in the file
            if parameter not in  params:
                raise Exception(" Parameter %s is not recognised " % (parameter))     
            else :
                paramaeter_type=params[parameter]
                if paramaeter_type=="bool":
                    if value.lower()=="true":
                        new_params[parameter]=True
                    else :
                        new_params[parameter]=False
                elif paramaeter_type=="str":
                    new_params[parameter]=value                                  
                elif paramaeter_type=="float":
                    try:
                        new_params[parameter]=float(value)
                    except:
                        raise Exception(" Parameter %s is expecting a float value but the current could nto be converted: %s " % (parameter,value))                                       
                elif paramaeter_type=="int":
                    try:
                        new_params[parameter]=int(value)
                    except:
                        raise Exception(" Parameter %s is expecting a int value but the current could nto be converted: %s " % (parameter,value))                                       
                    #special condition for 'max_leaf_nodes' parameter
                    if parameter=="max_leaf_nodes" and  new_params[parameter]<=0:
                        new_params[parameter]=None
                else :
                    raise Exception(" Parameter type %s is not recognised " % (paramaeter_type)) 
    f_file.close()
    
    if 'gamma'  in new_params and    new_params['gamma']<=0.0:
         new_params['gamma']='auto'
    new_params['probability']=True        
         
         
         
    return  Use_dense,use_scale,task,model_name,data_name,prediction_name,columns, new_params               

"""
Loads svmlight data
fname: filename to load
returns X, y
"""                            

def get_data(fname,cols):
    data = load_svmlight_file(fname,n_features =cols)
    return data[0], data[1]
            
#main method to get executed when calling python
def main():
    

    config_file=""   
    acceptable_parameters={"shrinking" : "bool" ,
                           "kernel" : "str" ,
                           "degree" : "int" ,
                           "C" : "float" ,                           
                           "gamma" : "float" ,  
                           "tol" : "float" ,     
                           "max_iter" : "int" , 
                           "coef0" : "float" ,                                      
                           "random_state" : "int" ,   
                           "verbose" : "int"                           
                           }
    

    #use_scale=true
    
    
    arguments=sys.argv
    print ("arguments: ",arguments )
    if len(arguments)!=2:
        raise Exception(" was expecting only one argument pointing to the config file... process will terminate")

    else :
        config_file=arguments[1] 
        dense,use_scale,task_type,model_file,data_file,prediction_file,column, model_parameters=read_file_end_return_parameters(config_file, acceptable_parameters)   
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
        if len(model_parameters)==0:
            raise Exception("model parameters cannot be empty") 
        if column<1:
            raise Exception("columns cannot be less than 1...")             
        sca=MaxAbsScaler()    
        ################### Model training ###############
        if  task_type =="train":
            X,y=get_data(data_file, column) #load data
            model=SVC(**model_parameters) # set model parameters
            if dense: #convert to dense - useful if the data does nto have high dimensionality .
            #Also sklearn models are not optimzied for sparse data in tree-cased algos
               X=X.toarray()
            if use_scale:
               X= sca.fit_transform(X)
               
            model.fit(X,y) #fitting model
            joblib.dump((model,sca) , model_file)
            if not os.path.isfile(model_file):
                raise Exception(" %s model file could not be exported - check permissions ... " % (model_file))             
                
            sys.exit(-1)# exit script
        ################### predicting ###############            
        else :
            if not os.path.isfile(model_file):
                raise Exception(" %s model file could not be imported " % (model_file))              
            X,y=get_data(data_file, column) #load data
            if dense: #convert to dense - useful if the data does nto have high dimensionality .
            #Also sklearn models are not optimzied for sparse data in tree-cased algos
               X=X.toarray()            
            model,sca=joblib.load(model_file)
            if use_scale:
               X= sca.transform(X)  
               
            preds=model.predict_proba(X)
            np.savetxt(prediction_file, preds, delimiter=",", fmt='%.9f')
            if not os.path.isfile(prediction_file):
                raise Exception(" %s prediction file could not be exported - check permissions ... " % (prediction_file))             
            sys.exit(-1)# exit script        
                     
if __name__=="__main__":
  main()
  


