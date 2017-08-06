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


Supplementary python script to perform sklearn's ExtraTreesClassifier, to berun in conjuction with 
StackNet's SklearnExtraTreesClassifier.java class

Parameters:
    
bootstrap=True,
criterion='gini'
max_depth=None
max_features=1.0
max_leaf_nodes=0
min_impurity_split=1e-07,
min_samples_leaf=1
min_samples_split=2
min_weight_fraction_leaf=0.0
n_estimators=10
n_jobs=1
random_state=None
verbose=0

"""

from sklearn.ensemble import ExtraTreesClassifier
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

def read_file_end_return_parameters(conf_name, params):
    new_params={}
    Use_dense=False
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
                
    return  Use_dense,task,model_name,data_name,prediction_name,columns, new_params               

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
    acceptable_parameters={"bootstrap" : "bool" ,
                           "criterion" : "str" ,
                           "max_depth" : "int" ,
                           "max_features" : "float" ,                           
                           "max_leaf_nodes" : "int" ,  
                           "min_impurity_split" : "float" ,     
                           "min_samples_leaf" : "int" ,
                           "min_samples_split" : "int" ,  
                           "min_weight_fraction_leaf" : "float" ,   
                           "n_estimators" : "int" ,      
                           "n_jobs" : "int" ,                                
                           "random_state" : "int" ,   
                           "verbose" : "int"                           
                           }
    
    arguments=sys.argv
    print ("arguments: ",arguments )
    if len(arguments)!=2:
        raise Exception(" was expecting only one argument pointing to the config file... process will terminate")

    else :
        config_file=arguments[1] 
        dense,task_type,model_file,data_file,prediction_file,column, model_parameters=read_file_end_return_parameters(config_file, acceptable_parameters)   
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
            
        ################### Model training ###############
        if  task_type =="train":
            X,y=get_data(data_file, column) #load data
            model=ExtraTreesClassifier(**model_parameters) # set model parameters
            if dense: #convert to dense - useful if the data does nto have high dimensionality .
            #Also sklearn models are not optimzied for sparse data in tree-cased algos
               X=X.toarray()
            model.fit(X,y) #fitting model
            joblib.dump((model) , model_file)
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
            model=joblib.load(model_file)
            preds=model.predict_proba(X)
            np.savetxt(prediction_file, preds, delimiter=",", fmt='%.9f')
            if not os.path.isfile(prediction_file):
                raise Exception(" %s prediction file could not be exported - check permissions ... " % (prediction_file))             
            sys.exit(-1)# exit script        
                     
if __name__=="__main__":
  main()
  


