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


Supplementary python script to perform keras's ANN, to be run in conjuction with 
StackNet's KerasnnClassifier.java class

Parameters:
    
l2=0.0,0.0
momentum=0.9
epochs=10
batch_size=64
stopping_rounds=0
threads=1
use_dense=True		
validation_split=0.0
copy=false
seed=1
lr=0.01		
shuffle=true
standardize=true			
batch_normalization=true		
use_log1p=false			
hidden=50,25		
droupouts=0.4,02	
weight_init=lecun_uniform		
activation=relu,relu
optimizer=adam
loss=categorical_crossentropy
verbose=0	
"""

import sys
import os
from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.preprocessing import StandardScaler,MaxAbsScaler
from scipy.sparse import csr_matrix 
import keras
from keras.optimizers import Adam,Adagrad,SGD,Nadam,Adadelta
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py
import gc
from sklearn import cross_validation
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

                elif paramaeter_type=="float_list":
                    splits=value.replace(" ","").split(",")
                    float_values=[]
                    for values in splits: 
                        try:
                           float_values.append(float(values))
                        except:
                            raise Exception(" Parameter %s is expecting a comma separated string of floats but the current string count not be converted to a list: %s " % (parameter,value))                                       
                    
                    new_params[parameter]=float_values
                elif paramaeter_type=="int_list":
                    splits=value.replace(" ","").split(",")
                    int_values=[]
                    for values in splits: 
                        try:
                           int_values.append(int(values))
                        except:
                            raise Exception(" Parameter %s is expecting a comma separated string of ints but the current string count not be converted to a list: %s " % (parameter,value))                                       
                    
                    new_params[parameter]=int_values                    
                elif paramaeter_type=="str_list":
                    splits=value.replace(" ","").split(",")
                    str_values=[]
                    for values in splits: 
                           str_values.append(values)
                    new_params[parameter]=str_values                      
                     
                else :
                    raise Exception(" Parameter type %s is not recognised " % (paramaeter_type)) 
    f_file.close()
                
    return  Use_dense,task,model_name,data_name,prediction_name,columns, new_params               


"""
Creates a keras model with 'sequential' format
input_dum: column dimension of training data
output_dum: output of the last layer
params : model parameters
returns : keras sequential model
""" 

def build_model(input_d, output_d, params):
    
    hidden=[]
    if "hidden" not in params :
        raise Exception("hidden is a mandatory parameter")
    hidden=params["hidden"]
    dropouts=[]
    if "droupouts"  in params :
        dropouts=   params["droupouts"]
    else :
        dropouts=[0.0001 for s in range(0,len(hidden)) ]
    if len(dropouts)<len(hidden):
        while len(dropouts)!=len(hidden):
            dropouts.append(0.0001)
    if len(dropouts)>len(hidden):
        while len(dropouts)!=len(hidden):
            dropouts.pop()   

    activations=[]
    if "activation"  in params :
        activations=   params["activation"]
    else :
        activations=["relu" for s in range(0,len(hidden)) ]
    if len(activations)<len(hidden):
        while len(activations)!=len(hidden):
            activations.append("relu")
    if len(activations)>len(hidden):
        while len(activations)!=len(hidden):
            activations.pop()   
            
    l2s=[]
    if "l2"  in params :
        l2s=   params["l2"]
    else :
        l2s=[0.0 for s in range(0,len(hidden)) ]
    if len(l2s)<len(hidden):
        while len(l2s)!=len(hidden):
            l2s.append(0.0)
    if len(l2s)>len(hidden):
        while len(l2s)!=len(hidden):
            l2s.pop()   
            
    weight_in='lecun_uniform'
    if "weight_init" in params :
        weight_in= params["weight_init"]
        
    loss= 'categorical_crossentropy'
    if "loss" in params :
        loss= params["loss"]  
        
    lr=0.01
    if "lr" in params :
       lr = params["lr"]
       
    optimzer= 'Adam'
    if "optimzer" in params :
        optimzer= params["optimzer"] 

    momentum=0.01
    if "momentum" in params :
       momentum = params["momentum"]
       
    use_batch=False   
    if "batch_normalization" in params :
       use_batch = params["batch_normalization"]
       
       
       
    opt= Adam (lr=lr) 
    if optimzer.lower()=="adagrad":
        opt= Adagrad (lr=lr)         
    elif optimzer.lower()=="nadam":
        opt= Nadam (lr=lr)   
    elif optimzer.lower()=="adadelta":
        opt= Adadelta (lr=lr)   
    elif optimzer.lower()=="sgd":
        opt= SGD (lr=lr, momentum=momentum, nesterov=True) 
        
    models = Sequential()
    
    for h in range (len(hidden)):
        unit=hidden[h]
        dropout=dropouts[h]
        l_2=l2s[h]
        active=activations[h]
        if h==0:
            models.add(Dense(input_dim=input_d, units=unit, kernel_initializer=weight_in,kernel_regularizer=regularizers.l2(l_2) ))
        else :
            models.add(Dense( units=unit, kernel_initializer=weight_in,kernel_regularizer=regularizers.l2(l_2) ))            
    
        models.add(Activation(active))
        if use_batch:
            models.add(BatchNormalization())  
        
        models.add(Dropout(dropout))
        

    models.add(Dense(output_d, init=weight_in))
    models.add(Activation('linear'))

    models.compile(loss=loss, optimizer=opt)
    return models    


def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0



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
    acceptable_parameters={"standardize" : "bool" ,
                           "use_log1p" : "bool" ,
                           "shuffle" : "bool" ,    
                           "batch_normalization" : "bool" ,                            
                           "weight_init" : "str" ,
                           "activation" : "str_list" ,                           
                           "optimizer" : "str" ,    
                           "loss" : "str" ,                            
                           "l2" : "float_list" ,
                           "hidden" : "int_list" ,    
                           "droupouts" : "float_list" ,                            
                           "momentum" : "float" ,                           
                           "epochs" : "int" ,  
                           "lr" : "float" ,     
                           "batch_size" : "int" ,
                           "stopping_rounds" : "int" ,  
                           "validation_split" : "float" ,                                       
                           "seed" : "int" ,   
                           "verbose" : "int",    
                           "threads" : "int"                            
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

        batch_size=32
        if  "batch_size" in model_parameters:
            batch_size=model_parameters["batch_size"]                
        verbose=0
        if  "verbose" in model_parameters:
            verbose=model_parameters["verbose"] 
            
        ################### Model training ###############
        if  task_type =="train":
            X,y=get_data(data_file, column) #load data
            st=StandardScaler() 
            ab=MaxAbsScaler()
           
            shuffle=True
            if  "shuffle" in model_parameters:
                shuffle=model_parameters["shuffle"]
            epochs=1
            if  "epochs" in model_parameters:
                epochs=model_parameters["epochs"]    
            stopping_rounds=0
            if  "stopping_rounds" in model_parameters:
                stopping_rounds=model_parameters["stopping_rounds"]
            validation_split=0
            if  "validation_split" in model_parameters:
                validation_split=model_parameters["validation_split"]    
            seed=1
            if  "seed" in model_parameters:
                seed=model_parameters["seed"]                    
                
            np.random.seed(seed)             
            if dense: #convert to dense - useful if the data does nto have high dimensionality .

               X=X.toarray()
               if "use_log1p" in model_parameters and model_parameters["use_log1p"]==True:
                   X[X<0]=0
                   X=np.log1p(X)                  
               if "standardize" in model_parameters and model_parameters["standardize"]==True:
                   X=st.fit_transform(X)
                   
               model=build_model(X.shape[1],1,model_parameters )     
               
               if validation_split<=0.0:
                                      
                     model.fit( X, 
                                y, 
                                epochs=epochs,
                                verbose=verbose,
                                batch_size=batch_size,
                                shuffle=shuffle)

               else :
                  
                   x_train_oof, x_valid_oof, y_train_oof_nn, y_valid_oof_nn = cross_validation.train_test_split(
                            X, y, test_size=validation_split, random_state=seed) 

                   callbacks = [
                    EarlyStopping(
                        monitor='val_loss', 
                        patience=stopping_rounds,
                        verbose=verbose,
                         mode='auto'),
                    ModelCheckpoint(
                        model_file, 
                        monitor='val_loss', 
                        save_best_only=True, 
                        verbose=verbose)
                    ]
                    
                   model.fit(
                        x_train_oof, 
                        y_train_oof_nn, 
                        epochs=epochs,
                        validation_data=(x_valid_oof, y_valid_oof_nn),
                        verbose=verbose,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        shuffle=shuffle)  
                   
               joblib.dump((st) , model_file+".scaler")                   
               keras.models.save_model(model,model_file)
               model=None
               gc.collect()                     
               if not os.path.isfile(model_file):
                    raise Exception(" %s model file could not be exported - check permissions ... " % (model_file))
        
            else :
               
               if "use_log1p" in model_parameters and model_parameters["use_log1p"]==True:
                   X[X<0]=0
                   X=csr_matrix(X).log1p()               
               if "standardize" in model_parameters and model_parameters["standardize"]==True:
                   X=ab.fit_transform(X)
                   
               model=build_model(X.shape[1],1,model_parameters )     
               
               if validation_split<=0.0:
                   
                    model.fit_generator(generator=batch_generator(X,  y, batch_size, shuffle),
                                        epochs=epochs,
                                        steps_per_epoch=int(np.ceil(X.shape[0]/batch_size)),
                                        verbose=verbose)                   
                                      
                     
               else :
                  
                   x_train_oof, x_valid_oof, y_train_oof_nn, y_valid_oof_nn = cross_validation.train_test_split(
                            X, y, test_size=validation_split, random_state=seed) 

                   callbacks = [
                    EarlyStopping(
                        monitor='val_loss', 
                        patience=stopping_rounds,
                        verbose=verbose,
                         mode='auto'),
                    ModelCheckpoint(
                        model_file, 
                        monitor='val_loss', 
                        save_best_only=True, 
                        verbose=verbose)
                    ]
                   
                   model.fit_generator(generator=batch_generator(x_train_oof, y_train_oof_nn, batch_size, shuffle),
                                        epochs=epochs,
                                        steps_per_epoch=int(np.ceil(x_train_oof.shape[0]/batch_size)),
                                        validation_data=batch_generator(x_valid_oof, y_valid_oof_nn, batch_size, False),
                                        validation_steps=int(np.ceil(x_valid_oof.shape[0]/batch_size)),
                                        callbacks=callbacks,
                                        verbose=verbose)                   
                                  
            
               joblib.dump((ab) , model_file+".scaler")                   
               keras.models.save_model(model,model_file)
               model=None
               gc.collect()                     
               if not os.path.isfile(model_file):
                    raise Exception(" %s model file could not be exported - check permissions ... " % (model_file))
                    
            sys.exit(-1)# exit script
        ################### predicting ###############            
        else :
            if not os.path.isfile(model_file):
                raise Exception(" %s model file could not be imported " % (model_file))    
            if not os.path.isfile(model_file+".scaler"):
                raise Exception(" %s sclaer file could not be imported " % (model_file+".scaler"))                 
            X,y=get_data(data_file, column) #load data
            scaler=joblib.load(model_file+".scaler")
            model=keras.models.load_model(model_file)	
            
            if dense: #convert to dense - useful if the data does nto have high dimensionality .
            #Also sklearn models are not optimzied for sparse data in tree-cased algos
               X=X.toarray()
               if "use_log1p" in model_parameters and model_parameters["use_log1p"]==True:
                   X[X<0]=0
                   X=np.log1p(X)                  
               if "standardize" in model_parameters and model_parameters["standardize"]==True:
                   X=scaler.transform(X)
                   
               preds=model.predict(X,
                                   verbose=verbose,
                                   batch_size=batch_size)
               np.savetxt(prediction_file, preds, delimiter=",", fmt='%.9f')               
            
            else :
               if "use_log1p" in model_parameters and model_parameters["use_log1p"]==True:
                   X[X<0]=0
                   X=csr_matrix(X).log1p()               
               if "standardize" in model_parameters and model_parameters["standardize"]==True:
                   X=scaler.transform(X)

                   
                   
               preds = model.predict_generator(generator=batch_generatorp(X, batch_size, False),
                                               steps=int(np.ceil(X.shape[0]/batch_size))+1,
                                               verbose=verbose)              
               np.savetxt(prediction_file, preds, delimiter=",", fmt='%.9f')
               
            if not os.path.isfile(prediction_file):
                raise Exception(" %s prediction file could not be exported - check permissions ... " % (prediction_file))             
            sys.exit(-1)# exit script        
                 
if __name__=="__main__":
  main()
  


