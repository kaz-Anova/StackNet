# -*- coding: utf-8 -*-
"""
Created 22/4/2017

@author: Marios Michailidis

Script that prepares 2 data sets to get a top 11 position in the Amazon Classification Challenge:
Link:https://www.kaggle.com/c/amazon-employee-access-challenge

First source of data is via selecting the best up for up to 4-way interractions fo all
categorical variables ysing a linear model. Then the results are printes as sparse files

The scource are counts and likelihood features created per fold for up to 3way interractions 
( no feature selection ) .This produces 5 pairs of train/cv files that will be used to do 
the stacking semi-manually . E.g. in the first level no kfolding will take place.       

"""

#amazon helper functions

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix,csc_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

"""
Converts a dataset to weights of evidence (actuall computations) :
Good explanation here :http://ucanalytics.com/blogs/information-value-and-weight-of-evidencebanking-case/
These are likelihood type of features
"""    
    
def convert_dataset_to_woe(xc,yc,xt, rounding=2,cols=None):
    xc=xc.tolist()
    xt=xt.tolist()
    yc=yc.tolist()
    if cols==None:
        cols=[k for k in range(0,len(xc[0]))]
    woe=[ [0.0 for k in range(0,len(cols))] for g in range(0,len(xt))]
    good=[]
    bads=[]
    for col in cols:
        dictsgoouds=defaultdict(int)        
        dictsbads=defaultdict(int)
        good.append(dictsgoouds)
        bads.append(dictsbads)        
    total_goods=0
    total_bads =0

    for a in range (0,len(xc)):
        target=yc[a]
        if target>0.0:
           total_goods+=1.0
        else :
           total_bads+=1.0
        for j in range(0,len(cols)):
            col=cols[j]
            if target>0:
                good[j][xc[a][col]]+=1.0
            else :
                bads[j][xc[a][col]]+=1.0  
    #print(total_goods,total_bads)            
    
    for a in range (0,len(xt)):    
        for j in range(0,len(cols)):
            col=cols[j]
            tempgood=0.0
            tempbad=0.0
            if xt[a][col] in good[j]:
                tempgood=float(good[j][xt[a][col]])
            if xt[a][col] in bads[j]:
                tempbad=float(bads[j][xt[a][col]])  
            if tempgood>0.0 and tempbad>0.0:
                #print(tempgood,tempbad)
                woe[a][j]=round(np.log((tempgood/total_goods) /(tempbad/total_bads)) , rounding)
            elif tempgood>0.0 :
                 woe[a][j]=3.0
            elif tempbad>0.0:
                woe[a][j]=-3.0
            else :
                 woe[a][j]=round(np.log(0.9421/0.0579))
    return woe            
    
"""
Converts a dataset to weights of evidence (general):
Good explanation here :http://ucanalytics.com/blogs/information-value-and-weight-of-evidencebanking-case/
These are likelihood type of features
"""  
def convert_to_woe(X,y, Xt, seed=1, cvals=5, roundings=2, columns=None):
    
    if columns==None:
        columns=[k for k in range(0,(X.shape[1]))]    
        
    X=X.tolist()
    Xt=Xt.tolist() 
    woetrain=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(X))]
    woetest=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(Xt))]    
    
    kfolder=StratifiedKFold(y, n_folds=cvals,shuffle=True, random_state=seed)
    for train_index, test_index in kfolder:
        # creaning and validation sets
        X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
        y_train = np.array(y)[train_index]

        woecv=convert_dataset_to_woe(X_train,y_train,X_cv, rounding=roundings,cols=columns)
        X_cv=X_cv.tolist()
        no=0
        for real_index in test_index:
            for j in range(0,len(X_cv[0])):
                woetrain[real_index][j]=X_cv[no][j]
            no+=1
        no=0
        for real_index in test_index:
            for j in range(0,len(columns)):
                col=columns[j]
                woetrain[real_index][col]=woecv[no][j]
            no+=1      
    woefinal=convert_dataset_to_woe(np.array(X),np.array(y),np.array(Xt), rounding=roundings,cols=columns) 

    for real_index in range(0,len(Xt)):
        for j in range(0,len(Xt[0])):           
            woetest[real_index][j]=Xt[real_index][j]
            
    for real_index in range(0,len(Xt)):
        for j in range(0,len(columns)):
            col=columns[j]
            woetest[real_index][col]=woefinal[real_index][j]
            
    return np.array(woetrain), np.array(woetest)

"""
converts sparse data to StackNet format
Better use this one than standard svmlight.

"""
def fromsparsetofile(filename, array, deli1=" ", deli2=":",ytarget=None):    
    zsparse=csr_matrix(csc_matrix(array))
    indptr = zsparse.indptr
    indices = zsparse.indices
    data = zsparse.data
    print(" data lenth %d" % (len(data)))
    print(" indices lenth %d" % (len(indices)))    
    print(" indptr lenth %d" % (len(indptr)))
    
    f=open(filename,"w")
    counter_row=0
    for b in range(0,len(indptr)-1):
        #if there is a target, print it else , print nothing
        if ytarget!=None:
             f.write(str(ytarget[b]) + deli1)     
             
        for k in range(indptr[b],indptr[b+1]):
            if (k==indptr[b]):
                if np.isnan(data[k]):
                    f.write("%d%s%f" % (indices[k],deli2,-1))
                else :
                    f.write("%d%s%f" % (indices[k],deli2,data[k]))                    
            else :
                if np.isnan(data[k]):
                     f.write("%s%d%s%f" % (deli1,indices[k],deli2,-1))  
                else :
                    f.write("%s%d%s%f" % (deli1,indices[k],deli2,data[k]))
        f.write("\n")
        counter_row+=1
        if counter_row%10000==0:    
            print(" row : %d " % (counter_row))    
    f.close()  
    
"""
Load training and test data. Then create in a brute force way to cerate all possible 4-way 
categorical interractions and test whether auc improves when adding them. 
Once it finds the best interractions, it prints them as sparse data
as:
    train.sparse
    test.sparse
"""
  
def create_4way_interractions(path=""):
    
    train_df=pd.read_csv(path + "train.csv")
    test_df=pd.read_csv(path + "test.csv")
    train_df.drop("ROLE_CODE", axis=1, inplace=True)
    test_df.drop("ROLE_CODE", axis=1, inplace=True)
    
    y=np.array(train_df['ACTION'])
    train_df.drop("ACTION", axis=1, inplace=True)
    test_df.drop("id", axis=1, inplace=True)  
    
    columns=train_df.columns.values
    columns=[columns[k] for k in range(0,len(columns))] # we exclude the first column
    
    kfolder=StratifiedKFold(y, n_folds=5,shuffle=True, random_state=1) 
    
    grand_auc=0
    
    X=np.array(train_df)
    #X,y=shuffle(X,y, random_state=SEED) # Shuffle since the data is ordered by time
    i=0 # iterator counter
    model=SGDClassifier(loss='log', penalty='l2', alpha=0.0000225, n_iter=50, random_state=1)
    for train_index, test_index in kfolder:    
            X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
            y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]     
            one=OneHotEncoder(handle_unknown='ignore')
            one.fit(X_train)
            X_train=one.transform(X_train)
            X_cv=one.transform(X_cv) 
            model.fit(X_train,y_train)
            preds=model.predict_proba(X_cv)[:,1]
            auc=roc_auc_score(y_cv,preds)
            print (" fold %d/%d auc %f " % (i+1,5,auc))
            grand_auc+=auc
            i+=1
    grand_auc/=5
    print ("grand AUC is %f " % (grand_auc))
    
    columns=train_df.columns.values
    columns=[columns[k] for k in range(0,len(columns))] # we exclude the first column
    cols=[k for k in columns]
    newcols=cols[:]
    print(cols)
    for j1 in range(0,len(columns)):
        for j2 in range(j1+1,len(columns)):
                name1=columns[j1] + "_plus_" + columns[j2]
                cols.append(name1)

                train_df[name1]=train_df[columns[j1]].apply(lambda x:str(x)) + "_" + train_df[columns[j2]].apply(lambda x:str(x))
                test_df[name1]=test_df[columns[j1]].apply(lambda x:str(x))+ "_" + test_df[columns[j2]].apply(lambda x:str(x)) 
                lbl = LabelEncoder()
                lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
                train_df[name1] = lbl.transform(list(train_df[name1].values))
                test_df[name1] = lbl.transform(list(test_df[name1].values))                
                
                mean_auc=0
                X=np.array(train_df)
                i=0 # iterator counter    
                for train_index, test_index in kfolder:    
                        X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
                        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]     
                        one=OneHotEncoder(handle_unknown='ignore')
                        one.fit(X_train)
                        X_train=one.transform(X_train)
                        X_cv=one.transform(X_cv) 
                        model.fit(X_train,y_train)
                        preds=model.predict_proba(X_cv)[:,1]
                        auc=roc_auc_score(y_cv,preds)
                        print (" %s fold %d/%d auc %f " % (name1,i+1,5,auc))
                        mean_auc+=auc
                        i+=1
                mean_auc/=5  
                if (mean_auc>grand_auc+0.00001):
                    print (" %s will remain fold new Auc %f versus old Auc %f " % (name1,mean_auc,grand_auc))
                    grand_auc=mean_auc
                    newcols.append(name1)
                else :
                   print( "dropping %s as %f is NOT big enough to %f " %  (name1,mean_auc,grand_auc))
                   train_df.drop(name1, inplace=True,axis=1) 
                   test_df.drop(name1, inplace=True,axis=1) 
                   
                
    for j1 in range(0,len(columns)):
        for j2 in range(j1+1,len(columns)):
            for j3 in range(j2+1,len(columns)):            
                name1=columns[j1] + "_plus_" + columns[j2]+ "_plus_" + columns[j3]
                cols.append(name1)
                
                train_df[name1]=train_df[columns[j1]].apply(lambda x:str(x)) + "_" + train_df[columns[j2]].apply(lambda x:str(x))+ "_" + train_df[columns[j3]].apply(lambda x:str(x))
                test_df[name1]=test_df[columns[j1]].apply(lambda x:str(x))+ "_" + test_df[columns[j2]].apply(lambda x:str(x)) + "_" + test_df[columns[j3]].apply(lambda x:str(x)) 
                lbl = LabelEncoder()
                lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
                train_df[name1] = lbl.transform(list(train_df[name1].values))
                test_df[name1] = lbl.transform(list(test_df[name1].values))                       
                
                mean_auc=0
                X=np.array(train_df)
                i=0 # iterator counter    
                for train_index, test_index in kfolder:    
                        X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
                        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]     
                        one=OneHotEncoder(handle_unknown='ignore')
                        one.fit(X_train)
                        X_train=one.transform(X_train)
                        X_cv=one.transform(X_cv) 
                        model.fit(X_train,y_train)
                        preds=model.predict_proba(X_cv)[:,1]
                        auc=roc_auc_score(y_cv,preds)
                        print (" %s fold %d/%d auc %f " % (name1,i+1,5,auc))
                        mean_auc+=auc
                        i+=1
                mean_auc/=5  
                if (mean_auc>grand_auc+0.00001):
                    print (" %s will remain fold new Auc %f versus old Auc %f " % (name1,mean_auc,grand_auc))
                    grand_auc=mean_auc
                    newcols.append(name1)
                else :
                   print( "dropping %s as %f is NOT big enough to %f " %  (name1,mean_auc,grand_auc))
                   train_df.drop(name1, inplace=True,axis=1) 
                   test_df.drop(name1, inplace=True,axis=1) 

    for j1 in range(0,len(columns)):
        for j2 in range(j1+1,len(columns)):
            for j3 in range(j2+1,len(columns)):            
                for j4 in range(j3+1,len(columns)):                
                    name1=columns[j1] + "_plus_" + columns[j2]+ "_plus_" + columns[j3]+ "_plus_" + columns[j4]
                    cols.append(name1)

                    train_df[name1]=train_df[columns[j1]].apply(lambda x:str(x)) + "_" + train_df[columns[j2]].apply(lambda x:str(x))+ "_" + train_df[columns[j3]].apply(lambda x:str(x))+ "_" + train_df[columns[j4]].apply(lambda x:str(x))
                    test_df[name1]=test_df[columns[j1]].apply(lambda x:str(x))+ "_" + test_df[columns[j2]].apply(lambda x:str(x)) + "_" + test_df[columns[j3]].apply(lambda x:str(x)) + "_" + test_df[columns[j4]].apply(lambda x:str(x)) 
                    lbl = LabelEncoder()
                    lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
                    train_df[name1] = lbl.transform(list(train_df[name1].values))
                    test_df[name1] = lbl.transform(list(test_df[name1].values))                
                    
                    mean_auc=0
                    X=np.array(train_df)
                    i=0 # iterator counter    
                    for train_index, test_index in kfolder:    
                            X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
                            y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]     
                            one=OneHotEncoder(handle_unknown='ignore')
                            one.fit(X_train)
                            X_train=one.transform(X_train)
                            X_cv=one.transform(X_cv) 
                            model.fit(X_train,y_train)
                            preds=model.predict_proba(X_cv)[:,1]
                            auc=roc_auc_score(y_cv,preds)
                            print (" %s fold %d/%d auc %f " % (name1,i+1,5,auc))
                            mean_auc+=auc
                            i+=1
                    mean_auc/=5  
                    if (mean_auc>grand_auc+0.00001):
                        print (" %s will remain fold new Auc %f versus old Auc %f " % (name1,mean_auc,grand_auc))
                        grand_auc=mean_auc
                        newcols.append(name1)
                    else :
                       print( "dropping %s as %f is NOT big enough to %f " %  (name1,mean_auc,grand_auc))
                       train_df.drop(name1, inplace=True,axis=1) 
                       test_df.drop(name1, inplace=True,axis=1) 

    for j1 in range(0,len(columns)):
        for j2 in range(j1+1,len(columns)):
            for j3 in range(j2+1,len(columns)):            
                for j4 in range(j3+1,len(columns)):     
                    for j5 in range(j4+1,len(columns)):                       
                        name1=columns[j1] + "_plus_" + columns[j2]+ "_plus_" + columns[j3]+ "_plus_" + columns[j4]+ "_plus_" + columns[j5]
                        cols.append(name1)
                        
                        train_df[name1]=train_df[columns[j1]].apply(lambda x:str(x)) + "_" + train_df[columns[j2]].apply(lambda x:str(x))+ "_" + train_df[columns[j3]].apply(lambda x:str(x))+ "_" + train_df[columns[j4]].apply(lambda x:str(x))+ "_" + train_df[columns[j5]].apply(lambda x:str(x))
                        test_df[name1]=test_df[columns[j1]].apply(lambda x:str(x))+ "_" + test_df[columns[j2]].apply(lambda x:str(x)) + "_" + test_df[columns[j3]].apply(lambda x:str(x)) + "_" + test_df[columns[j4]].apply(lambda x:str(x)) + "_" + test_df[columns[j5]].apply(lambda x:str(x))
                        lbl = LabelEncoder()
                        lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
                        train_df[name1] = lbl.transform(list(train_df[name1].values))
                        test_df[name1] = lbl.transform(list(test_df[name1].values))                             
                        
                        mean_auc=0
                        X=np.array(train_df)
                        i=0 # iterator counter    
                        for train_index, test_index in kfolder:    
                                X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
                                y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]     
                                one=OneHotEncoder(handle_unknown='ignore')
                                one.fit(X_train)
                                X_train=one.transform(X_train)
                                X_cv=one.transform(X_cv) 
                                model.fit(X_train,y_train)
                                preds=model.predict_proba(X_cv)[:,1]
                                auc=roc_auc_score(y_cv,preds)
                                print (" %s fold %d/%d auc %f " % (name1,i+1,5,auc))
                                mean_auc+=auc
                                i+=1
                        mean_auc/=5  
                        if (mean_auc>grand_auc+0.00001):
                            print (" %s will remain fold new Auc %f versus old Auc %f " % (name1,mean_auc,grand_auc))
                            grand_auc=mean_auc
                            newcols.append(name1)
                        else :
                           print( "dropping %s as %f is NOT big enough to %f " %  (name1,mean_auc,grand_auc))
                           train_df.drop(name1, inplace=True,axis=1) 
                           test_df.drop(name1, inplace=True,axis=1) 
                       
    train_df.to_csv("trainid.csv",index=False)
    test_df.to_csv("testid.csv",index=False) 
      
    print ("one hot encoding")
    train=np.array(train_df)
    test=np.array(test_df) 
    
    for j in range(0,train.shape[1]):
        dicter=defaultdict(lambda:0)
        for i in range(0,train.shape[0]):
           dicter[str(train[i,j])]+=1 
        for i in range(0,test.shape[0]):
           dicter[str(test[i,j])]+=1 
        for i in range(0,train.shape[0]):
          train[i,j]=9999999 if dicter[str(train[i,j])]<=1 else  train[i,j]
        for i in range(0,test.shape[0]):
           test[i,j]=9999999 if dicter[str(test[i,j])]<=1 else  test[i,j]   
          
    one=OneHotEncoder(handle_unknown='ignore', sparse=True)
    test=one.fit_transform(test)
    train=one.transform(train)   
    test=csr_matrix(test)
    train=csr_matrix(train)    
    fromsparsetofile(path + "train.sparse", train, deli1=" ", deli2=":",ytarget=y)    
    fromsparsetofile(path + "test.sparse", test, deli1=" ", deli2=":",ytarget=None)      
    print (train.shape)
    print (test.shape)
    
  
    print ("counts")   
    result = pd.concat([train_df,test_df])    
    for f in newcols:
                cases=defaultdict(int)
                temp=np.array(result[f]).tolist()
                for k in temp:
                    cases[k]+=1
                print (f, len(cases)) 
                
                train_df[f]=train_df[f].apply(lambda x: cases[x])
                test_df[f]=test_df[f].apply(lambda x: cases[x])     
    
    train_df.to_csv("traincount.csv",index=False)
    test_df.to_csv("testcount.csv",index=False)  
       

"""
Computes all possible 3-way interractions
and finds the counts of each category
Then it perfoms k-fold and produces likelihood (woe)
values for all features and stacks them next to the counts. Then
it prints them in dense format as  :
    amazon_counts_train" + str(fold_number) + ".txt"
    amazon_counts_cv" + str(fold_number) + ".txt"
It also produces an amazon_counts_train.txt and amazon_counts_test.txt file too.
(so 12 in total - 5 pairs of train/cv and a final train and test file)
The aim is to prepare StackNet to run stacking with our own folds.

The data is also standardized. 
"""

def create_likelihoods_with_counts(path=""):
    
    number_of_folds=5
    SEED=15
    train_df=pd.read_csv(path + "train.csv")
    test_df=pd.read_csv(path + "test.csv")
    train_df.drop("ROLE_CODE", axis=1, inplace=True)
    test_df.drop("ROLE_CODE", axis=1, inplace=True)
    
    y=np.array(train_df['ACTION'])
    
    train_df.drop("ACTION", axis=1, inplace=True)
    test_df.drop("id", axis=1, inplace=True)  
    
    columns=train_df.columns.values
    columns=[columns[k] for k in range(0,len(columns))] # we exclude the first column
    cols=[k for k in columns]
    print(cols)
    for j1 in range(0,len(columns)):
        for j2 in range(j1+1,len(columns)):
                name1=columns[j1] + "_plus_" + columns[j2]
                cols.append(name1)
                train_df[name1]=train_df[columns[j1]].apply(lambda x:str(x)) + "_" + train_df[columns[j2]].apply(lambda x:str(x))
                test_df[name1]=test_df[columns[j1]].apply(lambda x:str(x))+ "_" + test_df[columns[j2]].apply(lambda x:str(x)) 
                lbl = LabelEncoder()
                lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
                train_df[name1] = lbl.transform(list(train_df[name1].values))
                test_df[name1] = lbl.transform(list(test_df[name1].values))                     
                

                
    for j1 in range(0,len(columns)):
        for j2 in range(j1+1,len(columns)):
            for j3 in range(j2+1,len(columns)):            
                name1=columns[j1] + "_plus_" + columns[j2]+ "_plus_" + columns[j3]
                cols.append(name1)

                train_df[name1]=train_df[columns[j1]].apply(lambda x:str(x)) + "_" + train_df[columns[j2]].apply(lambda x:str(x))+ "_" + train_df[columns[j3]].apply(lambda x:str(x))
                test_df[name1]=test_df[columns[j1]].apply(lambda x:str(x))+ "_" + test_df[columns[j2]].apply(lambda x:str(x)) + "_" + test_df[columns[j3]].apply(lambda x:str(x))
                lbl = LabelEncoder()
                lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
                train_df[name1] = lbl.transform(list(train_df[name1].values))
                test_df[name1] = lbl.transform(list(test_df[name1].values))     
    
 
    for j1 in range(0,len(columns)):
        for j2 in range(j1+1,len(columns)):
            for j3 in range(j2+1,len(columns)):           
                for j4 in range(j3+1,len(columns)):                   
                    name1=columns[j1] + "_plus_" + columns[j2]+ "_plus_" + columns[j3]+ "_plus_" + columns[j4]
                    cols.append(name1)
    
                    train_df[name1]=train_df[columns[j1]].apply(lambda x:str(x)) + "_" + train_df[columns[j2]].apply(lambda x:str(x))+ "_" + train_df[columns[j3]].apply(lambda x:str(x))+ "_" + train_df[columns[j4]].apply(lambda x:str(x))
                    test_df[name1]=test_df[columns[j1]].apply(lambda x:str(x))+ "_" + test_df[columns[j2]].apply(lambda x:str(x)) + "_" + test_df[columns[j3]].apply(lambda x:str(x))+ "_" + test_df[columns[j4]].apply(lambda x:str(x))
                    lbl = LabelEncoder()
                    lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
                    train_df[name1] = lbl.transform(list(train_df[name1].values))
                    test_df[name1] = lbl.transform(list(test_df[name1].values))    
                
    X=np.array(train_df)
    X_Test=np.array(test_df)
    
    print ("counts")   
    result = pd.concat([train_df,test_df])    
    for f in cols:
                cases=defaultdict(int)
                temp=np.array(result[f]).tolist()
                for k in temp:
                    cases[k]+=1
                print (f, len(cases)) 
                
                train_df[f]=train_df[f].apply(lambda x: cases[x])
                test_df[f]=test_df[f].apply(lambda x: cases[x])  
                
    X_count=np.array(train_df)
    X_Test_count=np.array(test_df)  
    
    X_count[X_count<=1]=0
    X_Test_count[X_Test_count<=1]=0  
                
               
    bigy=None     
    print(" creating likelihoods ")
    kfolder=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=SEED)
    #number_of_folds=0
    #X,y=shuffle(X,y, random_state=SEED) # Shuffle since the data is ordered by time
    i=0 # iterator counter
    print ("printing files for %d kfolds " % (number_of_folds))
    if number_of_folds>0:
        for train_index, test_index in kfolder:    
            X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
            X_train_count, X_cv_count = np.array(X_count)[train_index], np.array(X_count)[test_index]            
            y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index] 
            
            if bigy==None      :
                bigy=y_cv
            else :
                bigy=np.concatenate((bigy,y_cv))
                

            X_train,X_cv= convert_to_woe(X_train,y_train , X_cv, seed=1, cvals=5, roundings=2)


            
            temp_array_train=X_train
            temp_array_cv=X_cv    
            temp_array_train=np.column_stack((temp_array_train,X_train_count))
            temp_array_cv=np.column_stack((temp_array_cv,X_cv_count))
            
            for a in range(0,temp_array_train.shape[0]):
                for b in range(0,temp_array_train.shape[1]):
                    if temp_array_train[a,b]>0:
                        temp_array_train[a,b]=np.log1p(temp_array_train[a,b])
                    else :
                        temp_array_train[a,b]=-np.log1p(-temp_array_train[a,b])                        
                        
            for a in range(0,temp_array_cv.shape[0]):
                for b in range(0,temp_array_cv.shape[1]):
                    if temp_array_cv[a,b]>0:
                        temp_array_cv[a,b]=np.log1p(temp_array_cv[a,b])               
                    else :
                        temp_array_cv[a,b]= -np.log1p(-temp_array_cv[a,b])          
             
            stda=StandardScaler()
            stda.fit(temp_array_train)
            temp_array_train=stda.transform(temp_array_train)
            temp_array_cv=stda.transform(temp_array_cv)             
            
            #temp_array_train=csr_matrix(temp_array_train)
            #temp_array_cv=csr_matrix(temp_array_cv)      
            np.savetxt ("amazon_counts_train" + str(i) + ".txt",np.column_stack((y_train,temp_array_train)),delimiter=",")
            np.savetxt ("amazon_counts_cv" + str(i) + ".txt",np.column_stack((y_cv,temp_array_cv)),delimiter=",")      
            
            #fromsparsetofile("amazon_count_train" + str(i) + ".txt", temp_array_train, deli1=" ", deli2=":",ytarget=y_train)
            #fromsparsetofile("amazon_count_cv" + str(i) + ".txt", temp_array_cv,deli1=" ", deli2=":",ytarget=y_cv)  
           
            i+=1              
    
    np.savetxt("labcv.txt",bigy)
  
    X,X_Test= convert_to_woe(X,y , X_Test, seed=1, cvals=5, roundings=2)        
    temp_array_train=X
    temp_array_cv=X_Test
    
    temp_array_train=np.column_stack((temp_array_train,X_count))
    temp_array_cv=np.column_stack((temp_array_cv,X_Test_count))
    

    for a in range(0,temp_array_train.shape[0]):
        for b in range(0,temp_array_train.shape[1]):
            if temp_array_train[a,b]>0:
                temp_array_train[a,b]=np.log1p(temp_array_train[a,b])
            else :
                temp_array_train[a,b]=-np.log1p(-temp_array_train[a,b])                        
                
    for a in range(0,temp_array_cv.shape[0]):
        for b in range(0,temp_array_cv.shape[1]):
            if temp_array_cv[a,b]>0:
                temp_array_cv[a,b]=np.log1p(temp_array_cv[a,b])               
            else :
                temp_array_cv[a,b]= -np.log1p(-temp_array_cv[a,b])          
     
    stda=StandardScaler()
    stda.fit(temp_array_train)
    temp_array_train=stda.transform(temp_array_train)
    temp_array_cv=stda.transform(temp_array_cv) 

    #temp_array_train=csr_matrix(temp_array_train)
    #temp_array_cv=csr_matrix(temp_array_cv)  

    np.savetxt ("amazon_counts_train.txt",np.column_stack((y,temp_array_train)),delimiter=",")
    np.savetxt ("amazon_counts_test.txt",temp_array_cv,delimiter=",")     
    

    print ("done")
    

############ code runs here############
    
create_4way_interractions() # compute 4way interractions
create_likelihoods_with_counts()  # compute likelihoods and counts per fold and print 5 pairs of train/cv files 
    
    
    
    
    
    
