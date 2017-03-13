import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix
import sys
reload(sys)
sys.setdefaultencoding('utf8')

#create the data based on the raw json files
def load_data_sparse(data_path=""):
  
    
    train_file = data_path + "train.json"
    test_file = data_path + "test.json"
    train_df = pd.read_json(train_file)
    test_df = pd.read_json(test_file)
    print(train_df.shape)
    print(test_df.shape)
    features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]
    
    # count of photos #
    train_df["num_photos"] = train_df["photos"].apply(len)
    test_df["num_photos"] = test_df["photos"].apply(len)
    
    # count of "features" #
    train_df["num_features"] = train_df["features"].apply(len)
    test_df["num_features"] = test_df["features"].apply(len)

    train_df["listing_id"] = train_df["listing_id"] - 68119576.0
    test_df["listing_id"] =  test_df["listing_id"] - 68119576.0
    
    # count of words present in description column #
    train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
    test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))
 
    train_df["num_price_by_furniture"] = (train_df["price"])/ (train_df["bathrooms"] + train_df["bedrooms"] + 1.0)
    test_df["num_price_by_furniture"] =  (test_df["price"])/ (test_df["bathrooms"] + test_df["bedrooms"] +  1.0)
    
    train_df["price_latitue"] = (train_df["price"])/ (train_df["latitude"]+1.0)
    test_df["price_latitue"] =  (test_df["price"])/ (test_df["latitude"]+1.0)
    
    train_df["price_longtitude"] = (train_df["price"])/ (train_df["longitude"]-1.0)
    test_df["price_longtitude"] =  (test_df["price"])/ (test_df["longitude"]-1.0)  

    train_df["num_furniture"] =  train_df["bathrooms"] + train_df["bedrooms"] 
    test_df["num_furniture"] =   test_df["bathrooms"] + test_df["bedrooms"] 
    
    train_df["num_furniture"] = train_df["num_furniture"].apply(lambda x:  str(x) if float(x)<9.5 else '10')
    test_df["num_furniture"] = test_df["num_furniture"].apply(lambda x:  str(x) if float(x)<9.5 else '10')
            
    # convert the created column to datetime object so as to extract more features 
    train_df["created"] = pd.to_datetime(train_df["created"])
    test_df["created"] = pd.to_datetime(test_df["created"])
    
    # Let us extract some features like year, month, day, hour from date columns #
    train_df["created_month"] = train_df["created"].dt.month
    test_df["created_month"] = test_df["created"].dt.month
    train_df["created_day"] = train_df["created"].dt.day
    test_df["created_day"] = test_df["created"].dt.day
           
    train_df["created_hour"] = train_df["created"].dt.hour
    test_df["created_hour"] = test_df["created"].dt.hour
    train_df["total_days"] =   (train_df["created_month"] -4.0)*30 + train_df["created_day"] +  train_df["created_hour"] /25.0
    test_df["total_days"] =(test_df["created_month"] -4.0)*30 + test_df["created_day"] +  test_df["created_hour"] /25.0        
    train_df["diff_rank"]= train_df["total_days"]/train_df["listing_id"]
    test_df["diff_rank"]= test_df["total_days"]/test_df["listing_id"]
     
    categorical = [ "display_address", "manager_id", "building_id","street_address","num_furniture"]#,"num_furniture","latitude_binned"]#"", "","street_address"
    lencat=len(categorical)

    for f in range (0,lencat):
        for s in range (f+1,lencat): 
            train_df[categorical[f] + "_" +categorical[s]] =train_df[categorical[f]]+"_" + train_df[categorical[s]]
            test_df[categorical[f] + "_" +categorical[s]] =test_df[categorical[f]]+"_" + test_df[categorical[s]]            
            categorical.append(categorical[f] + "_" +categorical[s])
       
    # adding all these new features to use list #
    features_to_use.extend(["num_photos", "num_features", "num_description_words", "created_month", "created_day", "listing_id", "created_hour","total_days","diff_rank",#"listing_rank","total_days_rank",
    "num_price_by_furniture","price_latitue","price_longtitude"])#,"price_latitue_longtitude"]) "created_year", #,"num_description_length"
    result = pd.concat([train_df,test_df])

    for f in categorical:
            if train_df[f].dtype=='object':

                cases=defaultdict(int)
                temp=np.array(result[f]).tolist()
                for k in temp:
                    cases[k]+=1
                print f, len(cases) 
                
                train_df[f]=train_df[f].apply(lambda x: cases[x])
                test_df[f]=test_df[f].apply(lambda x: cases[x])               
                
                features_to_use.append(f)  

    train_df['features'] =  train_df['features'].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    test_df['features'] =test_df['features'].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))  

    train_df['description'] =  train_df['description'].apply(lambda x: str(x).encode('utf-8') if len(x)>2 else "nulldesc") 
    test_df['description'] =test_df['description'].apply(lambda x: str(x).encode('utf-8') if len(x)>2 else "nulldesc") 
    
    tfidfdesc=TfidfVectorizer(min_df=20, max_features=50, strip_accents='unicode',lowercase =True,
                        analyzer='word', token_pattern=r'\w{16,}', ngram_range=(1, 2), use_idf=False,smooth_idf=False, 
    sublinear_tf=True, stop_words = 'english')  
    
    print(train_df["features"].head())
       
    tfidf = CountVectorizer(stop_words='english', max_features=200)
    
    te_sparse = tfidf.fit_transform (test_df["features"])  
    tr_sparse = tfidf.transform(train_df["features"])   

    te_sparsed = tfidfdesc. fit_transform (test_df["description"])  
    tr_sparsed = tfidfdesc.transform(train_df["description"])
    print(features_to_use)
    

    train_X = sparse.hstack([train_df[features_to_use], tr_sparse,tr_sparsed]).tocsr()#
    test_X = sparse.hstack([test_df[features_to_use], te_sparse,te_sparsed]).tocsr()#
    
    target_num_map = {'high':0, 'medium':1, 'low':2}
    train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
    ids= test_df.listing_id.values
    print(train_X.shape, test_X.shape)    
    return train_X,test_X,train_y,ids

#create average value of the target variabe given a categorical feature        
def convert_dataset_to_avg(xc,yc,xt, rounding=2,cols=None):
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
    total_count=0.0
    total_sum =0.0

    for a in range (0,len(xc)):
        target=yc[a]
        total_sum+=target
        total_count+=1.0
        for j in range(0,len(cols)):
            col=cols[j]
            good[j][round(xc[a][col],rounding)]+=target
            bads[j][round(xc[a][col],rounding)]+=1.0  
    #print(total_goods,total_bads)            
    
    for a in range (0,len(xt)):    
        for j in range(0,len(cols)):
            col=cols[j]
            if round(xt[a][col],rounding) in good[j]:
                 woe[a][j]=float(good[j][round(xt[a][col],rounding)])/float(bads[j][round(xt[a][col],rounding)])  
            else :
                 woe[a][j]=round(total_sum/total_count)
    return woe            
    

#converts the select categorical features to numerical via creating averages based on the target variable within kfold. 

def convert_to_avg(X,y, Xt, seed=1, cvals=5, roundings=2, columns=None):
    
    if columns==None:
        columns=[k for k in range(0,(X.shape[1]))]    
    #print("it is not!!")        
    X=X.tolist()
    Xt=Xt.tolist() 
    woetrain=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(X))]
    woetest=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(Xt))]    
    
    kfolder=StratifiedKFold(y, n_folds=cvals,shuffle=True, random_state=seed)
    for train_index, test_index in kfolder:
        # creaning and validation sets
        X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
        y_train = np.array(y)[train_index]

        woecv=convert_dataset_to_avg(X_train,y_train,X_cv, rounding=roundings,cols=columns)
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
    woefinal=convert_dataset_to_avg(np.array(X),np.array(y),np.array(Xt), rounding=roundings,cols=columns) 

    for real_index in range(0,len(Xt)):
        for j in range(0,len(Xt[0])):           
            woetest[real_index][j]=Xt[real_index][j]
            
    for real_index in range(0,len(Xt)):
        for j in range(0,len(columns)):
            col=columns[j]
            woetest[real_index][col]=woefinal[real_index][j]
            
    return np.array(woetrain), np.array(woetest)

        
def main():
    

        #training and test files, created using SRK's python script
        train_file="train_stacknet.csv"
        test_file="test_stacknet.csv"
        
        ######### Load files ############

        X,X_test,y,ids=load_data_sparse (data_path="input/")# you might need to change that to whatever folder the json files are in
        ids= np.array([int(k)+68119576 for k in ids ]) # we add the id value we removed before for scaling reasons.
        print(X.shape, X_test.shape) 
        
        #create to numpy arrays (dense format)        
        X=X.toarray()
        X_test=X_test.toarray()  
        
        print ("scalling") 
        #scale the data
        stda=StandardScaler()  
        X_test=stda.fit_transform (X_test)          
        X=stda.transform(X)

        
        CO=[0,14,21] # columns to create averages on
        
        #Create Arrays for meta
        train_stacker=[ [0.0 for s in range(3)]  for k in range (0,(X.shape[0])) ]
        test_stacker=[[0.0 for s in range(3)]   for k in range (0,(X_test.shape[0]))]
        
        number_of_folds=5 # number of folds to use
        print("kfolder")
        #cerate 5 fold object
        mean_logloss = 0.0
        kfolder=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=15)   

        #xgboost_params
        param = {}
        param['booster']='gbtree'
        param['objective'] = 'multi:softprob'
        param['bst:eta'] = 0.04
        param['seed']=  1
        param['bst:max_depth'] = 6
        param['bst:min_child_weight']= 1.
        param['silent'] =  1  
        param['nthread'] = 12 # put more if you have
        param['bst:subsample'] = 0.7
        param['gamma'] = 1.0
        param['colsample_bytree']= 1.0
        param['num_parallel_tree']= 3   
        param['colsample_bylevel']= 0.7                  
        param['lambda']=5  
        param['num_class']= 3 

        
        i=0 # iterator counter
        print ("starting cross validation with %d kfolds " % (number_of_folds))
        for train_index, test_index in kfolder:
                # creaning and validation sets
                X_train, X_cv = X[train_index], X[test_index]
                y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
                #create past averages for some fetaures
                W_train,W_cv=convert_to_avg(X_train,y_train, X_cv, seed=1, cvals=5, roundings=2, columns=CO)
                W_train=np.column_stack((X_train,W_train[:,CO]))
                W_cv=np.column_stack((X_cv,W_cv[:,CO])) 
                print (" train size: %d. test size: %d, cols: %d " % ((W_train.shape[0]) ,(W_cv.shape[0]) ,(W_train.shape[1]) ))
                #training
                X1=xgb.DMatrix(csr_matrix(W_train), label=np.array(y_train),missing =-999.0)
                X1cv=xgb.DMatrix(csr_matrix(W_cv), missing =-999.0)
                bst = xgb.train(param.items(), X1, 1000) 
                #predictions
                predictions = bst.predict(X1cv)     
                preds=predictions.reshape( W_cv.shape[0], 3)

                #scalepreds(preds)     
                logs = log_loss(y_cv,preds)
                print "size train: %d size cv: %d loglikelihood (fold %d/%d): %f" % ((W_train.shape[0]), (W_cv.shape[0]), i + 1, number_of_folds, logs)
             
                mean_logloss += logs
                #save the results
                no=0
                for real_index in test_index:
                    for d in range (0,3):
                        train_stacker[real_index][d]=(preds[no][d])
                    no+=1
                i+=1
        mean_logloss/=number_of_folds
        print (" Average Lolikelihood: %f" % (mean_logloss) )
                
        #calculating averages for the train data
        W,W_test=convert_to_avg(X,y, X_test, seed=1, cvals=5, roundings=2, columns=CO)
        W=np.column_stack((X,W[:,CO]))
        W_test=np.column_stack((X_test,W_test[:,CO]))          
        #X_test=np.column_stack((X_test,woe_cv))      
        print (" making test predictions ")
        
        X1=xgb.DMatrix(csr_matrix(W), label=np.array(y) , missing =-999.0)
        X1cv=xgb.DMatrix(csr_matrix(W_test), missing =-999.0)
        bst = xgb.train(param.items(), X1, 1000) 
        predictions = bst.predict(X1cv)     
        preds=predictions.reshape( W_test.shape[0], 3)        
       
        for pr in range (0,len(preds)):  
                for d in range (0,3):            
                    test_stacker[pr][d]=(preds[pr][d]) 
        
        
        
        print ("merging columns")   
        #stack xgboost predictions
        X=np.column_stack((X,train_stacker))
        # stack id to test
        X_test=np.column_stack((X_test,test_stacker))        
        
        # stack target to train
        X=np.column_stack((y,X))
        # stack id to test
        X_test=np.column_stack((ids,X_test))
        
        #export to txt files (, del.)
        print ("exporting files")
        np.savetxt(train_file, X, delimiter=",", fmt='%.5f')
        np.savetxt(test_file, X_test, delimiter=",", fmt='%.5f')        

        print("Write results...")
        output_file = "submission_"+str( (mean_logloss ))+".csv"
        print("Writing submission to %s" % output_file)
        f = open(output_file, "w")   
        f.write("listing_id,high,medium,low\n")# the header   
        for g in range(0, len(test_stacker))  :
          f.write("%s" % (ids[g]))
          for prediction in test_stacker[g]:
             f.write(",%f" % (prediction))    
          f.write("\n")
        f.close()
        print("Done.")
                             
                  

       



if __name__=="__main__":
  main()
