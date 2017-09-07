# Based on this kaggle script : https://www.kaggle.com/danieleewww/xgboost-lightgbm-and-olsv107-w-month-features/code

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix


directory="input/" # hodls the data

## converts arrayo to sparse svmlight format
def fromsparsetofile(filename, array, deli1=" ", deli2=":",ytarget=None):    
    zsparse=csr_matrix(array)
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
    
#creates the main dataset abd prints 2 files to dataset2_train.txt and  dataset2_test.txt

def dataset2():

    ##### RE-READ PROPERTIES FILE
    
    print( "\nRe-reading properties file ...")
    properties = pd.read_csv(directory +'properties_2016.csv')

    train = pd.read_csv(directory +"train_2016_v2.csv")      
    ##### PROCESS DATA FOR XGBOOST
    
    print( "\nProcessing data for XGBoost ...")
    for c in properties.columns:
        properties[c]=properties[c].fillna(-1)
        if properties[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(properties[c].values))
            properties[c] = lbl.transform(list(properties[c].values))
    
    train_df = train.merge(properties, how='left', on='parcelid')
    
    train_df["transactiondate"] = pd.to_datetime(train_df["transactiondate"])
    train_df["Month"] = train_df["transactiondate"].dt.month
    
    x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
    x_test = properties.drop(['parcelid'], axis=1)
    
    x_test["transactiondate"] = '2016-07-01'
    x_test["transactiondate"] = pd.to_datetime(x_test["transactiondate"])
    x_test["Month"] = x_test["transactiondate"].dt.month #should use the most common training date 2016-10-01
    x_test = x_test.drop(['transactiondate'], axis=1)
    
    # shape        
    print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
    
    # drop out ouliers
    train_df=train_df[ train_df.logerror > -0.4 ]
    train_df=train_df[ train_df.logerror < 0.419 ]
    x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
    y_train = train_df["logerror"].values.astype(np.float32)
    x_train = x_train.values.astype(np.float32, copy=False)
    x_test = x_test.values.astype(np.float32, copy=False)  
  
    print('After removing outliers:')     
    print (" shapes of dataset 2 ", x_train.shape, y_train.shape, x_test.shape)
    
    print (" printing %s " % ("dataset2_train.txt") )
    fromsparsetofile("dataset2_train.txt", x_train, deli1=" ", deli2=":",ytarget=y_train)     
    print (" printing %s " % ("dataset2_test.txt") )    
    fromsparsetofile("dataset2_test.txt", x_test, deli1=" ", deli2=":",ytarget=None)         
    print (" finished with daatset2 " )      
    return
 


def main():
    

    dataset2()

    
    print( "\nFinished ...")
    
    
    

if __name__ == '__main__':
   main()