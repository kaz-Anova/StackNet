# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:23:59 2017

@author: mariosm
"""
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import sys
from nltk.corpus import stopwords
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix,hstack
#stops = set(stopwords.words("english"))
stops = set(["http","www","img","border","home","body","a","about","above","after","again","against","all","am","an",
"and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't",
"cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from",
"further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers",
"herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
"itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought",
"our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such",
"than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're",
"they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were",
"weren't","what","what's","when","when's""where","where's","which","while","who","who's","whom","why","why's","with","won't","would",
"wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves" ])

weights={}

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
    


# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=5000.0, min_count=2.0):
    if count < min_count:
        return 0.0
    else:
        return 1.0 / (count + eps)


def word_shares(row,wei,stop):
    
 
		q1 = set(str(row['question1']).lower().split())
		q1words = q1.difference(stop)
		if len(q1words) == 0:
			return '0:0:0:0:0'

		q2 = set(str(row['question2']).lower().split())
		q2words = q2.difference(stop)
		if len(q2words) == 0:
			return '0:0:0:0:0'

		q1stops = q1.intersection(stop)
		q2stops = q2.intersection(stop)

		shared_words = q1words.intersection(q2words)
		#print(len(shared_words))
		shared_weights = [wei.get(w, 0) for w in shared_words]
		total_weights = [wei.get(w, 0) for w in q1words] + [wei.get(w, 0) for w in q2words]
        
		R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
		R2 = float(len(shared_words)) / (float(len(q1words)) + float(len(q2words))) #count share
		R31 = float(len(q1stops)) / float(len(q1words)) #stops in q1
		R32 = float(len(q2stops)) / float(len(q2words)) #stops in q2
		return '{}:{}:{}:{}:{}'.format(R1, R2, float(len(shared_words)), R31, R32)


def main():

    input_folder="input/" # set your input folder here
    df_train = pd.read_csv(input_folder + 'train.csv')
    df_test  = pd.read_csv(input_folder + 'test.csv')
    print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))
	
                        
    train_mix = (df_train['question1']+ " " +  df_train['question2']).astype(str).values
    test_mix = (df_test['question1']+ " " +  df_test['question2'] ).astype(str).values   	
    print("Features processing, be patient...")


    
    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    #stops = set(stopwords.words("english"))

    X = pd.DataFrame()
    X_test = pd.DataFrame()
    df_train['word_shares'] = df_train.apply(word_shares, args = (weights,stops,),axis=1, raw=True)
    df_test['word_shares'] = df_test.apply(word_shares, args = (weights,stops,),axis=1, raw=True)
    
    X['word_match']       = df_train['word_shares'].apply(lambda x: float(x.split(':')[0]))
    X['tfidf_word_match'] = df_train['word_shares'].apply(lambda x: float(x.split(':')[1]))
    X['shared_count']     = df_train['word_shares'].apply(lambda x: float(x.split(':')[2]))
    X['stops1_ratio']     = df_train['word_shares'].apply(lambda x: float(x.split(':')[3]))
    X['stops2_ratio']     = df_train['word_shares'].apply(lambda x: float(x.split(':')[4]))
    X['diff_stops_r']     = X['stops1_ratio'] - X['stops2_ratio']
    X['len_q1'] = df_train['question1'].apply(lambda x: len(str(x)))
    X['len_q2'] = df_train['question2'].apply(lambda x: len(str(x)))
    X['diff_len'] = X['len_q1'] - X['len_q2']
    X['len_char_q1'] = df_train['question1'].apply(lambda x: len(str(x).replace(' ', '')))
    X['len_char_q2'] = df_train['question2'].apply(lambda x: len(str(x).replace(' ', '')))
    X['diff_len_char'] = X['len_char_q1'] - X['len_char_q2']
    X['len_word_q1'] = df_train['question1'].apply(lambda x: len(str(x).split()))
    X['len_word_q2'] = df_train['question2'].apply(lambda x: len(str(x).split()))
    X['diff_len_word'] = X['len_word_q1'] - X['len_word_q2']
    X['avg_world_len1'] = X['len_char_q1'] / X['len_word_q1']
    X['avg_world_len2'] = X['len_char_q2'] / X['len_word_q2']
    X['diff_avg_word'] = X['avg_world_len1'] - X['avg_world_len2']
    X['exactly_same'] = (df_train['question1'] == df_train['question2']).astype(int)
 
    X_test['word_match']       = df_test['word_shares'].apply(lambda x: float(x.split(':')[0]))
    X_test['tfidf_word_match'] = df_test['word_shares'].apply(lambda x: float(x.split(':')[1]))
    X_test['shared_count']     = df_test['word_shares'].apply(lambda x: float(x.split(':')[2]))
    X_test['stops1_ratio']     = df_test['word_shares'].apply(lambda x: float(x.split(':')[3]))
    X_test['stops2_ratio']     = df_test['word_shares'].apply(lambda x: float(x.split(':')[4]))
    X_test['diff_stops_r']     = X_test['stops1_ratio'] - X_test['stops2_ratio']
    X_test['len_q1'] = df_test['question1'].apply(lambda x: len(str(x)))
    X_test['len_q2'] = df_test['question2'].apply(lambda x: len(str(x)))
    X_test['diff_len'] = X_test['len_q1'] - X_test['len_q2']
    X_test['len_char_q1'] = df_test['question1'].apply(lambda x: len(str(x).replace(' ', '')))
    X_test['len_char_q2'] = df_test['question2'].apply(lambda x: len(str(x).replace(' ', '')))
    X_test['diff_len_char'] = X_test['len_char_q1'] - X_test['len_char_q2']
    X_test['len_word_q1'] = df_test['question1'].apply(lambda x: len(str(x).split()))
    X_test['len_word_q2'] = df_test['question2'].apply(lambda x: len(str(x).split()))
    X_test['diff_len_word'] = X_test['len_word_q1'] - X_test['len_word_q2']
    X_test['avg_world_len1'] = X_test['len_char_q1'] / X_test['len_word_q1']
    X_test['avg_world_len2'] = X_test['len_char_q2'] / X_test['len_word_q2']
    X_test['diff_avg_word'] = X_test['avg_world_len1'] - X_test['avg_world_len2']
    X_test['exactly_same'] = (df_test['question1'] == df_test['question2']).astype(int)   
    
   
    print (np.mean(X['word_match']) , np.mean(X['tfidf_word_match']),np.mean(X_test['word_match']) , np.mean(X_test['tfidf_word_match']))
    #convert to csr
    X=csr_matrix(X)
    X_test=csr_matrix(X_test)
    # the tfidf object
    tfidf=TfidfVectorizer(min_df=1, max_features=None, strip_accents='unicode',lowercase =True,
                        analyzer='word', token_pattern=r'\w{2,}', ngram_range=(1, 1), use_idf=True,smooth_idf=True, 
    sublinear_tf=True, stop_words = 'english')  
    
    # aplied tf-idf
    tr_sparsed  = tfidf. fit_transform (train_mix)  
    te_sparsed = tfidf.transform(test_mix)
    print (tr_sparsed.shape, te_sparsed.shape, X.shape, X_test.shape)  
    #join the the tfidf with the remaining data
    X =hstack([X,tr_sparsed]).tocsr()#
    X_test = hstack([X_test, te_sparsed]).tocsr()#

    #retrieve target
    y = df_train['is_duplicate'].values  
    print (X.shape, X_test.shape, y.shape) 
    
    #export sparse data to stacknet format (which is Libsvm format)
    fromsparsetofile("train.sparse", X, deli1=" ", deli2=":",ytarget=y)    
    fromsparsetofile("test.sparse", X_test, deli1=" ", deli2=":",ytarget=None)       
    
                     
if __name__=="__main__":
  main()
  