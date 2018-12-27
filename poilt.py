# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:10:16 2018

@author: paprasad
"""


import numpy as np
import pandas as pd
import nltk
import  re
from math import * 

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

from scipy import spatial


from math import*
 
def square_rooted(x):
 
    return round(sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
   x = x[0]
   y = y[0]
   
   numerator = sum(a*b for a,b in zip(x,y))
   denominator = square_rooted(x)*square_rooted(y)

   ret =  round(numerator/float(denominator),3)
   return ret



def cosine_similarity1(list1,list2):
    result = 1 - spatial.distance.cosine(list1[0], list2[0])
    return result



def lemmatize_text(text):
    '''
    This method applies lemmatizeation on the input text value
    '''
    temp=''
    for w in w_tokenizer.tokenize(text):
        temp = temp+' '+lemmatizer.lemmatize(w)
        temp = temp.lower()
    return temp

def text_stemming(text):
    '''
    This method applies stemming on the input text value
    '''
    temp=''
    #print(text)
    for w in w_tokenizer.tokenize(text):
        temp = temp+' '+ ps.stem(w)
    return temp  


def create_list(text):
    ret_dict = []
    for word in text.split():     
        if word not in stop_words and word not in ret_dict:
            ret_dict.append(word)
        
        
        
    return ret_dict



def feature(sentences,feature_set):
    temp = []
    for sentence in sentences:
        temp1= []
        for feature in feature_set:
            if feature in sentence:
                temp1.append(1)
            else:
                temp1.append(0)
        temp.append(temp1)    
            
    return temp    

def ques_feature(ques,feature_set):
    temp = []
    for feature in feature_set:
        if feature in ques:
            temp.append(1)
        else:
            temp.append(0)
    return temp

def clean_text(text):
    regex = re.compile('(<!--(.|\\n)*-->)')
    text = regex.sub('',text)
    regex = re.compile('\n')
    clean_text = regex.sub(' ',text)
    clean_text = ''.join(x for x in clean_text if( x.isalpha() or (x ==' ')))
    return clean_text



file = open('sample.txt','r',encoding="utf8")
file = file.read()
file = file.lower()
#file = lemmatize_text(file)
#file = text_stemming(file)

file = file.strip()

sentences = file.split('.') 
sentences.pop()
file = clean_text(file)


for index, sentence in enumerate(sentences):
    sentence = clean_text(sentence)
    sentences[index]  = sentence

stop_words = nltk.corpus.stopwords.words('english')


feature_set = create_list(file)

#for sentence in sentences:
feature_mat = feature(sentences,feature_set) 

df_passage = pd.DataFrame(feature_mat)



ques = input("enter your question\n")
ques = lemmatize_text(ques)
ques = text_stemming(ques)

ques =  ques_feature(ques,feature_set) 


df_ques = pd.DataFrame([ques])

temp = []
for row in range(df_passage.shape[0]):  
   # res = cosine_similarity(df_ques,df_passage[row])
    print(row)
    X= [list(df_ques.iloc[0,:])]
    #print(type(X))
   # X = np.shape(X)
    Y = [list(df_passage.iloc[row,:])]
    #print(type(Y))
   
    #res = np.dot(X,Y)
    res = cosine_similarity(X,Y)
    temp.append(res)
    
res = pd.DataFrame(temp)    
print(res[res[0]==np.max(res[0])])
