# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:28:07 2020

@author: Sayan Mondal
"""
## Import important Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = pd.read_csv("C:/Users/Sayan Mondal/Desktop/fake news/data.csv")


data.head()

data.isna().sum()

data.dropna(axis=0,inplace=True) ### dropping all the na values..

data.info()


X=data.drop('Label', axis=1) ## taking all values except Label....


X.shape

y=data.Label

y.shape

import tensorflow as tf
tf.__version__


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences ## for making input length fixed in embedding layer
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.text import one_hot  ## converting the sentences to one_hot representation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


## Lets take the vocab size..
vocab_size=8000

messages=X.copy()

messages.shape

messages.reset_index(inplace=True)


messages.head(6)


import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

messages.Body[5]

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
lemmatizer= WordNetLemmatizer()
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['Headline'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review= [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


## one hot representation....##    
one_hot_wrds= [one_hot(words,vocab_size) for words in corpus]    
one_hot_wrds
    
    
## Embeded representation....###
sent_length=18

embeded_docs= pad_sequences(one_hot_wrds, padding='pre', maxlen=sent_length)
embeded_docs[97]

## Model building..####
embedding_vector_features=50
model=Sequential()
model.add(Embedding(vocab_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(80))  ## one LSTM layer with 80 neurons...
model.add(Dense(1,activation='sigmoid')) ## Dense layer used for classification problems
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

## Adding dropout layer....###
embedding_vector_features=50
model=Sequential()
model.add(Embedding(vocab_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.2))
model.add(LSTM(80)) 
model.add(Dropout(0.2)) ## one LSTM layer with 80 neurons...
model.add(Dense(1,activation='sigmoid')) ## Dense layer used for classification problems
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


X_final=np.array(embeded_docs)
y_final=np.array(y)

X_final.shape
y_final.shape

## Train Test Split...####
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_state=60)

## Fitting the model...##
model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=100, batch_size=80)

y_pred=model.predict_classes(X_test)

## Confusion Matrix...###
confusion_matrix(y_test,y_pred)

## Accuracy
accuracy_score(y_test,y_pred) ## 83.45%...##

## Classification report...
print(classification_report(y_test,y_pred))


#######.......... Bi-Directional LSTM....######################
from tensorflow.keras.layers import Bidirectional

## Creating model
embedding_vector_features=60
model1=Sequential()
model1.add(Embedding(vocab_size,embedding_vector_features,input_length=sent_length))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dropout(0.2))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())

X_final=np.array(embeded_docs)
y_final=np.array(y)

X_final.shape
y_final.shape

## Train Test Split...####
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.30, random_state=60)

## Fitting the model...##
model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=100, batch_size=80)

y_pred=model.predict_classes(X_test)

## Confusion Matrix...###
confusion_matrix(y_test,y_pred)

## Accuracy
accuracy_score(y_test,y_pred) ## 86.38%...##

## Classification report...
print(classification_report(y_test,y_pred)) 

