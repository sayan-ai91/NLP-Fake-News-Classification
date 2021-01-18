## Import important Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

data = pd.read_csv("data.csv")


data.head()

data.isna().sum()

data.dropna(axis=0,inplace=True) ### dropping all the na values..

data.info()


X=X=data.drop('Label', axis=1)
y=data.Label



## Lets take the vocab size..
vocab_size=15000

messages=X.copy()

messages.shape

messages.reset_index(inplace=True)


### Text cleanning..##

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
pattern = r'[0-9]'





corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['Headline'][i])
    review= re.sub(pattern, '', messages['Headline'][i])
   
    review = review.lower()
    review = review.split()
   
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review= [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    #review =' '.join((item for item in review if not item.isdigit()))
    corpus.append(review)
    


## lets find the unique words##
corpus_word_count= corpus   

count =' '.join([str(elm) for elm in corpus_word_count ]) 

from nltk.tokenize.regexp import WordPunctTokenizer

#This tokenizer also splits our string into tokens:

my_toks = WordPunctTokenizer().tokenize(count)

len(my_toks)

## unique word count

my_vocab = set(my_toks)
len(my_vocab)    ## 6087
    
type(corpus)


##logistic regression....###
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
y=data.Label

type(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


lr=LogisticRegression(n_jobs=-1, C=1.0)
X_train.shape, y_train.shape


%%time
model=lr.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))
#### Accuracy of the model is 84.46...###

### kappa score...##
cohen_kappa_score(y_test,y_pred) ##.68445


print(classification_report(y_test,y_pred)) ## precision 0-> .82, 1-> .88... recall 0--> .85, 1-->> .83
# confusion Matrix...##

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

n_errors=(y_pred!=y_test).sum()
print(n_errors)  # 124

## ROC Curve...

lr_roc_auc = roc_auc_score(y_test, lr.predict(X_test)) 
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='logistic regression(area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Review Classification')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show() 
  
print (lr_roc_auc) # .83999




### Random Forest##
### Fitting TF-IDF on corpus...##

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
y=data.Label
type(X)

rf= RandomForestClassifier(n_estimators=100,n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=rf.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))
 #### Accuracy of the model is 84.21...###
 
### kappa score...##
cohen_kappa_score(y_test,y_pred) ##.6825


print(classification_report(y_test,y_pred)) ## precision 0-> .82, 1-> .88... recall 0--> .85, 1-->> .83
# confusion Matrix...##

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

n_errors=(y_pred!=y_test).sum()
print(n_errors)  # 140

## ROC Curve...

rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test)) #print(rf_roc_auc ) ## .8284
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest(area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('NEWS Classification')
plt.legend(loc="lower right")
plt.savefig('rf_ROC')
plt.show() 
  
print (rf_roc_auc) # .83999


### Xgboost implementation...##
import xgboost as xgb 

#X = cv.fit_transform(corpus)
X = vectorizer.fit_transform(corpus)
y=data.Label

## Hyper Parameter Optimization
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    }

XGBC=xgb.XGBClassifier() 

random_search=RandomizedSearchCV(XGBC,param_distributions=params,n_iter=10,scoring='roc_auc',n_jobs=-1,cv=5,verbose=1)

%%time
random_search.fit(X,y)

# finding best estimator..#
random_search.best_estimator_


XGBC=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.1,
              learning_rate=0.25, max_delta_step=0, max_depth=12,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1) 


X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.20, random_state=42)

model=XGBC.fit(X_train,y_train)

y_pred=model.predict(X_test)

n_errors=(y_pred!=y_test).sum()
print(n_errors) #166


print("Accuracy:",accuracy_score(y_test, y_pred)) ## 79.19%..

print(classification_report(y_test,y_pred))

cohen_kappa_score(y_test,y_pred)
# kappa score is .5830



#### catboost implementation.....###

from catboost import CatBoostClassifier
from catboost.utils import eval_metric
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
    
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
y=data.Label    

cb=CatBoostClassifier()

"""parameters =params = {'depth': [4,7,8,9],
          'learning_rate' : [0.03, 0.05, 0.10],
         'l2_leaf_reg': [4,7,8,9],
         'iterations': [500]}
randm = RandomizedSearchCV(estimator=cb, param_distributions = parameters, 
                               cv = 3, n_iter = 10, n_jobs=-1)

%%time
randm.fit(X_train, y_train)

# finding best estimator..#
print(randm.best_estimator_)

randm.best_params_""" ## Run this part only if your system config is good..##

cb=CatBoostClassifier(eval_metric="AUC",
                         depth=8, iterations=500, l2_leaf_reg=8, learning_rate= 0.05)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

%%time                            
model=cb.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred)) ## 81%..

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test,y_pred)) ### precision 0 --80, 1-- 83; recall 0--87, 1--74

cohen_kappa_score(y_test,y_pred) ## .6154

n_errors=(y_pred!=y_test).sum()
print(n_errors)  # 151


## ROC Curve...

cb_roc_auc = roc_auc_score(y_test, cb.predict(X_test)) #print(rf_roc_auc ) ## .8284
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='CatBoost(area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('NEWS Classification')
plt.legend(loc="lower right")
plt.savefig('cb_ROC')
plt.show() 
  
print (cb_roc_auc) # .7806




import tensorflow as tf
tf.__version__


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences ## for making input length fixed in embedding layer
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.text import one_hot  ## converting the sentences to one_hot representation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout



## one hot representation....##  
## Lets take the vocab size..
vocab_size=7000
one_hot_wrds= [one_hot(words,vocab_size) for words in corpus]    
one_hot_wrds
    
    
## Embeded representation....###
sent_length=25

embeded_docs= pad_sequences(one_hot_wrds, padding='pre', maxlen=sent_length)
type(embeded_docs)
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
embedding_vector_features=100
model=Sequential()
model.add(Embedding(vocab_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.2))
model.add(LSTM(80)) 
model.add(Dropout(0.2)) ## one LSTM layer with 80 neurons...
model.add(Dense(1,activation='sigmoid')) ## Dense layer used for classification problems
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


X_final=np.array(embeded_docs)
type(X_final)
y_final=np.array(y)

X_final.shape
y_final.shape

## Train Test Split...####
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_state=60)

## Fitting the model...##
model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=200, batch_size=100)

y_pred=model.predict_classes(X_test)

## Confusion Matrix...###
confusion_matrix(y_test,y_pred)

## Accuracy
accuracy_score(y_test,y_pred) ## 82%...##

## Classification report...
print(classification_report(y_test,y_pred))

cohen_kappa_score(y_test,y_pred) ## .6550



#######.......... Bi-Directional LSTM....######################
from tensorflow.keras.layers import Bidirectional

## Creating model
embedding_vector_features=100
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
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.20)

## Fitting the model...##
model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=200, batch_size=80)

y_pred=model.predict_classes(X_test)

## Confusion Matrix...###
confusion_matrix(y_test,y_pred)

## Accuracy
accuracy_score(y_test,y_pred) ## 90%...##

## Classification report...
print(classification_report(y_test,y_pred)) 

#kappa score
cohen_kappa_score(y_test,y_pred) ## .7930


## ROC Curve...## PLOTTING ROC AUC IN TENSORFLOW LITTLE DIFF...

y_pred_keras = model.predict(X_test).ravel()  ## need to add ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

BILSTM_roc_auc = roc_auc_score(y_test, y_pred_keras) #print(rf_roc_auc ) ## .8284
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
plt.plot(fpr, tpr, label='BiLSTM(area = %0.2f)' % BILSTM_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('NEWS Classification')
plt.legend(loc="lower right")
plt.savefig('BiLSTM')
plt.show() 

print(BILSTM_roc_auc)  ## .9576
  
