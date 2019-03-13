# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:00:51 2019

@author: arpit
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:55:48 2019

@author: arpit
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

 #   python -m textblob.download_corpora
def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

data = pd.read_csv('../dataset/dataset.csv')
x = data['news'].tolist()
y = data['type'].tolist()

for index,value in enumerate(x):
    print ("processing data:",index)
    x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])

vect = CountVectorizer(stop_words='english',min_df=2)
X = vect.fit_transform(x)
Y = np.array(y)

print( "no of features extracted:",X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print ("train size:", X_train.shape)
print("test size:", X_test.shape)
no_topics = 10

lda_model = NMF(n_components=no_topics, init = "nndsvd")
lda_train = lda_model.fit_transform(X_train)
lda_test = lda_model.transform(X_test)

model = RandomForestClassifier(n_estimators=300, max_depth=150,n_jobs=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
c_mat = confusion_matrix(y_test,y_pred)
kappa = cohen_kappa_score(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print("Confusion Matrix:\n", c_mat)
print("\nKappa: ",kappa)

print("F1 Score:"+str(f1_score(y_test, y_pred, average='macro')))
print("Precision:"+str(precision_score(y_test,y_pred, average='macro')))
print("Recall:"+str(recall_score(y_test,y_pred, average='macro')))
print("\nAccuracy: ",acc)