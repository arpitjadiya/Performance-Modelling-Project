import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from textblob import Word
import numpy as np
import time as time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score,recall_score
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

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

data = pd.read_csv('dataset.csv')
data_test = pd.read_csv('dataset_test.csv')
x = data['news'].tolist()
y = data['type'].tolist()
x_test = data_test['news'].tolist()
y_test = data_test['type'].tolist()

for index,value in enumerate(x):
    print("processing data:",index)
    x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])

for index,value in enumerate(x_test):
    print("processing data:",index)
    x_test[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])

vect = TfidfVectorizer(stop_words='english',min_df=2)
# vect = CountVectorizer(stop_words='english',min_df=2)

X_train = vect.fit_transform(x)
y_train = np.array(y)

X_test = vect.transform(x_test)
y_test = np.array(y_test)

print("train size:", X_train.shape)
print("test size:", X_test.shape)

no_topics = 10

start1 = time.time()
lda_model = TruncatedSVD(n_components=no_topics,
                         algorithm='randomized',
                         n_iter=50)
lda_train = lda_model.fit_transform(X_train)
lda_test = lda_model.transform(X_test)
scaler = MinMaxScaler()        # Scale features to make them positive
lda_train = scaler.fit_transform(lda_train)
lda_test = scaler.transform(lda_test)
end1 = time.time()

model = RandomForestClassifier(n_estimators=50, max_depth=10,n_jobs=1)
model.fit(lda_train, y_train)

y_pred = model.predict(lda_test)
c_mat = confusion_matrix(y_test,y_pred)
kappa = cohen_kappa_score(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)

print("Confusion Matrix:\n", c_mat)
print("Precision:"+str(precision_score(y_test,y_pred, average='macro')))
print("Recall:"+str(recall_score(y_test,y_pred, average='macro')))
print("F1 Score:"+str(f1_score(y_test, y_pred, average='macro')))
print("Accuracy: ",acc)
print("Time:"+str((end1-start1)))
print("Matthew's correlation coefficient:"+str(matthews_corrcoef(y_test,y_pred)))
print("Kappa:",kappa)