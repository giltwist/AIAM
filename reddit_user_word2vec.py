# gensim additions from https://medium.com/@dilip.voleti/classification-using-word2vec-b1d79d375381

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import gensim

import numpy as np
import pandas as pd
import time

from reddit_user_common import load_dataset

df=load_dataset(red_row=20000, user_row=20000)

# Clean data using the built in cleaner in gensim
df['text_clean'] = df['text'].apply(lambda x: gensim.utils.simple_preprocess(x))

print("Dataset size:", len(df))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split (df['text_clean'], df['label'] , test_size=0.3, random_state=42)

# Train the word2vec model
train_start = time.time()
w2v_model = gensim.models.Word2Vec(X_train,
                                   vector_size=100,
                                   window=5,
                                   min_count=2)

# Generate aggregated sentence vectors
words = set(w2v_model.wv.index_to_key )
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_train],dtype=object)
X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_test],dtype=object)

# Compute sentence vectors by averaging the word vectors for the words contained in the sentence
X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(100, dtype=float))
        
X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(100, dtype=float))

rf = RandomForestClassifier()
rf_model = rf.fit(X_train_vect_avg, y_train.values.ravel())

train_end=time.time();

print("\tTraining time: ", train_end-train_start)

# Use the trained model to make predictions on the test data
y_pred = rf_model.predict(X_test_vect_avg)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['reddit','user'])

print(f'\tAccuracy: {accuracy:.4f} \n')
