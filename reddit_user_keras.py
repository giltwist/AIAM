# Initially adapted from https://www.geeksforgeeks.org/nlp/text-classification-using-scikit-learn-in-nlp/
# Kernel testing insight from https://sklearner.com/sklearn-svc-kernel-parameter/
# Keras info from https://realpython.com/python-keras-text-classification/
# TFIDF with Keras from https://www.geeksforgeeks.org/nlp/tf-idf-representations-in-tensorflow/

from reddit_user_common import load_dataset, tk_init, show_result

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import time

import joblib
import tkinter as tk

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Bidirectional, LSTM

from tensorflow.keras.utils import to_categorical

REDDIT=20000
USER=20000

class RU_KERAS:
    
    def train_keras(self, verbose):
        df=load_dataset(red_row=REDDIT, user_row=USER)
        #df=pd.read_csv("./datasets/pruned.csv")
        
        print("Dataset size:", len(df))

        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', min_df=0.0001, max_df=0.7, ngram_range=(1,3))
        # Transform the text data to feature vectors
        X = vectorizer.fit_transform(df['text'])

        # Labels
        df['label'] = [1 if L=='user' else 0 for L in df['label']]
        y=df['label']

        train_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = Sequential()
        #model.add(Embedding(input_dim=X_train.shape[1], output_dim=128))
        #model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['f1_score'])

        #print(y_train.head())
        history = model.fit(X_train, to_categorical(y_train), epochs=4)

        loss, accuracy = model.evaluate(X_test, to_categorical(y_test))
        print(f"Accuracy: {accuracy} | Loss: {loss}")
        joblib.dump(model,'./datasets/tfidf-keras.pkl')
        joblib.dump(vectorizer,'./datasets/tfidf-vectorizer-keras.pkl')
        return model, vectorizer
              
    
    def predict(self):
        X = self.vectorizer.transform([self.text_entry.get("1.0", tk.END)])
        prediction = self.model.predict(X)[0]
        is_Me = bool(prediction[1]>prediction[0])
        #confidence = self.model.predict_proba(X)[0]
        show_result(self,is_Me,prediction)


if __name__ == "__main__":
    ru = RU_KERAS()
    try:
        ru.model=joblib.load('./datasets/tfidf-keras.pkl')
        ru.vectorizer=joblib.load('./datasets/tfidf-vectorizer-keras.pkl')
        print("Loading existing keras model")
    except:
        print("Generating keras model")
        ru.model, ru.vectorizer = ru.train_keras(True)

   

    tk_init(ru,ru.predict)

