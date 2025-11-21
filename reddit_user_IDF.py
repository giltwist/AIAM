# Initially adapted from https://www.geeksforgeeks.org/nlp/text-classification-using-scikit-learn-in-nlp/
# Kernel testing insight from https://sklearner.com/sklearn-svc-kernel-parameter/
# gensim additions from https://medium.com/@dilip.voleti/classification-using-word2vec-b1d79d375381

from reddit_user_common import load_dataset, tk_init, show_result

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import gensim

import pandas as pd
import time

import joblib
import tkinter as tk

class RU_IDF:

    def train_idf(self, verbose):
        df=load_dataset(red_row=20000, user_row=20000)
        #df=pd.read_csv("./datasets/pruned.csv")

        print("Dataset size:", len(df))

        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', min_df=0.0001, max_df=0.7, ngram_range=(1,3))
        # Transform the text data to feature vectors
        X = vectorizer.fit_transform(df['text'])
        # Labels
        y = df['label']

        # Split the dataset into training and testing sets
        train_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and train the classifier
        # only rbf had better accuracy but at double time
        idf = SVC(kernel='linear',probability=True)
        idf.fit(X_train, y_train)
        train_end=time.time();

        print("\tTraining time: ", train_end-train_start)

        if verbose:
            # Predict on the test set
            y_pred = idf.predict(X_test)
            predict_end = time.time()

            # Evaluate the performance
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['reddit','user'])
            report_end=time.time()

            print(f'\tAccuracy: {accuracy:.4f} \n')
            print('Classification Report:')
            print(report)

        joblib.dump(idf,'./datasets/tfidf-model.pkl')
        joblib.dump(vectorizer,'./datasets/tfidf-vectorizer.pkl')
        return idf, vectorizer
    
    def predict(self):
        X = self.vectorizer.transform([self.text_entry.get("1.0", tk.END)])
        prediction = self.model.predict(X)
        confidence = self.model.predict_proba(X)[0]
        show_result(self,prediction[0]=='user',confidence)


if __name__ == "__main__":
    ru = RU_IDF()
    try:
        ru.model=joblib.load('./datasets/tfidf-model.pkl')
        ru.vectorizer=joblib.load('./datasets/tfidf-vectorizer.pkl')
        print("Loading existing model")
    except:
        print("Generating new model")
        ru.model, ru.vectorizer = ru.train_idf(True)

    tk_init(ru,ru.predict)

