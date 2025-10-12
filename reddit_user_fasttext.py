# fasttext basics from https://rukshanjayasekara.wordpress.com/2022/05/13/text-classification-with-fasttext/

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import fasttext
import pprint
import numpy as np
import pandas as pd
import time
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from reddit_user_common import load_dataset, tk_init, show_result

import joblib
import tkinter as tk

class RU_FT:

    def train_ft(self, verbose):
        df=load_dataset(red_row=20000, user_row=20000)

        print("Dataset size:", len(df))
        prep_start=time.time()

        df['text'] = df['text'].apply(word_tokenize)
        df['text'] = df['text'].apply(lambda x: [word.lower() for word in x if word.isalpha() and word.lower() not in stopwords.words('english')])
        df['label'] = df['label'].apply(lambda x: "__label__" + x)


        # Split data into train and test sets
        df_train = df.sample(frac=0.7, random_state=42)
        df_test = df.drop(df_train.index)

        df_train.to_csv('reddit_train_ft.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ", encoding = 'utf-8')
        df_test.to_csv('reddit_test_ft.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ", encoding = 'utf-8')
        prep_end=time.time()
        print("\tPrep time: ", prep_end-prep_start)

        # Train the fasttext model
        #train_start = time.time()
        model = fasttext.train_supervised('reddit_train_ft.txt',wordNgrams=2,epoch=20,lr=0.7)
        #model = fasttext.train_supervised('reddit_train_ft.txt',autotuneValidationFile='reddit_test_ft.txt', autotuneMetric="f1:__label__user")
        #train_end=time.time();
        #print("\tTraining time: ", train_end-train_start)

        model.save_model('./datasets/ft-model.pkl')
        # Use the trained model to make predictions on the test data
        if verbose:
            pprint.pp(model.test_label('reddit_test_ft.txt'))

        return model

        #print(dict(list(y_pred.items())[0:5]))

        # Evaluate the performance
        #accuracy = accuracy_score(y_test, y_pred)
        #report = classification_report(y_test, y_pred, target_names=['reddit','user'])

        #print(f'\tAccuracy: {accuracy:.4f} \n')
    
    def predict(self):
        X = self.text_entry.get("1.0", tk.END).replace("\n", "")
        prediction = self.model.predict(X)
        
        print(prediction)
        #show_result(self,prediction[0]=='user')


if __name__ == "__main__":
    ru = RU_FT()
    try:
        ru.model=fasttext.load_model('./datasets/ft-model.pkl')
        
        print("Loading existing model")
    except:
        print("Generating new model")
        ru.model = ru.train_ft(True)

    tk_init(ru,ru.predict)
