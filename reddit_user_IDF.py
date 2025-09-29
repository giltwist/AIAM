# Initially adapted from https://www.geeksforgeeks.org/nlp/text-classification-using-scikit-learn-in-nlp/
# Kernel testing insight from https://sklearner.com/sklearn-svc-kernel-parameter/
# gensim additions from https://medium.com/@dilip.voleti/classification-using-word2vec-b1d79d375381

from reddit_user_common import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import gensim

import pandas as pd
import time

df=load_dataset(red_row=10000, user_row=1000)
# Clean data using the built in cleaner in gensim
df['text_clean'] = df['text'].apply(lambda x: " ".join(gensim.utils.simple_preprocess(x)))

print("Dataset size:", len(df))



# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Transform the text data to feature vectors
X_unclean = vectorizer.fit_transform(df['text'])
X_clean = vectorizer.fit_transform(df['text_clean'])
# Labels
y = df['label']

X_sets=[X_unclean,X_clean]

for trial, X in enumerate(X_sets):

    print (f'Beginning attempt {trial}')
    # Split the dataset into training and testing sets
    train_start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the classifier
    # only rbf had better accuracy but at double time
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    train_end=time.time();

    print("\tTraining time: ", train_end-train_start)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    predict_end = time.time()

    # Evaluate the performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['reddit','user'])
    report_end=time.time()

    print(f'\tAccuracy: {accuracy:.4f} \n')
    #print('Classification Report:')
    #print(report)

