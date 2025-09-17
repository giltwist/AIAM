# Adapted from https://www.geeksforgeeks.org/nlp/text-classification-using-scikit-learn-in-nlp/
# Further insight from https://sklearner.com/sklearn-svc-kernel-parameter/


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import time

# Load dataset
redditCSV = pd.read_csv("./datasets/reddit-comments-bodyonly.csv") 
userCSV = pd.read_csv("./datasets/user-comments-bodyonly.csv") 
redditCSV.columns=['text']
redditCSV['label']='reddit'
userCSV.columns=['text']
userCSV['label']='user'

rows=1000


df=pd.concat([redditCSV.head(n=rows),userCSV.head(n=rows)],ignore_index=True)

print("Dataset size:", len(df))

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Transform the text data to feature vectors
X = vectorizer.fit_transform(df['text'])
# Labels
y = df['label']


# Train with different kernel values
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernels:
    print (f'Beginning {kernel} attempt')
    # Split the dataset into training and testing sets
    train_start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the classifier
    clf = SVC(kernel=kernel)
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