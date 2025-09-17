# Adapted from https://www.geeksforgeeks.org/nlp/text-classification-using-scikit-learn-in-nlp/

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd

# Load dataset
redditCSV = pd.read_csv("./datasets/reddit-comments-bodyonly.csv") 
userCSV = pd.read_csv("./datasets/user-comments-bodyonly.csv") 
redditCSV.columns=['text']
redditCSV['label']='reddit'
userCSV.columns=['text']
userCSV['label']='user'

df=pd.concat([redditCSV,userCSV],ignore_index=True)

print(df.head())
print("-----")
print(df.tail())
