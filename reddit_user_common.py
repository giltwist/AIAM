# Initially adapted from https://www.geeksforgeeks.org/nlp/text-classification-using-scikit-learn-in-nlp/
# Kernel testing insight from https://sklearner.com/sklearn-svc-kernel-parameter/
# gensim additions from https://medium.com/@dilip.voleti/classification-using-word2vec-b1d79d375381

import pandas as pd

def load_dataset(red_row=100,user_row=100):
    # Load dataset
    redditCSV = pd.read_csv("./datasets/reddit-comments-bodyonly.csv") 
    userCSV = pd.read_csv("./datasets/user-comments-bodyonly.csv") 
    redditCSV.columns=['text']
    redditCSV['label']='reddit'
    userCSV.columns=['text']
    userCSV['label']='user'


    df=pd.concat([redditCSV.head(n=red_row),userCSV.head(n=user_row)],ignore_index=True)
    return df
