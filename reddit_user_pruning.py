# Adapted from https://github.com/he-y/NLP-Dataset-Pruning
# https://arxiv.org/html/2501.02432v1


from reddit_user_common import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from geom_median.numpy import compute_geometric_median
import pandas as pd

if __name__ == "__main__":
    df=load_dataset(red_row=1000000, user_row=20000)
    reddit_df=load_dataset(red_row=500000, user_row=0)
    user_df=load_dataset(red_row=0, user_row=20000)

    print("Dataset size:", len(df))

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', min_df = .001, max_df=0.7, ngram_range=(1,3))
    # Transform the text data to feature vectors
    vectorizer = vectorizer.fit(df['text'])


    user_matrix = vectorizer.transform(user_df['text'])
    user_matrix = np.array(user_matrix.todense())


    user_median=compute_geometric_median(user_matrix)
    print(user_median.median)

    #print("=====USER=====")

    user_distances = []
    for i in range(user_matrix.shape[0]):
        dist = np.linalg.norm(user_matrix[i]-user_median.median)
        user_distances.append(dist)

    user_df['distance']=user_distances

    user_df['bin']=pd.qcut(user_df['distance'], q=10, labels = range(10))
 
    user_sample = user_df.groupby('bin', group_keys=False).apply(lambda x: x.sample(100))

    #sorted_user = user_sample.sort_values('distance')

    #pd.options.display.max_colwidth = 500

    #print(sorted_user.shape)
    #print(sorted_user.head())
    #print("-----")
    #print(sorted_user.tail())


    #print("=====REDDIT=====")
    
    reddit_matrix= vectorizer.transform(reddit_df['text'])
    reddit_matrix = np.array(reddit_matrix.todense())

    reddit_median=compute_geometric_median(reddit_matrix)

    reddit_distances = []
    for i in range(reddit_matrix.shape[0]):
        dist = np.linalg.norm(reddit_matrix[i]-reddit_median.median)
        reddit_distances.append(dist)

    reddit_df['distance']=reddit_distances

    reddit_df['bin']=pd.qcut(reddit_df['distance'], q=10, labels = range(10))

    reddit_sample = reddit_df.groupby('bin', group_keys=False).apply(lambda x: x.sample(100))

    # sorted_reddit = reddit_df.sort_values('distance')

    # pd.options.display.max_colwidth = 500

    # print(sorted_reddit.iloc[:5, df.columns.get_loc('text')])
    # print("-----")
    # print(sorted_reddit.iloc[-5:, df.columns.get_loc('text')])

    sampled_df = pd.concat([reddit_sample,user_sample],ignore_index=True)
    sampled_df.to_csv("./datasets/pruned.csv")



