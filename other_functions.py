"""
Other functions

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', 50)

plt.rcParams['figure.figsize'] = (9, 6)
sns.set(style='white',font_scale=1.2)



def yosemite_attraction_reco(df_attractions, user_weights = defaultdict(int)):
    """
    Returns a recommendation of attractions to prioritize on based on desired characteristics input by the user.
    
    Input: df_attractions [DataFrame] - Scores across each topic & attraction
           user_weights [Default Dictionary] - Weights input by the user for the desired topics
           
    Output: Recommendations [List] - Top 3 recommendations sorted in order based on user weight inputs.
    
    Please note, the code for this recommenders was sourced from Julia Qiao's implementation
    for news outlet recommendations.
    Source Link: https://github.com/JuliaQiao/NLP_News_Recommender/
    """
       
    # Taking the average scores across each topic that will be weighed against the user inputs
    average_topic_scores  = df_attractions.mean(axis = 0)
    
    # Using the user input weights to create a user vector where the topic scores are weighed by the user inputs
    user_vector = [average_topic_scores[index]*user_weights[index] for index in average_topic_scores.index]
    
    # The above input has to be reshaped such that it has 1 row and 'n' columns
    # This makes it compatible with the cosine similarity matrix
    user_vector = np.array(user_vector).reshape(1, -1)
    similarity_matrix = cosine_similarity(df_attractions, user_vector).flatten()
    
    # Getting the indices in a sorted order from lowest to highest 
    index_sort = np.argsort(similarity_matrix)

    # List of top 3 suggested attractions. The negative indexing reverses the order as the indices are sorted smallest to biggest.
    suggested_attractions = df_attractions.index[index_sort][-1:-4:-1].values.tolist()
    
    return suggested_attractions
