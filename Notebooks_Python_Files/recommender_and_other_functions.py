"""
Other functions

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV,  StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import roc_auc_score, roc_curve, fbeta_score, make_scorer, classification_report, confusion_matrix
from sklearn.metrics import log_loss, precision_score, recall_score, accuracy_score

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

def classification_common_model (X_train_val, y_train_val, model, oversampler = RandomOverSampler(), scaler = StandardScaler(), \
                          threshold = 0.5, return_type = None):
    
    """
    Function does a kfold=5 split, scales & oversamples the data if requested, 
    fits the transformed data on the model specified & prints the average of the 5-Fold along various metrics.
    
    Inputs:
    X_train_val: Features to be used
    Y_train_val: Target variable
    model: Classification model to be used
    oversampler: Oversampling technique
    scalar: Scaling method to be used
    threshold: probability point at which prediction should be put into a specific class
    return_type: If the function should return a dictionary with all values; the scores only get printed.
    
    Outputs:
    Using data from the last of the 5 KFolds, returns a dictionary containing the fitted model, data used for validation,
    and prediction probabilities. Also returns the mean train, validaton, precision, recall, roc_auc and logloss scores
    along with the classification report.
    
    """
    
    #use stratified kfold to splice up train-val into train and val
    
    skfold = StratifiedKFold(n_splits=5, shuffle = True, random_state=42)
    skfold.get_n_splits(X_train_val, y_train_val)
    
    #create a list to store our scores
    train_scores= []
    val_scores= []
    precision_scores= []
    recall_scores= []
    roc_auc_scores = []
    logloss_scores = []
    
    #fit and score model on each fold
    for train, val in skfold.split(X_train_val, y_train_val):
        
        #set up train and val for each fold
        X_train, X_val = X_train_val.iloc[train], X_train_val.iloc[val]
        y_train, y_val = y_train_val.iloc[train], y_train_val.iloc[val]
        
        #oversample train data
        if oversampler:
            X_train, y_train = oversampler.fit_sample(X_train, y_train)

        #Scale data
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
                
        #fit model
        model.fit(X_train, y_train)
        
        #make prediction using y-val
        #try-except used incase model.predict_proba throws an error
        try:
            y_pred_proba = model.predict_proba(X_val)[:,1]
            y_pred = y_pred_proba > threshold
        except:
            y_pred_proba = np.zeros(len(y_val))
            y_pred = model.predict(X_val)
        
        #append scores onto list
        train_scores.append(accuracy_score(y_train,  model.predict(X_train)))
        val_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred, average='binary'))
        recall_scores.append(recall_score(y_val, y_pred, average='binary'))
        roc_auc_scores.append(roc_auc_score(y_val, y_pred_proba))
        logloss_scores.append(log_loss(y_val, y_pred_proba))
        
    #find the means for our scores
    mean_train = np.mean(train_scores)
    mean_val = np.mean(val_scores)
    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)
    roc_auc = np.mean(roc_auc_scores)
    logloss = np.mean(logloss_scores)
    
    #print our mean accuracy score, our train/test ratio precision, and recall
    print(f'Scores fit on {model}')
    print(f'Accuracy: {mean_val:.2f}')
    print(f'Train/Test ratio: {(mean_train)/(mean_val):.2f}')
    
    print(f'Precision: {precision:.2f}')
    print(f'RECALL: {recall:.2f}')
    print(f'Log Loss: {logloss:.2f}')
    print(f'ROC AUC: {roc_auc:.2f}')
    print(classification_report(y_val,y_pred))
    print('-----')      
    
    if return_type:
        return {'model':model, 'X_val':X_val, 'y_pred_proba':y_pred_proba, 'mean_train':mean_train, \
                'mean_val': mean_val, 'precision':precision, 'recall':recall, 'roc_auc':roc_auc, \
                'logloss':logloss, 'classification_report': classification_report(y_val,y_pred) }
    else:
        return None
