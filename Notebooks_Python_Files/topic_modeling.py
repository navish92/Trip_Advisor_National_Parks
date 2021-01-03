"""
    Python module to perform topic modelling using various algorithms, as implemented in the sklearn package
    along with other functions to display the topics in an easy-to-use interpret manner.

    Author: Navish Agarwal
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity

def lda_topic_modeling(word_matrix, vocab, n = 5):
    """
        Perform LDA topic modelling using sklearn on the provided doc-word vector.
    Args:
        word_matrix ([Numpy Matrix]): TF-IDF or Word Count Frequency Vector
        vocab ([List of strings]): Contains all the words that make up the entire vocabulary. Equals # of columns in the above vector
        n (int, optional): Number of topics to be generated. Defaults to 5.

    Returns:
        Returns a tuple containing 4 elements
        lda [Sklearn LDA Model]: The fitted LDA model
        lda.bound_ [float]: LDA score
        topic_matrix [Pandas Dataframe]: Dataframe containing topics scores of every document (columns=topics, rows=documents)
        word_matrix [Pandas Dataframe]: Dataframe containing topics scores of "every" word in the corpus (columns=topics, rows=words)
    """
    lda = LatentDirichletAllocation(n_components=n, random_state=0, max_iter = 100, n_jobs = -1, verbose = 1)
    lda.fit(word_matrix)
    topic_matrix = pd.DataFrame(lda.transform(word_matrix)).add_prefix("topic_")
    word_matrix = pd.DataFrame(lda.components_, \
        columns = vocab).T.add_prefix('topic_')

    return lda, lda.bound_, topic_matrix, word_matrix

def nmf_topic_modeling (word_matrix, vocab, n = 5):
    """
        Perform NMF topic modelling using sklearn on the provided doc-word vector.
    Args:
        word_matrix ([Numpy Matrix]): TF-IDF or Word Count Frequency Vector
        vocab ([List of strings]): Contains all the words that make up the entire vocabulary. Equals # of columns in the above vector
        n (int, optional): Number of topics to be generated. Defaults to 5.

    Returns:
        Returns a tuple containing 4 elements
        nmf [Sklearn NMF Model]: The fitted LDA model
        nmf.reconstruction_err_ [float]: NMF score
        topic_matrix [Pandas Dataframe]: Dataframe containing topics scores of every document (columns=topics, rows=documents)
        word_matrix [Pandas Dataframe]: Dataframe containing topics scores of "every" word in the corpus (columns=topics, rows=words)
    """

    nmf = NMF(n_components = n, max_iter = 1000)
    nmf.fit(word_matrix)

    topic_matrix = pd.DataFrame(nmf.transform(word_matrix)).add_prefix("topic_")
    word_matrix = pd.DataFrame(nmf.components_, \
        columns = vocab).T.add_prefix('topic_')

    return nmf, nmf.reconstruction_err_, topic_matrix, word_matrix

def top_reviews(topic_matrix_df, topic = 0, n_reviews = 5):
    """
        Function to return the top scoring documents under the provided topic #.
    Args:
        topic_matrix_df ([Pandas Dataframe]): Contains the documents (reviews) as rows, topics as columns and topic scores as values.
        topic (int, optional): The topic # whose top documents (reviews) should be returned. Defaults to 0.
        n_reviews (int, optional): Number of documents (reviews) to be returned. Defaults to 5.

    Returns:
        [Numpy array of strings]: Top scoring documents (reviews) under the specified topic, arranged in descending order of score
    """

    return (topic_matrix_df
            .sort_values(by=f'topic_{topic}', ascending=False)
            .head(n_reviews)['raw_review']
            .values)

def top_words(word_topic_matrix_df, topic = 0, n_words = 5):
    """
        Function to return the top scoring words under the provided topic #.
    Args:
        word_topic_matrix_df ([Pandas Dataframe]): Contains the vocabulary words as rows, topics as columns and topic scores as values.
        topic (int, optional): The topic # whose top scoring words should be returned. Defaults to 0.
        n_reviews (int, optional): Number of words to be returned. Defaults to 5.

    Returns:
        [Pandas Series]: Top scoring words under the specified topic, arranged in descending order of score
    """

    return (word_topic_matrix_df
            .sort_values(by=f'topic_{topic}', ascending=False)
            .head(n_words))[f'topic_{topic}']