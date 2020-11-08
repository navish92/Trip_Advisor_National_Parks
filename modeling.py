import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

def nmf_topic_modeling (word_matrix, vocab, n = 5):

    nmf = NMF(n_components = n)
    nmf.fit(word_matrix)

    topic_matrix = pd.DataFrame(nmf.transform(word_matrix)).add_prefix("topic_")
    word_matrix = pd.DataFrame(nmf.components_, \
        columns = vocab).T.add_prefix('topic_')

    return nmf, nmf.reconstruction_err_, topic_matrix, word_matrix

def top_reviews(topic_matrix_df, topic, n_reviews):
    return (topic_matrix_df
            .sort_values(by=topic, ascending=False)
            .head(n_reviews)['raw_review']
            .values)

def top_words(word_topic_matrix_df, topic, n_words):
    return (word_topic_matrix_df
            .sort_values(by=topic, ascending=False)
            .head(n_words))[topic]

def lsa_topic_modeling(word_matrix, vocab, n = 5):

    lsa = TruncatedSVD(10)
    lsa.fit(word_matrix)
    topic_matrix = pd.DataFrame(lsa.transform(word_matrix)).add_prefix('topic_')
    # lsa.explained_variance_ratio_

    word_matrix = pd.DataFrame(lsa.components_, columns = vocab).T.add_prefix('topic_')

    return lsa, lsa.explained_variance_, topic_matrix, word_matrix