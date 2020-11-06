"""
    Functions to help cleanup text as part of preprocessing in NLP, 
    before proceeding with the next steps.

    Author: Navish Agarwal
    Create Date: 11/05/2020
    Modified Date:
"""

import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize, RegexpTokenizer, regexp_tokenize, WhitespaceTokenizer
from nltk.tokenize import MWETokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import nlp_cleaning
import string
import spacy

sp_nlp = spacy.load('en', disable=[ 'parser', 'ner'])

def remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    tokens = [token for token in tokens if token not in stopwords_list]
    return ' '.join(tokens)


def spacy_lemmatization(text):
    spacy_text = sp_nlp(text)
    tokens_lemma = [token.lemma_ for token in spacy_text ]
    text_lemma = ' '.join(tokens_lemma)
    tokens_lemma_remove_pron = re.sub(r'-PRON-', '', text_lemma)
    
    return tokens_lemma_remove_pron

def cleaning(text):
    """
    Cleans the provided text by lowering all letters, removing urls, removing emails,
    substituing punctuations with spaces
    Args:
        text (str): Input string to be cleaned

    Returns:
        text: cleaned text
    """

    text = text.lower()
    text = re.sub('http\S+', '' , text)
    text = re.sub('\S*@\S+', '', text)

    text = re.sub(r'[<.*?>;\-!()/,:&—\\]+', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)

    # Example statement for removing words with numbers in them
    # re.sub('\w*\d\w*', ' ', clean_text)

    # Alternate version of removing all punctuations only
    # re.sub('[%s]' % re.escape(string.punctuation), ' ', my_text)

    return text

def remove_stop_words(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
# def cleaning(text, regexp = r'[^A-Za-z\s]', sub = ''):
#     """
#     Clean the provided text using the regexp

#     Args:
#         text (str): Text string to be cleaned
#         regexp (regexp, optional):  Defaults to r'[^A-Za-z\s]'.

#     Returns:
#         text: cleaned text based on the regex provided
#     """

#     return re.sub(regexp, '', text)

                
