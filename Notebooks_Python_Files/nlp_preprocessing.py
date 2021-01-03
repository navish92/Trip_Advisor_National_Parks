"""
    Functions to help cleanup text as part of preprocessing in NLP, 
    before proceeding with the next steps.

    Author: Navish Agarwal
    Create Date: 11/05/2020
    Modified Date: 11/15/2020
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
import string
import spacy

sp_nlp = spacy.load('en', disable=['parser'])

def remove_stopwords(text, remove_words_list = stopwords.words('english')):
    """
    Removes stop works from the provided text, using the list provided in the calling function 
    or the default list of stop words obtained from NLTK.
    Args:
        text ([str]): sentence (review) from which stop words have to be removed
        remove_words_list ([list of strings], optional): Custom list of stop words to be used. Defaults to stopwords list from the NLTK corpus.

    Returns:
        [str]: processed text with stop words removed
    """
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    tokens = [token for token in tokens if token not in remove_words_list]
    return ' '.join(tokens)


def spacy_lemmatization(text):
    """
    Uses Spacy to lemmatize each token in the text provided

    Args:
        text ([str]): text string whose tokens will be lemmatized using Spacy

    Returns:
        tokens_lemma_remove_pron [str]: Return lemmatized version of the tokens present in the input 'text' argument
    """
    
    spacy_text = sp_nlp(text)
    tokens_lemma = [token.lemma_ for token in spacy_text]
    text_lemma = ' '.join(tokens_lemma)
    tokens_lemma_remove_pron = re.sub(r'-PRON-', '', text_lemma)
    
    return tokens_lemma_remove_pron

def spacy_pos_filtering(text, pos = [], ent_label = []):
    """
    Takes the text provided and KEEPS only the pos types specified in the calling function.
    Additionally, the entity types described in the argument are REMOVED from the text.

    Args:
        text (str): Text string whose tokens will be analyzed POS & Entity Labels and then processed on.
        pos (list, optional): List of POS types that should be kept. Defaults to empty list i.e. all POS types will be kept
        ent_label (list, optional): List of Named Entity types that should be removed. Defaults to empty list i.e. all entity types will be kept

    Returns:
        filtered_text [str]: Processed text
    """

    spacy_text = sp_nlp(text)
    
    # [print(token, token.text, token.pos_, token.ent_type_) for token in spacy_text]
    tokens = [token.text for token in spacy_text if ((token.pos_ in pos) and (token.ent_type_ not in ent_label))]
    
    filtered_text = ' '.join(tokens)
    
    return filtered_text


def cleaning(text):
    """
    Cleans the provided text by lowering all letters, removing urls, removing emails,
    substituing punctuations with spaces
    Args:
        text (str): Input string to be cleaned

    Returns:
        text (str): cleaned text
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

                
