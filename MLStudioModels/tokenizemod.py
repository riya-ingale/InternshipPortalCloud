# downloading resources and importing libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')

import pandas as pd
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import re
def tokenize(sentences):
    '''
    tokenizes a bunch of sentences after normalizing them and returns stemmed tokens.
    
    INPUT:
    sentences - a paragraph that need to be tokenized
    
    OUTPUT:
    tokens - list of stemmed tokens
    
    '''
    # normalizing, tokenizing, lemmatizing 
    sentences = re.sub('\W',' ',sentences) 
    sentences = re.sub('[0-9]',' ',sentences)

    tokens = word_tokenize(sentences)
    tokens = [i.strip() for i in tokens]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(i) for i in tokens]
    return tokens