# coding=utf-8
import pandas as pd
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from pprint import pprint
np.random.seed(2018)

path = 'abcnews-date-text.csv'


def load_data_for_testing(path):
    data = pd.read_csv(path, error_bad_lines=False);
    data_text = data[['headline_text']]
    data_text['index'] = data_text.index
    documents = data_text
    print(len(documents))
    print(documents[:5])
    return documents


def lemmatize_stemming(text):       # lemmatization
    stemmer = PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):       # preprocessing
    """
    Text preprocessing
    1) Remove words that have fewer than 3 characters, 2) Remove stopwords, 3) Lemmatization, 4) Stemming
    """
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    # if you want to have a look at a document after preprocessing, uncomment this line
    preview_document(result)

    return result


def preview_document(documents):
    doc_sample = documents[documents['index'] == 4310].values[0][0]
    print('original document: ')
    words = []
    for word in doc_sample.split(' '):
        words.append(word)
    print(words)

    print('\n\n tokenized and lemmatized document: ')
    print(preprocess(doc_sample))

    return


def gensim_doc2bow(dictionary, documents):
    """
    For each document create a dictionary reporting how many words and how many times those words appear.
    :return: bow_corpus
    """
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
    print(bow_corpus[4310])

    bow_doc_4310 = bow_corpus[4310]
    for i in range(len(bow_doc_4310)):
        print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0],
                                                         dictionary[bow_doc_4310[i][0]], bow_doc_4310[i][1]))
    return bow_corpus


def bow_model(documents):
    """ BOW model on the dataset"""

    dictionary = gensim.corpora.Dictionary(documents)  # contains the number of times a word appears in the training set

    # Filter tokens that appear in less than 15 documents and more than 0.5 documents (fraction of total corpus size).
    # Keep only the first 100000 most frequent tokens.
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    bow_corpus = gensim_doc2bow(dictionary, documents)

    return bow_corpus, dictionary


def tfidf(bow_corpus):
    """ TF-IDF model on the dataset"""

    tfidf = models.TfidfModel(bow_corpus)       # Create tf-idf model
    tfidf_corpus = tfidf[bow_corpus]        # apply transformation to the entire corpus

    # preview TF-IDF scores for the first document
    for doc in tfidf_corpus:
        pprint(doc)
        break

    return tfidf_corpus


def print_results(lda_model):
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWord: {}'.format(idx, topic))
    return


def lda_with_bow(bow_corpus, dictionary):
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    print_results(lda_model)
    return


def lda_with_tfidf(corpus_tfidf, dictionary):
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
    print_results(lda_model_tfidf)
    return


def main():

    # load data
    documents = load_data_for_testing(path)

    # preprocess the headline text, saving the results as ‘processed_docs’
    processed_docs = documents['headline_text'].map(preprocess)
    print(processed_docs[:10])

    # BOW corpus
    bow_corpus, dictionary = bow_model(processed_docs)

    # lda model with BOW corpus
    lda_with_bow(bow_corpus, dictionary)

    # tfidf corpus
    tfidf_corpus = tfidf(bow_corpus)

    # lda model with t-idf corpus
    lda_with_tfidf(tfidf_corpus, dictionary)

