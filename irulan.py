import numpy as np
import string
import sklearn
from corextopic import corextopic as ct
import pandas as pd
import sklearn.feature_extraction
from nltk import FreqDist

def clean_split_mins(data, secs):

    # create documents by second to input into a topic model
    # input data in the form of a dataframe [text, date]
    # output a list split into documents of length [secs]
    if len(data) == 0:
        return [], []
        
    # initialise
    date0 = data["date"][0]
    txt_list = [""]
    date_list = [date0]
    i = 0

    # make into documents of length 'secs'
    for text, date in zip(data["text"], data["date"]):

        if date - date0 > secs:

            txt_list.append(str(text) + " ")
            i += 1
            date0 = date
            date_list.append(date0)

        else:

            txt_list[i] += str(text) + " "

    txt_list = [line.replace("'", "").lower() for line in txt_list]

    for c in string.punctuation:

        txt_list = [line.replace(c, " ") for line in txt_list]


    return txt_list, date_list



def moving_average(arr, a) :

    # input an array and return a moving average 

    ret = np.cumsum(arr, dtype=float)
    ret[a:] = ret[a:] - ret[:-a]
    return ret[a - 1:] / a



# define function to create a topic model
def tm(text, anchors = None, num_topics = 40, seed = 0):

    # initialise vectorizer 
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(max_features = 10000, binary = True) 

    # get sparse matrix of number of times each word appears in each document
    matrix = vectorizer.fit_transform(text)

    # get the list of words
    words = sorted(vectorizer.vocabulary_.keys())

    # Train the CorEx topic model
    topic_model = ct.Corex(n_hidden = num_topics, words = words, seed = seed)  # Define the number of latent (hidden) topics to use.
    topic_model.fit(matrix, words = words, anchors = anchors, anchor_strength = 10)

    # get words with highest mutual information with the topic - not words with most probability

    print("Total correlation:", topic_model.tc)
    
    return vectorizer, topic_model



def create_word2vec(text):

    from time import time
    import re
    from gensim.models.phrases import Phrases
    import multiprocessing
    from gensim.models import Word2Vec
    from collections import defaultdict

    txt = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in text)

    t = time()

    df_clean = pd.DataFrame({'clean': txt})
    df_clean = df_clean.dropna().drop_duplicates()

    sent = [row.split() for row in df_clean['clean']]

    phrases = Phrases(sent, min_count=1, progress_per=10000)

    sentences = phrases[sent]

    print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

    word_freq = defaultdict(int)
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1
    len(word_freq)

    sorted(word_freq, key=word_freq.get, reverse=True)[:10]

    cores = multiprocessing.cpu_count() # number of cores for training

    print("Starting training")

    w2v_model = Word2Vec(min_count=10,
                        window=2,
                        vector_size=300,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=cores-1)

    t = time()

    w2v_model.build_vocab(sentences, progress_per=10000)

    w2v_model.init_sims(replace=True)

    print('Time to build model: {} mins'.format(round((time() - t) / 60, 2)))

    return w2v_model



def sentiment(word, model,
                pos_list = ["good", "excellent", "correct", "best", "happy", "positive", "fortunate"], 
                neg_list = ["bad", "terrible", "wrong", "worst", "disappointed", "negative", "unfortunate"]):
    
    pos_total = 0
    neg_total = 0
    
    # convert words to vectors
    word = model.wv.__getitem__(word)
    pos_list = [model.wv.__getitem__(i) for i in pos_list]
    neg_list = [model.wv.__getitem__(i) for i in neg_list]
    
    # sum over the cosine similarities of positive words
    for v in pos_list:
        pos = np.dot(word, v)
        pos /= (np.linalg.norm(word) * np.linalg.norm(v))
        
        pos_total += pos
        
    # sum over the cosine similarities of negative words
    for u in neg_list:
        
        neg = np.dot(word, u)
        neg /= (np.linalg.norm(word) * np.linalg.norm(u))
        
        neg_total += neg
        
    # difference in positive and negative
    s = pos_total - neg_total
    
    return s

pos_list = ["good", "excellent", "correct", "best", "happy", "positive", "fortunate"]

neg_list = ["bad", "terrible", "wrong", "worst", "disappointed", "negative", "unfortunate"]

def glove_sentiment(word, model, pos_list = pos_list, neg_list = neg_list):
    
    if word not in model.keys():
        return 0
    
    else:
        pos_total = 0
        neg_total = 0

        # convert words to vectors
        word = model[word]
        pos_list = [model[i] for i in pos_list]
        neg_list = [model[i] for i in neg_list]

        # sum over the cosine similarities of positive words
        for v in pos_list:
            pos = np.dot(word, v)
            pos /= (np.linalg.norm(word) * np.linalg.norm(v))

            pos_total += pos

        # sum over the cosine similarities of negative words
        for u in neg_list:

            neg = np.dot(word, u)
            neg /= (np.linalg.norm(word) * np.linalg.norm(u))

            neg_total += neg

        # difference in positive and negative
        s = pos_total - neg_total

        return s
    

def doc_sentiment(doc, lexicon):

    s = 0
    k = 0

    for w in doc.split():

        if w in lexicon.keys():

            s += lexicon[w]
            k += 1

    return s/np.max([1, k])


def get_counts(text1, text2):

    text1 = " ".join(text1)
    text2 = " ".join(text2)

    counts1 = dict() # initialise dictionary
    # get frequency counts
    counts1 = FreqDist(text1.split())


    counts2 = dict() # initialise dictionary
    # get frequency counts
    counts2 = FreqDist(text2.split())

    return counts1, counts2