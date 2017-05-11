# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import json
import codecs
import nltk
import pymorphy2

from scipy.linalg import eigvals
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer

from sklearn.manifold import TSNE
from re import sub

import warnings

# Suppress warnings from pandas library
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas", lineno=570)


PATH_DIR = 'src'
FILENAME = 'LavrentPaper'
FILENAME_parse = 'LavrentPaperParsed_noun'

data_dict = {}


def collect_text():

    cur_dir = os.path.abspath(os.path.curdir)

    file_path = os.path.join(cur_dir, PATH_DIR, FILENAME)

    log = codecs.open(file_path, "r", "utf-8")

    text = log.read()

    log.close()

    morph = pymorphy2.MorphAnalyzer(lang='ru')

    sentences = nltk.sent_tokenize(text)

    sent_tokens = []

    for sent in sentences:

        words = nltk.word_tokenize(sent)

        word_list = []

        for word in words:

            cword = sub(u'”|“|«|»|«|\xab', '', word)

            if cword:

                p = morph.parse(cword)[0]

                if p.tag.POS and p.tag.POS not in ["CONJ", "PREP", "NPRO"]:

                    if p.tag.POS == "NOUN":
                        word_list.append(p.normal_form)

        sent_tokens.append(word_list)

    cur_dir = os.path.abspath(os.path.curdir)

    file_path = os.path.join(cur_dir, PATH_DIR, FILENAME_parse)

    text_dump = codecs.open(file_path, "w", "utf-8")

    for sent in sent_tokens:

        text_dump.write(json.dumps(sent, encoding="utf-8") + "\n")

    text_dump.close()


def extract_text():

    cur_dir = os.path.abspath(os.path.curdir)

    file_path = os.path.join(cur_dir, PATH_DIR, FILENAME_parse)

    text_dump = codecs.open(file_path, "r", "utf-8")

    sent_tokens = []

    for sent in text_dump.readlines():

        sent_val = json.loads(sent.strip(),  encoding="utf-8")

        sent_tokens.append(" ".join(sent_val))

    text_dump.close()

    return sent_tokens


if __name__ == '__main__':

    #collect_text()

    sent_tokens = extract_text()

    vectorizer = CountVectorizer(min_df=2)

    dtm = vectorizer.fit_transform(sent_tokens)

    #print np.asarray(np.asmatrix(dtm.T) * np.asmatrix(dtm))

    lsa = TruncatedSVD(500, algorithm='randomized', n_iter=10)

    dtm_lsa = lsa.fit_transform(dtm.T)

    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

    #print eigvals(np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T))

    similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)

    df = pd.DataFrame(similarity, index=vectorizer.get_feature_names(), columns=vectorizer.get_feature_names())

    print df[u'глеб '].sort_values(ascending=False).head(50)

    print u'борис', u'глеб', df[u'борис'][u'глеб']
    print u'борис', u'князь', df[u'борис'][u'князь']
    print u'князь', u'глеб', df[u'князь'][u'глеб']
    print u'аскольд', u'дир', df[u'аскольд'][u'дир']
    print u'ольга', u'игорь', df[u'ольга'][u'игорь']
    print u'ольга', u'святослав', df[u'ольга'][u'святослав']
    print u'ольга', u'владимир', df[u'ольга'][u'владимир']
    print u'ольга', u'древлянин', df[u'ольга'][u'древлянин']
    print u'владимир', u'древлянин', df[u'владимир'][u'древлянин']
    print u'святослав', u'древлянин', df[u'святослав'][u'древлянин']
    print u'игорь', u'древлянин', df[u'игорь'][u'древлянин']
    print u'владимир', u'ярополк', df[u'владимир'][u'ярополк']
    print u'владимир', u'бог', df[u'владимир'][u'бог']
    print u'владимир', u'христианин', df[u'владимир'][u'христианин']

    '''
    word_coords = np.zeros((len(model.vocab), word_dim))

    word_list = []

    for _ord, word in enumerate(model.vocab):

        word_list.append(word)

        word_coords[_ord, :] = model[word]

    model_tsne = TSNE(n_components=3, random_state=0)
    np.set_printoptions(suppress=True)

    fit_transform_data = model_tsne.fit_transform(word_coords)

    cur_path = '/home/eugene/PycharmProjects/History/src/coord_data'

    log = codecs.open(cur_path, "w", "utf-8")

    for _ord, line in enumerate(fit_transform_data):

        line_coord = ",".join([word_list[_ord]] + [str(item) for item in line.tolist()])

        log.write(line_coord + "\n")

    log.close()
    '''
