# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import json
import codecs
import nltk
import pymorphy2

from sklearn.manifold import TSNE
from re import sub
from gensim.models import word2vec

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

        sent_tokens.append(json.loads(sent.strip(),  encoding="utf-8"))

    text_dump.close()

    return sent_tokens


if __name__ == '__main__':

    #collect_text()

    sent_tokens = extract_text()

    #df = pd.DataFrame({"word": data_dict.keys(), "freq": data_dict.values()})

    #df = df.sort_values(["freq"], ascending=False)

    #df.to_csv('freq_dict', encoding="utf-8", sep="\t", columns=["word", "freq"])

    word_dim = 30

    model = word2vec.Word2Vec(iter=50, min_count=2, window=10, size=word_dim)

    model.build_vocab(sent_tokens)

    #print model.similarity(u'аскольд', u'дир')

    print u'ольга', u'игорь', model.similarity(u'ольга', u'игорь')
    print u'ольга', u'святослав', model.similarity(u'ольга', u'святослав')
    print u'ольга', u'владимир', model.similarity(u'ольга', u'владимир')
    print u'ольга', u'древлянин', model.similarity(u'ольга', u'древлянин')
    print u'владимир', u'древлянин', model.similarity(u'владимир', u'древлянин')
    print u'святослав', u'древлянин', model.similarity(u'святослав', u'древлянин')
    print u'игорь', u'древлянин', model.similarity(u'игорь', u'древлянин')
    print u'владимир', u'ярополк', model.similarity(u'владимир', u'ярополк')

    print "$$$$$$"
    print u'киев', u'ярослав', model.similarity(u'киев', u'ярослав')
    print u'киев', u'собор', model.similarity(u'киев', u'собор')
    print u'киев', u'андрей', model.similarity(u'киев', u'андрей')
    print u'киев', u'израиль', model.similarity(u'киев', u'израиль')
    print u'киев', u'днепр', model.similarity(u'киев', u'днепр')
    print u'киев', u'новгород', model.similarity(u'киев', u'новгород')

    print "$$$$$$"
    print u'киев', u'глеб', model.similarity(u'киев', u'глеб')
    print u'киев', u'борис', model.similarity(u'киев', u'борис')

    print "$$$$$$"

    lval = model.most_similar([u'киев'], topn=50)

    for item in lval:
        print item[0], model.similarity(u'киев', item[0])

    print("###########")

    lval = model.most_similar([u'ярослав'], topn=50)

    for item in lval:
        print item[0], model.similarity(u'ярослав', item[0])

    print("###########")
    print(u'владимир')

    lval = model.most_similar([u'владимир'], topn=50)

    for item in lval:
        print item[0], model.similarity(u'владимир', item[0])

    print("###########")

    print(u'христианин')

    print("###########")

    print model.similarity(u'кровопролитие', u'церковь')

    print("###########")

    print(u'киев')

    lval = model.most_similar([u'киев'], topn=25)

    for item in lval:
        print item[0], model.similarity(u'киев', item[0])

    print("###########")

    print(u'новгород')

    lval = model.most_similar([u'новгород'], topn=25)

    for item in lval:
        print item[0], model.similarity(u'новгород', item[0])

    #lval = model.most_similar(positive=[u'ольга'],
                              #negative=[u'игорь'])

    model.save('/home/eugene/lavr_model_noun.bin')

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
