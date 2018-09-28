import os
import sys
import io
import re
import MeCab
import mojimoji
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

from config import PAD, UNK, WIKIQA_DIR

WAKATI = MeCab.Tagger('-Ochasen')
STOPWORDS = stopwords.words("english")


def get_logger(outdir):
    file_name = sys.argv[0]

    os.makedirs(outdir, exist_ok=True)

    logger = getLogger(__name__)
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler('{0}/{1}.log'.format(outdir, file_name), 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    return logger


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    embed_matrix = np.zeros((n + 2, d))  # 語彙数はPAD, UNKを足す
    vocab = [PAD, UNK]
    i = 2
    for line in fin:
        tokens = line.rstrip().split(' ')
        # 語彙リストに単語を追加
        w = tokens.pop(0)
        vocab.append(w)
        # embed_matrixにベクトルを追加
        v = np.asarray(tokens, dtype='float32')
        embed_matrix[i] = v
        i += 1
    return embed_matrix, vocab


def to_sep_space(txt):
    txt = mojimoji.zen_to_han(txt)  # 英数字は全て半角
    txt = re.sub(r'\d+', '0', txt)  # 連続した数字を0で置換
    parsed = WAKATI.parse(txt)
    sep_txt = [i.split('\t')[0] for i in parsed.split('\n') if i not in ['', 'EOS']]
    return ' '.join(sep_txt)


def get_sim_index(vec, vecs):
    sim = cosine_similarity(vec, vecs)
    sort_i = np.argsort(sim)[0][::-1]
    sim = np.sort(sim)[0][::-1]
    return sort_i, sim


def show_sim_faq(df, sim, n_top=3):
    index = 0
    for _, row in df.iterrows():
        print('-----------------------------')
        print('[{0}位] {1}: {2} ({3})'.format(index + 1, row['q_id'], row['q_txt'], sim[index]))
        print()
        print('{0}'.format(row['a_txt']))
        print('-----------------------------')
        index += 1
        if index >= n_top:
            break


"""
WikiQA
"""


def load_glove_vectors(fname, d):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    fin = [line for line in fin]
    n = len(fin)
    embed_matrix = np.zeros((n + 2, d))  # 語彙数はPAD, UNKを足す
    vocab = [PAD, UNK]
    i = 2
    for line in fin:
        tokens = line.rstrip().split(' ')
        # 語彙リストに単語を追加
        w = tokens.pop(0)
        vocab.append(w)
        # embed_matrixにベクトルを追加
        v = np.asarray(tokens, dtype='float32')
        embed_matrix[i] = v
        i += 1
    return embed_matrix, vocab


def load_text(fname):
    txt_lst = []
    with open(WIKIQA_DIR + '/' + fname, 'r') as f:
        for line in f.readlines()[1:]:
            sep_line = line.split('\t')
            txt_lst.append(sep_line[1])
            txt_lst.append(sep_line[5])
    return [trim(i) for i in txt_lst]


def trim(sentence):
    sentence = sentence.lower()
    _ignores = re.compile("[.,-/\"'>()&;:]")
    sep = ['' if w in STOPWORDS or _ignores.match(w) else w
           for w in sentence.split()]
    sep = [i for i in sep if i != '']
    return ' '.join(sep)


def load_wikiqa():
    txt = load_text('WikiQA-train.tsv')
    # txt += load_text('WikiQA-dev.tsv')
    # txt += load_text('WikiQA-test.tsv')
    txt = [i for i in txt if len(i.split()) > 0]
    txt = list(set(txt))
    return txt


"""
NNLM
"""


def execute(tensor):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        return sess.run(tensor)
