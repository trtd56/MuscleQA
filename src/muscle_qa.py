import six
import pandas as pd
import numpy as np

from functions import get_logger, to_sep_space, get_sim_index, show_sim_faq
from config import LOGDIR, MUSCLE_QA, MUSCLE_CORPUS, MUSCLE_MODEL
from net import AutoEncoder
from return_corpus import ReutersMuscleCorpus


class Infer():

    def __init__(self, encoder, corpus):
        self.encoder = encoder
        self.corpus = corpus

    def __call__(self, text):
        text_sp = to_sep_space(text)
        ids = self.corpus.doc2ids(text_sp)
        vec = self.corpus.embed_matrix[ids]
        vec = np.reshape(vec, (1, vec.shape[0], vec.shape[1]))
        feat = self.encoder.predict(vec)
        return feat[0]


def main():
    logger = get_logger(LOGDIR)
    logger.info('start')

    logger.info('1. Load the trained model.')
    ae = AutoEncoder.load(MUSCLE_MODEL)
    encoder = ae.get_encoder()

    logger.info('2. Load the corpus.')
    corpus = ReutersMuscleCorpus.load(MUSCLE_CORPUS)

    logger.info('3. Set the infer model.')
    infer = Infer(encoder, corpus)

    qa_df = pd.read_csv(MUSCLE_QA)
    q_txts = qa_df['q_txt'].tolist()
    vecs = np.array([infer(d) for d in q_txts])

    # 超回復とは
    # 夏までに痩せたい
    # 睡眠時間はどのくらいが良いですか？
    while True:
        text = six.moves.input('>> ')
        if text == '':
            break
        vec = infer(text)
        sort_i, sim = get_sim_index([vec], vecs)

        df = qa_df.loc[sort_i]
        show_sim_faq(df, sim)

    logger.info('end')


if __name__ == '__main__':
    main()
