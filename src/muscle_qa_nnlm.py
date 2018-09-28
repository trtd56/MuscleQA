import six
import tensorflow_hub as hub
import pandas as pd
from functions import get_logger, execute, to_sep_space, get_sim_index, show_sim_faq
from config import LOGDIR, JA_NNLM_MODEL, MUSCLE_QA


def main():
    logger = get_logger(LOGDIR)
    logger.info('start')

    logger.info('load faq data')
    qa_df = pd.read_csv(MUSCLE_QA)
    q_txt = qa_df['q_txt'].tolist()
    sep_q_txt = [to_sep_space(i) for i in q_txt]

    logger.info('load NN Language Model')
    embed = hub.Module(JA_NNLM_MODEL)
    embeddings = embed(sep_q_txt)

    logger.info('to vectors')
    vecs = execute(embeddings)
    logger.info('vector shape: {}'.format(vecs.shape))

    while True:
        text = six.moves.input('>> ')
        if text == '':
            break
        sep_input = to_sep_space(text)
        embeddings = embed([sep_input])
        vec = execute(embeddings)

        sort_i, sim = get_sim_index(vec, vecs)
        df = qa_df.loc[sort_i]
        show_sim_faq(df, sim)

    logger.info('end')


if __name__ == '__main__':
    main()
