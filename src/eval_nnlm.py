import tensorflow_hub as hub
import pandas as pd
from functions import get_logger, execute, get_sim_index
from config import LOGDIR, WIKIQA_DIR, EN_NNLM_MODEL


def main():
    logger = get_logger(LOGDIR)
    logger.info('start')

    logger.info('load NN Language Model')
    embed = hub.Module(EN_NNLM_MODEL)

    qa_df = pd.read_csv(WIKIQA_DIR + '/WikiQA-test.tsv', sep='\t')
    maps = []
    mrrs = []
    for q_id in qa_df['QuestionID'].unique():
        df = qa_df[qa_df['QuestionID'] == q_id]
        if 1 not in df['Label'].unique():
            logger.debug('{0}: not answer'.format(q_id))
            continue
        q_doc = df['Question'].iloc[0].lower()
        embeddings = embed([q_doc])
        q_vec = execute(embeddings)
        a_docs = df['Sentence'].map(lambda x: x.lower()).tolist()
        embeddings = embed(a_docs)
        a_vecs = execute(embeddings)
        sort_i, sim = get_sim_index(q_vec, a_vecs)
        labels = [i for i, v in enumerate(df['Label']) if v == 1]
        rank = [i + 1 for i, v in enumerate(sort_i) if v in labels]
        _mrr = 1 / rank[0]
        _map = sum([1 / i for i in rank]) / len(rank)
        maps.append(_map)
        mrrs.append(_mrr)
        logger.info('{0}: MAP {1}, MRR {2}'.format(q_id, _map, _mrr))
    map_avg = sum(maps) / len(maps)
    mrr_avg = sum(mrrs) / len(mrrs)
    logger.info('MAP AVG {0} / MRR AVG {1}'.format(map_avg, mrr_avg))

    logger.info('end')


if __name__ == '__main__':
    main()
