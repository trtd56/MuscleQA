import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, ModelCheckpoint

from functions import get_logger, load_glove_vectors, load_wikiqa, get_sim_index, trim
from config import LOGDIR, WIKIQA_MODEL, GLOVE_MODEL, GLOVE_SIZE, WIKIQA_CORPUS, WIKIQA_DIR
from net import AutoEncoder
from return_corpus import ReutersMuscleCorpus

seq_size = 100
batch_size = 64
n_epoch = 20
latent_size = 512


class Infer():

    def __init__(self, encoder, corpus):
        self.encoder = encoder
        self.corpus = corpus

    def __call__(self, text):
        text_sp = ' '.join([trim(i) for i in text.split()])
        ids = self.corpus.doc2ids(text_sp)
        vec = self.corpus.embed_matrix[ids]
        vec = np.reshape(vec, (1, vec.shape[0], vec.shape[1]))
        feat = self.encoder.predict(vec)
        return feat[0]


def main():
    logger = get_logger(LOGDIR)
    logger.info('start')

    logger.info('1. Load WikiQA text')
    wikiqa_text = load_wikiqa()
    min_w = min([len(i.split()) for i in wikiqa_text])
    max_w = max([len(i.split()) for i in wikiqa_text])
    logger.info('{0} sentence, {1}-{2} words'.format(len(wikiqa_text), min_w, max_w))

    logger.info('2. Load GloVe embeddings.')
    embed_matrix, vocab = load_glove_vectors(GLOVE_MODEL, d=GLOVE_SIZE)
    logger.info('embedding shape is {}'.format(embed_matrix.shape))

    logger.info('3. Prepare the corpus.')
    corpus = ReutersMuscleCorpus()
    corpus.build(embed_matrix, vocab, seq_size)
    corpus.documents = wikiqa_text
    corpus.save(WIKIQA_CORPUS)

    logger.info('4. Make autoencoder model.')
    ae = AutoEncoder(seq_size=seq_size, embed_size=embed_matrix.shape[1], latent_size=latent_size)
    ae.build()

    logger.info('5. Train model.')
    ae.model.compile(optimizer="adam", loss="mse")
    train_iter = corpus.batch_iter(batch_size)
    train_step = corpus.get_step_count(batch_size)

    ae.model.fit_generator(
        train_iter,
        train_step,
        epochs=n_epoch,
        # validation_data=train_iter,
        # validation_steps=train_step,
        callbacks=[
            TensorBoard(log_dir=LOGDIR),
            ModelCheckpoint(filepath=WIKIQA_MODEL, save_best_only=True)
        ]
    )

    logger.info('6. Load the encoder.')
    encoder = ae.get_encoder()

    logger.info('7. Set the infer model.')
    infer = Infer(encoder, corpus)

    logger.info('8. Evaluate the model.')
    qa_df = pd.read_csv(WIKIQA_DIR + '/WikiQA-test.tsv', sep='\t')
    maps = []
    mrrs = []
    for q_id in qa_df['QuestionID'].unique():
        df = qa_df[qa_df['QuestionID'] == q_id]
        if 1 not in df['Label'].unique():
            logger.debug('{0}: not answer'.format(q_id))
            continue
        q_doc = df['Question'].iloc[0].lower()
        q_vec = infer(q_doc)
        a_docs = df['Sentence'].map(lambda x: x.lower()).tolist()
        a_vecs = [infer(d) for d in a_docs]
        sort_i, sim = get_sim_index([q_vec], a_vecs)
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
