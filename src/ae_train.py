from keras.callbacks import TensorBoard, ModelCheckpoint

from net import AutoEncoder
from return_corpus import ReutersMuscleCorpus
from functions import get_logger, load_vectors
from config import LOGDIR, JAWIKI_MODEL, MUSCLE_CORPUS, MUSCLE_MODEL

seq_size = 15
batch_size = 4
n_epoch = 20
latent_size = 512


def main():
    logger = get_logger(LOGDIR)
    logger.info('start')

    logger.info('1. Load Japanese word2vec embeddings.')
    embed_matrix, vocab = load_vectors(JAWIKI_MODEL)
    logger.info('embedding shape is {}'.format(embed_matrix.shape))

    logger.info('2. Prepare the corpus.')
    corpus = ReutersMuscleCorpus()
    corpus.build(embed_matrix, vocab, seq_size)
    corpus.save(MUSCLE_CORPUS)

    logger.info('3. Make autoencoder model.')
    ae = AutoEncoder(seq_size=seq_size, embed_size=embed_matrix.shape[1], latent_size=latent_size)
    ae.build()

    logger.info('4. Train model.')
    ae.model.compile(optimizer="adam", loss="mse")
    train_iter = corpus.batch_iter(batch_size)
    train_step = corpus.get_step_count(batch_size)
    valid_iter = corpus.batch_iter(batch_size)
    valid_step = corpus.get_step_count(batch_size)

    ae.model.fit_generator(
        train_iter,
        train_step,
        epochs=n_epoch,
        validation_data=valid_iter,
        validation_steps=valid_step,
        callbacks=[
            TensorBoard(log_dir=LOGDIR),
            ModelCheckpoint(filepath=MUSCLE_MODEL, save_best_only=True)
        ]
    )

    logger.info('end')


if __name__ == '__main__':
    main()
