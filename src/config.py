PAD = 'PAD'
UNK = 'UNK'
LOGDIR = './result'

JAWIKI_MODEL = '../data/entity_vector/entity_vector.model.txt'
MUSCLE_TEXT = '../data/muscle_text.csv'
MUSCLE_MODEL = LOGDIR + '/autoencoder.h5'
MUSCLE_CORPUS = LOGDIR + '/muscle_corpus.pkl'
MUSCLE_QA = '../data/muscle_qa.csv'

GLOVE_MODEL = '../data/glove.6B/glove.6B.300d.txt'
GLOVE_SIZE = 300
WIKIQA_DIR = '../data/WikiQACorpus/'
WIKIQA_MODEL = LOGDIR + '/wikiqa.h5'
WIKIQA_CORPUS = LOGDIR + '/wikiqa_corpus.pkl'

EN_NNLM_MODEL = 'https://tfhub.dev/google/nnlm-en-dim128/1'
JA_NNLM_MODEL = 'https://tfhub.dev/google/nnlm-ja-dim128/1'
