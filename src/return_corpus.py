import pickle
import numpy as np
import pandas as pd

from functions import to_sep_space
from config import PAD, UNK, MUSCLE_TEXT


class ReutersMuscleCorpus():

    def __init__(self):
        self.vocab = []
        self.documents = []
        self.embed_matrix = None
        self.pad_id = None
        self.unk_id = None
        self.seq_size = None

    def build(self, embed_matrix, vocab, seq_size):
        self.vocab = vocab
        self.pad_id = self.vocab.index(PAD)
        self.unk_id = self.vocab.index(UNK)
        self.embed_matrix = embed_matrix
        self.seq_size = seq_size
        self.set_documents()

    def set_documents(self):
        df = pd.read_csv(MUSCLE_TEXT)
        self.documents = df['text'].tolist()
        self.documents = [to_sep_space(doc)
                          for doc in self.documents]

    def doc2ids(self, sentence):
        ids = [self.vocab.index(word) if word in self.vocab else self.unk_id
               for word in sentence.split()][:self.seq_size]
        if len(ids) < self.seq_size:
            ids += [self.pad_id] * (self.seq_size - len(ids))
        return np.array(ids)

    def batch_iter(self, batch_size):
        n_step = self.get_step_count(batch_size)
        docs_i = np.array([self.doc2ids(doc)
                           for doc in self.documents])
        while True:
            indices = np.random.permutation(np.arange(len(docs_i)))
            for s in range(n_step):
                index = s * batch_size
                x = docs_i[indices[index:(index + batch_size)]]
                x_vec = self.embed_matrix[x]
                yield x_vec, x_vec

    def get_step_count(self, batch_size):
        n_docs = len(self.documents)
        return n_docs // batch_size

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            corpus = pickle.load(f)
        return corpus
