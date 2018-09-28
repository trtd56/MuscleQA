from keras.models import Model
from keras.models import load_model
from keras.layers import Input, LSTM, Bidirectional, RepeatVector


class AutoEncoder():

    def __init__(self, seq_size=50, embed_size=100, latent_size=256):
        self.seq_size = seq_size
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.model = None

    def build(self):
        inputs = Input(shape=(self.seq_size, self.embed_size), name="input")
        encoded = Bidirectional(
            LSTM(self.latent_size),
            merge_mode="concat", name="encoder")(inputs)
        encoded = RepeatVector(self.seq_size, name="replicate")(encoded)
        decoded = Bidirectional(
            LSTM(self.embed_size, return_sequences=True),
            merge_mode="sum", name="decoder")(encoded)

        self.model = Model(inputs, decoded)

    @classmethod
    def load(cls, path):
        model = load_model(path)
        _, seq_size, embed_size = model.input.shape  # top is batch size
        latent_size = model.get_layer("encoder").input_shape[1]
        ae = AutoEncoder(seq_size, embed_size, latent_size)
        ae.model = model
        return ae

    def get_encoder(self):
        if self.model:
            m = self.model
            encoder = Model(m.input, m.get_layer("encoder").output)
            return encoder
        else:
            raise Exception("Model is not built/loaded")
