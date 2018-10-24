import keras
import numpy as np
import keras.backend as K


def _get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
         range(max_len)])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


class Embedding:
    def __init__(self, output_dim, dropout: float = 0.1, vocab_size: int = 30000, num_segments: int = 2,
                 max_len: int = 512, trainable_pos_embedding: bool = True):
        self.token_emb = keras.layers.Embedding(vocab_size, output_dim, input_length=max_len)
        self.segment_emb = keras.layers.Embedding(num_segments, output_dim, input_length=max_len)
        self.max_len = max_len
        if not trainable_pos_embedding:
            self.pos_emb = keras.layers.Embedding(max_len, output_dim, trainable=False, input_length=max_len,
                                                  weights=[_get_pos_encoding_matrix(max_len, output_dim)])
        else:
            self.pos_emb = keras.layers.Embedding(max_len, output_dim, input_length=max_len)
        self.dropout = keras.layers.Dropout(dropout)

    def __call__(self, tokens, segment_ids):
        token_embedding = self.token_emb(tokens)
        segment_embedding = self.segment_emb(segment_ids)
        pos_embedding = self.pos_emb(K.variable(np.arange(self.max_len).reshape((1, -1))))
        pos_embedding = K.repeat_elements(pos_embedding, K.shape(token_embedding)[0], 0)
        # TODO how should I apply dropout? dropout(sum) or sum(dropout), I think it should be a flag
        return self.dropout(keras.layers.add([token_embedding, segment_embedding, pos_embedding]))
