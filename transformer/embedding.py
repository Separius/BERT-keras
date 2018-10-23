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
                 max_len: int = 512, trainable_pos_embedding: bool = True, batch_size: int = 256):
        self.emb = keras.layers.Embedding(vocab_size + num_segments + (max_len if trainable_pos_embedding else 0),
                                          output_dim, input_length=max_len)
        self.max_len = max_len
        if not trainable_pos_embedding:
            self.pos_emb = keras.layers.Embedding(max_len, output_dim, trainable=False,
                                                  weights=[_get_pos_encoding_matrix(max_len, output_dim)],
                                                  input_length=max_len)
            self.pos_start = 0
        else:
            self.pos_emb = self.emb
            self.pos_start = vocab_size + num_segments
        self.dropout = keras.layers.Dropout(dropout)
        self.batch_size = batch_size

    def __call__(self, tokens, segment_ids):
        token_embedding = self.emb(tokens)
        segment_embedding = self.emb(segment_ids)
        pos_embedding = self.pos_emb(K.variable((np.arange(0, self.max_len) + self.pos_start).reshape((1, -1))))
        pos_embedding = K.repeat_elements(pos_embedding, self.batch_size, 0)
        return self.dropout(keras.layers.add([token_embedding, segment_embedding, pos_embedding]))
