import keras
import numpy as np
from data.vocab import TextEncoder
from transformer.layers import LayerNormalization


def _get_pos_encoding_matrix(max_len: int, d_emb: int) -> np.array:
    pos_enc = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
         range(max_len)], dtype=np.float32)
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


# NOTE that for vocab_size you should also add special_count
class Embedding(keras.layers.Layer):
    def __init__(self, output_dim: int = 768, dropout: float = 0.1, vocab_size: int = 30000 + TextEncoder.SPECIAL_COUNT,
                 max_len: int = 512, trainable_pos_embedding: bool = True, use_one_dropout: bool = False,
                 use_embedding_layer_norm: bool = False, layer_norm_epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.use_one_dropout = use_one_dropout
        self.output_dim = output_dim
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.trainable_pos_embedding = trainable_pos_embedding

        self.segment_emb = keras.layers.Embedding(TextEncoder.NUM_SEGMENTS, output_dim, input_length=max_len,
                                                  name='SegmentEmbedding')
        if not trainable_pos_embedding:
            self.pos_emb = keras.layers.Embedding(max_len, output_dim, trainable=False, input_length=max_len,
                                                  name='PositionEmbedding',
                                                  weights=[_get_pos_encoding_matrix(max_len, output_dim)])
        else:
            self.pos_emb = keras.layers.Embedding(max_len, output_dim, input_length=max_len, name='PositionEmbedding')
        self.token_emb = keras.layers.Embedding(vocab_size, output_dim, input_length=max_len, name='TokenEmbedding')
        self.embedding_dropout = keras.layers.Dropout(dropout, name='EmbeddingDropOut')
        self.add_embeddings = keras.layers.Add(name='AddEmbeddings')
        self.use_embedding_layer_norm = use_embedding_layer_norm
        if self.use_embedding_layer_norm:
            self.embedding_layer_norm = LayerNormalization(layer_norm_epsilon)
        else:
            self.embedding_layer_norm = None
        self.layer_norm_epsilon = layer_norm_epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.output_dim

    def get_config(self):
        config = {
            'max_len': self.max_len,
            'use_one_dropout': self.use_one_dropout,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'vocab_size': self.vocab_size,
            'trainable_pos_embedding': self.trainable_pos_embedding,
            'embedding_layer_norm': self.use_embedding_layer_norm,
            'layer_norm_epsilon': self.layer_norm_epsilon
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __call__(self, inputs, **kwargs):
        tokens, segment_ids, pos_ids = inputs
        segment_embedding = self.segment_emb(segment_ids)
        pos_embedding = self.pos_emb(pos_ids)
        token_embedding = self.token_emb(tokens)
        if self.use_one_dropout:
            summation = self.add_embeddings([segment_embedding, pos_embedding, token_embedding])
            if self.embedding_layer_norm:
                summation = self.embedding_layer_norm(summation)
            return self.embedding_dropout(summation)
        summation = self.add_embeddings(
            [self.embedding_dropout(segment_embedding), self.embedding_dropout(pos_embedding),
             self.embedding_dropout(token_embedding)])
        if self.embedding_layer_norm:
            summation = self.embedding_layer_norm(summation)
        return summation
