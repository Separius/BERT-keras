import keras
import numpy as np
from typing import Optional
from data.vocab import TextEncoder
from transformer.config import BERTConfig


def _get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
         range(max_len)])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


# NOTE that for vocab_size you should also add special_count
class Embedding:
    def __init__(self, output_dim: int = 768, dropout: float = 0.1, vocab_size: int = 30004, max_len: int = 512,
                 trainable_pos_embedding: bool = True, use_one_dropout: bool = BERTConfig.USE_ONE_DROPOUT,
                 token_emb_weight: Optional[np.array] = None, segment_emb_weight: Optional[np.array] = None,
                 pos_emb_weight: Optional[np.array] = None):
        self.segment_emb = keras.layers.Embedding(TextEncoder.NUM_SEGMENTS, output_dim, input_length=max_len,
                                                  name='SegmentEmbedding',
                                                  weights=None if segment_emb_weight is None else [segment_emb_weight])
        if not trainable_pos_embedding:
            self.pos_emb = keras.layers.Embedding(max_len, output_dim, trainable=False, input_length=max_len,
                                                  name='PositionEmbedding', weights=[
                    _get_pos_encoding_matrix(max_len, output_dim) if pos_emb_weight is None else [pos_emb_weight]])
        else:
            self.pos_emb = keras.layers.Embedding(max_len, output_dim, input_length=max_len, name='PositionEmbedding')
        self.token_emb = keras.layers.Embedding(vocab_size, output_dim, input_length=max_len, name='TokenEmbedding',
                                                weights=None if token_emb_weight is None else [token_emb_weight])
        self.max_len = max_len
        self.dropout = keras.layers.Dropout(dropout, name='EmbeddingDropOut')
        self.use_one_dropout = use_one_dropout
        self.add_embeddings = keras.layers.Add(name='AddEmbeddings')

    def __call__(self, tokens, segment_ids, pos_ids):
        segment_embedding = self.segment_emb(segment_ids)
        pos_embedding = self.pos_emb(pos_ids)
        token_embedding = self.token_emb(tokens)
        if self.use_one_dropout:
            return self.dropout(self.add_embeddings([segment_embedding, pos_embedding, token_embedding]))
        else:
            return self.add_embeddings(
                [self.dropout(segment_embedding), self.dropout(pos_embedding), self.dropout(token_embedding)])
