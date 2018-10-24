from encoder import AbstractEncoder
from transformer.embedding import Embedding
from transformer.multihead_attention import Encoder


class Transformer(AbstractEncoder):
    def __init__(self, d_inner_hid: int, d_k: int, d_v: int, embedding_dim: int, embedding_dropout: float = 0.1,
                 vocab_size: int = 30000, num_segments: int = 2, max_len: int = 512,
                 trainable_pos_embedding: bool = True, n_head: int = 12, n_layers: int = 12,
                 transformer_dropout: float = 0.1):
        self.embedding = Embedding(embedding_dim, embedding_dropout, vocab_size, num_segments, max_len,
                                   trainable_pos_embedding)
        self.encoder = Encoder(embedding_dim, d_inner_hid, n_head, d_k, d_v, n_layers, transformer_dropout)

    def process(self, tokens, segment_ids, masks):
        x = self.embedding(tokens, segment_ids)
        return self.encoder(x, masks)
