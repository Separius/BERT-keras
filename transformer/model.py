import keras
import keras.backend as K
from data.vocab import TextEncoder
from transformer.embedding import Embedding
from keras.layers import Conv1D, Dropout, Add, Input
from transformer.layers import MultiHeadAttention, Gelu, LayerNormalization


class MultiHeadSelfAttention:
    def __init__(self, n_state: int, n_head: int, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float) -> None:
        assert n_state % n_head == 0
        self.c_attn = Conv1D(3 * n_state, 1, name='layer_{}/c_attn'.format(layer_id))
        self.attn = MultiHeadAttention(n_head, n_state, attention_dropout, use_attn_mask,
                                       neg_inf, name='layer_{}/self_attention'.format(layer_id))
        self.c_attn_proj = Conv1D(n_state, 1, name='layer_{}/c_attn_proj'.format(layer_id))

    def __call__(self, x, mask):
        output = self.c_attn(x)
        output = self.attn(output) if mask is None else self.attn([output, mask])
        return self.c_attn_proj(output)


class PositionWiseFF:
    def __init__(self, n_state: int, d_hid: int, layer_id: int, accurate_gelu: bool) -> None:
        self.c_fc = Conv1D(d_hid, 1, name='layer_{}/c_fc'.format(layer_id))
        self.activation = Gelu(accurate=accurate_gelu, name='layer_{}/gelu'.format(layer_id))
        self.c_ffn_proj = Conv1D(n_state, 1, name='layer_{}/c_ffn_proj'.format(layer_id))

    def __call__(self, x):
        output = self.activation(self.c_fc(x))
        return self.c_ffn_proj(output)


class EncoderLayer:
    def __init__(self, n_state: int, n_head: int, d_hid: int, residual_dropout: float, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float, ln_epsilon: float, accurate_gelu: bool) -> None:
        self.attention = MultiHeadSelfAttention(n_state, n_head, attention_dropout, use_attn_mask, layer_id, neg_inf)
        self.drop1 = Dropout(residual_dropout, name='layer_{}/ln_1_drop'.format(layer_id))
        self.add1 = Add(name='layer_{}/ln_1_add'.format(layer_id))
        self.ln1 = LayerNormalization(ln_epsilon, name='layer_{}/ln_1'.format(layer_id))
        self.ffn = PositionWiseFF(n_state, d_hid, layer_id, accurate_gelu)
        self.drop2 = Dropout(residual_dropout, name='layer_{}/ln_2_drop'.format(layer_id))
        self.add2 = Add(name='layer_{}/ln_2_add'.format(layer_id))
        self.ln2 = LayerNormalization(ln_epsilon, name='layer_{}/ln_2'.format(layer_id))

    def __call__(self, x, mask):
        a = self.attention(x, mask)
        n = self.ln1(self.add1([x, self.drop1(a)]))
        f = self.ffn(n)
        return self.ln2(self.add2([n, self.drop2(f)]))


def create_transformer(embedding_dim: int = 768, embedding_dropout: float = 0.1, vocab_size: int = 30000,
                       max_len: int = 512, trainable_pos_embedding: bool = True, num_heads: int = 12,
                       num_layers: int = 12, attention_dropout: float = 0.1, use_one_embedding_dropout: bool = False,
                       d_hid: int = 768 * 4, residual_dropout: float = 0.1, use_attn_mask: bool = True,
                       embedding_layer_norm: bool = False, neg_inf: float = -1e9, layer_norm_epsilon: float = 1e-5,
                       accurate_gelu: bool = False) -> keras.Model:
    vocab_size += TextEncoder.SPECIAL_COUNT
    tokens = Input(batch_shape=(None, max_len), name='token_input', dtype='int32')
    segment_ids = Input(batch_shape=(None, max_len), name='segment_input', dtype='int32')
    pos_ids = Input(batch_shape=(None, max_len), name='position_input', dtype='int32')
    attn_mask = Input(batch_shape=(None, 1, max_len, max_len), name='attention_mask_input',
                      dtype=K.floatx()) if use_attn_mask else None
    inputs = [tokens, segment_ids, pos_ids]
    embedding_layer = Embedding(embedding_dim, embedding_dropout, vocab_size, max_len, trainable_pos_embedding,
                                use_one_embedding_dropout, embedding_layer_norm, layer_norm_epsilon)
    x = embedding_layer(inputs)
    for i in range(num_layers):
        x = EncoderLayer(embedding_dim, num_heads, d_hid, residual_dropout,
                         attention_dropout, use_attn_mask, i, neg_inf, layer_norm_epsilon, accurate_gelu)(x, attn_mask)
    if use_attn_mask:
        inputs.append(attn_mask)
    return keras.Model(inputs=inputs, outputs=[x], name='Transformer')
