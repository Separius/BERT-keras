import math
import keras
import numpy as np
import keras.backend as K
from typing import Optional
from transformer.embedding import Embedding
from keras.layers import Lambda, Conv1D, Dropout, Add, Layer, Input


def shape_list(x):
    ps = x.get_shape().as_list()
    ts = K.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def split_heads(x, n, k=False):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    new_x = K.reshape(x, new_x_shape)
    return K.permute_dimensions(new_x, [0, 2, 3, 1] if k else [0, 2, 1, 3])


def create_mask(pad_mask: Optional[np.array], is_causal: bool, batch_size: Optional[int] = None,
                length: Optional[int] = None):
    if pad_mask is not None:
        batch_size, length = pad_mask.shape
    else:
        assert batch_size and length
    if is_causal:
        b = np.cumsum(np.eye(length), axis=0)
    else:
        b = np.ones((length, length))
    b = np.reshape(b, [1, 1, length, length])  # 1, 1, L, L
    b = np.repeat(b, batch_size, axis=0)
    if pad_mask is not None:
        _pad_mask = 1.0 - pad_mask[..., np.newaxis]
        _pad_mask = np.repeat(_pad_mask, length, 2)
        _pad_mask_t = np.transpose(_pad_mask, [0, 2, 1])
        tmp = _pad_mask * _pad_mask_t
        tmp = tmp[:, np.newaxis, ...]
        if b is None:
            b = tmp
        else:
            b = b * tmp
    return b


def merge_heads(x):
    new_x = K.permute_dimensions(x, [0, 2, 1, 3])
    x_shape = shape_list(new_x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return K.reshape(new_x, new_x_shape)


def scaled_dot_product_attention(q, k, v, mask, attention_dropout):
    w = K.batch_dot(q, k) / K.sqrt(K.cast(shape_list(v)[-1], K.floatx()))  # w is B, H, L, L
    if mask is not None:
        w = mask * w + (1.0 - mask) * 10e-9
    w = K.softmax(w)
    w = Dropout(attention_dropout)(w)
    a = K.batch_dot(w, v)
    return a


def self_attention(x, n_head, n_state, mask):
    _q, _k, _v = x[:, :, :n_state], x[:, :, n_state:2 * n_state], x[:, :, -n_state:]
    q = split_heads(_q, n_head)  # q is B, H, L, C//H
    k = split_heads(_k, n_head, k=True)  # k is B, H, C//H, L
    v = split_heads(_v, n_head)
    a = scaled_dot_product_attention(q, k, v, mask, 0.1)
    return merge_heads(a)


def multi_head_attention(x, n_state, n_head, mask, residual_dropout=0.1):  # x is batch, seq_len, channels(n_state)
    assert n_state % n_head == 0
    x = Conv1D(3 * n_state, 1)(x)  # x will be batch, seq_len, 3*channels
    a = Lambda(lambda input: self_attention(input, n_head, n_state, mask),
               output_shape=lambda input_shape: [input_shape[0], input_shape[1], input_shape[2] // 3])(x)
    a = Conv1D(n_state, 1)(a)
    return Dropout(residual_dropout)(a)


def gelu(x):
    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=keras.initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=keras.initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


def position_wise(x, d_hid, dropout=0.1):
    output = Lambda(lambda x: gelu(Conv1D(d_hid, 1)(x)))(x)
    output = Conv1D(d_hid, 1)(output)
    output = Dropout(dropout)(output)
    output = Add()([output, x])
    return LayerNormalization()(output)


def encoder_layer(x, n_state, n_head, mask, dropout=0.1):
    a = multi_head_attention(x, n_state, n_head, mask, residual_dropout=dropout)
    return position_wise(a, n_state, dropout=dropout)


# TODO use a config for default values
def create_model(embedding_dim: int, embedding_dropout: float = 0.1, vocab_size: int = 30000, num_segments: int = 2,
                 max_len: int = 512, trainable_pos_embedding: bool = True, n_head: int = 12, n_layers: int = 12,
                 transformer_dropout: float = 0.1, use_one_embedding_dropout: bool = True):
    tokens = Input(shape=(max_len,))
    segment_ids = Input(shape=(max_len,))
    # NOTE mask is created via mask_attn_weights_numpy
    mask = Input(shape=(1, max_len, max_len))
    pos_ids = Lambda(lambda length: K.reshape(K.arange(shape_list(tokens)[1]), (1, -1)))(tokens)

    x = Embedding(embedding_dim, embedding_dropout, vocab_size, num_segments, max_len,
                  trainable_pos_embedding, use_one_embedding_dropout)(tokens, segment_ids, pos_ids)
    for _ in range(n_layers):
        x = encoder_layer(x, embedding_dim, n_head, mask, dropout=transformer_dropout)
    return keras.Model(inputs=[tokens, segment_ids, mask], outputs=[x])


a = create_model(300, vocab_size=10, max_len=64, trainable_pos_embedding=True, n_head=3, n_layers=3)
