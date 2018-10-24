import math
import keras
import numpy as np
import keras.backend as K
from transformer.embedding import Embedding


def shape_list(x):
    ps = x.get_shape().as_list()
    ts = K.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return K.reshape(x, new_x_shape)


def split_heads(x, n, k=False):
    return K.permute_dimensions(split_states(x, n), [0, 2, 3, 1] if k else [0, 2, 1, 3])


def mask_attn_weights(pad_mask, is_causal, batch_size, length):
    if is_causal:
        b = K.cumsum(K.eye(length), 0)
        b = K.reshape(b, [1, 1, length, length])  # 1, 1, L, L
        b = K.repeat_elements(b, batch_size, 0)
    else:
        b = None
    if pad_mask is not None:
        _pad_mask = K.expand_dims(pad_mask, -1)
        _pad_mask = K.repeat_elements(_pad_mask, length, 2)
        _pad_mask_t = K.permute_dimensions(_pad_mask, [0, 2, 1])
        tmp = K.expand_dims(keras.layers.multiply([_pad_mask, _pad_mask_t]), 1)
        if b is None:
            b = tmp
        else:
            b = keras.layers.multiply([b, tmp])
    return b


def _attn(q, k, v, mask, attention_dropout):
    w = keras.layers.Lambda(lambda input: K.batch_dot(q, k) / K.sqrt(K.cast(shape_list(input)[-1], K.floatx())))(
        v)  # w is B, H, L, L
    if mask is not None:
        w = mask * w + (1.0 - mask) * 10e-9
    w = keras.layers.Softmax()(w)
    w = keras.layers.Dropout(attention_dropout)(w)
    a = K.batch_dot(w, v)
    return a


def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return K.reshape(x, new_x_shape)


def merge_heads(x):
    return merge_states(K.permute_dimensions(x, [0, 2, 1, 3]))


def neo_attn(x, n_state, n_head, mask, residual_dropout=0.1):  # x is batch, seq_len, channels(n_state)
    assert n_state % n_head == 0
    x = keras.layers.Conv1D(3 * n_state, 1)(x)  # x will be batch, seq_len, 3*channels

    def attn(input):
        _q, _k, _v = input[:, :, :n_state], input[:, :, n_state:2 * n_state], input[:, :, -n_state:]
        q = split_heads(_q, n_head)  # q is B, H, L, C//H
        k = split_heads(_k, n_head, k=True)  # k is B, H, C//H, L
        v = split_heads(_v, n_head)
        a = _attn(q, k, v, mask, 0.1)
        return merge_heads(a)

    a = keras.layers.Lambda(lambda input: attn(input),
                            output_shape=lambda input_shape: [input_shape[0], input_shape[1], input_shape[2] // 3])
    a = a(x)
    a = keras.layers.Conv1D(n_state, 1)(a)
    a = keras.layers.Dropout(residual_dropout)(a)
    return a


def gelu(x):
    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


class LayerNormalization(keras.layers.Layer):
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
    output = keras.layers.Lambda(lambda x: gelu(keras.layers.Conv1D(d_hid, 1)(x)))(x)
    output = keras.layers.Conv1D(d_hid, 1)(output)
    output = keras.layers.Dropout(dropout)(output)
    output = keras.layers.Add()([output, x])
    return LayerNormalization()(output)


def encoder_layer(x, n_state, n_head, mask, dropout=0.1):
    a = neo_attn(x, n_state, n_head, mask, residual_dropout=dropout)
    return keras.layers.Lambda(lambda x: position_wise(x, n_state, dropout=dropout))(a)


# TODO use a config for default values
def create_model(embedding_dim: int, embedding_dropout: float = 0.1,
                 vocab_size: int = 30000, num_segments: int = 2, max_len: int = 512,
                 trainable_pos_embedding: bool = True, n_head: int = 12, n_layers: int = 12,
                 transformer_dropout: float = 0.1, use_one_embedding_dropout: bool = True, is_causal: bool = False):
    tokens = keras.layers.Input(shape=(max_len,), tensor=K.variable(np.random.randint(0, 3, (32, max_len))))
    segment_ids = keras.layers.Input(shape=(max_len,), tensor=K.variable(np.random.randint(0, 2, (32, max_len))))
    masks = keras.layers.Input(shape=(max_len,),
                               tensor=K.variable(np.random.randint(0, 1, (32, max_len)).astype(np.float32)))
    x = Embedding(embedding_dim, embedding_dropout, vocab_size, num_segments, max_len,
                  trainable_pos_embedding, use_one_embedding_dropout)(tokens, segment_ids)
    shape = shape_list(x)
    mask = keras.layers.Lambda(lambda input: mask_attn_weights(input, is_causal, shape[0], shape[1]))(masks)
    for _ in range(n_layers):
        x = encoder_layer(x, embedding_dim, n_head, mask, dropout=transformer_dropout)
    return keras.Model(inputs=[tokens, segment_ids, masks], outputs=[x])


a = create_model(300, vocab_size=10, max_len=64, trainable_pos_embedding=False, n_head=3, n_layers=3, is_causal=True)
