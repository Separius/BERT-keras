import math
import numpy as np
import keras.backend as K
from keras.layers import Dropout


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
    # TODO which one is better?
    # def dropped_inputs():
    #     return K.dropout(w, attention_dropout, None, seed=None)
    #
    # w = K.in_train_phase(dropped_inputs, w, training=None)
    w = Dropout(attention_dropout)(w)
    return K.batch_dot(w, v)


def self_attention(x, mask, n_head, n_state, attention_dropout):
    _q, _k, _v = x[:, :, :n_state], x[:, :, n_state:2 * n_state], x[:, :, -n_state:]
    q = split_heads(_q, n_head)  # q is B, H, L, C//H
    k = split_heads(_k, n_head, k=True)  # k is B, H, C//H, L
    v = split_heads(_v, n_head)
    a = scaled_dot_product_attention(q, k, v, mask, attention_dropout)
    return merge_heads(a)


def gelu(x):
    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))
