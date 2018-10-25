import math
import numpy as np
import keras.backend as K
from keras.layers import Dropout


# TODO this does not work for theano :( [and I was unable to install cntk to check it]
def shape_list(x):
    if K.backend() == 'tensorflow':
        tmp = K.int_shape(x)
    else:
        tmp = x.shape
    tmp = list(tmp)
    tmp[0] = -1
    return tmp


def split_heads(x, n: int, k: bool = False):  # B, L, C
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


def scaled_dot_product_attention(q, k, v, mask, attention_dropout: float):
    w = K.batch_dot(q, k)  # w is B, H, L, L
    w = w / K.sqrt(K.cast(shape_list(v)[-1], K.floatx()))
    if mask is not None:
        w = mask * w + (1.0 - mask) * -1e9
    if K.backend() == 'tensorflow':
        w = K.softmax(w)
    else:
        w = K.T.exp(w - w.max()) / K.T.exp(w - w.max()).sum(axis=-1, keepdims=True)
    w = Dropout(attention_dropout)(w)
    return K.batch_dot(w, v)  # v is B, H, L, C//H


def self_attention(x, mask, n_head: int, n_state: int, attention_dropout: float):
    _q, _k, _v = x[:, :, :n_state], x[:, :, n_state:2 * n_state], x[:, :, -n_state:]
    q = split_heads(_q, n_head)  # B, H, L, C//H
    k = split_heads(_k, n_head, k=True)  # B, H, C//H, L
    v = split_heads(_v, n_head)  # B, H, L, C//H
    a = scaled_dot_product_attention(q, k, v, mask, attention_dropout)
    return merge_heads(a)


def gelu(x):
    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))
