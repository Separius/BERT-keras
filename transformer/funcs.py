import math
import numpy as np
import keras.backend as K
from keras.layers import Dropout


def shape_list(x):
    if K.backend() != 'theano':
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


# q and v are B, H, L, C//H ; k is B, H, C//H, L ; mask is B, 1, L, L
def scaled_dot_product_attention_tf(q, k, v, attn_mask, attention_dropout: float, neg_inf: float):
    w = K.batch_dot(q, k)  # w is B, H, L, L
    w = w / K.sqrt(K.cast(shape_list(v)[-1], K.floatx()))
    if attn_mask is not None:
        w = attn_mask * w + (1.0 - attn_mask) * neg_inf
    w = K.softmax(w)
    w = Dropout(attention_dropout)(w)
    return K.batch_dot(w, v)  # it is B, H, L, C//H [like v]


def scaled_dot_product_attention_th(q, k, v, attn_mask, attention_dropout: float, neg_inf: float):
    w = theano_matmul(q, k)
    w = w / K.sqrt(K.cast(shape_list(v)[-1], K.floatx()))
    if attn_mask is not None:
        attn_mask = K.repeat_elements(attn_mask, shape_list(v)[1], 1)
        w = attn_mask * w + (1.0 - attn_mask) * neg_inf
    w = K.T.exp(w - w.max()) / K.T.exp(w - w.max()).sum(axis=-1, keepdims=True)
    w = Dropout(attention_dropout)(w)
    return theano_matmul(w, v)


def multihead_attention(x, attn_mask, n_head: int, n_state: int, attention_dropout: float, neg_inf: float):
    _q, _k, _v = x[:, :, :n_state], x[:, :, n_state:2 * n_state], x[:, :, -n_state:]
    q = split_heads(_q, n_head)  # B, H, L, C//H
    k = split_heads(_k, n_head, k=True)  # B, H, C//H, L
    v = split_heads(_v, n_head)  # B, H, L, C//H
    if K.backend() == 'tensorflow':
        a = scaled_dot_product_attention_tf(q, k, v, attn_mask, attention_dropout, neg_inf)
    else:
        a = scaled_dot_product_attention_th(q, k, v, attn_mask, attention_dropout, neg_inf)
    return merge_heads(a)


def gelu(x):
    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


# https://stackoverflow.com/a/42194662/2796084
def theano_matmul(a, b, _left=False):
    assert a.ndim == b.ndim
    ndim = a.ndim
    assert ndim >= 2
    if _left:
        b, a = a, b
    if ndim == 2:
        return K.T.dot(a, b)
    else:
        # If a is broadcastable but b is not.
        if a.broadcastable[0] and not b.broadcastable[0]:
            # Scan b, but hold a steady.
            # Because b will be passed in as a, we need to left multiply to maintain
            #  matrix orientation.
            output, _ = K.theano.scan(theano_matmul, sequences=[b], non_sequences=[a[0], 1])
        # If b is broadcastable but a is not.
        elif b.broadcastable[0] and not a.broadcastable[0]:
            # Scan a, but hold b steady.
            output, _ = K.theano.scan(theano_matmul, sequences=[a], non_sequences=[b[0]])
        # If neither dimension is broadcastable or they both are.
        else:
            # Scan through the sequences, assuming the shape for this dimension is equal.
            output, _ = K.theano.scan(theano_matmul, sequences=[a, b])
        return output
