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


def scaled_dot_product_attention(q, k, v, mask, attention_dropout, layer_id):
    w = K.batch_dot(q, k) / K.sqrt(K.cast(shape_list(v)[-1], K.floatx()))  # w is B, H, L, L
    if mask is not None:
        w = mask * w + (1.0 - mask) * 10e-9
    w = Dropout(attention_dropout, name='layer_{}/attention_dropout'.format(layer_id))(K.softmax(w))
    return K.batch_dot(w, v)


def self_attention(x, n_head, n_state, mask, attention_dropout, layer_id):
    _q, _k, _v = x[:, :, :n_state], x[:, :, n_state:2 * n_state], x[:, :, -n_state:]
    q = split_heads(_q, n_head)  # q is B, H, L, C//H
    k = split_heads(_k, n_head, k=True)  # k is B, H, C//H, L
    v = split_heads(_v, n_head)
    a = scaled_dot_product_attention(q, k, v, mask, attention_dropout, layer_id)
    return merge_heads(a)


def multi_head_attention(x, n_state, n_head, mask, attention_dropout, layer_id):
    assert n_state % n_head == 0
    x = Conv1D(3 * n_state, 1, name='layer_{}/c_attn'.format(layer_id))(x)  # x will be batch, seq_len, 3*channels
    a = Lambda(lambda input: self_attention(input, n_head, n_state, mask, attention_dropout, layer_id),
               name='layer_{}/self_attention'.format(layer_id),
               output_shape=lambda input_shape: [input_shape[0], input_shape[1], input_shape[2] // 3])(x)
    return Conv1D(n_state, 1, name='layer_{}/c_attn_proj'.format(layer_id))(a)


def gelu(x):
    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=keras.initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=keras.initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


def position_wise(x, n_state, d_hid, layer_id: int):
    output = Conv1D(d_hid, 1, name='layer_{}/c_fc'.format(layer_id))(x)
    output = Lambda(lambda x: gelu(x), name='layer_{}/gelu'.format(layer_id))(output)
    return Conv1D(n_state, 1, name='layer_{}/c_ffn_proj'.format(layer_id))(output)


def encoder_layer(x, n_state, n_head, mask, d_hid, residual_dropout, attention_dropout, layer_id: int):
    a = multi_head_attention(x, n_state, n_head, mask, attention_dropout, layer_id)
    n = LayerNormalization(name='layer_{}/ln_1'.format(layer_id))(Add(name='layer_{}/ln_1_add'.format(layer_id))(
        [x, Dropout(residual_dropout, name='layer_{}/ln_1_drop'.format(layer_id))(a)]))
    f = position_wise(n, n_state, d_hid, layer_id)
    output = LayerNormalization(name='layer_{}/ln_2'.format(layer_id))(Add(name='layer_{}/ln_2_add'.format(layer_id))(
        [n, Dropout(residual_dropout, name='layer_{}/ln_2_drop'.format(layer_id))(f)]))
    return output


# TODO use a config for default values
def create_model(embedding_dim: int, embedding_dropout: float, vocab_size: int, max_len: int,
                 trainable_pos_embedding: bool, num_heads: int, num_layers: int, attention_dropout: float,
                 use_one_embedding_dropout: bool, d_hid: int, residual_dropout: float = 0.1, num_segments: int = 2):
    # NOTE mask is created via mask_attn_weights_numpy
    mask = Input(shape=(1, max_len, max_len), name='MaskInput')
    tokens = Input(shape=(max_len,), name='TokenInput')
    segment_ids = Input(shape=(max_len,), name='SegmentInput')
    pos_ids = Lambda(lambda length: K.reshape(K.arange(shape_list(tokens)[1]), (1, -1)), name='PositionInput')(tokens)
    x = Embedding(embedding_dim, embedding_dropout, vocab_size, num_segments, max_len,
                  trainable_pos_embedding, use_one_embedding_dropout)(tokens, segment_ids, pos_ids)
    for i in range(num_layers):
        x = encoder_layer(x, embedding_dim, num_heads, mask, d_hid, residual_dropout, attention_dropout, i)
    return keras.Model(inputs=[tokens, segment_ids, mask], outputs=[x], name='Transformer')

# TODO load, pretrain, test, sentence_level_train, QRNN, release! :D
