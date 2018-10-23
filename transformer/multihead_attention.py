# this is from https://github.com/Lsdefine/attention-is-all-you-need-keras
import numpy as np
import keras.backend as K
from keras.initializers import Ones, Zeros
from keras.layers import Dense, Dropout, Lambda, Add, Activation, Layer, TimeDistributed, Conv1D


class ScaledDotProductAttention:
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10) * x)(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention:
    def __init__(self, n_head, d_model, d_k, d_v, dropout, use_norm=True):
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.qs_layer = Dense(n_head * d_k, use_bias=False)
        self.ks_layer = Dense(n_head * d_k, use_bias=False)
        self.vs_layer = Dense(n_head * d_v, use_bias=False)
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
        ks = self.ks_layer(k)
        vs = self.vs_layer(v)

        def reshape1(x):
            s = K.shape(x)  # [batch_size, len_q, n_head * d_k]
            x = K.reshape(x, [s[0], s[1], n_head, d_k])
            x = K.permute_dimensions(x, [2, 0, 1, 3])
            x = K.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
            return x

        qs = Lambda(reshape1)(qs)
        ks = Lambda(reshape1)(ks)
        vs = Lambda(reshape1)(vs)

        if mask is not None:
            mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
        head, attn = self.attention(qs, ks, vs, mask=mask)

        def reshape2(x):
            s = K.shape(x)  # [n_head * batch_size, len_v, d_v]
            x = K.reshape(x, [n_head, -1, s[1], s[2]])
            x = K.permute_dimensions(x, [1, 2, 0, 3])
            x = K.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
            return x

        head = Lambda(reshape2)(head)
        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn


class PositionwiseFeedForward:
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer:
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


class Encoder:
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, layers=6, dropout=0.1):
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, x, mask):
        for enc_layer in self.layers:
            x, att = enc_layer(x, mask)
        return x
