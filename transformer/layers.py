import math
import keras.backend as K
from keras.layers import Layer
from keras.initializers import Ones, Zeros
from transformer.funcs import gelu, multihead_attention


class MultiHeadAttention(Layer):
    def __init__(self, n_head: int, n_state: int, attention_dropout: float, use_attn_mask: bool, neg_inf: float,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_head = n_head
        self.n_state = n_state
        self.attention_dropout = attention_dropout
        self.use_attn_mask = use_attn_mask
        self.neg_inf = neg_inf

    def compute_output_shape(self, input_shape):
        x = input_shape[0] if self.use_attn_mask else input_shape
        return x[0], x[1], x[2] // 3

    def call(self, inputs, **kwargs):
        x = inputs[0] if self.use_attn_mask else inputs
        attn_mask = inputs[1] if self.use_attn_mask else None
        return multihead_attention(x, attn_mask, self.n_head, self.n_state, self.attention_dropout, self.neg_inf)

    def get_config(self):
        config = {
            'n_head': self.n_head,
            'n_state': self.n_state,
            'attention_dropout': self.attention_dropout,
            'use_attn_mask': self.use_attn_mask,
            'neg_inf': self.neg_inf,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalization(Layer):
    def __init__(self, eps: float = 1e-5, **kwargs) -> None:
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super().build(input_shape)

    def call(self, x, **kwargs):
        u = K.mean(x, axis=-1, keepdims=True)
        s = K.mean(K.square(x - u), axis=-1, keepdims=True)
        z = (x - u) / K.sqrt(s + self.eps)
        return self.gamma * z + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'eps': self.eps,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Gelu(Layer):
    def __init__(self, accurate: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.accurate = accurate

    def call(self, inputs, **kwargs):
        if not self.accurate:
            return gelu(inputs)
        if K.backend() == 'tensorflow':
            erf = K.tf.erf
        else:
            erf = K.T.erf
        return inputs * 0.5 * (1.0 + erf(inputs / math.sqrt(2.0)))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'accurate': self.accurate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
