import keras.backend as K
from keras.layers import Layer, Dense
from keras.initializers import Ones, Zeros
from transformer.funcs import multihead_attention, gelu


class MultiHeadAttention(Layer):
    def __init__(self, n_head: int, n_state: int, attention_dropout: float, ignore_mask: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_head = n_head
        self.n_state = n_state
        self.attention_dropout = attention_dropout
        self.ignore_mask = ignore_mask

    def compute_output_shape(self, input_shape):
        x = input_shape if self.ignore_mask else input_shape[0]
        return x[0], x[1], x[2] // 3

    def call(self, inputs, **kwargs):
        x = inputs if self.ignore_mask else inputs[0]
        mask = None if self.ignore_mask else inputs[1]
        return multihead_attention(x, mask, self.n_head, self.n_state, self.attention_dropout)

    def get_config(self):
        config = {
            'n_head': self.n_head,
            'n_state': self.n_state,
            'attention_dropout': self.attention_dropout,
            'ignore_mask': self.ignore_mask,
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return gelu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class TiedEmbeddingsTransposed(Dense):
    def __init__(self, tied_to, units: int, use_bias: bool, **kwargs):
        super().__init__(units, use_bias=use_bias, **kwargs)
        self.tied_to = None if tied_to is None else K.transpose(tied_to)

    def build(self, input_shape):
        super().build(input_shape)
        if self.tied_to is not None:
            self.kernel = self.tied_to
            if self.use_bias:
                self.trainable_weights = [self.trainable_weights[1]]
            else:
                self.trainable_weights = []

    def get_config(self):
        config = {
            'tied_to': None,  # TODO correct this somehow
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
