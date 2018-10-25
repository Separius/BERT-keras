import keras.backend as K
from keras.layers import Layer, Dense
from keras.initializers import Ones, Zeros
from transformer.funcs import shape_list, self_attention, gelu


class SelfAttention(Layer):
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
        return self_attention(x, mask, self.n_head, self.n_state, self.attention_dropout)


class LayerNormalization(Layer):
    def __init__(self, eps: float = 1e-6, **kwargs) -> None:
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class Gelu(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return gelu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class TiedEmbeddingsTransposed(Dense):
    def __init__(self, tied_to, units: int, **kwargs):
        super().__init__(units, **kwargs)
        self.tied_to = tied_to

    def build(self, input_shape):
        super().build(input_shape)
        if self.tied_to is not None:
            self.kernel = K.transpose(self.tied_to)
            self.trainable_weights = [self.trainable_weights[1]]
