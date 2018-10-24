import json
import math
import keras
import numpy as np
import keras.backend as K
from data.vocab import TextEncoder
from transformer.config import BERTConfig
from transformer.embedding import Embedding
from keras.layers import Lambda, Conv1D, Dropout, Add, Layer, Input


def _shape_list(x):
    ps = x.get_shape().as_list()
    ts = K.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def _split_heads(x, n, k=False):
    x_shape = _shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    new_x = K.reshape(x, new_x_shape)
    return K.permute_dimensions(new_x, [0, 2, 3, 1] if k else [0, 2, 1, 3])


def _merge_heads(x):
    new_x = K.permute_dimensions(x, [0, 2, 1, 3])
    x_shape = _shape_list(new_x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return K.reshape(new_x, new_x_shape)


def _scaled_dot_product_attention(q, k, v, mask, attention_dropout, layer_id):
    w = K.batch_dot(q, k) / K.sqrt(K.cast(_shape_list(v)[-1], K.floatx()))  # w is B, H, L, L
    if mask is not None:
        w = mask * w + (1.0 - mask) * 10e-9
    w = Dropout(attention_dropout, name='layer_{}/attention_dropout'.format(layer_id))(K.softmax(w))
    return K.batch_dot(w, v)


def _self_attention(x, n_head, n_state, mask, attention_dropout, layer_id):
    _q, _k, _v = x[:, :, :n_state], x[:, :, n_state:2 * n_state], x[:, :, -n_state:]
    q = _split_heads(_q, n_head)  # q is B, H, L, C//H
    k = _split_heads(_k, n_head, k=True)  # k is B, H, C//H, L
    v = _split_heads(_v, n_head)
    a = _scaled_dot_product_attention(q, k, v, mask, attention_dropout, layer_id)
    return _merge_heads(a)


def _multi_head_attention(x, n_state, n_head, mask, attention_dropout, layer_id):
    assert n_state % n_head == 0
    x = Conv1D(3 * n_state, 1, name='layer_{}/c_attn'.format(layer_id))(x)  # x will be batch, seq_len, 3*channels
    a = Lambda(lambda input: _self_attention(input, n_head, n_state, mask, attention_dropout, layer_id),
               name='layer_{}/self_attention'.format(layer_id),
               output_shape=lambda input_shape: [input_shape[0], input_shape[1], input_shape[2] // 3])(x)
    return Conv1D(n_state, 1, name='layer_{}/c_attn_proj'.format(layer_id))(a)


def _gelu(x):
    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


class _LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=keras.initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=keras.initializers.Zeros(), trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


def _position_wise(x, n_state: int, d_hid: int, layer_id: int):
    output = Conv1D(d_hid, 1, name='layer_{}/c_fc'.format(layer_id))(x)
    output = Lambda(lambda x: _gelu(x), name='layer_{}/gelu'.format(layer_id))(output)
    return Conv1D(n_state, 1, name='layer_{}/c_ffn_proj'.format(layer_id))(output)


def _encoder_layer(x, n_state, n_head, mask, d_hid, residual_dropout, attention_dropout, layer_id: int):
    a = _multi_head_attention(x, n_state, n_head, mask, attention_dropout, layer_id)
    n = _LayerNormalization(name='layer_{}/ln_1'.format(layer_id))(Add(name='layer_{}/ln_1_add'.format(layer_id))(
        [x, Dropout(residual_dropout, name='layer_{}/ln_1_drop'.format(layer_id))(a)]))
    f = _position_wise(n, n_state, d_hid, layer_id)
    output = _LayerNormalization(name='layer_{}/ln_2'.format(layer_id))(Add(name='layer_{}/ln_2_add'.format(layer_id))(
        [n, Dropout(residual_dropout, name='layer_{}/ln_2_drop'.format(layer_id))(f)]))
    return output


def create_model(embedding_dim: int = 768, embedding_dropout: float = 0.1, vocab_size: int = 30004, max_len: int = 512,
                 trainable_pos_embedding: bool = True, num_heads: int = 12, num_layers: int = 12,
                 attention_dropout: float = 0.1, use_one_embedding_dropout: bool = BERTConfig.USE_ONE_DROPOUT,
                 d_hid: int = 768 * 4, residual_dropout: float = 0.1,
                 ignore_mask: bool = BERTConfig.IGNORE_MASK) -> keras.Model:
    # NOTE mask is created via create_mask
    mask = None if ignore_mask else Input(shape=(1, max_len, max_len), name='MaskInput')
    tokens = Input(shape=(max_len,), name='TokenInput')
    segment_ids = Input(shape=(max_len,), name='SegmentInput')
    pos_ids = Lambda(lambda length: K.reshape(K.arange(_shape_list(tokens)[1]), (1, -1)), name='PositionInput')(tokens)
    x = Embedding(embedding_dim, embedding_dropout, vocab_size, max_len, trainable_pos_embedding,
                  use_one_embedding_dropout)(tokens, segment_ids, pos_ids)
    for i in range(num_layers):
        x = _encoder_layer(x, embedding_dim, num_heads, mask, d_hid, residual_dropout, attention_dropout, i)
    return keras.Model(inputs=[tokens, segment_ids] + ([] if ignore_mask else [mask]), outputs=[x],
                       name='Transformer')


def load_openai_model(path: str = './openai_weights/', ignore_mask: bool = False,
                      use_one_embedding_dropout: bool = False) -> keras.Model:
    shapes = json.load(open(path + 'params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(path + 'params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    init_params[1] = np.concatenate(
        (init_params[1], np.random.randn(TextEncoder.SPECIAL_COUNT, 768).astype(np.float32) * 0.02), axis=0)
    init_params = [np.zeros((TextEncoder.NUM_SEGMENTS, 768)).astype(np.float32)] + init_params
    model = create_model(embedding_dim=768, embedding_dropout=0.1, vocab_size=40478 + TextEncoder.SPECIAL_COUNT,
                         max_len=512, ignore_mask=ignore_mask, trainable_pos_embedding=True, num_heads=12,
                         num_layers=12, use_one_embedding_dropout=use_one_embedding_dropout, d_hid=4 * 768,
                         attention_dropout=0.1, residual_dropout=0.1)
    model.set_weights(init_params)
    return model
