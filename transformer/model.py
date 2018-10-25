import json
import math
import keras
import numpy as np
import keras.backend as K
from data.vocab import TextEncoder
from transformer.config import BERTConfig
from transformer.embedding import Embedding
from keras.layers import Lambda, Conv1D, Dropout, Add, Layer, Input, TimeDistributed, Dense


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


def _scaled_dot_product_attention(q, k, v, mask, attention_dropout):
    w = K.batch_dot(q, k) / K.sqrt(K.cast(_shape_list(v)[-1], K.floatx()))  # w is B, H, L, L
    if mask is not None:
        w = mask * w + (1.0 - mask) * 10e-9
    w = Dropout(attention_dropout)(K.softmax(w))
    return K.batch_dot(w, v)


def _self_attention(x, mask, n_head, n_state, attention_dropout):
    _q, _k, _v = x[:, :, :n_state], x[:, :, n_state:2 * n_state], x[:, :, -n_state:]
    q = _split_heads(_q, n_head)  # q is B, H, L, C//H
    k = _split_heads(_k, n_head, k=True)  # k is B, H, C//H, L
    v = _split_heads(_v, n_head)
    a = _scaled_dot_product_attention(q, k, v, mask, attention_dropout)
    return _merge_heads(a)


class SelfAttention(Layer):
    def __init__(self, n_head, n_state, attention_dropout, **kwargs):
        super().__init__(**kwargs)
        self.n_head = n_head
        self.n_state = n_state
        self.attention_dropout = attention_dropout

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2] // 3

    def call(self, inputs, **kwargs):
        return _self_attention(inputs[0], inputs[1], self.n_head, self.n_state, self.attention_dropout)


class MultiHeadAttention:
    def __init__(self, n_state, n_head, attention_dropout, layer_id):
        assert n_state % n_head == 0
        self.c_attn = Conv1D(3 * n_state, 1, name='layer_{}/c_attn'.format(layer_id))
        self.self_attn = SelfAttention(n_head, n_state, attention_dropout,
                                       name='layer_{}/self_attention'.format(layer_id))
        self.c_attn_proj = Conv1D(n_state, 1, name='layer_{}/c_attn_proj'.format(layer_id))

    def __call__(self, x, mask):
        output = self.c_attn(x)
        output = self.self_attn([output, mask])
        return self.c_attn_proj(output)


def _gelu(x):
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
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class PositionWiseFF:
    def __init__(self, n_state: int, d_hid: int, layer_id: int):
        self.c_fc = Conv1D(d_hid, 1, name='layer_{}/c_fc'.format(layer_id))
        self.activation = Lambda(lambda x: _gelu(x), name='layer_{}/gelu'.format(layer_id),
                                 output_shape=lambda input_shape: input_shape)
        self.c_ffn_proj = Conv1D(n_state, 1, name='layer_{}/c_ffn_proj'.format(layer_id))

    def __call__(self, x):
        output = self.activation(self.c_fc(x))
        return self.c_ffn_proj(output)


class EncoderLayer:
    def __init__(self, n_state, n_head, d_hid, residual_dropout, attention_dropout, layer_id: int):
        self.attention = MultiHeadAttention(n_state, n_head, attention_dropout, layer_id)
        self.drop1 = Dropout(residual_dropout, name='layer_{}/ln_1_drop'.format(layer_id))
        self.add1 = Add(name='layer_{}/ln_1_add'.format(layer_id))
        self.ln1 = LayerNormalization(name='layer_{}/ln_1'.format(layer_id))
        self.ffn = PositionWiseFF(n_state, d_hid, layer_id)
        self.drop2 = Dropout(residual_dropout, name='layer_{}/ln_2_drop'.format(layer_id))
        self.add2 = Add(name='layer_{}/ln_2_add'.format(layer_id))
        self.ln2 = LayerNormalization(name='layer_{}/ln_2'.format(layer_id))

    def __call__(self, x, mask):
        a = self.attention(x, mask)
        n = self.ln1(self.add1([x, self.drop1(a)]))
        f = self.ffn(n)
        return self.ln2(self.add2([n, self.drop2(f)]))


class TiedEmbeddingsTransposed(Dense):
    def __init__(self, tied_to, units, **kwargs):
        super().__init__(units, **kwargs)
        self.tied_to = tied_to

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel = K.transpose(self.tied_to.weights[0])
        self.trainable_weights = [self.trainable_weights[1]]


def create_model(embedding_dim: int = 768, embedding_dropout: float = 0.1,
                 vocab_size: int = 30000 + TextEncoder.SPECIAL_COUNT, max_len: int = 512,
                 trainable_pos_embedding: bool = True, num_heads: int = 12, num_layers: int = 12,
                 attention_dropout: float = 0.1, use_one_embedding_dropout: bool = BERTConfig.USE_ONE_DROPOUT,
                 d_hid: int = 768 * 4, residual_dropout: float = 0.1,
                 ignore_mask: bool = BERTConfig.IGNORE_MASK, debug: bool = False) -> keras.Model:
    # NOTE mask is created via create_mask
    mask = None if ignore_mask or debug else Input(batch_shape=(None, 1, max_len, max_len), name='MaskInput')
    tokens = Input(batch_shape=(None, max_len), name='TokenInput',
                   tensor=K.variable(np.random.randint(0, vocab_size, (3, max_len))) if debug else None)
    segment_ids = Input(batch_shape=(None, max_len), name='SegmentInput',
                        tensor=K.variable(np.random.randint(0, 2, (3, max_len))) if debug else None)
    pos_ids = Lambda(lambda x: K.reshape(K.arange(_shape_list(x)[1]), (1, -1)), name='PositionInput',
                     output_shape=lambda input_shape: (1, input_shape[1]))(tokens)
    embedding_layer = Embedding(embedding_dim, embedding_dropout, vocab_size, max_len, trainable_pos_embedding,
                                use_one_embedding_dropout)
    x = embedding_layer(tokens, segment_ids, pos_ids)
    for i in range(num_layers):
        x = EncoderLayer(embedding_dim, num_heads, d_hid, residual_dropout, attention_dropout, i)(x, mask)
    logits = TimeDistributed(TiedEmbeddingsTransposed(embedding_layer.token_emb, units=vocab_size, name='TiedDecoder'),
                             name='TiedDecoderDistributed')(x)
    if debug:
        print(K.eval(x).shape, K.eval(logits).shape)
    return keras.Model(inputs=[tokens, segment_ids] + ([] if ignore_mask or debug else [mask]), outputs=[x, logits],
                       name='Transformer')


def load_openai_model(path: str = './openai_weights/', ignore_mask: bool = False,
                      use_one_embedding_dropout: bool = False, debug: bool = False) -> keras.Model:
    shapes = json.load(open(path + 'params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(path + 'params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    # add special token embedding to token embedding
    init_params[1] = np.concatenate(
        (init_params[1], np.random.randn(TextEncoder.SPECIAL_COUNT, 768).astype(np.float32) * 0.02), axis=0)
    init_params = [np.zeros((TextEncoder.NUM_SEGMENTS, 768)).astype(np.float32)] + init_params  # segment embedding
    init_params = init_params + [np.zeros((40478 + TextEncoder.SPECIAL_COUNT,)).astype(np.float32)]  # decoder's bias
    model = create_model(embedding_dim=768, embedding_dropout=0.1, vocab_size=40478 + TextEncoder.SPECIAL_COUNT,
                         max_len=512, ignore_mask=ignore_mask, trainable_pos_embedding=True, num_heads=12,
                         num_layers=12, use_one_embedding_dropout=use_one_embedding_dropout, d_hid=4 * 768,
                         attention_dropout=0.1, residual_dropout=0.1, debug=debug)
    if debug:
        assert len(model.weights) == len(init_params)
        for a, b in zip(model.weights, init_params):
            assert a.shape == b.shape
    model.set_weights(init_params)
    return model
