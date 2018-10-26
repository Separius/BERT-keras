import json
import keras
import numpy as np
import keras.backend as K
from data.vocab import TextEncoder
from transformer.config import BertConfig
from transformer.embedding import Embedding
from keras.layers import Conv1D, Dropout, Add, Input, TimeDistributed, Dense
from transformer.layers import MultiHeadAttention, Gelu, LayerNormalization, TiedDecoder


class MultiHeadSelfAttention:
    def __init__(self, n_state: int, n_head: int, attention_dropout: float, ignore_mask: bool, layer_id: int) -> None:
        assert n_state % n_head == 0
        self.c_attn = Conv1D(3 * n_state, 1, name='layer_{}/c_attn'.format(layer_id))
        self.attn = MultiHeadAttention(n_head, n_state, attention_dropout, ignore_mask,
                                       name='layer_{}/self_attention'.format(layer_id))
        self.c_attn_proj = Conv1D(n_state, 1, name='layer_{}/c_attn_proj'.format(layer_id))

    def __call__(self, x, mask):
        output = self.c_attn(x)
        output = self.attn(output) if mask is None else self.attn([output, mask])
        return self.c_attn_proj(output)


class PositionWiseFF:
    def __init__(self, n_state: int, d_hid: int, layer_id: int) -> None:
        self.c_fc = Conv1D(d_hid, 1, name='layer_{}/c_fc'.format(layer_id))
        self.activation = Gelu(name='layer_{}/gelu'.format(layer_id))
        self.c_ffn_proj = Conv1D(n_state, 1, name='layer_{}/c_ffn_proj'.format(layer_id))

    def __call__(self, x):
        output = self.activation(self.c_fc(x))
        return self.c_ffn_proj(output)


class EncoderLayer:
    def __init__(self, n_state: int, n_head: int, d_hid: int, residual_dropout: float, attention_dropout: float,
                 ignore_mask: bool, layer_id: int, **kwargs) -> None:
        self.attention = MultiHeadSelfAttention(n_state, n_head, attention_dropout, ignore_mask, layer_id)
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


__test_batch_size = 3


def load_openai_model(path: str = './openai/model/', ignore_mask: bool = False, use_one_embedding_dropout: bool = False,
                      debug: bool = False, max_len: int = 512, compute_logit: bool = True) -> keras.Model:
    with open(path + 'params_shapes.json') as f:
        shapes = json.load(f)
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(path + 'params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    init_params[0] = init_params[0][:min(512, max_len)]
    # add special token embedding to token embedding
    init_params[1] = np.concatenate(
        (init_params[1], np.random.randn(TextEncoder.SPECIAL_COUNT, 768).astype(np.float32) * 0.02), axis=0)
    init_params = [np.zeros((TextEncoder.NUM_SEGMENTS, 768)).astype(np.float32)] + init_params  # segment embedding
    model = create_model(embedding_dim=768, embedding_dropout=0.1, vocab_size=40478 + TextEncoder.SPECIAL_COUNT,
                         max_len=min(512, max_len), ignore_mask=ignore_mask, trainable_pos_embedding=True, num_heads=12,
                         num_layers=12, use_one_embedding_dropout=use_one_embedding_dropout, d_hid=4 * 768,
                         attention_dropout=0.1, residual_dropout=0.1, debug=debug, use_tied_decoder=True,
                         compute_logit=compute_logit)
    if debug:
        assert len(model.weights) == len(init_params)
        for a, b in zip(model.weights, init_params):
            assert a.shape == b.shape
    model.set_weights(init_params)
    return model


class LanguageModelingModel(keras.layers.Layer):
    def __init__(self, compute_logit, embedding_dim, embedding_dropout, vocab_size, max_len, trainable_pos_embedding,
                 use_one_embedding_dropout, ignore_mask, num_layers, num_heads, d_hid, residual_dropout,
                 attention_dropout, use_tied_decoder, **kwargs):
        super().__init__(**kwargs)
        self.compute_logit = compute_logit
        self.ignore_mask = ignore_mask
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding_dropout = embedding_dropout
        self.max_len = max_len
        self.trainable_pos_embedding = trainable_pos_embedding
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_hid = d_hid
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self.use_tied_decoder = use_tied_decoder
        self.use_one_embedding_dropout = use_one_embedding_dropout
        self.embedding_layer = Embedding(embedding_dim, embedding_dropout, vocab_size, max_len, trainable_pos_embedding,
                                         use_one_embedding_dropout)
        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append(
                EncoderLayer(embedding_dim, num_heads, d_hid, residual_dropout, attention_dropout, ignore_mask, i))
        if self.compute_logit:
            if not use_tied_decoder:
                self.decoder = Conv1D(filters=vocab_size, kernel_size=1, use_bias=False, name='Decoder')
            else:
                self.decoder = None

    def get_config(self):
        config = {
            'compute_logit': self.compute_logit,
            'embedding_dim': self.embedding_dim,
            'embedding_dropout': self.embedding_dropout,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'trainable_pos_embedding': self.trainable_pos_embedding,
            'use_one_embedding_dropout': self.use_one_embedding_dropout,
            'ignore_mask': self.ignore_mask,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'd_hid': self.d_hid,
            'residual_dropout': self.residual_dropout,
            'attention_dropout': self.attention_dropout,
            'use_tied_decoder': self.use_tied_decoder,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        default_output = (input_shape[0], input_shape[1], self.embedding_dim)
        if self.compute_logit:
            return [default_output, (input_shape[0], input_shape[1], self.vocab_size)]
        return default_output

    def call(self, inputs, **kwargs):
        x = self.embedding_layer([inputs[0], inputs[1], inputs[2]])
        mask = None if self.ignore_mask else inputs[3]
        for encoder in self.encoder_layers:
            x = encoder(x, mask)
        if self.compute_logit:
            if self.decoder:
                return [x, self.decoder(x)]
            return [x, K.conv1d(x, K.expand_dims(K.transpose(self.embedding_layer.token_emb.weights[0]), 0), strides=1,
                                padding='valid', data_format=K.normalize_data_format(None), dilation_rate=1)]
        return x


def create_model(embedding_dim: int = 768, embedding_dropout: float = 0.1,
                 vocab_size: int = 30000 + TextEncoder.SPECIAL_COUNT, max_len: int = 512,
                 trainable_pos_embedding: bool = True, num_heads: int = 12, num_layers: int = 12,
                 attention_dropout: float = 0.1, use_one_embedding_dropout: bool = BertConfig.USE_ONE_DROPOUT,
                 d_hid: int = 768 * 4, residual_dropout: float = 0.1, use_tied_decoder: bool = True,
                 ignore_mask: bool = BertConfig.IGNORE_MASK, compute_logit: bool = True,
                 debug: bool = False) -> keras.Model:
    # NOTE mask is created via create_mask
    mask = None if ignore_mask else Input(batch_shape=(None, 1, max_len, max_len), name='MaskInput', tensor=K.variable(
        np.random.randint(0, 2, (__test_batch_size, 1, max_len, max_len)).astype(np.float32)) if debug else None)
    tokens = Input(batch_shape=(None, max_len), name='TokenInput', tensor=K.variable(
        np.random.randint(0, vocab_size - TextEncoder.SPECIAL_COUNT, (__test_batch_size, max_len))) if debug else None)
    segment_ids = Input(batch_shape=(None, max_len), name='SegmentInput',
                        tensor=K.variable(np.random.randint(0, 2, (__test_batch_size, max_len))) if debug else None)
    pos_ids = Input(batch_shape=(None, max_len), name='PositionInput', tensor=K.variable(
        np.repeat(np.arange(max_len, dtype=np.int64).reshape(1, -1), __test_batch_size, 0)) if debug else None)
    inputs = [tokens, segment_ids, pos_ids] + ([] if ignore_mask else [mask])

    # lm_model = LanguageModelingModel(compute_logit, embedding_dim, embedding_dropout, vocab_size, max_len,
    #                                  trainable_pos_embedding, use_one_embedding_dropout, ignore_mask, num_layers,
    #                                  num_heads, d_hid, residual_dropout, attention_dropout, use_tied_decoder)
    # outputs = lm_model(inputs)
    # if compute_logit:
    #     x, logits = outputs
    # else:
    #     x = outputs
    embedding_layer = Embedding(embedding_dim, embedding_dropout, vocab_size, max_len, trainable_pos_embedding,
                                use_one_embedding_dropout)
    x = embedding_layer([tokens, segment_ids, pos_ids])
    for i in range(num_layers):
        x = EncoderLayer(embedding_dim, num_heads, d_hid, residual_dropout,
                         attention_dropout, ignore_mask, i)(x, mask)
    if compute_logit:
        if use_tied_decoder:
            logits = TiedDecoder(vocab_size)([x, embedding_layer.token_emb.weights[0]])
        else:
            logits = Conv1D(filters=vocab_size, kernel_size=1, use_bias=False, name='Decoder')(x)
        outputs = [x, logits]
    else:
        outputs = x

    if debug:
        x_eval = K.eval(x)
        assert x_eval.shape == (__test_batch_size, max_len, embedding_dim), x_eval.shape
        if compute_logit:
            logits_eval = K.eval(logits)
            assert logits_eval.shape == (__test_batch_size, max_len, vocab_size), logits_eval.shape
    return keras.Model(inputs=inputs, outputs=outputs, name='Transformer')
