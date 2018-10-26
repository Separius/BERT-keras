import json
import keras
import numpy as np
import keras.backend as K
from data.vocab import TextEncoder
from transformer.config import BertConfig
from transformer.embedding import Embedding
from transformer.layers import MultiHeadAttention, Gelu, LayerNormalization
from keras.layers import Conv1D, Dropout, Add, Input, Lambda, TimeDistributed


class MultiHeadSelfAttention:
    def __init__(self, n_state: int, n_head: int, attention_dropout: float, use_attn_mask: bool, layer_id: int) -> None:
        assert n_state % n_head == 0
        self.c_attn = Conv1D(3 * n_state, 1, name='layer_{}/c_attn'.format(layer_id))
        self.attn = MultiHeadAttention(n_head, n_state, attention_dropout, use_attn_mask,
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
                 use_attn_mask: bool, layer_id: int, **kwargs) -> None:
        self.attention = MultiHeadSelfAttention(n_state, n_head, attention_dropout, use_attn_mask, layer_id)
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


def load_openai_transformer(path: str = './openai/model/', use_attn_mask: bool = True,
                            use_one_embedding_dropout: bool = False, max_len: int = 512) -> keras.Model:
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
    model = create_transformer(embedding_dim=768, embedding_dropout=0.1, vocab_size=40478 + TextEncoder.SPECIAL_COUNT,
                               max_len=min(512, max_len), use_attn_mask=use_attn_mask, trainable_pos_embedding=True,
                               num_heads=12, num_layers=12, use_one_embedding_dropout=use_one_embedding_dropout,
                               d_hid=4 * 768, attention_dropout=0.1, residual_dropout=0.1)
    model.set_weights(init_params)
    return model


def create_transformer(embedding_dim: int = 768, embedding_dropout: float = 0.1,
                       vocab_size: int = 30000 + TextEncoder.SPECIAL_COUNT, max_len: int = 512,
                       trainable_pos_embedding: bool = True, num_heads: int = 12, num_layers: int = 12,
                       attention_dropout: float = 0.1, use_one_embedding_dropout: bool = BertConfig.USE_ONE_DROPOUT,
                       d_hid: int = 768 * 4, residual_dropout: float = 0.1,
                       use_attn_mask: bool = BertConfig.USE_ATTN_MASK) -> keras.Model:
    tokens = Input(batch_shape=(None, max_len), name='token_input', dtype=np.int32)
    segment_ids = Input(batch_shape=(None, max_len), name='segment_input', dtype=np.int32)
    pos_ids = Input(batch_shape=(None, max_len), name='position_input', dtype=np.int32)
    attn_mask = Input(batch_shape=(None, 1, max_len, max_len), name='attention_mask_input',
                      dtype=K.floatx()) if use_attn_mask else None
    inputs = [tokens, segment_ids, pos_ids]
    embedding_layer = Embedding(embedding_dim, embedding_dropout, vocab_size, max_len, trainable_pos_embedding,
                                use_one_embedding_dropout)
    x = embedding_layer(inputs)
    for i in range(num_layers):
        x = EncoderLayer(embedding_dim, num_heads, d_hid, residual_dropout,
                         attention_dropout, use_attn_mask, i)(x, attn_mask)
    inputs = inputs + ([attn_mask] if use_attn_mask else [])
    return keras.Model(inputs=inputs, outputs=x, name='Transformer')


def create_lm(given_model=None):
    if given_model is None:
        given_model = load_openai_transformer()
    logits = TimeDistributed(Lambda(
        lambda x: K.dot(x, K.transpose(
            given_model.get_layer('TokenEmbedding').weights[0])), name='logits'))(
        given_model.outputs[0])
    return keras.Model(inputs=given_model.inputs, outputs=logits, name='TransformerLM')


def train_model(base_model, pretrain_generator, pretrain_steps, finetune_generator, finetune_steps,
                pretrain_compile_kwargs, finetune_compile_kwargs, task_list, saving_config):
    # create all the dense classifiers and based on weights make place holders for them
    # create pretrain keras.model; compile it
    # for step times get batch and train (hopefully with some loss monitoring and stuff)
    # create finetune keras.model; compile it
    # for step times get batch and train (hopefully with some loss monitoring and stuff)
    pass
