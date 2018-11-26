import tensorflow.keras as keras
import numpy as np
import tensorflow.keras.backend as K
from data.vocab import TextEncoder
from transformer.embedding import Embedding
from tensorflow.keras.layers import Conv1D, Dropout, Add, Input
from transformer.layers import MultiHeadAttention, Gelu, LayerNormalization


class MultiHeadSelfAttention:
    def __init__(self, n_state: int, n_head: int, attention_dropout: float, use_attn_mask: bool, layer_id: int,
                 neg_inf: float) -> None:
        assert n_state % n_head == 0
        self.c_attn = Conv1D(3 * n_state, 1, name='layer_{}/c_attn'.format(layer_id))
        self.attn = MultiHeadAttention(n_head, n_state, attention_dropout, use_attn_mask,
                                       neg_inf, name='layer_{}/self_attention'.format(layer_id))
        self.c_attn_proj = Conv1D(n_state, 1, name='layer_{}/c_attn_proj'.format(layer_id))

    def __call__(self, x, mask):
        output = self.c_attn(x)
        output = self.attn(output) if mask is None else self.attn([output, mask])
        return self.c_attn_proj(output)


class PositionWiseFF:
    def __init__(self, n_state: int, d_hid: int, layer_id: int, accurate_gelu: bool) -> None:
        self.c_fc = Conv1D(d_hid, 1, name='layer_{}/c_fc'.format(layer_id))
        self.activation = Gelu(accurate=accurate_gelu, name='layer_{}/gelu'.format(layer_id))
        self.c_ffn_proj = Conv1D(n_state, 1, name='layer_{}/c_ffn_proj'.format(layer_id))

    def __call__(self, x):
        output = self.activation(self.c_fc(x))
        return self.c_ffn_proj(output)


class EncoderLayer:
    def __init__(self, n_state: int, n_head: int, d_hid: int, residual_dropout: float, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float, ln_epsilon: float, accurate_gelu: bool) -> None:
        self.attention = MultiHeadSelfAttention(n_state, n_head, attention_dropout, use_attn_mask, layer_id, neg_inf)
        self.drop1 = Dropout(residual_dropout, name='layer_{}/ln_1_drop'.format(layer_id))
        self.add1 = Add(name='layer_{}/ln_1_add'.format(layer_id))
        self.ln1 = LayerNormalization(ln_epsilon, name='layer_{}/ln_1'.format(layer_id))
        self.ffn = PositionWiseFF(n_state, d_hid, layer_id, accurate_gelu)
        self.drop2 = Dropout(residual_dropout, name='layer_{}/ln_2_drop'.format(layer_id))
        self.add2 = Add(name='layer_{}/ln_2_add'.format(layer_id))
        self.ln2 = LayerNormalization(ln_epsilon, name='layer_{}/ln_2'.format(layer_id))

    def __call__(self, x, mask):
        a = self.attention(x, mask)
        n = self.ln1(self.add1([x, self.drop1(a)]))
        f = self.ffn(n)
        return self.ln2(self.add2([n, self.drop2(f)]))


def create_transformer(embedding_dim: int = 768, embedding_dropout: float = 0.1,
                       vocab_size: int = 30000, max_len: int = 512,
                       trainable_pos_embedding: bool = True, num_heads: int = 12, num_layers: int = 12,
                       attention_dropout: float = 0.1, use_one_embedding_dropout: bool = False,
                       d_hid: int = 768 * 4, residual_dropout: float = 0.1,
                       use_attn_mask: bool = True, embedding_layer_norm: bool = False,
                       neg_inf: float = -1e9, ln_epsilon: float = 1e-5, accurate_gelu: bool = False) -> keras.Model:
    vocab_size += TextEncoder.SPECIAL_COUNT
    tokens = Input(batch_shape=(None, max_len), name='token_input', dtype='int32')
    segment_ids = Input(batch_shape=(None, max_len), name='segment_input', dtype='int32')
    pos_ids = Input(batch_shape=(None, max_len), name='position_input', dtype='int32')
    attn_mask = Input(batch_shape=(None, 1, max_len, max_len), name='attention_mask_input',
                      dtype=K.floatx()) if use_attn_mask else None
    inputs = [tokens, segment_ids, pos_ids]
    embedding_layer = Embedding(embedding_dim, embedding_dropout, vocab_size, max_len, trainable_pos_embedding,
                                use_one_embedding_dropout, embedding_layer_norm, ln_epsilon)
    x = embedding_layer(inputs)
    for i in range(num_layers):
        x = EncoderLayer(embedding_dim, num_heads, d_hid, residual_dropout,
                         attention_dropout, use_attn_mask, i, neg_inf, ln_epsilon, accurate_gelu)(x, attn_mask)
    inputs = inputs + ([attn_mask] if use_attn_mask else [])
    return keras.Model(inputs=inputs, outputs=[x], name='Transformer')


def load_bert(base_location: str = './google_bert/model/uncased_L-12_H-768_A-12/', use_attn_mask: bool = True,
              max_len: int = 512) -> keras.Model:
    print(1)
    import tensorflow as tf
    from google_bert.modeling import BertConfig
    print(2)
    bert_config = BertConfig.from_json_file(base_location + 'bert_config.json')
    init_checkpoint = base_location + 'bert_model.ckpt'
    print(3)
    var_names = tf.train.list_variables(init_checkpoint)
    print(4)
    check_point = tf.train.load_checkpoint(init_checkpoint)
    print(5)
    model = create_transformer(embedding_layer_norm=True, neg_inf=-10000.0, use_attn_mask=use_attn_mask,
                               vocab_size=bert_config.vocab_size - TextEncoder.SPECIAL_COUNT, accurate_gelu=True,
                               ln_epsilon=1e-12, max_len=max_len, use_one_embedding_dropout=True,
                               d_hid=bert_config.intermediate_size, embedding_dim=bert_config.hidden_size,
                               num_layers=bert_config.hidden_size, num_heads=bert_config.num_attention_heads,
                               residual_dropout=bert_config.hidden_dropout_prob,
                               attention_dropout=bert_config.attention_probs_dropout_prob)
    print(6)
    if K.backend() == 'tensorflow':
        weights = [np.zeros(w.shape) for w in model.weights]
    else:
        weights = [np.zeros(w.get_value().shape) for w in model.weights]
    for var_name, _ in var_names:
        w_id = None
        qkv = None
        is_pos_embedding = False
        unsqueeze = False
        parts = var_name.split('/')
        first_vars_size = 5
        if parts[1] == 'embeddings':
            n = parts[-1]
            if n == 'token_type_embeddings':  # TODO handle special_tokens
                w_id = 0
            elif n == 'position_embeddings':
                w_id = 1
                is_pos_embedding = True
            elif n == 'word_embeddings':
                w_id = 2
            elif n == 'gamma':
                w_id = 3
            elif n == 'beta':
                w_id = 4
            else:
                raise ValueError()
        elif parts[2].startswith('layer_'):
            layer_number = int(parts[2][len('layer_'):])
            if parts[3] == 'attention':
                if parts[-1] == 'beta':
                    w_id = first_vars_size + layer_number * 12 + 5
                elif parts[-1] == 'gamma':
                    w_id = first_vars_size + layer_number * 12 + 4
                elif parts[-2] == 'dense':
                    if parts[-1] == 'bias':
                        w_id = first_vars_size + layer_number * 12 + 3
                    elif parts[-1] == 'kernel':
                        w_id = first_vars_size + layer_number * 12 + 2
                        unsqueeze = True
                    else:
                        raise ValueError()
                elif parts[-2] == 'key' or parts[-2] == 'query' or parts[-2] == 'value':
                    w_id = first_vars_size + layer_number * 12 + (0 if parts[-1] == 'kernel' else 1)
                    unsqueeze = parts[-1] == 'kernel'
                    qkv = parts[-2][0]
                else:
                    raise ValueError()
            elif parts[3] == 'intermediate':
                if parts[-1] == 'bias':
                    w_id = first_vars_size + layer_number * 12 + 7
                elif parts[-1] == 'kernel':
                    w_id = first_vars_size + layer_number * 12 + 6
                    unsqueeze = True
                else:
                    raise ValueError()
            elif parts[3] == 'output':
                if parts[-1] == 'beta':
                    w_id = first_vars_size + layer_number * 12 + 11
                elif parts[-1] == 'gamma':
                    w_id = first_vars_size + layer_number * 12 + 10
                elif parts[-1] == 'bias':
                    w_id = first_vars_size + layer_number * 12 + 9
                elif parts[-1] == 'kernel':
                    w_id = first_vars_size + layer_number * 12 + 8
                    unsqueeze = True
                else:
                    raise ValueError()

        if w_id is not None and qkv is None:
            print(var_name, ' -> ', model.weights[w_id].name)
            if is_pos_embedding:
                weights[w_id][:max_len, :] = check_point.get_tensor(var_name)[:max_len,
                                             :] if not unsqueeze else check_point.get_tensor(var_name)[
                                                                      None, :max_len, :]
            else:
                weights[w_id][:] = check_point.get_tensor(var_name) if not unsqueeze else \
                    check_point.get_tensor(var_name)[
                        None, ...]
        elif w_id is not None:
            print(var_name, ' -> ', model.weights[w_id].name, '::', qkv)
            p = {'q': 0, 'k': 1, 'v': 2}[qkv]
            if weights[w_id].ndim == 3:
                dim_size = weights[w_id].shape[1]
                weights[w_id][0, :, p * dim_size:(p + 1) * dim_size] = check_point.get_tensor(
                    var_name) if not unsqueeze else \
                    check_point.get_tensor(var_name)[
                        None, ...]
            else:
                dim_size = weights[w_id].shape[0] // 3
                weights[w_id][p * dim_size:(p + 1) * dim_size] = check_point.get_tensor(var_name)
        else:
            print('not mapped: ', var_name)  # TODO pooler, cls/predictions, cls/seq_relationship
    model.set_weights(weights)
    return model
