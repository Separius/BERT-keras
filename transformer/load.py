import json
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
from data.vocab import TextEncoder
from google_bert.modeling import BertConfig
from transformer.model import create_transformer


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
    model = create_transformer(embedding_dim=768, embedding_dropout=0.1, vocab_size=40478,
                               max_len=min(512, max_len), use_attn_mask=use_attn_mask, trainable_pos_embedding=True,
                               num_heads=12, num_layers=12, use_one_embedding_dropout=use_one_embedding_dropout,
                               d_hid=4 * 768, attention_dropout=0.1, residual_dropout=0.1)
    model.set_weights(init_params)
    return model


def load_google_bert(base_location: str = './google_bert/downloads/multilingual_L-12_H-768_A-12/',
                     use_attn_mask: bool = True, max_len: int = 512, verbose: bool = False) -> keras.Model:
    bert_config = BertConfig.from_json_file(base_location + 'bert_config.json')
    init_checkpoint = base_location + 'bert_model.ckpt'
    var_names = tf.train.list_variables(init_checkpoint)
    check_point = tf.train.load_checkpoint(init_checkpoint)
    vocab_size = bert_config.vocab_size - TextEncoder.BERT_SPECIAL_COUNT - TextEncoder.BERT_UNUSED_COUNT
    model = create_transformer(embedding_layer_norm=True, neg_inf=-10000.0, use_attn_mask=use_attn_mask,
                               vocab_size=vocab_size, accurate_gelu=True, layer_norm_epsilon=1e-12, max_len=max_len,
                               use_one_embedding_dropout=True, d_hid=bert_config.intermediate_size,
                               embedding_dim=bert_config.hidden_size, num_layers=bert_config.num_hidden_layers,
                               num_heads=bert_config.num_attention_heads,
                               residual_dropout=bert_config.hidden_dropout_prob,
                               attention_dropout=bert_config.attention_probs_dropout_prob)
    if K.backend() == 'tensorflow':
        weights = [np.zeros(w.shape) for w in model.weights]
    else:
        weights = [np.zeros(w.get_value().shape) for w in model.weights]
    for var_name, _ in var_names:
        w_id = None
        qkv = None
        unsqueeze = False
        parts = var_name.split('/')
        first_vars_size = 5
        if parts[1] == 'embeddings':
            n = parts[-1]
            if n == 'token_type_embeddings':
                w_id = 0
            elif n == 'position_embeddings':
                w_id = 1
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
            if verbose:
                print(var_name, ' -> ', model.weights[w_id].name)
            if w_id == 1:  # pos embedding
                weights[w_id][:max_len, :] = check_point.get_tensor(var_name)[:max_len,
                                             :] if not unsqueeze else check_point.get_tensor(var_name)[
                                                                      None, :max_len, :]
            elif w_id == 2:  # word embedding
                # ours: unk, [vocab], pad, msk(mask), bos(cls), del(use sep again), eos(sep)
                # theirs: pad, 99 unused, unk, cls, sep, mask, [vocab]
                saved = check_point.get_tensor(var_name)  # vocab_size, emb_size
                # weights[our_position] = saved[their_position]
                weights[w_id][0] = saved[1 + TextEncoder.BERT_UNUSED_COUNT]  # unk
                weights[w_id][1:vocab_size] = saved[-vocab_size + 1:]
                weights[w_id][vocab_size + TextEncoder.PAD_OFFSET] = saved[0]
                weights[w_id][vocab_size + TextEncoder.MSK_OFFSET] = saved[4 + TextEncoder.BERT_UNUSED_COUNT]
                weights[w_id][vocab_size + TextEncoder.BOS_OFFSET] = saved[2 + TextEncoder.BERT_UNUSED_COUNT]
                weights[w_id][vocab_size + TextEncoder.DEL_OFFSET] = saved[3 + TextEncoder.BERT_UNUSED_COUNT]
                weights[w_id][vocab_size + TextEncoder.EOS_OFFSET] = saved[3 + TextEncoder.BERT_UNUSED_COUNT]
            else:
                weights[w_id][:] = check_point.get_tensor(var_name) if not unsqueeze else \
                    check_point.get_tensor(var_name)[
                        None, ...]
        elif w_id is not None:
            if verbose:
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
            if verbose:
                print('not mapped: ', var_name)  # TODO pooler, cls/predictions, cls/seq_relationship
    model.set_weights(weights)
    return model
