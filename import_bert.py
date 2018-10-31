import numpy as np
import tensorflow as tf
from data.dataset import TextEncoder
from transformer.model import create_transformer

base_location = '/home/sepehr/Downloads/uncased_L-12_H-768_A-12/'
pre_ckpt = base_location + 'bert_model.ckpt'

var_names = tf.train.list_variables(pre_ckpt)
check_point = tf.train.load_checkpoint(pre_ckpt)
my_model = create_transformer(embedding_layer_norm=True, neg_inf=-10000.0, vocab_size=30522 - TextEncoder.SPECIAL_COUNT,
                              ln_epsilon=1e-12)
# our weights: seg, pos, token, emb_ln_g, emb_ln_b
#   per layer: qkv_w, qkv_b, aproj_w, aproj_b, att_ln_g, att_ln_b,
#               fc_w, fc_b, fc_proj_w, fc_proj_b, fc_ln_g, fc_ln_b

# tf weights: emb_ln_b, emb_ln_g, pos, seg, token,
#  per layer: att_ln_b, att_ln_g, aproj_b, aproj_w, kb, kw, qb, qw, vb, vw
#               fc_b, fc_w, fc_ln_b, fc_ln_g, fc_proj_b, fc_proj_w

weights = [np.zeros(w.shape) for w in my_model.weights]

for var_name, _ in var_names:
    w_id = None
    qkv = None
    unsqueeze = False
    parts = var_name.split('/')
    if parts[1] == 'embeddings':
        n = parts[-1]
        if n == 'token_type_embeddings':  # TODO handle special_tokens
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
                w_id = 5 + layer_number * 12 + 5
            elif parts[-1] == 'gamma':
                w_id = 5 + layer_number * 12 + 4
            elif parts[-2] == 'dense':
                if parts[-1] == 'bias':
                    w_id = 5 + layer_number * 12 + 3
                elif parts[-1] == 'kernel':
                    w_id = 5 + layer_number * 12 + 2
                    unsqueeze = True
                else:
                    raise ValueError()
            elif parts[-2] == 'key' or parts[-2] == 'query' or parts[-2] == 'value':
                w_id = 5 + layer_number * 12 + (0 if parts[-1] == 'kernel' else 1)
                unsqueeze = parts[-1] == 'kernel'
                qkv = parts[-2][0]
            else:
                raise ValueError()
        elif parts[3] == 'intermediate':
            if parts[-1] == 'bias':
                w_id = 5 + layer_number * 12 + 7
            elif parts[-1] == 'kernel':
                w_id = 5 + layer_number * 12 + 6
                unsqueeze = True
            else:
                raise ValueError()
        elif parts[3] == 'output':
            if parts[-1] == 'beta':
                w_id = 5 + layer_number * 12 + 11
            elif parts[-1] == 'gamma':
                w_id = 5 + layer_number * 12 + 10
            elif parts[-1] == 'bias':
                w_id = 5 + layer_number * 12 + 9
            elif parts[-1] == 'kernel':
                w_id = 5 + layer_number * 12 + 8
                unsqueeze = True
            else:
                raise ValueError()

    if w_id is not None and qkv is None:
        print(var_name, ' -> ', my_model.weights[w_id].name)
        weights[w_id][:] = check_point.get_tensor(var_name) if not unsqueeze else check_point.get_tensor(var_name)[
            None, ...]
    elif w_id is not None:
        print(var_name, ' -> ', my_model.weights[w_id].name, '::', qkv)
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

from google.modeling import BertModel, BertConfig, get_assigment_map_from_checkpoint

sess = tf.Session()
bert_config = BertConfig.from_json_file(base_location + 'bert_config.json')
batch_size = 3
seq_len = 11
input_ids = tf.placeholder(tf.int32, [batch_size, seq_len])
input_mask = tf.placeholder(tf.int32, [batch_size, seq_len])
segment_ids = tf.placeholder(tf.int32, [batch_size, seq_len])


def get_google_model(input_ids, input_mask, segment_ids):
    google_model = BertModel(config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask,
                             token_type_ids=segment_ids, use_one_hot_embeddings=False)
    return google_model.get_sequence_output()


res = get_google_model(input_ids, input_mask, segment_ids)
tvars = tf.trainable_variables()
(assignment_map, initialized_variable_names) = get_assigment_map_from_checkpoint(tvars,
                                                                                 pre_ckpt)
tf.train.init_from_checkpoint(pre_ckpt, assignment_map)
sess.run(tf.global_variables_initializer())  # TODO this is wrong
sess.run(tf.local_variables_initializer())
xmb = np.random.randint(0, 10000, (batch_size, seq_len))
xmb2 = np.random.randint(0, 2, (batch_size, seq_len))
xmb3 = np.random.randint(0, 2, (batch_size, seq_len))
tf_result = sess.run(res, {input_ids: xmb, input_mask: xmb2, segment_ids: xmb3})
