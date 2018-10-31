import numpy as np
import tensorflow as tf
from google.modeling import BertConfig, BertModel, get_assigment_map_from_checkpoint

base_location = '/home/sepehr/Downloads/uncased_L-12_H-768_A-12/'
bert_config = BertConfig.from_json_file(base_location + 'bert_config.json')
init_checkpoint = base_location + 'bert_model.ckpt'


def model_fn_builder(bert_config, init_checkpoint, layer_indexes):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=False)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map, _) = get_assigment_map_from_checkpoint(
            tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        predictions = {
            "unique_id": unique_ids,
            "seq_out": model.get_sequence_output()
        }

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


batch_size = 3
seq_len = 5
xmb = np.random.randint(0, 10000, (batch_size, seq_len))
xmb2 = np.ones((batch_size, seq_len), dtype=np.int32)
xmb3 = np.ones((batch_size, seq_len), dtype=np.int32)


# xmb2 = np.random.randint(0, 2, (batch_size, seq_len))
# xmb3 = np.random.randint(0, 2, (batch_size, seq_len))


def input_fn(params):
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant([0, 1, 2], shape=[batch_size], dtype=tf.int32),
        "input_ids":
            tf.constant(
                xmb, shape=[batch_size, seq_len],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                xmb2,
                shape=[batch_size, seq_len],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                xmb3,
                shape=[batch_size, seq_len],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d


model_fn = model_fn_builder(bert_config=bert_config, init_checkpoint=init_checkpoint, layer_indexes=[0])
is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(master=None, tpu_config=tf.contrib.tpu.TPUConfig(num_shards=8,
                                                                                       per_host_input_for_training=is_per_host))
estimator = tf.contrib.tpu.TPUEstimator(use_tpu=False, model_fn=model_fn, config=run_config,
                                        predict_batch_size=batch_size)
tf_result = [r['seq_out'] for r in estimator.predict(input_fn)]

from data.dataset import TextEncoder
from transformer.model import create_transformer

pre_ckpt = base_location + 'bert_model.ckpt'

var_names = tf.train.list_variables(pre_ckpt)
check_point = tf.train.load_checkpoint(pre_ckpt)
my_model = create_transformer(embedding_layer_norm=True, neg_inf=-10000.0, vocab_size=30522 - TextEncoder.SPECIAL_COUNT,
                              ln_epsilon=1e-12, max_len=seq_len)
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
    is_pos_embedding = False
    unsqueeze = False
    parts = var_name.split('/')
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
        if is_pos_embedding:
            weights[w_id][:seq_len, :] = check_point.get_tensor(var_name)[:seq_len,
                                         :] if not unsqueeze else check_point.get_tensor(var_name)[
                                                                  None, :seq_len, :]
        else:
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

import keras.backend as K

K.set_learning_phase(0)
from data.dataset import create_attention_mask, generate_pos_ids

pos = generate_pos_ids(batch_size, seq_len)

k_output = my_model.predict([xmb, xmb3, pos, create_attention_mask(xmb2, False, None, None)])
print(k_output[0] - tf_result[0])
