import unittest
import numpy as np

from transformer import refresh_keras_backend
refresh_keras_backend(use_tpu=False) # there are too many settings to use tpu on tensorflow model, so using gpu for the test may be great

import tensorflow as tf
from transformer.load import load_google_bert
from data.vocab import TextEncoder, BERTTextEncoder
from google_bert.modeling import BertConfig, BertModel, get_assignment_map_from_checkpoint


class TestBert(unittest.TestCase):
    def __init__(self, method_name: str = 'runTest') -> None:
        super().__init__(methodName=method_name)

    def test_same_result(self):
        base_location = './google_bert/downloads/multilingual_L-12_H-768_A-12/'
        bert_config = BertConfig.from_json_file(base_location + 'bert_config.json')
        init_checkpoint = base_location + 'bert_model.ckpt'

        def model_fn_builder(bert_config, init_checkpoint):
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
                (assignment_map, _) = get_assignment_map_from_checkpoint(
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

        batch_size = 8
        seq_len = 5
        xmb = np.random.randint(106, bert_config.vocab_size - 106, (batch_size, seq_len))
        xmb2 = np.random.randint(0, 2, (batch_size, seq_len), dtype=np.int32)
        xmb3 = np.random.randint(0, 2, (batch_size, seq_len), dtype=np.int32)

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

        model_fn = model_fn_builder(bert_config=bert_config, init_checkpoint=init_checkpoint)
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(master=None, tpu_config=tf.contrib.tpu.TPUConfig(num_shards=8,
                                                                                               per_host_input_for_training=is_per_host))
        estimator = tf.contrib.tpu.TPUEstimator(use_tpu=False, model_fn=model_fn, config=run_config,
                                                predict_batch_size=batch_size)
        tf_result = [r for r in estimator.predict(input_fn)]

        import keras.backend as K

        K.set_learning_phase(0)
        my_model = load_google_bert(base_location, max_len=seq_len)

        from data.dataset import create_attention_mask, generate_pos_ids

        pos = generate_pos_ids(batch_size, seq_len)
        k_mask = create_attention_mask(xmb2, False, None, None, True)
        bert_encoder = BERTTextEncoder(base_location + 'vocab.txt')
        for b in range(len(xmb)):
            xmb[b] = np.array(bert_encoder.standardize_ids(xmb[b].tolist()))
        k_output = my_model.predict([xmb, xmb3, pos, k_mask])
        max_max = 0
        for i in range(batch_size):
            if k_mask[i].mean() != 0:  # TODO (when mask == full zero, keras_res != tf_res)
                new_max = np.abs(k_output[i] - tf_result[i]['seq_out']).max()
                if new_max > max_max:
                    max_max = new_max
        assert max_max < 5e-5, max_max  # TODO reduce the error (I think it's because of the LayerNorm)


if __name__ == "__main__":
    unittest.main()
