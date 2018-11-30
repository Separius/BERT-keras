import os
import uuid
import json

from transformer import refresh_keras_backend
refresh_keras_backend(use_tpu=False) # tpu mode doesn't support switch backend to theano

import keras
import numpy as np
from importlib import reload
from keras import backend as K
from data.vocab import TextEncoder
from unittest import TestCase, SkipTest
from data.lm_dataset import dummy_lm_generator
from transformer.train import train_model, load_model
from transformer.model import create_transformer
from transformer.load import load_openai_transformer
from transformer.layers import MultiHeadAttention, LayerNormalization, Gelu
from data.dataset import create_attention_mask, TaskMetadata, TaskWeightScheduler


def set_keras_backend(backend):
    global K
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend


class TestTransformer(TestCase):
    def __init__(self, method_name: str = 'runTest') -> None:
        super().__init__(methodName=method_name)
        self.vocab_size = 23
        self.num_heads = 2
        self.num_layers = 2
        self.embedding_dim = 6
        self.d_hid = 12
        self.max_len = 7
        self.supported_backends = {'tensorflow', 'theano'}
        self.original_backend = K.backend()

    def tearDown(self):
        set_keras_backend(self.original_backend)

    def list_backends(self, orig_backend=None):
        if orig_backend is None:
            orig_backend = K.backend()
        # always start from the default backend
        return [orig_backend] + list(self.supported_backends - {orig_backend})

    def create_small_model(self, use_attn_mask: bool):
        return create_transformer(vocab_size=self.vocab_size,
                                  num_heads=self.num_heads, num_layers=self.num_layers,
                                  embedding_dim=self.embedding_dim, d_hid=self.d_hid,
                                  max_len=self.max_len, use_attn_mask=use_attn_mask)

    @staticmethod
    def compare_two_models(model_a, model_b):
        assert len(model_a.weights) == len(model_b.weights)
        for x, y in zip(model_a.weights, model_b.weights):
            assert (K.eval(x) == K.eval(y)).all()

    def test_train(self):
        model = self.create_small_model(use_attn_mask=True)
        batch_size = 3
        generator = dummy_lm_generator(self.vocab_size, self.max_len, batch_size, 10000, False)
        tasks_meta_data = [TaskMetadata('lm', True, self.vocab_size + TextEncoder.SPECIAL_COUNT, 0.1,
                                        TaskWeightScheduler(True, False)),
                           TaskMetadata('lm_untied', True, self.vocab_size + TextEncoder.SPECIAL_COUNT, 0.3,
                                        TaskWeightScheduler(False, True)),
                           TaskMetadata('count', False, 2, 0.1, TaskWeightScheduler(True, True))]
        model = train_model(model, True, tasks_meta_data, generator, generator, pretrain_steps=100, pretrain_epochs=3,
                            finetune_steps=50, finetune_epochs=2, verbose=0)
        path = '/tmp/{}.model'.format(uuid.uuid4())
        model.save_weights(path)
        loaded_model = load_model(path, self.create_small_model(use_attn_mask=True), tasks_meta_data)
        assert len(model.inputs) == len(loaded_model.inputs)
        assert len(model.outputs) == len(loaded_model.outputs)
        self.compare_two_models(model, loaded_model)

    def test_save_load_all(self):
        for backend in self.list_backends():
            try:
                set_keras_backend(backend)
            except ModuleNotFoundError:
                continue
            K.set_learning_phase(0)  # test
            for use_attn_mask in [True, False]:
                model = self.create_small_model(use_attn_mask)
                path = '/tmp/{}.model'.format(uuid.uuid4())
                try:
                    model.save(path)
                    new_model = keras.models.load_model(path, custom_objects={'MultiHeadAttention': MultiHeadAttention,
                                                                              'LayerNormalization': LayerNormalization,
                                                                              'Gelu': Gelu})
                    TestTransformer.compare_two_models(model, new_model)
                except Exception as e:
                    raise e
                finally:
                    if os.path.exists(path):
                        os.remove(path)

    def test_save_load_weights(self):
        for backend in self.list_backends():
            try:
                set_keras_backend(backend)
            except ModuleNotFoundError:
                continue
            K.set_learning_phase(0)  # test
            for use_attn_mask in [True, False]:
                model = self.create_small_model(use_attn_mask)
                path = '/tmp/{}.model'.format(uuid.uuid4())
                try:
                    model.save_weights(path)
                    model.load_weights(path)
                except Exception as e:
                    raise e
                finally:
                    if os.path.exists(path):
                        os.remove(path)

    def test_same_result(self):
        orig_backend = K.backend()
        batch_size = 3
        xmb = np.random.randint(0, self.vocab_size, (batch_size, self.max_len, 2), dtype=np.int32)
        xmb[:, :, 1] = np.random.randint(0, self.max_len, (batch_size, self.max_len), dtype=np.int32)
        for use_attn_mask in [True, False]:
            inputs = [xmb[:, :, 0], np.zeros((batch_size, self.max_len), dtype=np.int32), xmb[:, :, 1]]
            results_x = {}
            if use_attn_mask:
                mask = create_attention_mask(None, True, batch_size, self.max_len)
                inputs.append(mask)
            for backend in self.list_backends(orig_backend):
                try:
                    set_keras_backend(backend)
                except ModuleNotFoundError:
                    continue
                K.set_learning_phase(0)  # test
                model = self.create_small_model(use_attn_mask)
                model = load_openai_transformer(use_attn_mask=use_attn_mask, max_len=self.max_len,
                                                use_one_embedding_dropout=True)
                results_x[backend] = model.predict(inputs, batch_size=batch_size)
                del model
            set_keras_backend(orig_backend)
            for k1 in results_x.keys():
                for k2 in results_x.keys():
                    if k1 == k2:
                        continue
                    assert np.allclose(results_x[k1], results_x[k2], atol=1.e-4, rtol=1.e-4)

    def test_different_backends_work(self):
        for use_attn_mask in [True, False]:
            orig_backend = K.backend()
            for backend in self.list_backends(orig_backend):
                try:
                    set_keras_backend(backend)
                except ModuleNotFoundError:
                    pass
                K.set_learning_phase(0)  # test
                model = self.create_small_model(use_attn_mask)
                del model
            set_keras_backend(orig_backend)

    def test_different_backends_load_openai(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise SkipTest('tensorflow is not installed, so we can not compare results with the released model')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from openai.train import dropout, embed, block, find_trainable_variables

        n_vocab = 40478
        n_ctx = 7
        n_embd = 768
        embd_pdrop = 0.1
        n_layer = 12
        n_batch_train = 2
        n_transfer = 1 + 12 * 12

        def model(X, train=False, reuse=False):
            with tf.variable_scope('model', reuse=reuse):
                we = tf.get_variable("we", [n_vocab + TextEncoder.SPECIAL_COUNT + n_ctx, n_embd],
                                     initializer=tf.random_normal_initializer(stddev=0.02))
                we = dropout(we, embd_pdrop, train)
                h = embed(X, we)
                for layer in range(n_layer):
                    h = block(h, 'h%d' % layer, train=train, scale=True)
                return h

        X_train = tf.placeholder(tf.int32, [n_batch_train, n_ctx, 2])
        res = model(X_train)

        params = find_trainable_variables('model')
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.global_variables_initializer())

        with open('openai/model/params_shapes.json') as f:
            shapes = json.load(f)
        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        init_params = [np.load('openai/model/params_{}.npy'.format(n)) for n in range(10)]
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        init_params[0] = init_params[0][:n_ctx]
        init_params[0] = np.concatenate(
            [init_params[1], (np.random.randn(TextEncoder.SPECIAL_COUNT, n_embd) * 0.02).astype(np.float32),
             init_params[0]], 0)
        del init_params[1]

        sess.run([p.assign(ip) for p, ip in zip(params[:n_transfer], init_params[:n_transfer])])
        xmb = np.random.randint(0, n_vocab, (n_batch_train, n_ctx, 2))
        xmb[:, :, 1] = np.random.randint(0, n_ctx, (n_batch_train, n_ctx))
        xmb_tf = xmb.copy()
        xmb_tf[:, :, 1] += n_vocab + TextEncoder.SPECIAL_COUNT
        tf_result = sess.run(res, {X_train: xmb_tf})

        for backend in self.list_backends():
            try:
                set_keras_backend(backend)
            except ModuleNotFoundError:
                continue
            K.set_learning_phase(0)
            keras_model = load_openai_transformer(use_attn_mask=True, use_one_embedding_dropout=False, max_len=n_ctx)
            mask = create_attention_mask(None, True, n_batch_train, n_ctx)
            k_result = keras_model.predict(
                [xmb[:, :, 0], np.zeros((n_batch_train, n_ctx), dtype=np.int64), xmb[:, :, 1], mask],
                batch_size=n_batch_train)

            if K.backend() != 'tensorflow':
                assert np.allclose(tf_result, k_result, atol=1.e-4, rtol=1.e-4)
            else:
                assert (tf_result == k_result).all()
