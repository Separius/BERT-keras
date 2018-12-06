'''This file is for compatibility.'''

import sys


def tpu_compatible():
    '''Fit the tpu problems we meet while using keras tpu model'''
    if not hasattr(tpu_compatible, 'once'):
        tpu_compatible.once = True
    else:
        return
    import tensorflow as tf
    import tensorflow.keras.backend as K
    _version = tf.__version__.split('.')
    is_correct_version = int(_version[0]) >= 1 and (int(_version[0]) >= 2 or int(_version[1]) >= 13)
    from tensorflow.contrib.tpu.python.tpu.keras_support import KerasTPUModel
    def initialize_uninitialized_variables():
        sess = K.get_session()
        uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
        init_op = tf.variables_initializer(
            [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
        )
        sess.run(init_op)

    _tpu_compile = KerasTPUModel.compile

    def tpu_compile(self,
                    optimizer,
                    loss=None,
                    metrics=None,
                    loss_weights=None,
                    sample_weight_mode=None,
                    weighted_metrics=None,
                    target_tensors=None,
                    **kwargs):
        if not is_correct_version:
            raise ValueError('You need tensorflow >= 1.3 for better keras tpu support!')
        _tpu_compile(self, optimizer, loss, metrics, loss_weights,
                     sample_weight_mode, weighted_metrics,
                     target_tensors, **kwargs)
        initialize_uninitialized_variables()  # for unknown reason, we should run this after compile sometimes

    KerasTPUModel.compile = tpu_compile


def replace_keras_to_tf_keras():
    tpu_compatible()
    import tensorflow as tf
    sys.modules['keras'] = tf.keras
    globals()['keras'] = tf.keras
    import keras.backend as K
    K.tf = tf


def clean_keras_module():
    modules = [i for i in sys.modules.keys()]
    for i in modules:
        if i.split('.')[0] == 'keras':
            del sys.modules[i]


def refresh_keras_backend(use_tpu=True):
    clean_keras_module()
    import keras.backend as K
    if use_tpu and K.backend() != 'theano':
        clean_keras_module()
        replace_keras_to_tf_keras()
        import keras.backend as K
    return K


refresh_keras_backend()
