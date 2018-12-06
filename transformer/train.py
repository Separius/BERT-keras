import keras
import numpy as np
import keras.backend as K
from typing import List, Generator, Optional
from keras.layers import Dropout, Input, Lambda, TimeDistributed, Dense
from data.dataset import TaskMetadata, SentenceBatch, create_attention_mask, generate_pos_ids


def _mask_loss(y_true, y_pred, y_mask, element_wise_loss):
    l = K.switch(y_mask, element_wise_loss(y_true, y_pred), K.zeros_like(y_mask, dtype=K.floatx()))
    return K.sum(l) / (K.cast(K.sum(y_mask), dtype='float32') + K.epsilon())


def classification_loss(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


def masked_classification_loss(y_true, y_pred, y_mask):
    return _mask_loss(y_true, y_pred, y_mask, classification_loss)


def sparse_gather(y_pred, target_indices, task_name):
    clf_h = Lambda(lambda x: K.reshape(x, (-1, K.int_shape(x)[-1])), name=task_name + '_flatten')(y_pred)
    return Lambda(lambda x: K.gather(x[0], K.cast(x[1], 'int32')), name=task_name + '_gather')([clf_h, target_indices])


def pass_through_loss(y_true, y_pred):
    return y_pred


def load_model(weights_path: str, base_model: keras.Model, tasks_meta_data: List[TaskMetadata]):
    model = train_model(base_model, is_causal=False, tasks_meta_data=tasks_meta_data,
                        pretrain_generator=None, finetune_generator=None)
    model.load_weights(weights_path)
    return model


def train_model(base_model: keras.Model, is_causal: bool, tasks_meta_data: List[TaskMetadata], pretrain_generator,
                finetune_generator, pretrain_epochs: int = 1, pretrain_optimizer='adam', pretrain_steps: int = 1000000,
                pretrain_callbacks=None, finetune_epochs: int = 1, finetune_optimizer='adam',
                finetune_steps: int = 10000, finetune_callbacks=None, verbose: int = 0,
                TPUStrategy: Optional['tf.contrib.tpu.TPUDistributionStrategy'] = None):
    if TPUStrategy is not None:
        import tensorflow as tf
    token_input = base_model.inputs[0]
    segment_input = base_model.inputs[1]
    position_input = base_model.inputs[2]
    uses_attn_mask = len(base_model.inputs) == 4
    max_len = K.int_shape(base_model.inputs[0])[1]
    if uses_attn_mask:
        attention_mask_input = base_model.inputs[3]
    all_logits = []
    all_tasks = {task.name: task for task in tasks_meta_data}
    task_nodes = {}
    sent_level_mask_inputs = []
    assert len(all_tasks) == len(tasks_meta_data)
    for task in all_tasks.values():
        task_loss_weight = Input(batch_shape=(None, 1), dtype='float32', name=task.name + '_loss_weight')
        if task.is_token_level:
            if task.name == 'lm':
                decoder = Lambda(lambda x: K.dot(x, K.transpose(base_model.get_layer('TokenEmbedding').weights[0])),
                                 name='lm_logits')
            else:
                decoder = Dense(units=task.num_classes, name=task.name + '_logits')
            logits = TimeDistributed(decoder, name=task.name + '_logits_time_distributed')(
                Dropout(task.dropout)(base_model.outputs[0]))
            task_target = Input(batch_shape=(None, max_len,), dtype='int32', name=task.name + '_target_input')
            task_mask = Input(batch_shape=(None, max_len), dtype='int8' if TPUStrategy is None else 'int32',
                              name=task.name + '_mask_input')
            task_loss = Lambda(lambda x: x[0] * masked_classification_loss(x[1], x[2], x[3]), name=task.name + '_loss')(
                [task_loss_weight, task_target, logits, task_mask])
        else:
            task_mask = Input(batch_shape=(None, 1), dtype='int32', name=task.name + '_mask_input')
            decoder_input = sparse_gather(base_model.outputs[0], task_mask, task.name)
            logits = Dense(units=task.num_classes, name=task.name + '_logits')(Dropout(task.dropout)(decoder_input))
            task_target = Input(batch_shape=(None, 1), dtype='int32', name=task.name + '_target_input')
            task_loss = Lambda(lambda x: x[0] * classification_loss(x[1], x[2]), name=task.name + '_loss')(
                [task_loss_weight, task_target, logits])
            sent_level_mask_inputs.append(task_mask)
        task_nodes[task.name] = {
            'target': task_target,
            'mask': task_mask,
            'loss_weight': task_loss_weight,
            'loss': task_loss,
        }
        all_logits.append(logits)

    def get_generator(sentence_generator: Generator[SentenceBatch, None, None], is_pretrain: bool):
        for i, batch in enumerate(sentence_generator):
            batch_size, seq_len = batch.tokens.shape
            x = [batch.tokens, batch.segments, generate_pos_ids(batch_size, max_len)]
            y = []
            if uses_attn_mask:
                x.append(create_attention_mask(batch.padding_mask, is_causal))
            for task_name in task_nodes.keys():
                if is_pretrain:
                    cond = all_tasks[task_name].weight_scheduler.active_in_pretrain
                else:
                    cond = all_tasks[task_name].weight_scheduler.active_in_finetune
                if cond:
                    if task_name in batch.sentence_classification:
                        task_data_batch = batch.sentence_classification[task_name]
                    else:
                        task_data_batch = batch.token_classification[task_name]
                    x.append(task_data_batch.target)
                    if all_tasks[task_name].is_token_level:
                        x.append(task_data_batch.target_mask)
                    else:
                        x.append((task_data_batch.target_mask + np.arange(batch_size) * seq_len).astype(np.int32))
                    x.append(
                        np.repeat(np.array(
                            [all_tasks[task_name].weight_scheduler.get(is_pretrain, i)]), batch_size,
                            0))
                    y.append(np.repeat(np.array([0.0]), batch_size, 0))
            yield x, y

    def train_step(is_pretrain: bool):
        _inputs = [token_input, segment_input, position_input]
        _outputs = []
        if uses_attn_mask:
            _inputs.append(attention_mask_input)
        for task_name in task_nodes.keys():
            if is_pretrain:
                cond = all_tasks[task_name].weight_scheduler.active_in_pretrain
            else:
                cond = all_tasks[task_name].weight_scheduler.active_in_finetune
            if cond:
                _inputs.append(task_nodes[task_name]['target'])
                _inputs.append(task_nodes[task_name]['mask'])
                _inputs.append(task_nodes[task_name]['loss_weight'])
                _outputs.append(task_nodes[task_name]['loss'])
        _generator = get_generator(pretrain_generator if is_pretrain else finetune_generator, is_pretrain)
        _model = keras.Model(inputs=_inputs, outputs=_outputs)
        if TPUStrategy is not None:
            '''
            Create TPUStrategy like this:
            tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
            TPUStrategy = tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
            )
            '''
            _model = tf.contrib.tpu.keras_to_tpu_model(_model, strategy=TPUStrategy)
        _model.compile(pretrain_optimizer if is_pretrain else finetune_optimizer, loss=pass_through_loss)
        _model.fit_generator(_generator, steps_per_epoch=pretrain_steps if is_pretrain else finetune_steps,
                             verbose=verbose, callbacks=pretrain_callbacks if is_pretrain else finetune_callbacks,
                             shuffle=False, epochs=pretrain_epochs if is_pretrain else finetune_epochs)

    if pretrain_generator is not None:
        train_step(True)
    if finetune_generator is not None:
        train_step(False)

    ret_model = keras.Model(inputs=base_model.inputs + sent_level_mask_inputs, outputs=all_logits)
    if TPUStrategy is not None:
        ret_model = tf.contrib.tpu.keras_to_tpu_model(ret_model, strategy=TPUStrategy)
        # Compile for TPU model predicting for the first time. Also you can call compile for training after this
        ret_model.compile(finetune_optimizer, loss=pass_through_loss)
    return ret_model
