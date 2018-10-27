import keras
import numpy as np
import keras.backend as K
from typing import List, Generator
from keras.layers import Dropout, Input, Lambda, TimeDistributed, Dense
from data.dataset import TaskMetadata, SentenceBatch, create_attention_mask


def _mask_loss(y_true, y_pred, y_mask, element_wise_loss):
    l = K.switch(y_mask, element_wise_loss(y_true, y_pred), K.zeros_like(y_mask, dtype=K.floatx()))
    return K.sum(l)  # / (K.sum(y_mask) + K.epsilon()) # TODO


def classification_loss(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


def masked_classification_loss(y_true, y_pred, y_mask):
    return _mask_loss(y_true, y_pred, y_mask, classification_loss)


# TODO
def sparse_gather(y_pred, target_indices, seq_len):
    clf_h = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    return K.gather(clf_h, K.arange(K.int_shape(target_indices)[0], dtype='int32') * seq_len + target_indices)


def model_loss(y_true, y_pred):
    return y_pred


def train_model(base_model: keras.Model, is_causal: bool,
                tasks_meta_data: List[TaskMetadata], pretrain_generator, finetune_generator,
                pretrain_optimizer='adam', pretrain_steps: int = 1000000, pretrain_callbacks=None,
                finetune_optimizer='adam', finetune_steps: int = 10000, finetune_callbacks=None):
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
    assert len(all_tasks) == len(tasks_meta_data)
    for task in all_tasks.values():
        task_loss_weight = Input(batch_shape=(1,), dtype='float32', name=task.name + '_loss_weight')
        if task.is_token_level:
            if task.name == 'lm':
                decoder = Lambda(lambda x: K.dot(x, K.transpose(base_model.get_layer('TokenEmbedding').weights[0])),
                                 name='lm_logits')
            else:
                decoder = Dense(units=task.num_classes, name=task.name + '_logits')
            logits = TimeDistributed(decoder, name=task.name + '_logits_time_distributed')(
                Dropout(task.dropout)(base_model.outputs[0]))
            task_target = Input(batch_shape=(None, max_len,), dtype='int32', name=task.name + '_target_input')
            task_mask = Input(batch_shape=(None, max_len), dtype='int8', name=task.name + '_mask_input')
            task_loss = Lambda(lambda x: x[0] * masked_classification_loss(x[1], x[2], x[3]))(
                [task_loss_weight, task_target, logits, task_mask])
            # task_loss = task_loss_weight * masked_classification_loss(task_target, logits,
            #                                                           task_mask)  # TODO make this lambda
        else:
            task_mask = Input(batch_shape=(None,), dtype='int32', name=task.name + '_mask_input')
            decoder_input = sparse_gather(base_model.outputs[0], task_mask, max_len)
            logits = Dense(units=task.num_classes, name=task.name + '_logits')(Dropout(task.dropout)(decoder_input))
            task_target = Input(batch_shape=(None,), dtype='int32', name=task.name + '_target_input')
            task_loss = task_loss_weight * classification_loss(task_target, logits)  # TODO make this lambda
        task_nodes[task.name] = {
            'target': task_target,
            'mask': task_mask,
            'loss_weight': task_loss_weight,
            'loss': task_loss
        }
        all_logits.append(logits)

    def get_generator(sentence_generator: Generator[SentenceBatch, None, None], is_pretrain: bool):
        for i, batch in enumerate(sentence_generator):
            x = [batch.tokens, batch.segments,
                 np.repeat(np.arange(max_len, dtype=np.int32).reshape(1, -1), len(batch.tokens), 0)]
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
                    x.append(task_data_batch.target_mask)
                    x.append(np.array([all_tasks[task_name].weight_scheduler.get(is_pretrain, i)]))
                    y.append(np.array([0.0]))
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
        _model.compile(pretrain_optimizer if is_pretrain else finetune_optimizer, loss=model_loss)
        _model.fit_generator(_generator, steps_per_epoch=pretrain_steps if is_pretrain else finetune_steps,
                             callbacks=pretrain_callbacks if is_pretrain else finetune_callbacks, shuffle=False)

    if pretrain_generator is not None:
        train_step(True)
    if finetune_generator is not None:
        train_step(False)
    return keras.Model(inputs=base_model.inputs, outputs=all_logits)
