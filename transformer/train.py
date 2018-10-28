import keras
import numpy as np
import keras.backend as K
from typing import List, Generator
from data.lm_dataset import dummy_lm_generator
from transformer.model import create_transformer
from keras.layers import Dropout, Input, Lambda, TimeDistributed, Dense
from data.dataset import TaskMetadata, SentenceBatch, create_attention_mask, TextEncoder, TaskWeightScheduler


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
            task_mask = Input(batch_shape=(None, max_len), dtype='int8', name=task.name + '_mask_input')
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
            x = [batch.tokens, batch.segments,
                 np.repeat(np.arange(max_len, dtype=np.int32).reshape(1, -1), batch_size, 0)]
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
                            0))  # [np.array()] didn't work
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
        _model.compile(pretrain_optimizer if is_pretrain else finetune_optimizer, loss=model_loss)
        _model.fit_generator(_generator, steps_per_epoch=pretrain_steps if is_pretrain else finetune_steps, verbose=2,
                             callbacks=pretrain_callbacks if is_pretrain else finetune_callbacks, shuffle=False,
                             epochs=5)
        # _model.fit_generator(_generator, steps_per_epoch=pretrain_steps if is_pretrain else finetune_steps,
        #                      callbacks=pretrain_callbacks if is_pretrain else finetune_callbacks, shuffle=False)

    if pretrain_generator is not None:
        train_step(True)
    if finetune_generator is not None:
        train_step(False)
    return keras.Model(inputs=base_model.inputs + sent_level_mask_inputs, outputs=all_logits)


if __name__ == '__main__':
    vocab_size = 23
    num_heads = 2
    num_layers = 2
    embedding_dim = 6
    d_hid = 12
    max_len = 7
    model = create_transformer(vocab_size=vocab_size + TextEncoder.SPECIAL_COUNT,
                               num_heads=num_heads, num_layers=num_layers,
                               embedding_dim=embedding_dim, d_hid=d_hid,
                               max_len=max_len, use_attn_mask=True)
    batch_size = 3
    steps = 1000000
    generator = dummy_lm_generator(vocab_size, max_len, batch_size, steps, False)
    # model = train_model(model, True, [TaskMetadata('lm', True, vocab_size + TextEncoder.SPECIAL_COUNT, 0.1,
    #                                                TaskWeightScheduler(True, False))], generator,
    #                     None, pretrain_steps=100)
    model = train_model(model, True, [
        TaskMetadata('lm', True, vocab_size + TextEncoder.SPECIAL_COUNT, 0.1, TaskWeightScheduler(True, False)),
        TaskMetadata('count', False, 2, 0.1, TaskWeightScheduler(True, True))], generator, None, pretrain_steps=100)
