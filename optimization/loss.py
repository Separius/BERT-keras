import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Layer


def tmp(max_len, batch_size):
    pos_id = K.variable(np.repeat(np.arange(max_len, dtype=np.int64).reshape(1, -1), batch_size, 0))

def tmp2():
    pass
    # logits = TimeDistributed(
    #     TiedEmbeddingsTransposed(embedding_layer.token_emb.weights[0] if use_tied_decoder else None,
    #                              units=vocab_size,
    #                              name='Decoder'), name='DecoderTimeDistributed')(x)


# tokens(B, L)[to find extraction point], x = (B, L, C)
# Optional(logits(B, L, V)), lm_targets(B, L), is_next = Optional((B,))
# masks(B, L)[padding indicator], token_classification = Optional({'t_name': (B, L)})
# sentence_classification(B), task_weights{lm, next, lm_after, is_next_after==0, others}
# task_target_sizes = {num_classes + 1}, outputs_logit?

class Task:
    def __init__(self, name, max_len, is_classification):
        self.name = name
        # B(label) ; B, L(classification_target) ; B, L, C (regression_target)
        # layer (C -> L(num_classes/num_channels)); dropout
        # target_mask(B or B,L); loss_func; pretrain_weight: 0; supervised_weight: 1
        self.target = Input(shape=(max_len,))


def dummy_loss(y_true, y_pred):
    return y_pred


# y_true is int of (B,L) with elements in [0, ignore_id]
def masked_classification(y_true, y_pred, y_mask):
    return masked_loss(y_true, y_pred, y_mask, masked_classification_loss)


def multichannel_mse(y_true, y_pred):
    return K.sqrt(K.sum(K.pow(y_true - y_pred, 2), axis=-1))


def masked_classification_loss(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


def sparse_gather(y_pred, target_indices, seq_len):
    clf_h = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    return K.gather(clf_h, K.arange(K.int_shape(target_indices)[0], dtype=np.int64) * seq_len + target_indices)


def masked_loss(y_true, y_pred, y_mask, element_wise_loss=multichannel_mse):
    l = K.switch(y_mask, element_wise_loss(y_true, y_pred), K.zeros_like(y_mask, dtype=K.floatx()))
    return K.sum(l) / (K.sum(y_mask) + K.epsilon())
    # K.switch(y_pred_mask)K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


def get_loss_network(base_model: Model, ignore_mask: bool, max_len: int):
    # TODO later merge the two networks
    mask = None if ignore_mask else Input(batch_shape=(None, 1, max_len, max_len))
    tokens = Input(batch_shape=(None, max_len))
    segment_ids = Input(batch_shape=(None, max_len))
    pos_ids = Input(batch_shape=(None, max_len))
    base_model_inputs = [tokens, segment_ids, pos_ids] + ([] if ignore_mask else [mask])

    h, logit = base_model(base_model_inputs)
    # h is None, max_len, channels; logit is None, max_len, vocab_size
    # TODO extract sentence_level_pred based on h, tokens => None, then compare that with Y and calc loss
