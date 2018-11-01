import random
import numpy as np
from data.vocab import TextEncoder
from typing import List, NamedTuple, Optional, Dict, Any


class TaskWeightScheduler:
    def __init__(self, active_in_pretrain: bool, active_in_finetune: bool,
                 pretrain_value: float = 1.0, finetune_value: float = 1.0):
        self.active_in_pretrain = active_in_pretrain
        self.active_in_finetune = active_in_finetune
        self.pretrain_value = pretrain_value
        self.finetune_value = finetune_value

    def get(self, is_pretrain: bool, step: int) -> float:
        if is_pretrain and self.active_in_pretrain:
            return self.pretrain_value
        if not is_pretrain and self.active_in_finetune:
            return self.finetune_value
        raise ValueError()


class TaskMetadata(NamedTuple):
    name: str  # "lm" will be considered differently (will use tied decoder)
    is_token_level: bool
    num_classes: int
    dropout: float
    weight_scheduler: TaskWeightScheduler


class TokenTaskData(NamedTuple):
    target: List[int]
    target_mask: List[bool]


class SentenceTaskData(NamedTuple):
    target: int
    target_index: int


class TaskDataBatch(NamedTuple):
    target: np.array  # (int32) batch_size for sentence level tasks or batch_size, seq_len for token level tasks
    target_mask: np.array  # (int8) same as target (will ignore zeros)


class Sentence(NamedTuple):
    tokens: List[int]
    padding_mask: List[bool]
    segments: Optional[List[int]] = None
    token_classification: Optional[Dict[str, TokenTaskData]] = None
    sentence_classification: Optional[Dict[str, SentenceTaskData]] = None


class SentenceBatch(NamedTuple):
    tokens: np.array  # (int32) batch_size, seq_len
    padding_mask: np.array  # (int8) batch_size, seq_len (0 or 1, zeros should be ignored (1 == use, 0 == padded))
    segments: np.array  # (int32) batch_size, seq_len
    token_classification: Dict[str, TaskDataBatch]  # task_name('lm' is special) : task_data
    sentence_classification: Dict[str, TaskDataBatch]  # task_name : task_data


def create_attention_mask(pad_mask: Optional[np.array], is_causal: bool, batch_size: Optional[int] = None,
                          length: Optional[int] = None, bert_attention: bool = False) -> np.array:
    if pad_mask is not None:
        assert pad_mask.ndim == 2
        batch_size, length = pad_mask.shape
    if is_causal:
        b = np.cumsum(np.eye(length, dtype=np.float32), axis=0)
    else:
        b = np.ones((length, length), dtype=np.float32)
    b = np.reshape(b, [1, 1, length, length])
    b = np.repeat(b, batch_size, axis=0)  # B, 1, L, L
    if pad_mask is not None:
        _pad_mask = pad_mask[..., np.newaxis]
        _pad_mask = np.repeat(_pad_mask, length, 2)
        _pad_mask_t = np.transpose(_pad_mask, [0, 2, 1])
        if bert_attention:
            tmp = _pad_mask_t
        else:
            tmp = _pad_mask * _pad_mask_t
        tmp = tmp[:, np.newaxis, ...]
        if b is None:
            b = tmp.astype(np.float32)
        else:
            b = b * tmp
    return b


def _trim_seq(seq: Optional[List[Any]], length: int, from_end: bool = True) -> Optional[List[Any]]:
    if seq is None:
        return None
    return seq[:length] if from_end else seq[-length:]


def _trim_sentence_target(task_dict: Dict[str, SentenceTaskData], desired_len: int,
                          orig_seq_len: int, from_end: bool = True) -> Dict[
    str, SentenceTaskData]:
    trimmed_task_dict = {}
    for k, v in task_dict.items():
        target_index = v.target_index
        if orig_seq_len > desired_len:
            if from_end and target_index > desired_len:
                target_index = -1
            if not from_end:
                target_index -= orig_seq_len - desired_len
        if target_index >= 0:
            trimmed_task_dict[k] = SentenceTaskData(v.target, target_index)
    return trimmed_task_dict


def _trim_sentence(sentence: Sentence, length: int, from_end: bool = True) -> Sentence:
    return Sentence(_trim_seq(sentence.tokens, length, from_end),
                    _trim_seq(sentence.padding_mask, length, from_end),
                    _trim_seq(sentence.segments, length, from_end),
                    {k: TokenTaskData(_trim_seq(v.target, length, from_end),
                                      _trim_seq(v.target_mask, length, from_end)) for k, v in
                     sentence.token_classification.items()} if sentence.token_classification is not None else {},
                    _trim_sentence_target(sentence.sentence_classification, length, len(sentence.tokens),
                                          from_end) if sentence.sentence_classification is not None else {})


def check_sent_len(sentence: Sentence, min_len: Optional[int], max_len: Optional[int], from_end: bool = True) -> \
        Optional[Sentence]:
    if min_len is not None and len(sentence.tokens) < min_len:
        return None
    if max_len is not None and len(sentence.tokens) > max_len:
        return _trim_sentence(sentence, max_len, from_end)
    return sentence


def msk_sentence(sentence: List[int], vocab_size: int, keep_prob: float,
                 mask_prob: float, rand_prob: float) -> Sentence:
    prediction_target = [0] * len(sentence)
    prediction_mask = [False] * len(sentence)
    new_sent = sentence.copy()
    for i in range(len(sentence)):
        probability = random.random()
        if probability > keep_prob:
            prediction_target[i] = sentence[i]
            prediction_mask[i] = True
            if probability < (mask_prob + keep_prob):
                new_sent[i] = vocab_size + TextEncoder.MSK_OFFSET
            elif probability < (mask_prob + rand_prob + keep_prob):
                new_sent[i] = random.randrange(vocab_size)
    return Sentence(new_sent, [True] * len(new_sent), None,
                    token_classification={'lm': TokenTaskData(prediction_target, prediction_mask)},
                    sentence_classification={})


def _pad_seq(seq: List[Any], pad_token: Any, pad_len: int, is_post_pad: bool = True) -> List[Any]:
    return (seq + [pad_token] * pad_len) if is_post_pad else ([pad_token] * pad_len + seq)


def pad(sentence: Sentence, pad_id: int, max_len: int, is_post_pad: bool = True) -> Sentence:
    pad_len = max_len - len(sentence.tokens)
    if pad_len == 0:
        return sentence
    return Sentence(_pad_seq(sentence.tokens, pad_id, pad_len, is_post_pad),
                    _pad_seq(sentence.padding_mask, False, pad_len, is_post_pad),
                    _pad_seq(sentence.segments, 0, pad_len, is_post_pad),
                    {k: TokenTaskData(_pad_seq(v.target, 0, pad_len, is_post_pad),
                                      _pad_seq(v.target_mask, False, pad_len, is_post_pad)) for k, v in
                     sentence.token_classification.items()} if sentence.token_classification is not None else {},
                    {k: SentenceTaskData(v.target, v.target_index + (0 if is_post_pad else pad_len)) for k, v in
                     sentence.sentence_classification.items()} if sentence.sentence_classification is not None else {})


def generate_pos_ids(batch_size: int, max_len: int) -> np.array:
    return np.repeat(np.arange(max_len, dtype=np.int32).reshape(1, -1), batch_size, 0)
