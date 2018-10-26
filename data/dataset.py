import random
import numpy as np
from data.vocab import TextEncoder
from typing import List, NamedTuple, Optional, Dict, Callable, Union


class TaskWeightScheduler:
    def __init__(self, active_in_pretrain: bool, active_in_finetune: bool, default_value: float = 1.0):
        self.active_in_pretrain = active_in_pretrain
        self.active_in_finetune = active_in_finetune
        self.default_value = default_value

    def get(self, is_finetune: bool, step: int) -> float:
        if is_finetune and self.active_in_finetune:
            return self.default_value
        if not is_finetune and self.active_in_pretrain:
            return self.default_value
        raise ValueError()


class TaskMetadata(NamedTuple):
    name: str  # "lm" will be considered differently (can use tied decoder)
    num_classes: int
    dropout: float
    weight_scheduler: TaskWeightScheduler


class TaskData(NamedTuple):
    target: np.array
    target_mask: np.array


class NeoBertBatch(NamedTuple):
    tokens: np.array
    padding_mask: np.array
    segments: np.array
    sentence_classification: Dict[TaskData]
    token_classification: Dict[TaskData]


class BertBatch(NamedTuple):
    tokens: np.array  # batch_size, seq (token_id)
    lm_targets: np.array  # batch_size, seq (vocab_size+TextEncoder.PAD_OFFSET should be ignored)
    is_next: np.array  # batch_size (0 or 1)
    segment_ids: np.array  # batch_size, seq (0 or 1)
    masks: np.array  # batch_size, seq (0 or 1, zeros should be ignored (1 == use))
    token_classification: Optional[Dict[str, np.array]] = None  # task_name: batch_size, seq(num_classes + 1 for ignore)
    sentence_classification: Optional[Dict[str, np.array]] = None  # task_name: batch_size


class Sentence(NamedTuple):
    tokens: List[int]
    lm_target: List[int]
    token_target: Optional[Dict[str, List[int]]] = None  # used for PoS and NER
    sentence_target: Optional[Dict[str, int]] = None  # used for sentiment and classification


class BertSentence(NamedTuple):
    sentence: Sentence
    is_next: bool
    segment_id: List[int]
    mask: Optional[List[bool]] = None  # used to indicate padding(not causality)


def create_attention_mask(pad_mask: Optional[np.array], is_causal: bool = True, batch_size: Optional[int] = 256,
                          length: Optional[int] = 512) -> np.array:
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
        tmp = _pad_mask * _pad_mask_t
        tmp = tmp[:, np.newaxis, ...]
        if b is None:
            b = tmp.astype(np.float32)
        else:
            b = b * tmp
    return b


def check_sent_len(sentence: Sentence, min_len: Optional[int], max_len: Optional[int], from_end: bool = False) -> \
        Optional[Sentence]:
    if min_len is not None and len(sentence.tokens) < min_len:
        return None
    if max_len is not None and len(sentence.tokens) > max_len:
        if from_end:
            return Sentence(sentence.tokens[-max_len:], sentence.lm_target[-max_len:])
        else:
            return Sentence(sentence.tokens[:max_len], sentence.lm_target[:max_len])


def msk_sentence(sentence: List[int], vocab_size: int, keep_prob: float,
                 mask_prob: float, rand_prob: float) -> Sentence:
    prediction_target = [vocab_size + TextEncoder.PAD_OFFSET] * len(sentence)
    new_sent = sentence.copy()
    for i in range(len(sentence)):
        probability = random.random()
        if probability > keep_prob:
            prediction_target[i] = sentence[i]
            if probability < (mask_prob + keep_prob):
                new_sent[i] = vocab_size + TextEncoder.MSK_OFFSET
            elif probability < (mask_prob + rand_prob + keep_prob):
                new_sent[i] = random.randrange(vocab_size)
    return Sentence(new_sent, prediction_target)


def pad(bert_sent: BertSentence, pad_id: int, max_len: int) -> BertSentence:
    pad_size = max_len - len(bert_sent.segment_id)
    return BertSentence(Sentence(bert_sent.sentence.tokens + [pad_id] * pad_size,
                                 bert_sent.sentence.lm_target + [pad_id] * pad_size),
                        bert_sent.is_next, bert_sent.segment_id + [pad_id] * pad_size,
                        [True] * (max_len - pad_size) + [False] * pad_size)
