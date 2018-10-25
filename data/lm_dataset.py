import os
import random
import numpy as np
from data.vocab import TextEncoder
from typing import List, NamedTuple, Optional, Generator, TextIO, Dict


class BERTSentenceBatch(NamedTuple):
    tokens: np.array  # batch_size, seq (token_id)
    lm_targets: np.array  # batch_size, seq (vocab_size+TextEncoder.PAD_OFFSET should be ignored)
    is_next: np.array  # batch_size (0 or 1)
    segment_ids: np.array  # batch_size, seq (0 or 1)
    masks: np.array  # batch_size, seq (0 or 1, zeros should be ignored (1 == use))
    token_classification: Optional[Dict[str, np.array]] = None  # task_name: batch_size, seq(num_classes + 1 for ignore)
    sentence_classification: Optional[Dict[str, np.array]] = None  # task_name: batch_size


def lm_generator(text_corpus_address: str, text_encoder: TextEncoder, keep_prob: float = 0.85,
                 mask_prob: float = 0.15 * 0.8, rand_prob: float = 0.15 * 0.1, min_len: Optional[int] = None,
                 max_len: Optional[int] = 512, steps: int = 1000000, file_jump_prob: float = 0.1,
                 mismatch_prob: float = 0.5, num_file_pointers: int = 8, is_causal: bool = False,
                 use_single_sentence: bool = False, batch_size: int = 256) -> Generator[BERTSentenceBatch, None, None]:
    if not (0.0 <= mask_prob <= 1.0 and
            0.0 <= rand_prob <= 1.0 and
            0.0 <= keep_prob <= 1.0 and
            0.0 <= file_jump_prob <= 1.0):
        raise ValueError('all probablities should be between [0, 1]')
    if mask_prob + rand_prob + keep_prob > 1.0:
        raise ValueError('sum of mask, rand and keep probablities should be less than 1.0')
    if use_single_sentence:
        generator = _get_lm_generator_single(text_corpus_address, text_encoder, keep_prob, mask_prob, rand_prob,
                                             min_len, max_len, steps, file_jump_prob, num_file_pointers)
    else:
        in_memory = file_jump_prob == 0.0 and num_file_pointers == 1
        generator = _get_lm_generator_double(text_corpus_address, text_encoder, keep_prob, mask_prob, rand_prob,
                                             min_len, max_len, steps, mismatch_prob, in_memory, file_jump_prob,
                                             num_file_pointers)
    batch = []
    for item in generator:
        batch.append(_make_causal(item, text_encoder.pad_id) if is_causal else item)
        if len(batch) == batch_size:
            yield _create_batch(batch, text_encoder.pad_id, max_len)
            batch = []


def create_mask(pad_mask: Optional[np.array], is_causal: bool = True, batch_size: Optional[int] = 256,
                length: Optional[int] = 512):
    if pad_mask is not None:
        batch_size, length = pad_mask.shape
    if is_causal:
        b = np.cumsum(np.eye(length, dtype=np.float32), axis=0)
    else:
        b = np.ones((length, length))
    b = np.reshape(b, [1, 1, length, length])  # 1, 1, L, L
    b = np.repeat(b, batch_size, axis=0)
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


class _MaskedSentence(NamedTuple):
    tokens: List[int]
    lm_target: List[int]
    token_target: Optional[Dict[str, List[int]]] = None  # used for PoS and NER
    sentence_target: Optional[Dict[str, int]] = None  # used for sentiment and classification


class _BERTSentence(NamedTuple):
    sentence: _MaskedSentence
    is_next: bool
    segment_id: List[int]
    mask: Optional[List[bool]] = None  # used to indicate padding


def _make_causal(item: _BERTSentence, pad_id: int) -> _BERTSentence:
    for i in range(len(item.sentence.tokens) - 1):
        item.sentence.lm_target[i] = item.sentence.tokens[i + 1]
    item.sentence.lm_target[-1] = pad_id
    return item


def _mask_sentence(sentence: List[int], vocab_size: int, keep_prob: float,
                   mask_prob: float, rand_prob: float) -> _MaskedSentence:
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
    return _MaskedSentence(new_sent, prediction_target)


def _check_len(sentence: _MaskedSentence, min_len: Optional[int], max_len: Optional[int], from_end: bool = False) -> \
        Optional[_MaskedSentence]:
    if min_len is not None and len(sentence.tokens) < min_len:
        return None
    if max_len is not None and len(sentence.tokens) > max_len:
        if from_end:
            return _MaskedSentence(sentence.tokens[-max_len:], sentence.lm_target[-max_len:])
        else:
            return _MaskedSentence(sentence.tokens[:max_len], sentence.lm_target[:max_len])


def _grab_line(files: List[TextIO], file_size: int, jump_prob: float):
    file = files[random.randrange(len(files))]
    if random.random() < jump_prob:
        file.seek(random.randrange(file_size))
        file.readline()  # discard - bound to be partial line
    random_line = file.readline()
    if len(random_line) == 0:  # we have hit the end
        file.seek(0)
        random_line = file.readline()
    return random_line


def _pad(bert_sent: _BERTSentence, pad_id: int, max_len: int):
    pad_size = max_len - len(bert_sent.segment_id)
    return _BERTSentence(_MaskedSentence(bert_sent.sentence.tokens + [pad_id] * pad_size,
                                         bert_sent.sentence.lm_target + [pad_id] * pad_size),
                         bert_sent.is_next, bert_sent.segment_id + [pad_id] * pad_size,
                         [True] * (max_len - pad_size) + [False] * pad_size)


def _create_batch(batch: List[_BERTSentence], pad_id: int, max_len: Optional[int] = None):
    sort_indices = np.argsort([len(item.segment_id) for item in batch])[::-1]
    if max_len is None:
        max_len = len(batch[sort_indices[0]].segment_id)
    sorted_batch = [_pad(batch[i], pad_id, max_len) for i in sort_indices]
    return BERTSentenceBatch(
        np.array([bert_sentence.sentence.tokens for bert_sentence in sorted_batch], dtype=np.int64),
        np.array([bert_sentence.sentence.lm_target for bert_sentence in sorted_batch], dtype=np.int64),
        np.array([bert_sentence.is_next for bert_sentence in sorted_batch], dtype=np.float32),
        np.array([bert_sentence.segment_id for bert_sentence in sorted_batch], dtype=np.int64),
        np.array([bert_sentence.mask for bert_sentence in sorted_batch], dtype=np.int8)
    )


def _get_lm_generator_single(text_corpus_address: str, text_encoder: TextEncoder, keep_prob: float, mask_prob: float,
                             rand_prob: float, min_len: Optional[int], max_len: Optional[int], steps: int, jump_prob,
                             num_files) -> Generator[_BERTSentence, None, None]:
    counter = 0
    _max_len = float('inf') if max_len is None else max_len - 2
    _min_len = 0 if min_len is None else min_len - 2
    files = [open(text_corpus_address) for _ in range(num_files)]
    file_size = os.stat(text_corpus_address).st_size

    def _encode_line(line):
        return _check_len(
            _mask_sentence(text_encoder.encode(line.rstrip()), len(text_encoder), keep_prob, mask_prob, rand_prob),
            _min_len, _max_len)

    def _yield_sentence(sent1: _MaskedSentence) -> _BERTSentence:
        masked_sentence = _MaskedSentence([text_encoder.bos_id] + sent1.tokens + [text_encoder.eos_id],
                                          [text_encoder.pad_id] + sent1.lm_target + [text_encoder.pad_id])
        return _BERTSentence(masked_sentence, True, [0] * len(masked_sentence.tokens))

    while True:
        sent = _grab_line(files, file_size, jump_prob)
        encoded = _encode_line(sent)
        if not encoded:
            continue
        counter += 1
        yield _yield_sentence(encoded)
        if counter > steps:
            break
    for f in files:
        f.close()


def _get_lm_generator_double(text_corpus_address: str, text_encoder: TextEncoder, keep_prob: float, mask_prob: float,
                             rand_prob: float, min_len: Optional[int], max_len: Optional[int], steps: int,
                             mismatch_prob: float, in_memory: bool, jump_prob: float, num_files: int) -> Generator[
    _BERTSentence, None, None]:
    counter = 0
    _max_len = float('inf') if max_len is None else max_len - 3
    _min_len = 0 if min_len is None else min_len - 3
    if in_memory:
        with open(text_corpus_address) as f:
            all_lines = [text_encoder.encode(line.rstrip()) for line in f]
        files = None
    else:
        all_lines = None
        files = [open(text_corpus_address) for _ in range(num_files)]
    file_size = os.stat(text_corpus_address).st_size

    def _encode_line(line: str, half: bool, from_end: bool = False) -> Optional[_MaskedSentence]:
        return _check_len(
            _mask_sentence(text_encoder.encode(line.rstrip()), len(text_encoder), keep_prob, mask_prob, rand_prob),
            _min_len // (2 if half else 1), _max_len // (2 if half else 1), from_end=from_end)

    def _yield_sentence(sent1: _MaskedSentence, sent2: Optional[_MaskedSentence] = None) -> _BERTSentence:
        if sent2 is None:
            split_idx = random.randint(_min_len // 2, len(sent1.tokens) - _min_len // 2)
            masked_sentence = _MaskedSentence(
                [text_encoder.bos_id] + sent1.tokens[:split_idx] + [text_encoder.del_id] + sent1.tokens[
                                                                                           split_idx:] + [
                    text_encoder.eos_id],
                [text_encoder.pad_id] + sent1.lm_target[:split_idx] + [text_encoder.pad_id] + sent1.lm_target[
                                                                                              split_idx:] + [
                    text_encoder.pad_id])
            return _BERTSentence(masked_sentence, True,
                                 [0] * (split_idx + 2) + [1] * (len(masked_sentence.tokens) - 2 - split_idx))
        masked_sentence = _MaskedSentence(
            [text_encoder.bos_id] + sent1.tokens + [text_encoder.del_id] + sent2.tokens + [text_encoder.eos_id],
            [text_encoder.pad_id] + sent1.lm_target + [text_encoder.pad_id] + sent2.lm_target + [text_encoder.pad_id])
        return _BERTSentence(masked_sentence, False, [0] * (2 + len(sent1.tokens)) + [1] * (1 + len(sent2.tokens)))

    def _calc_encoded(line, _all_lines=None, _files=None):
        if random.random() < mismatch_prob:
            _encoded1 = _encode_line(line, half=True)
            if _all_lines is not None:
                line2 = _all_lines[random.randrange(len(_all_lines))]
            else:
                line2 = _grab_line(_files, file_size, jump_prob)
            _encoded2 = _encode_line(line2, half=True, from_end=True)
            if _encoded2 is None:
                return None
        else:
            _encoded1 = _encode_line(line, half=False)
            _encoded2 = None
        return _encoded1, _encoded2

    current_line_number = 0
    max_line_number = len(all_lines) if all_lines else float('inf')
    while True:
        encoded1, encoded2 = _calc_encoded(
            all_lines[current_line_number] if all_lines else _grab_line(files, file_size, jump_prob), all_lines,
            files)
        if encoded1 is None:
            continue
        counter += 1
        if all_lines:
            current_line_number += 1
            if current_line_number == max_line_number:
                current_line_number = 0
        yield _yield_sentence(encoded1, encoded2)
        if counter > steps:
            break
    for f in files:
        f.close()
