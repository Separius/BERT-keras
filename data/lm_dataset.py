import os
import random
import numpy as np
import sentencepiece as spm
from data.vocab import SPECIAL_TOKEN_IDS
from typing import List, NamedTuple, Optional, Generator, TextIO


class BERTSentenceBatch(NamedTuple):
    tokens: np.array
    targets: np.array
    is_next: np.array
    segment_ids: np.array
    masks: np.array


def lm_generator(text_corpus_address: str, spm_model_name: str = 'spm', keep_prob: float = 0.85, mask_prob: float = 0.8,
                 rand_prob: float = 0.1, min_len: Optional[int] = None, max_len: Optional[int] = 512, steps=1000000,
                 jump_prob: float = 0.1, mismatch_prob: float = 0.5, num_files: int = 8,
                 use_single_sentence: bool = False, batch_size: int = 256) -> Generator[BERTSentenceBatch, None, None]:
    if use_single_sentence:
        generator = _get_lm_generator_single(text_corpus_address, spm_model_name, keep_prob, mask_prob, rand_prob,
                                             min_len, max_len, steps, jump_prob, num_files)
    else:
        in_memory = jump_prob == 0.0 and num_files == 1
        generator = _get_lm_generator_double(text_corpus_address, spm_model_name, keep_prob, mask_prob, rand_prob,
                                             min_len, max_len, steps, mismatch_prob, in_memory, jump_prob, num_files)
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            yield _create_batch(batch, max_len)
            batch = []


class _MaskedSentence(NamedTuple):
    sentence: List[int]
    target: List[int]


class _BERTSentence(NamedTuple):
    sentence: _MaskedSentence
    is_next: bool
    segment_id: List[int]
    mask: Optional[List[bool]] = None


def _mask_sentence(sentence: List[int], vocab_size: int, keep_prob: float,
                   mask_prob: float, rand_prob: float) -> _MaskedSentence:
    prediction_target = [0] * len(sentence)
    for i in range(len(sentence)):
        if random.random() > keep_prob:
            prediction_target[i] = sentence[i]
            probability = random.random()
            if probability < mask_prob:
                sentence[i] = SPECIAL_TOKEN_IDS['msk']
            elif probability < (mask_prob + rand_prob):
                sentence[i] = random.randint(SPECIAL_TOKEN_IDS['unk'], vocab_size)
    return _MaskedSentence(sentence, prediction_target)


def _check_len(sentence: _MaskedSentence, min_len: Optional[int], max_len: Optional[int], from_end: bool = False) -> \
        Optional[_MaskedSentence]:
    if min_len is not None and len(sentence.sentence) < min_len:
        return None
    if max_len is not None and len(sentence.sentence) > max_len:
        if from_end:
            return _MaskedSentence(sentence.sentence[-max_len:], sentence.target[-max_len:])
        else:
            return _MaskedSentence(sentence.sentence[:max_len], sentence.target[:max_len])


def _grad_line(files: List[TextIO], file_size: int, jump_prob: float):
    file = files[random.randrange(len(files))]
    if random.random() < jump_prob:
        file.seek(random.randrange(file_size))
        file.readline()  # discard - bound to be partial line
    random_line = file.readline()
    if len(random_line) == 0:  # we have hit the end
        file.seek(0)
        random_line = file.readline()
    return random_line


def _pad(bert_sent: _BERTSentence, max_len: int):
    pad_id = SPECIAL_TOKEN_IDS['pad']
    pad_size = max_len - len(bert_sent.segment_id)
    return _BERTSentence(_MaskedSentence([pad_id] * pad_size + bert_sent.sentence.sentence,
                                         [pad_id] * pad_size + bert_sent.sentence.target),
                         bert_sent.is_next, [pad_id] * pad_size + bert_sent.segment_id,
                         [False] * pad_size + [True] * (max_len - pad_size))


def _create_batch(batch: List[_BERTSentence], max_len: Optional[int]):
    sort_indices = np.argsort([len(item.segment_id) for item in batch])[::-1]
    if max_len is None:
        max_len = len(batch[sort_indices[0]].segment_id)
    sorted_batch = [_pad(batch[i], max_len) for i in sort_indices]
    return BERTSentenceBatch(
        np.array([bert_sentence.sentence.sentence for bert_sentence in sorted_batch], dtype=np.int64),
        np.array([bert_sentence.sentence.target for bert_sentence in sorted_batch], dtype=np.int64),
        np.array([bert_sentence.is_next for bert_sentence in sorted_batch], dtype=np.float32),
        np.array([bert_sentence.segment_id for bert_sentence in sorted_batch], dtype=np.int64),
        # TODO can we can set this to int 8?
        np.array([bert_sentence.mask for bert_sentence in sorted_batch], dtype=np.float32)
    )


def _get_lm_generator_single(text_corpus_address: str, spm_model_name: str = 'spm', keep_prob: float = 0.85,
                             mask_prob: float = 0.8, rand_prob: float = 0.1, min_len: Optional[int] = None,
                             max_len: Optional[int] = 512, steps=1000000, jump_prob: float = 0.1,
                             num_files: int = 8) -> Generator[_BERTSentence, None, None]:
    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(spm_model_name))
    counter = 0
    should_continue = True
    _max_len = float('inf') if max_len is None else max_len - 2
    _min_len = 0 if min_len is None else min_len - 2
    files = [open(text_corpus_address) for _ in range(num_files)]
    file_size = os.stat(text_corpus_address).st_size

    def _encode_line(line):
        return _check_len(_mask_sentence(sp.encode_as_ids(line.rstrip()), len(sp), keep_prob, mask_prob, rand_prob),
                          _min_len, _max_len)

    def _yield_sentence(sent1: _MaskedSentence) -> _BERTSentence:
        masked_sentence = _MaskedSentence([SPECIAL_TOKEN_IDS['bos']] + sent1.sentence + [SPECIAL_TOKEN_IDS['eos']],
                                          [0] + sent1.target + [0])
        return _BERTSentence(masked_sentence, True, [0] * len(masked_sentence.sentence))

    while should_continue:
        sent = _grad_line(files, file_size, jump_prob)
        encoded = _encode_line(sent)
        if not encoded:
            continue
        counter += 1
        yield _yield_sentence(encoded)
        if counter > steps:
            should_continue = False
            break
    for f in files:
        f.close()


def _get_lm_generator_double(text_corpus_address: str, spm_model_name: str = 'spm', keep_prob: float = 0.85,
                             mask_prob: float = 0.8, rand_prob: float = 0.1, min_len: Optional[int] = None,
                             max_len: Optional[int] = 512, steps=1000000, mismatch_prob: float = 0.5,
                             in_memory: bool = False, jump_prob: float = 0.1,
                             num_files: int = 8) -> Generator[_BERTSentence, None, None]:
    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(spm_model_name))
    counter = 0
    _max_len = float('inf') if max_len is None else max_len - 3
    _min_len = 0 if min_len is None else min_len - 3
    if in_memory:
        with open(text_corpus_address) as f:
            all_lines = [sp.encode_as_ids(line.rstrip()) for line in f]
        files = None
    else:
        all_lines = None
        files = [open(text_corpus_address) for _ in range(num_files)]
    file_size = os.stat(text_corpus_address).st_size

    def _encode_line(line: str, half: bool, from_end: bool = False) -> Optional[_MaskedSentence]:
        return _check_len(_mask_sentence(sp.encode_as_ids(line.rstrip()), len(sp), keep_prob, mask_prob, rand_prob),
                          _min_len // (2 if half else 1), _max_len // (2 if half else 1), from_end=from_end)

    def _yield_sentence(sent1: _MaskedSentence, sent2: Optional[_MaskedSentence] = None) -> _BERTSentence:
        if sent2 is None:
            split_idx = random.randint(_min_len // 2, len(sent1.sentence) - _min_len // 2)
            masked_sentence = _MaskedSentence(
                [SPECIAL_TOKEN_IDS['bos']] + sent1.sentence[:split_idx] + [
                    SPECIAL_TOKEN_IDS['eos']] + sent1.sentence[
                                                split_idx:] + [
                    SPECIAL_TOKEN_IDS['eos']],
                [0] + sent1.target[:split_idx] + [0] + sent1.target[split_idx:] + [0])
            return _BERTSentence(masked_sentence, True,
                                 [0] * (split_idx + 2) + [1] * (len(masked_sentence.sentence) - 2 - split_idx))
        masked_sentence = _MaskedSentence(
            [SPECIAL_TOKEN_IDS['bos']] + sent1.sentence + [SPECIAL_TOKEN_IDS['eos']] + sent2.sentence + [
                SPECIAL_TOKEN_IDS['eos']], [0] + sent1.target + [0] + sent2.target + [0])
        return _BERTSentence(masked_sentence, False,
                             [0] * (2 + len(sent1.sentence)) + [1] * (1 + len(sent2.sentence)))

    def _calc_encoded(line, _all_lines=None, _files=None):
        if random.random() < mismatch_prob:
            _encoded1 = _encode_line(line, half=True)
            if _all_lines is not None:
                line2 = _all_lines[random.randint(0, len(_all_lines) - 1)]
            else:
                line2 = _grad_line(_files, file_size, jump_prob)
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
            all_lines[current_line_number] if all_lines else _grad_line(files, file_size, jump_prob), all_lines,
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
