import os
import random
import numpy as np
from contextlib import ExitStack
from data.vocab import TextEncoder
from typing import List, Optional, Generator, TextIO, Tuple, Dict
from data.dataset import (Sentence, pad, msk_sentence, check_sent_len,
                          SentenceBatch, TaskDataBatch, TokenTaskData, SentenceTaskData)


def lm_generator(text_corpus_address: str, text_encoder: TextEncoder, keep_prob: float = 0.85,
                 mask_prob: float = 0.15 * 0.8, rand_prob: float = 0.15 * 0.1, min_len: Optional[int] = None,
                 max_len: Optional[int] = 512, file_jump_prob: float = 0.1, mismatch_prob: float = 0.5,
                 num_file_pointers: int = 8, is_causal: bool = False, use_single_sentence: bool = False,
                 batch_size: int = 256) -> Generator[SentenceBatch, None, None]:
    if not (0.0 <= mask_prob <= 1.0 and 0.0 <= rand_prob <= 1.0 and
            0.0 <= keep_prob <= 1.0 and 0.0 <= file_jump_prob <= 1.0):
        raise ValueError('all probablities should be between zero and one')
    if mask_prob + rand_prob + keep_prob > 1.0:
        raise ValueError('sum of mask, rand and keep probablities should be less than 1.0')
    if use_single_sentence:
        generator = _get_lm_generator_single(text_corpus_address, text_encoder, keep_prob, mask_prob, rand_prob,
                                             min_len, max_len, file_jump_prob, num_file_pointers)
    else:
        in_memory = file_jump_prob == 0.0 and num_file_pointers == 1
        generator = _get_lm_generator_double(text_corpus_address, text_encoder, keep_prob, mask_prob, rand_prob,
                                             min_len, max_len, mismatch_prob, in_memory, file_jump_prob,
                                             num_file_pointers)
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            batch = make_next_token_prediction(batch) if is_causal else batch
            batch = _create_batch(batch, text_encoder.pad_id, max_len)
            yield batch
            batch = []


def make_next_token_prediction(batch: List[Sentence]) -> List[Sentence]:
    for item in batch:
        for i in range(len(item.tokens) - 1):
            item.token_classification['lm'].target[i] = item.tokens[i + 1]
            item.token_classification['lm'].target_mask[i] = True
        item.token_classification['lm'].target[-1] = 0
        item.token_classification['lm'].target_mask[-1] = False
    return batch


def _grab_line(files: List[TextIO], file_size: int, jump_prob: float) -> str:
    file = files[random.randrange(len(files))]
    if random.random() < jump_prob:
        file.seek(random.randrange(file_size))
        file.readline()  # discard - bound to be partial line
    random_line = file.readline()
    if len(random_line) == 0:  # we have hit the end
        file.seek(0)
        random_line = file.readline()
    return random_line


def _create_token_task_batch(batch: List[Sentence]) -> Dict[str, TaskDataBatch]:
    batch_keys = set(batch[0].token_classification.keys())
    for item in batch:
        assert batch_keys == set(batch[0].token_classification.keys())
    result = {}
    for key in batch_keys:
        result[key] = TaskDataBatch(
            np.array([item.token_classification[key].target for item in batch], dtype=np.int32),
            np.array([item.token_classification[key].target_mask for item in batch], dtype=np.int32))
    return result


def _create_sent_task_batch(batch: List[Sentence]) -> Dict[str, TaskDataBatch]:
    batch_keys = set(batch[0].sentence_classification.keys())
    for item in batch:
        assert batch_keys == set(batch[0].sentence_classification.keys())
    result = {}
    for key in batch_keys:
        result[key] = TaskDataBatch(
            np.array([item.sentence_classification[key].target for item in batch], dtype=np.int32),
            np.array([item.sentence_classification[key].target_index for item in batch], dtype=np.int32))
    return result


def _create_batch(batch: List[Sentence], pad_id: int, max_len: Optional[int] = None) -> SentenceBatch:
    if max_len is None:
        max_len = max(len(item.tokens) for item in batch)
    padded_batch = [pad(item, pad_id, max_len) for item in batch]
    return SentenceBatch(
        np.array([item.tokens for item in padded_batch], dtype=np.int32),
        np.array([item.padding_mask for item in padded_batch], dtype=np.int8),
        np.array([item.segments for item in padded_batch], dtype=np.int32),
        _create_token_task_batch(padded_batch), _create_sent_task_batch(padded_batch)
    )


def _get_lm_generator_single(text_corpus_address: str, text_encoder: TextEncoder, keep_prob: float, mask_prob: float,
                             rand_prob: float, min_len: Optional[int], max_len: Optional[int], jump_prob,
                             num_files) -> Generator[Sentence, None, None]:
    _max_len = float('inf') if max_len is None else max_len - 2
    _min_len = 0 if min_len is None else min_len - 2
    file_size = os.stat(text_corpus_address).st_size
    with ExitStack() as stack:
        files = [stack.enter_context(open(text_corpus_address)) for _ in range(num_files)]

        def _encode_line(line: str) -> Optional[Sentence]:
            return check_sent_len(
                msk_sentence(text_encoder.encode(line.rstrip()), len(text_encoder), keep_prob, mask_prob, rand_prob),
                _min_len, _max_len)

        def _yield_sentence(sent: Sentence) -> Sentence:
            lm = sent.token_classification['lm']
            return Sentence(
                [text_encoder.bos_id] + sent.tokens + [text_encoder.eos_id],
                [True] + sent.padding_mask + [True],
                [0] * (len(sent.tokens) + 2),
                {'lm': TokenTaskData([0] + lm.target + [0], [False] + lm.target_mask + [False])},
                {}
            )

        while True:
            sent = _grab_line(files, file_size, jump_prob)
            encoded = _encode_line(sent)
            if not encoded:
                continue
            yield _yield_sentence(encoded)


def _get_lm_generator_double(text_corpus_address: str, text_encoder: TextEncoder, keep_prob: float, mask_prob: float,
                             rand_prob: float, min_len: Optional[int], max_len: Optional[int],
                             mismatch_prob: float, in_memory: bool, jump_prob: float, num_files: int) -> Generator[
    Sentence, None, None]:
    _max_len = float('inf') if max_len is None else max_len - 3
    _min_len = 0 if min_len is None else min_len - 3
    file_size = os.stat(text_corpus_address).st_size
    current_line_number = 0
    with ExitStack() as stack:
        if in_memory:
            with open(text_corpus_address) as f:
                all_lines = [text_encoder.encode(line.rstrip()) for line in f]
            files = None
        else:
            all_lines = None
            files = [stack.enter_context(open(text_corpus_address)) for _ in range(num_files)]
        max_line_number = len(all_lines) if all_lines else float('inf')

        def _encode_line(line: str, half: bool, from_end: bool = False) -> Optional[Sentence]:
            return check_sent_len(
                msk_sentence(text_encoder.encode(line.rstrip()), len(text_encoder), keep_prob, mask_prob, rand_prob),
                _min_len // (2 if half else 1), _max_len // (2 if half else 1), from_end=from_end)

        def _yield_sentence(sent1: Sentence, sent2: Optional[Sentence] = None) -> Sentence:
            lm = sent1.token_classification['lm']
            if sent2 is None:
                split_idx = random.randint(_min_len // 2, len(sent1.tokens) - _min_len // 2)
                return Sentence(
                    [text_encoder.bos_id] + sent1.tokens[:split_idx] + [text_encoder.del_id] + sent1.tokens[
                                                                                               split_idx:] + [
                        text_encoder.eos_id],
                    [True] + sent1.padding_mask[:split_idx] + [True] + sent1.padding_mask[split_idx:] + [True],
                    [0] * (split_idx + 2) + [1] * (1 + len(sent1.tokens) - split_idx),
                    {'lm': TokenTaskData([0] + lm.target[:split_idx] + [0] + lm.target[split_idx:] + [0],
                                         [False] + lm.target_mask[:split_idx] + [False] + lm.target_mask[split_idx:] + [
                                             False])},
                    {}
                )
            lm_ = sent2.token_classification['lm']
            return Sentence(
                [text_encoder.bos_id] + sent1.tokens + [text_encoder.del_id] + sent2.tokens + [text_encoder.eos_id],
                [True] + sent1.padding_mask + [True] + sent2.padding_mask + [True],
                [0] * (2 + len(sent1.tokens)) + [1] * (1 + len(sent2.tokens)),
                {'lm': TokenTaskData([0] + lm.target + [0] + lm_.target + [0],
                                     [False] + lm.target_mask + [False] + lm_.target_mask + [False])},
                {}
            )

        def _calc_encoded(line: str, _all_lines: Optional[List[str]] = None, _files: Optional[List[TextIO]] = None) -> \
                Optional[Tuple[Optional[Sentence], Optional[Sentence]]]:
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

        while True:
            encoded1, encoded2 = _calc_encoded(
                all_lines[current_line_number] if all_lines else _grab_line(files, file_size, jump_prob), all_lines,
                files)
            if encoded1 is None:
                continue
            if all_lines:
                current_line_number += 1
                if current_line_number == max_line_number:
                    current_line_number = 0
            yield _yield_sentence(encoded1, encoded2)


def dummy_lm_generator(vocab_size: int, max_len: int, batch_size: int, steps: int, easy: bool = True):  # identity
    def dummy_generator():
        for _ in range(steps):
            seq_len = random.randint(1, max_len - 1)
            tokens = [random.randrange(vocab_size) for i in range(seq_len)]
            tokens[-1] = eos_id
            yield Sentence(
                tokens=tokens,
                padding_mask=[True] * seq_len,
                segments=[0] * seq_len,
                token_classification={
                    'lm': TokenTaskData(tokens if easy else [random.randrange(vocab_size) for i in range(seq_len)],
                                        [True] * seq_len),
                    'lm_untied': TokenTaskData(
                        tokens if easy else [random.randrange(vocab_size) for i in range(seq_len)], [True] * seq_len)
                },
                sentence_classification={'count': SentenceTaskData(seq_len % 2, seq_len - 1)}
            )

    pad_id = vocab_size + TextEncoder.PAD_OFFSET
    eos_id = vocab_size + TextEncoder.EOS_OFFSET
    generator = dummy_generator()
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            batch = _create_batch(batch, pad_id, max_len)
            yield batch
            batch = []
