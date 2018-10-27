import os
import re
import random
import tempfile
import numpy as np
from unittest import TestCase
from typing import Optional, List
from data.vocab import TextEncoder
from data.dataset import (create_attention_mask, Sentence, pad, check_sent_len,
                          msk_sentence, SentenceTaskData, TokenTaskData)
from data.lm_dataset import _create_batch, _grab_line, make_next_token_prediction, dummy_lm_generator


class TestData(TestCase):
    def __init__(self, method_name: str = 'runTest') -> None:
        super().__init__(methodName=method_name)
        self.vocab_size = 100

    def setUp(self) -> None:
        pass

    def generate_random_seq(self, length: int, max: Optional[int] = None) -> List[int]:
        return [random.randrange(self.vocab_size if max is None else max) for i in range(length)]

    def generate_random_mask(self, length: int) -> List[bool]:
        return [random.random() < 0.5 for _ in range(length)]

    def generate_sentence(self, length: int) -> Sentence:
        return Sentence(self.generate_random_seq(length), [True] * length, [0] * length,
                        {'lm': TokenTaskData(self.generate_random_seq(length),
                                             self.generate_random_mask(length))}, {})

    def test_pad(self):
        bert_sent = self.generate_sentence(5)
        lm_orig = bert_sent.token_classification['lm']
        pad_id = self.vocab_size + TextEncoder.PAD_OFFSET
        padded_sent = pad(bert_sent, pad_id, 10)
        lm = padded_sent.token_classification['lm']
        assert len(padded_sent.padding_mask) == len(padded_sent.segments) == len(lm.target_mask) == len(
            padded_sent.tokens) == len(lm.target) == 10
        for i in range(5):
            assert padded_sent.padding_mask[i]
            assert padded_sent.segments[i] == bert_sent.segments[i]
            assert lm.target[i] == lm_orig.target[i]
            assert lm.target_mask[i] == lm_orig.target_mask[i]
            assert padded_sent.tokens[i] == bert_sent.tokens[i]
        for i in range(5, 10):
            assert not padded_sent.padding_mask[i]
            assert padded_sent.segments[i] == 0
            assert lm.target[i] == 0
            assert lm.target_mask[i] == 0
            assert padded_sent.tokens[i] == pad_id

    def test_create_batch(self):
        max_len = 64
        pad_id = self.vocab_size + TextEncoder.PAD_OFFSET
        for batch_size in [32, 1]:
            sentences = []
            for i in range(batch_size):
                sentences.append(self.generate_sentence(random.randint(1, max_len - 5)))
            for i in range(2):
                if i == 0:
                    batch = _create_batch(sentences, pad_id, max_len)
                else:
                    batch = _create_batch(sentences, pad_id)
                    max_len = max([len(sent.tokens) for sent in sentences])
                assert batch.tokens.shape == (batch_size, max_len)
                assert batch.tokens.dtype == np.int32
                assert batch.segments.shape == (batch_size, max_len)
                assert batch.segments.dtype == np.int32
                assert batch.padding_mask.shape == (batch_size, max_len)
                assert batch.padding_mask.dtype == np.int8
                assert batch.token_classification['lm'].target.shape == (batch_size, max_len)
                assert batch.token_classification['lm'].target.dtype == np.int32
                assert batch.token_classification['lm'].target_mask.shape == (batch_size, max_len)
                assert batch.token_classification['lm'].target_mask.dtype == np.int32

    def test_msk_sentence(self):
        seq_len = 32
        sentence = self.generate_random_seq(seq_len)

        masked_sentence = msk_sentence(sentence, vocab_size=self.vocab_size, keep_prob=1.0, mask_prob=0.0,
                                       rand_prob=0.0)
        assert len(sentence) == len(masked_sentence.tokens) == len(
            masked_sentence.token_classification['lm'].target) == len(
            masked_sentence.token_classification['lm'].target_mask)
        for i in range(seq_len):
            assert masked_sentence.tokens[i] == sentence[i]
            assert masked_sentence.token_classification['lm'].target[i] == 0
            assert masked_sentence.token_classification['lm'].target_mask[i] == 0

        masked_sentence = msk_sentence(sentence, vocab_size=self.vocab_size, keep_prob=0.0, mask_prob=1.0,
                                       rand_prob=0.0)
        assert len(sentence) == len(masked_sentence.tokens) == len(
            masked_sentence.token_classification['lm'].target) == len(
            masked_sentence.token_classification['lm'].target_mask)
        for i in range(seq_len):
            assert masked_sentence.tokens[i] == self.vocab_size + TextEncoder.MSK_OFFSET
            assert masked_sentence.token_classification['lm'].target[i] == sentence[i]
            assert masked_sentence.token_classification['lm'].target_mask[i] == 1

        masked_sentence = msk_sentence(sentence, vocab_size=self.vocab_size, keep_prob=0.0, mask_prob=0.0,
                                       rand_prob=0.0)
        assert len(sentence) == len(masked_sentence.tokens) == len(
            masked_sentence.token_classification['lm'].target) == len(
            masked_sentence.token_classification['lm'].target_mask)
        for i in range(seq_len):
            assert masked_sentence.tokens[i] == sentence[i]
            assert masked_sentence.token_classification['lm'].target[i] == sentence[i]
            assert masked_sentence.token_classification['lm'].target_mask[i] == 1

        sentence = [index + self.vocab_size for index in sentence]
        masked_sentence = msk_sentence(sentence, vocab_size=self.vocab_size, keep_prob=0.0, mask_prob=0.0,
                                       rand_prob=1.0)
        assert len(sentence) == len(masked_sentence.tokens) == len(
            masked_sentence.token_classification['lm'].target) == len(
            masked_sentence.token_classification['lm'].target_mask)
        for i in range(seq_len):
            assert masked_sentence.tokens[i] != sentence[i]
            assert masked_sentence.token_classification['lm'].target[i] == sentence[i]
            assert masked_sentence.token_classification['lm'].target_mask[i] == 1

    def test_make_causal(self):
        pad_id = self.vocab_size + TextEncoder.PAD_OFFSET
        orig_sentence = self.generate_sentence(5)
        result = _create_batch(make_next_token_prediction([orig_sentence]), pad_id)
        lm = result.token_classification['lm']
        assert (np.array(orig_sentence.tokens)[1:] == lm.target[0, :-1]).all()
        assert lm.target[0, -1] == 0
        assert (lm.target_mask[0, :-1] == 1).all()
        assert lm.target_mask[0, -1] == 0

    def test_grab_line(self):
        fp1 = tempfile.TemporaryFile(mode='w+')
        fp2 = tempfile.TemporaryFile(mode='w+')
        for i in range(100):
            fp1.write('hello world {}!\n'.format(i))
            fp2.write('hi universe {}!\n'.format(i))
        fp1.seek(0)
        fp2.seek(0)
        for i in range(200):
            line = _grab_line([fp1], os.stat(fp1.fileno()).st_size, jump_prob=0.0)
            assert line == 'hello world {}!\n'.format(i % 100)
        fp1.seek(0)
        i = j = 0
        for _ in range(200):
            line = _grab_line([fp1, fp2], os.stat(fp1.fileno()).st_size, jump_prob=0.0)
            if line.startswith('hello'):
                assert line == 'hello world {}!\n'.format(i % 100)
                i += 1
            else:
                assert line == 'hi universe {}!\n'.format(j % 100)
                j += 1
        fp1.seek(0)
        fp2.seek(0)
        pattern = re.compile('(hello world)|(hi universe) \d+!\\n')
        for _ in range(200):
            line = _grab_line([fp1, fp2], os.stat(fp1.fileno()).st_size, jump_prob=1.0)
            assert pattern.match(line) is not None
        fp1.close()
        fp2.close()

    def test_create_mask(self):
        batch_size = 3
        length = 5
        pad_mask = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=np.int8)
        is_causal = False
        mask = create_attention_mask(pad_mask, is_causal)
        assert mask.shape == (batch_size, 1, length, length)
        assert mask.dtype == np.float32
        assert (mask[0, 0] == np.array([[1, 1, 1, 0, 0],
                                        [1, 1, 1, 0, 0],
                                        [1, 1, 1, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]], dtype=np.float32)).all()
        assert (mask[1, 0] == np.array([[1, 1, 0, 0, 1],
                                        [1, 1, 0, 0, 1],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [1, 1, 0, 0, 1]], dtype=np.float32)).all()
        assert (mask[2, 0] == np.zeros((length, length), dtype=np.float32)).all()

        is_causal = True
        mask = create_attention_mask(pad_mask, is_causal)
        assert mask.shape == (batch_size, 1, length, length)
        assert mask.dtype == np.float32
        assert (mask[0, 0] == np.array([[1, 0, 0, 0, 0],
                                        [1, 1, 0, 0, 0],
                                        [1, 1, 1, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]], dtype=np.float32)).all()
        assert (mask[1, 0] == np.array([[1, 0, 0, 0, 0],
                                        [1, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [1, 1, 0, 0, 1]], dtype=np.float32)).all()
        assert (mask[2, 0] == np.zeros((length, length), dtype=np.float32)).all()

        is_causal = False
        mask = create_attention_mask(None, is_causal, batch_size, length)
        assert mask.shape == (batch_size, 1, length, length)
        assert mask.dtype == np.float32
        for i in range(3):
            assert (mask[i, 0] == np.ones((length, length), dtype=np.float32)).all()

        is_causal = True
        mask = create_attention_mask(None, is_causal, batch_size, length)
        assert mask.shape == (batch_size, 1, length, length)
        assert mask.dtype == np.float32
        tri = np.array([[1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1]], dtype=np.float32)
        for i in range(3):
            assert (mask[i, 0] == tri).all()

    def test_check_sent_len(self):
        orig_length = 10
        class_target = 2
        original_sent = self.generate_sentence(orig_length)
        original_sent.sentence_classification['sc'] = SentenceTaskData(class_target, 0)
        original_sent.sentence_classification['sc_ok'] = SentenceTaskData(class_target + 1, 5)
        assert check_sent_len(original_sent, min_len=10, max_len=None) is not None
        assert check_sent_len(original_sent, min_len=11, max_len=None) is None
        res = check_sent_len(original_sent, min_len=None, max_len=7, from_end=False)
        assert len(res.tokens) == len(res.padding_mask) == len(res.token_classification['lm'].target) == len(
            res.token_classification['lm'].target_mask) == 7
        assert res.tokens[0] == original_sent.tokens[3]
        assert set(res.sentence_classification.keys()) == {'sc_ok'}
        assert res.sentence_classification['sc_ok'].target == class_target + 1
        assert res.sentence_classification['sc_ok'].target_index == 5 - 3

    def test_generation(self):
        lm_generator = dummy_lm_generator(self.vocab_size, 32, 32, 100)
        for i, sentence_batch in enumerate(lm_generator):
            assert sentence_batch.tokens.shape == (32, 32)
        assert i == 100 // 32 - 1
