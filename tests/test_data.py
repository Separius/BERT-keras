import os
import re
import copy
import random
import tempfile
import numpy as np
from unittest import TestCase
from typing import Optional, List
from data.vocab import TextEncoder
from data.lm_dataset import _create_batch, _grab_line, make_next_token_prediction
from data.dataset import create_attention_mask, BertSentence, Sentence, pad, msk_sentence


class TestData(TestCase):
    def __init__(self, method_name: str = 'runTest') -> None:
        super().__init__(methodName=method_name)
        self.vocab_size = 100

    def setUp(self) -> None:
        pass

    def generate_random_seq(self, length: int, max: Optional[int] = None) -> List[int]:
        return [random.randrange(self.vocab_size if max is None else max) for i in range(length)]

    def get_a_bert_sentence(self, length: int) -> BertSentence:
        return BertSentence(Sentence(self.generate_random_seq(length), self.generate_random_seq(length)), True,
                            self.generate_random_seq(length, 2))

    def test_pad(self):
        bert_sent = self.get_a_bert_sentence(5)
        pad_id = self.vocab_size + TextEncoder.PAD_OFFSET
        padded_sent = pad(bert_sent, pad_id, 10)
        assert len(padded_sent.mask) == len(padded_sent.segment_id) == len(padded_sent.sentence.lm_target) == len(
            padded_sent.sentence.tokens) == 10
        for i in range(5):
            assert padded_sent.mask[i]
            assert padded_sent.segment_id[i] == bert_sent.segment_id[i]
            assert padded_sent.sentence.lm_target[i] == bert_sent.sentence.lm_target[i]
            assert padded_sent.sentence.tokens[i] == bert_sent.sentence.tokens[i]
        for i in range(5, 10):
            assert not padded_sent.mask[i]
            assert padded_sent.segment_id[i] == pad_id
            assert padded_sent.sentence.lm_target[i] == pad_id
            assert padded_sent.sentence.tokens[i] == pad_id

    def test_create_batch(self):
        batch_size = 32
        max_len = 64
        sentences = []
        pad_id = self.vocab_size + TextEncoder.PAD_OFFSET
        for i in range(batch_size):
            sentences.append(self.get_a_bert_sentence(random.randint(1, max_len - 5)))
        for i in range(2):
            if i == 0:
                batch = _create_batch(sentences, pad_id, max_len)
            else:
                batch = _create_batch(sentences, pad_id)
                max_len = max([len(sent.segment_id) for sent in sentences])
            assert batch.masks.shape == (batch_size, max_len)
            assert batch.masks.dtype == np.int8
            assert batch.segment_ids.shape == (batch_size, max_len)
            assert batch.segment_ids.dtype == np.int64
            assert batch.tokens.shape == (batch_size, max_len)
            assert batch.tokens.dtype == np.int64
            assert batch.is_next.shape == (batch_size,)
            assert batch.is_next.dtype == np.float32
            assert batch.lm_targets.shape == (batch_size, max_len)
            assert batch.lm_targets.dtype == np.int64

    def test_msk_sentence(self):
        seq_len = 32
        sentence = self.generate_random_seq(seq_len)

        masked_sentence = msk_sentence(sentence, vocab_size=self.vocab_size, keep_prob=1.0, mask_prob=0.0,
                                        rand_prob=0.0)
        assert len(sentence) == len(masked_sentence.tokens) == len(masked_sentence.lm_target)
        for i in range(seq_len):
            assert masked_sentence.tokens[i] == sentence[i]
            assert masked_sentence.lm_target[i] == self.vocab_size + TextEncoder.PAD_OFFSET

        masked_sentence = msk_sentence(sentence, vocab_size=self.vocab_size, keep_prob=0.0, mask_prob=1.0,
                                        rand_prob=0.0)
        assert len(sentence) == len(masked_sentence.tokens) == len(masked_sentence.lm_target)
        for i in range(seq_len):
            assert masked_sentence.tokens[i] == self.vocab_size + TextEncoder.MSK_OFFSET
            assert masked_sentence.lm_target[i] == sentence[i]

        masked_sentence = msk_sentence(sentence, vocab_size=self.vocab_size, keep_prob=0.0, mask_prob=0.0,
                                        rand_prob=0.0)
        assert len(sentence) == len(masked_sentence.tokens) == len(masked_sentence.lm_target)
        for i in range(seq_len):
            assert masked_sentence.tokens[i] == sentence[i]
            assert masked_sentence.lm_target[i] == sentence[i]

        sentence = [index + self.vocab_size for index in sentence]
        masked_sentence = msk_sentence(sentence, vocab_size=self.vocab_size, keep_prob=0.0, mask_prob=0.0,
                                        rand_prob=1.0)
        assert len(sentence) == len(masked_sentence.tokens) == len(masked_sentence.lm_target)
        for i in range(seq_len):
            assert masked_sentence.tokens[i] != sentence[i]
            assert masked_sentence.lm_target[i] == sentence[i]

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

    def test_make_causal(self):
        pad_id = self.vocab_size + TextEncoder.PAD_OFFSET
        orig_sentence = self.get_a_bert_sentence(5)
        result = make_next_token_prediction(copy.deepcopy(orig_sentence), pad_id=pad_id)
        assert (np.array(orig_sentence.sentence.tokens)[1:] == np.array(result.sentence.lm_target)[:-1]).all()
        assert result.sentence.lm_target[-1] == pad_id

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
