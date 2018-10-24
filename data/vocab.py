import os
import re
import json
from typing import List

try:
    import sentencepiece as spm
except ImportError:
    print('if you want sentencepiece encoder, please install sentencepiece')

try:
    import ftfy
    import spacy
except ImportError:
    print("if you want to use OpenAI's encoder and pretrained model, please install spacy, and ftfy")


# TOKEN_IDs = {unk=0, vocab={1..vocab_size-1}, specials(pad,bos,eos,msk), segments, positions}


class TextEncoder:
    PAD_OFFSET = 0
    MSK_OFFSET = 1
    BOS_OFFSET = 2
    EOS_OFFSET = 3
    SGA_OFFSET = 4  # Segment_A
    SGB_OFFSET = 5  # Segment_A
    POS_START_OFFSET = 6

    def __init__(self, vocab_size: int):
        # NOTE you MUST always put unk at 0, then regular vocab, then special chars, and then pos
        self.vocab_size = vocab_size
        self.unk_id = 0
        self.pad_id = vocab_size + self.PAD_OFFSET
        self.msk_id = vocab_size + self.MSK_OFFSET
        self.bos_id = vocab_size + self.BOS_OFFSET
        self.eos_id = vocab_size + self.EOS_OFFSET
        self.sga_id = vocab_size + self.SGA_OFFSET
        self.sgb_id = vocab_size + self.SGB_OFFSET
        self.pos_start_id = vocab_size + self.POS_START_OFFSET

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, sent: str) -> List[int]:
        raise NotImplementedError()


class SentencePieceTextEncoder(TextEncoder):
    def __init__(self, text_corpus_address: str, model_name: str = 'spm',
                 vocab_size: int = 30000, model_type: str = 'unigram'):
        super().__init__(vocab_size)
        if not os.path.exists('{}.model'.format(model_name)):
            if model_type.lower() not in ('unigram', 'bpe', 'char', 'word'):
                raise ValueError(
                    '{} is not a valid model_type for sentence piece, '
                    'valid options are: unigram, bpe, char, word'.format(model_type))
            spm.SentencePieceTrainer.Train(
                '--input={input} --model_prefix={model_name} --vocab_size={vocab_size} '
                '--character_coverage={coverage} --model_type={model_type} '
                '--pad_id=-1 --unk_id=0 --bos_id=-1 --eos_id=-1 --input_sentence_size=100000000 '
                '--training_sentence_size=100000000 --control_symbols=@@@<MASK>@@@'.format(
                    input=text_corpus_address, model_name=model_name, vocab_size=vocab_size, coverage=1,
                    model_type=model_type.lower()))
        self.sp = spm.SentencePieceProcessor()
        self.sp.load('{}.model'.format(model_name))

    def encode(self, sent: str) -> List[int]:
        return self.sp.encode_as_ids(sent)


# borrowed from https://github.com/openai/finetune-transformer-lm/blob/master/text_utils.py


class OpenAITextEncoder(TextEncoder):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path: str = './data/encoder_bpe_40000.json',
                 bpe_path: str = './data/vocab_40000.bpe') -> None:
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(encoder_path))
        super().__init__(len(self.encoder))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(bpe_path).read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    @staticmethod
    def _get_pairs(word):
        """
        Return set of symbol pairs in a word.
        word is represented as tuple of symbols (symbols being variable-length strings)
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    @staticmethod
    def _text_standardize(text: str) -> str:
        """
        fixes some issues the spacy tokenizer had on books corpus
        also does some whitespace standardization
        """
        text = text.replace('—', '-')
        text = text.replace('–', '-')
        text = text.replace('―', '-')
        text = text.replace('…', '...')
        text = text.replace('´', "'")
        text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
        text = re.sub('\s*\n\s*', ' \n ', text)
        text = re.sub('[^\S\n]+', ' ', text)
        return text.strip()

    def _bpe(self, token: str):
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = self._get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def encode(self, sent: str) -> List[int]:
        text = self.nlp(self._text_standardize(ftfy.fix_text(sent)))
        text_tokens = []
        for token in text:
            text_tokens.extend([self.encoder.get(t, 0) for t in self._bpe(token.text.lower()).split(' ')])
        return text_tokens
