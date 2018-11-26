import os
from typing import List, Optional

try:
    import sentencepiece as spm
except:
    print('if you want sentencepiece encoder, please install sentencepiece')

try:
    from openai.text_utils import TextEncoder as _OpenAITextEncoder
except:
    print('if you want to use OpenAI\'s encoder and pretrained model, please install spacy, and ftfy')

try:
    from google_bert.tokenization import FullTokenizer
except:
    print('if you want to use Google\'s encoder and pretrained models, please clone the bert submodule')


# TOKEN_IDs = {unk=0, vocab={1..vocab_size-1}, specials(pad,bos,del,eos,msk)}


class TextEncoder:
    PAD_OFFSET = 0
    MSK_OFFSET = 1
    BOS_OFFSET = 2
    DEL_OFFSET = 3  # delimiter
    EOS_OFFSET = 4
    SPECIAL_COUNT = 5
    NUM_SEGMENTS = 2
    BERT_UNUSED_COUNT = 99  # bert pretrained models
    BERT_SPECIAL_COUNT = 4  # they don't have DEL

    def __init__(self, vocab_size: int):
        # NOTE you MUST always put unk at 0, then regular vocab, then special tokens, and then pos
        self.vocab_size = vocab_size
        self.unk_id = 0
        self.pad_id = vocab_size + self.PAD_OFFSET
        self.msk_id = vocab_size + self.MSK_OFFSET
        self.bos_id = vocab_size + self.BOS_OFFSET
        self.del_id = vocab_size + self.DEL_OFFSET
        self.eos_id = vocab_size + self.EOS_OFFSET

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, sent: str) -> List[int]:
        raise NotImplementedError()


class SentencePieceTextEncoder(TextEncoder):
    def __init__(self, text_corpus_address: Optional[str], model_name: str = 'spm',
                 vocab_size: int = 30000, spm_model_type: str = 'unigram') -> None:
        super().__init__(vocab_size)
        if not os.path.exists('{}.model'.format(model_name)):
            if spm_model_type.lower() not in ('unigram', 'bpe', 'char', 'word'):
                raise ValueError(
                    '{} is not a valid model_type for sentence piece, '
                    'valid options are: unigram, bpe, char, word'.format(spm_model_type))
            spm.SentencePieceTrainer.Train(
                '--input={input} --model_prefix={model_name} --vocab_size={vocab_size} '
                '--character_coverage={coverage} --model_type={model_type} '
                '--pad_id=-1 --unk_id=0 --bos_id=-1 --eos_id=-1 --input_sentence_size=100000000 '
                '--training_sentence_size=100000000'.format(
                    input=text_corpus_address, model_name=model_name, vocab_size=vocab_size, coverage=1,
                    model_type=spm_model_type.lower()))
        self.sp = spm.SentencePieceProcessor()
        self.sp.load('{}.model'.format(model_name))

    def encode(self, sent: str) -> List[int]:
        return self.sp.encode_as_ids(sent)


class OpenAITextEncoder(TextEncoder):
    def __init__(self, encoder_path: str = './openai/model/encoder_bpe_40000.json',
                 bpe_path: str = './openai/model/vocab_40000.bpe') -> None:
        self.encoder = _OpenAITextEncoder(encoder_path, bpe_path)
        super().__init__(len(self.encoder.encoder))

    def encode(self, sent: str) -> List[int]:
        return self.encoder.encode([sent], verbose=False)[0]


class BERTTextEncoder(TextEncoder):
    def __init__(self, vocab_file: str, do_lower_case: bool = True) -> None:
        self.tokenizer = FullTokenizer(vocab_file, do_lower_case)
        super().__init__(len(self.tokenizer.vocab))
        self.bert_unk_id = self.tokenizer.vocab['[UNK]']
        self.bert_msk_id = self.tokenizer.vocab['[MASK]']

    def standardize_ids(self, ids: List[int]) -> List[int]:
        for i in range(len(ids)):
            if ids[i] == self.bert_unk_id:  # UNK
                ids[i] = 0
            else:  # VOCAB
                ids[i] -= self.bert_msk_id
        return ids

    def encode(self, sent: str) -> List[int]:
        return self.standardize_ids(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent)))
