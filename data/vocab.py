import sentencepiece as spm

SPECIAL_TOKEN_IDS = {'pad': 0, 'msk': 1, 'bos': 2, 'eos': 3, 'unk': 4}


def create_sentence_piece_model(text_corpus_address: str, model_name: str = 'spm',
                                vocab_size: int = 30000, model_type: str = 'unigram'):
    if model_type.lower() not in ('unigram', 'bpe', 'char', 'word'):
        raise ValueError(
            '{} is not a valid model_type for sentence piece, '
            'valid options are: unigram, bpe, char, word'.format(model_type))
    spm.SentencePieceTrainer.Train(
        '--input={input} --model_prefix={model_name} --vocab_size={vocab_size} '
        '--character_coverage={coverage} --model_type={model_type} '
        '--pad_id=0 --unk_id=4 --bos_id=2 --eos_id=3 --input_sentence_size=100000000 '
        '--training_sentence_size=100000000 --control_symbols=@@@<MASK>@@@'.format(
            input=text_corpus_address, model_name=model_name, vocab_size=vocab_size, coverage=1,
            model_type=model_type.lower()))
