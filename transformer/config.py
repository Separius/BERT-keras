# I'm putting unknown parameters here so they can easily be updated once the official code is out
class BERTConfig:
    USE_ONE_DROPOUT = False  # assuming it's like the open ai
    IGNORE_MASK = False  # TBH I think they can be ignored but I'm following bert-pytorch

# TODO pretrain*2, test(against openAI + unit tests), sentence_level_train, token_level_train, QRNN, readme
