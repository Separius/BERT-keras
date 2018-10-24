import json
import keras
import numpy as np
from data.vocab import TextEncoder
from transformer.embedding import Embedding
from transformer.multihead_attention import shape_list
from transformer.config import OpenAIConfig, BERTConfig


def load(path: str, is_causal: bool = True):
    shapes = json.load(open(path + 'params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(path + 'params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    pos_embedding = init_params[0]  # (512, 768)
    token_embedding = init_params[1]  # (40478, 768)
    token_embedding = np.concatenate(
        (np.random.randn(TextEncoder.SPECIAL_COUNT, OpenAIConfig.EMBEDDING_SIZE) * 0.02, token_embedding).astype(
            np.float32), axis=0)
    max_len = OpenAIConfig.MAX_LEN
    # tokens = keras.layers.Input(shape=(max_len,))
    # segment_ids = keras.layers.Input(shape=(max_len,))
    # masks = keras.layers.Input(shape=(max_len,))
    # x = Embedding(output_dim=OpenAIConfig.EMBEDDING_SIZE,
    #               dropout=OpenAIConfig.EMBEDDING_DROPOUT,
    #               vocab_size=OpenAIConfig.VOCAB_SIZE,
    #               num_segments=BERTConfig.NUM_SEGMENTS,
    #               max_len=OpenAIConfig.MAX_LEN,
    #               trainable_pos_embedding=OpenAIConfig.TRAINABLE_POS_EMBEDDING,
    #               use_one_dropout=OpenAIConfig.USE_ONE_EMBEDDING_DROPOUT,
    #               pos_emb_weight=pos_embedding,
    #               token_emb_weight=token_embedding,
    #               segment_emb_weight=np.zeros(
    #                   (BERTConfig.NUM_SEGMENTS, OpenAIConfig.EMBEDDING_SIZE)).astype(np.float32))(tokens, segment_ids)
    shape = shape_list(x)
    del init_params[1]
    del init_params[0]
    for _ in range(OpenAIConfig.NUM_LAYERS):
        pass
