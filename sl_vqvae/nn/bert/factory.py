from torch import nn

from sl_vqvae.config import BERTModelConfig
from sl_vqvae.nn.bert.transformer import BERTPoseTransformer


def build_model(config: BERTModelConfig) -> nn.Module:
    return BERTPoseTransformer(**config.model_dump())
