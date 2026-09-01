from torch import nn

from sl_vqvae.config import CSLRModelConfig
from sl_vqvae.nn.cslr.transformer import CSLRPoseTransformer


def build_model(config: CSLRModelConfig) -> nn.Module:
    kwargs = config.model_dump(exclude={"pretrained_bert_checkpoint"})
    kwargs["body_parts"] = tuple(kwargs["body_parts"])
    model = CSLRPoseTransformer(**kwargs)
    if config.pretrained_bert_checkpoint:
        model.load_pretrained_bert(config.pretrained_bert_checkpoint)
    return model
