from torch import nn

from sl_vqvae.config import BodyPartTransformerModelConfig, TransformerModelConfig
from sl_vqvae.nn.vqvae.body_part_transformer import BodyPartTransformerVQVAE
from sl_vqvae.nn.vqvae.transformer import TransformerVQVAE


def build_model(config: TransformerModelConfig | BodyPartTransformerModelConfig) -> nn.Module:
    if isinstance(config, TransformerModelConfig):
        kwargs = config.model_dump(exclude={"type"})
        kwargs["body_parts"] = tuple(kwargs["body_parts"])
        return TransformerVQVAE(**kwargs)
    if isinstance(config, BodyPartTransformerModelConfig):
        return BodyPartTransformerVQVAE(**config.model_dump(exclude={"type"}))
    raise ValueError(f"Unknown model config type: {config!r}")
