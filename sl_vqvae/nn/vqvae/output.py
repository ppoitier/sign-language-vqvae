from dataclasses import dataclass

from torch import Tensor

from sl_vqvae.nn.quantizers.quantizer import QuantizerOutput


@dataclass
class VQVAEOutput:
    reconstructions: dict[str, Tensor]
    quantizer_outputs: dict[str, QuantizerOutput]
    total_loss: Tensor
    reconstruction_loss: Tensor
    quantizer_loss: Tensor
