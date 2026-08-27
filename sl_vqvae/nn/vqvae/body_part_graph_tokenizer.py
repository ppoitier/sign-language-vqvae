from dataclasses import dataclass

from torch import nn, Tensor

from sl_vqvae.nn.encoders.gcn import PoseEmbeddingLayer
from sl_vqvae.nn.quantizers.body_parts_quantizer import BodyPartsQuantizer
from sl_vqvae.nn.quantizers.quantizer import QuantizerOutput
from sl_vqvae.nn.decoders.mlp_decoder import MLPDecoder


@dataclass
class TokenizerOutput:
    """Everything a downstream trainer / criterion / metric needs.

    reconstructions:    body_part -> (N, T, L, C) reconstructed coordinates.
    quantizer_outputs:  body_part -> QuantizerOutput, carrying the per-part VQ
                        loss and the discrete code indices (N, T).
    """

    reconstructions: dict[str, Tensor]
    quantizer_outputs: dict[str, QuantizerOutput]


class BodyPartGraphTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PoseEmbeddingLayer()
        self.quantizer = BodyPartsQuantizer()
        self.decoder = MLPDecoder()

    @property
    def codebook_sizes(self) -> dict[str, int]:
        return {
            body_part: quantizer.n_embeddings
            for body_part, quantizer in self.quantizer.quantizers.items()
        }

    def forward(self, x: dict[str, Tensor]) -> TokenizerOutput:
        z: dict[str, Tensor] = self.encoder(x)
        z_q: dict[str, QuantizerOutput] = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return TokenizerOutput(reconstructions=x_recon, quantizer_outputs=z_q)


if __name__ == '__main__':
    import torch

    N, T, L_pose, L_hand, C = 16, 500, 23, 21, 2
    x = {
        'upper_pose': torch.randn(N, T, L_pose, C),
        'left_hand': torch.randn(N, T, L_hand, C),
        'right_hand': torch.randn(N, T, L_hand, C),
    }

    model = BodyPartGraphTokenizer()
    out = model(x)

    print({k: v.shape for k, v in out.reconstructions.items()})
    print({k: q.loss.item() for k, q in out.quantizer_outputs.items()})
