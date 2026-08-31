import torch
from torch import nn, Tensor

from sl_vqvae.nn.decoders.transformer import PoseTransformerDecoder
from sl_vqvae.nn.quantizers.quantizer import QuantizerOutput


class BodyPartTransformerDecoder(nn.Module):
    """
    Transformer decoder with one reconstruction stream per modality (pose, hand).

    Mirrors `BodyPartTransformerEncoder`: the 'hand' sub-decoder is shared
    between 'left_hand' and 'right_hand'.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        max_length: int = 500,
        n_pose_landmarks: int = 23,
        n_hand_landmarks: int = 21,
        n_coordinates: int = 2,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pos_encoding: str = "rope",
        attn_mask_strategy: str | None = None,
    ):
        super().__init__()
        self.n_coordinates = n_coordinates
        self.n_landmarks = {
            "upper_pose": n_pose_landmarks,
            "left_hand": n_hand_landmarks,
            "right_hand": n_hand_landmarks,
        }

        pose_decoder = PoseTransformerDecoder(
            c_in=embedding_dim,
            c_out=n_pose_landmarks * n_coordinates,
            c_hidden=embedding_dim,
            max_length=max_length,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,
            attn_mask_strategy=attn_mask_strategy,
        )
        hand_decoder = PoseTransformerDecoder(
            c_in=embedding_dim,
            c_out=n_hand_landmarks * n_coordinates,
            c_hidden=embedding_dim,
            max_length=max_length,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,
            attn_mask_strategy=attn_mask_strategy,
        )
        self.decoders = nn.ModuleDict(
            {
                "upper_pose": pose_decoder,
                "left_hand": hand_decoder,
                "right_hand": hand_decoder,
            }
        )

    def forward(
        self, quantized: dict[str, QuantizerOutput], mask: Tensor
    ) -> dict[str, Tensor]:
        out = dict()
        for body_part, quantizer_out in quantized.items():
            z_q = quantizer_out.quantized_vectors
            N, T, _ = z_q.shape
            x_hat = self.decoders[body_part](z_q, mask)
            out[body_part] = x_hat.reshape(N, T, self.n_landmarks[body_part], self.n_coordinates)
        return out


if __name__ == "__main__":
    from sl_vqvae.nn.quantizers.quantizer import QuantizerOutput

    N, T, D = 16, 500, 256
    quantized = {
        "upper_pose": QuantizerOutput(torch.randn(N, T, D).cuda(), torch.zeros(N, T), torch.tensor(0.0)),
        "left_hand": QuantizerOutput(torch.randn(N, T, D).cuda(), torch.zeros(N, T), torch.tensor(0.0)),
        "right_hand": QuantizerOutput(torch.randn(N, T, D).cuda(), torch.zeros(N, T), torch.tensor(0.0)),
    }
    mask = torch.ones(N, T).bool().cuda()

    model = BodyPartTransformerDecoder(embedding_dim=D, max_length=T).cuda()
    out = model(quantized, mask)
    print({k: v.shape for k, v in out.items()})
