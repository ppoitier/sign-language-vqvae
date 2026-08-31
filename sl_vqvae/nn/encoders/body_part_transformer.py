import torch
from torch import nn, Tensor

from sl_vqvae.nn.encoders.transformer import PoseTransformerEncoder


class BodyPartTransformerEncoder(nn.Module):
    """
    Transformer encoder with one embedding stream per modality (pose, hand).

    The 'hand' sub-encoder is shared between 'left_hand' and 'right_hand' --
    the data loader already flips the left hand to the right-hand convention,
    so both can be modeled by the same weights.
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

        pose_encoder = PoseTransformerEncoder(
            c_in=n_pose_landmarks * n_coordinates,
            c_hidden=embedding_dim,
            max_length=max_length,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,
            attn_mask_strategy=attn_mask_strategy,
        )
        hand_encoder = PoseTransformerEncoder(
            c_in=n_hand_landmarks * n_coordinates,
            c_hidden=embedding_dim,
            max_length=max_length,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,
            attn_mask_strategy=attn_mask_strategy,
        )
        self.encoders = nn.ModuleDict(
            {
                "upper_pose": pose_encoder,
                "left_hand": hand_encoder,
                "right_hand": hand_encoder,
            }
        )

    def forward(self, poses: dict[str, Tensor], mask: Tensor) -> dict[str, Tensor]:
        out = dict()
        for body_part, x in poses.items():
            N, T, L, C = x.shape
            x = x.reshape(N, T, L * C)
            out[body_part] = self.encoders[body_part](x, mask)
        return out


if __name__ == "__main__":
    N, T, L_pose, L_hand, C = 16, 500, 23, 21, 2
    x = {
        "upper_pose": torch.randn(N, T, L_pose, C).cuda(),
        "left_hand": torch.randn(N, T, L_hand, C).cuda(),
        "right_hand": torch.randn(N, T, L_hand, C).cuda(),
    }
    mask = torch.ones(N, T).bool().cuda()

    model = BodyPartTransformerEncoder(embedding_dim=256, max_length=T).cuda()
    out = model(x, mask)
    print({k: v.shape for k, v in out.items()})
