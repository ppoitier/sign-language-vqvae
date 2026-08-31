from torch import nn, Tensor

from sl_vqvae.nn.quantizers.quantizer import Quantizer, QuantizerEMA, QuantizerOutput


class BodyPartsQuantizer(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 512,
        n_hand_embeddings: int = 1000,
        n_pose_embeddings: int = 500,
        commitment_loss_factor: float = 0.25,
        quantization_loss_factor: float = 1.0,
        use_quantizer_ema: bool = False,
        quantizer_ema_decay: float = 0.99,
    ):
        super().__init__()
        if use_quantizer_ema:
            hand_quantizer = QuantizerEMA(
                n_embeddings=n_hand_embeddings,
                embedding_dim=embedding_dim,
                commitment_loss_factor=commitment_loss_factor,
                decay=quantizer_ema_decay,
            )
            pose_quantizer = QuantizerEMA(
                n_embeddings=n_pose_embeddings,
                embedding_dim=embedding_dim,
                commitment_loss_factor=commitment_loss_factor,
                decay=quantizer_ema_decay,
            )
        else:
            hand_quantizer = Quantizer(
                n_embeddings=n_hand_embeddings,
                embedding_dim=embedding_dim,
                commitment_loss_factor=commitment_loss_factor,
                quantization_loss_factor=quantization_loss_factor,
            )
            pose_quantizer = Quantizer(
                n_embeddings=n_pose_embeddings,
                embedding_dim=embedding_dim,
                commitment_loss_factor=commitment_loss_factor,
                quantization_loss_factor=quantization_loss_factor,
            )
        self.quantizers = nn.ModuleDict(
            {
                "upper_pose": pose_quantizer,
                "left_hand": hand_quantizer,
                "right_hand": hand_quantizer,
            }
        )

    def forward(self, poses: dict[str, Tensor]) -> dict[str, QuantizerOutput]:
        out = dict()
        for body_part, quantizer in self.quantizers.items():
            out[body_part] = quantizer(poses[body_part])
        return out


if __name__ == "__main__":
    import torch

    N, T, D = 16, 500, 512
    x = {
        'upper_pose': torch.randn(N, T, D),
        'left_hand': torch.randn(N, T, D),
        'right_hand': torch.randn(N, T, D),
    }
    model = BodyPartsQuantizer()
    out = model(x)
    print(out)
