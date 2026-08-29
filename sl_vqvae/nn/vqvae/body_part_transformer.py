import torch
from torch import nn, Tensor

from sl_vqvae.criterions.reconstruction import masked_reconstruction_loss
from sl_vqvae.nn.encoders.body_part_transformer import BodyPartTransformerEncoder
from sl_vqvae.nn.decoders.body_part_transformer import BodyPartTransformerDecoder
from sl_vqvae.nn.quantizers.body_parts_quantizer import BodyPartsQuantizer
from sl_vqvae.nn.quantizers.quantizer import QuantizerOutput
from sl_vqvae.nn.vqvae.output import VQVAEOutput


class BodyPartTransformerVQVAE(nn.Module):
    """
    VQ-VAE over pose sequences, with one embedding dictionary per modality.

    Input / output: body_part -> (N, T, L, C), e.g. 'upper_pose' (16, 500, 23, 2).
    Latent:         body_part -> (N, T, embedding_dim), one code per frame.

    Two embedding dictionaries (pose, hand) and two decoders (pose, hand):
    'left_hand' and 'right_hand' share the same encoder, quantizer and decoder
    instance, since the data loader already flips the left hand to match the
    right-hand convention.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        n_pose_embeddings: int = 500,
        n_hand_embeddings: int = 1000,
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
        reconstruction: str = "l2",
        reconstruction_weights: dict[str, float] | None = None,
        commitment_loss_factor: float = 0.25,
        quantization_loss_factor: float = 1.0,
        use_quantizer_ema: bool = True,
        quantizer_ema_decay: float = 0.99,
    ):
        super().__init__()
        if reconstruction not in ("l1", "l2"):
            raise ValueError(f"Unknown reconstruction loss: {reconstruction!r}")
        self.reconstruction = reconstruction
        self.reconstruction_weights = reconstruction_weights
        self.embedding_dim = embedding_dim

        self.encoder = BodyPartTransformerEncoder(
            embedding_dim=embedding_dim,
            max_length=max_length,
            n_pose_landmarks=n_pose_landmarks,
            n_hand_landmarks=n_hand_landmarks,
            n_coordinates=n_coordinates,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,
            attn_mask_strategy=attn_mask_strategy,
        )

        self.quantizer = BodyPartsQuantizer(
            embedding_dim=embedding_dim,
            n_hand_embeddings=n_hand_embeddings,
            n_pose_embeddings=n_pose_embeddings,
            commitment_loss_factor=commitment_loss_factor,
            quantization_loss_factor=quantization_loss_factor,
            use_quantizer_ema=use_quantizer_ema,
            quantizer_ema_decay=quantizer_ema_decay,
        )

        self.decoder = BodyPartTransformerDecoder(
            embedding_dim=embedding_dim,
            max_length=max_length,
            n_pose_landmarks=n_pose_landmarks,
            n_hand_landmarks=n_hand_landmarks,
            n_coordinates=n_coordinates,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,
            attn_mask_strategy=attn_mask_strategy,
        )

    @property
    def modality_groups(self) -> dict[str, list[str]]:
        return {"pose": ["upper_pose"], "hand": ["left_hand", "right_hand"]}

    def n_embeddings(self, body_part: str) -> int:
        return self.quantizer.quantizers[body_part].n_embeddings

    def loss_function(
        self,
        poses: dict[str, Tensor],
        reconstructions: dict[str, Tensor],
        quantizer_outputs: dict[str, QuantizerOutput],
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        reconstruction_loss = masked_reconstruction_loss(
            poses, reconstructions, mask, kind=self.reconstruction, weights=self.reconstruction_weights
        )
        quantizer_loss = torch.stack([q.loss for q in quantizer_outputs.values()]).mean()
        total_loss = reconstruction_loss + quantizer_loss
        return total_loss, reconstruction_loss, quantizer_loss

    def forward(self, poses: dict[str, Tensor], mask: Tensor) -> VQVAEOutput:
        """
        Args:
            poses: body_part -> (N, T, L, C) pose sequence.
            mask:  (N, T) -- 1=valid, 0=pad.
        """
        z_e = self.encoder(poses, mask)

        quantizer_outputs = self.quantizer(z_e)

        reconstructions = self.decoder(quantizer_outputs, mask)

        total_loss, reconstruction_loss, quantizer_loss = self.loss_function(
            poses, reconstructions, quantizer_outputs, mask
        )

        return VQVAEOutput(
            reconstructions=reconstructions,
            quantizer_outputs=quantizer_outputs,
            total_loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            quantizer_loss=quantizer_loss,
        )


if __name__ == "__main__":
    N, T, L_pose, L_hand, C = 16, 500, 23, 21, 2
    poses = {
        "upper_pose": torch.randn(N, T, L_pose, C).cuda(),
        "left_hand": torch.randn(N, T, L_hand, C).cuda(),
        "right_hand": torch.randn(N, T, L_hand, C).cuda(),
    }
    mask = torch.ones(N, T).bool().cuda()

    model = BodyPartTransformerVQVAE(embedding_dim=256, max_length=T).cuda()
    out = model(poses, mask)

    print({k: v.shape for k, v in out.reconstructions.items()})
    print({k: q.quantized_indices.shape for k, q in out.quantizer_outputs.items()})
    print(out.total_loss)
