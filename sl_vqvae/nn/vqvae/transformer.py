import torch
from torch import nn, Tensor

from sl_vqvae.criterions.reconstruction import masked_reconstruction_loss
from sl_vqvae.nn.encoders.transformer import PoseTransformerEncoder
from sl_vqvae.nn.decoders.transformer import PoseTransformerDecoder
from sl_vqvae.nn.quantizers.quantizer import QuantizerEMA
from sl_vqvae.nn.vqvae.output import VQVAEOutput


class TransformerVQVAE(nn.Module):
    """
    VQ-VAE over pose sequences, with a single embedding dictionary shared
    across all body parts.

    Input / output: body_part -> (N, T, L, C), e.g. 'upper_pose' (16, 500, 23, 2).
    Latent:         (N, T, embedding_dim), one shared code per frame.

    All body parts are concatenated into one vector per frame before encoding,
    quantized against a single codebook, and split back into a reconstruction
    per body part after decoding. `PoseTransformerEncoder`/`PoseTransformerDecoder`
    stay plain sequence transformers (tensor in, tensor out) -- the same
    classes are reused as-is, one instance per modality, by
    `BodyPartTransformerEncoder`/`Decoder`, which is why the concatenation
    logic lives here rather than inside them.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        n_embeddings: int = 1000,
        max_length: int = 500,
        body_parts: tuple[str, ...] = ("upper_pose", "left_hand", "right_hand"),
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
        quantizer_ema_decay: float = 0.99,
    ):
        super().__init__()
        if reconstruction not in ("l1", "l2"):
            raise ValueError(f"Unknown reconstruction loss: {reconstruction!r}")
        self.reconstruction = reconstruction
        self.reconstruction_weights = reconstruction_weights
        self.body_parts = body_parts
        self.embedding_dim = embedding_dim
        self.n_coordinates = n_coordinates
        self.n_landmarks = {
            "upper_pose": n_pose_landmarks,
            "left_hand": n_hand_landmarks,
            "right_hand": n_hand_landmarks,
        }
        c_in = sum(self.n_landmarks[body_part] * n_coordinates for body_part in body_parts)

        transformer_kwargs = dict(
            c_hidden=embedding_dim,
            max_length=max_length,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,
            attn_mask_strategy=attn_mask_strategy,
        )
        self.encoder = PoseTransformerEncoder(c_in=c_in, **transformer_kwargs)

        self.quantizer = QuantizerEMA(
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
            commitment_loss_factor=commitment_loss_factor,
            decay=quantizer_ema_decay,
        )

        self.decoder = PoseTransformerDecoder(c_in=embedding_dim, c_out=c_in, **transformer_kwargs)

    @property
    def modality_groups(self) -> dict[str, list[str]]:
        """All body parts share the single codebook."""
        return {"all": list(self.body_parts)}

    def n_embeddings(self, body_part: str) -> int:
        return self.quantizer.n_embeddings

    def _concat(self, poses: dict[str, Tensor]) -> Tensor:
        return torch.cat([poses[body_part].flatten(2) for body_part in self.body_parts], dim=-1)

    def _split(self, x: Tensor) -> dict[str, Tensor]:
        N, T, _ = x.shape
        out = dict()
        offset = 0
        for body_part in self.body_parts:
            L = self.n_landmarks[body_part]
            width = L * self.n_coordinates
            out[body_part] = x[..., offset : offset + width].reshape(N, T, L, self.n_coordinates)
            offset += width
        return out

    def loss_function(
        self,
        poses: dict[str, Tensor],
        reconstructions: dict[str, Tensor],
        quantizer_loss: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        reconstruction_loss = masked_reconstruction_loss(
            poses, reconstructions, mask, kind=self.reconstruction, weights=self.reconstruction_weights
        )
        total_loss = reconstruction_loss + quantizer_loss
        return total_loss, reconstruction_loss

    def forward(self, poses: dict[str, Tensor], mask: Tensor) -> VQVAEOutput:
        """
        Args:
            poses: body_part -> (N, T, L, C) pose sequence.
            mask:  (N, T) -- 1=valid, 0=pad.
        """
        x = self._concat(poses)
        z_e = self.encoder(x, mask)

        quantizer_output = self.quantizer(z_e)

        x_hat = self.decoder(quantizer_output.quantized_vectors, mask)
        reconstructions = self._split(x_hat)

        total_loss, reconstruction_loss = self.loss_function(
            poses, reconstructions, quantizer_output.loss, mask
        )

        # Same QuantizerOutput under every key: all body parts share one
        # codebook, so downstream per-body-part metric code can still read
        # `quantizer_outputs[body_part]` uniformly.
        quantizer_outputs = {body_part: quantizer_output for body_part in self.body_parts}

        return VQVAEOutput(
            reconstructions=reconstructions,
            quantizer_outputs=quantizer_outputs,
            total_loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            quantizer_loss=quantizer_output.loss,
        )


if __name__ == "__main__":
    N, T, L_pose, L_hand, C = 16, 500, 23, 21, 2
    poses = {
        "upper_pose": torch.randn(N, T, L_pose, C).cuda(),
        "left_hand": torch.randn(N, T, L_hand, C).cuda(),
        "right_hand": torch.randn(N, T, L_hand, C).cuda(),
    }
    mask = torch.ones(N, T).bool().cuda()

    model = TransformerVQVAE(embedding_dim=256, n_embeddings=1000, max_length=T).cuda()
    out = model(poses, mask)

    print({k: v.shape for k, v in out.reconstructions.items()})
    print({k: q.quantized_indices.shape for k, q in out.quantizer_outputs.items()})
    print(out.total_loss)
