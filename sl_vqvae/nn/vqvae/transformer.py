from dataclasses import dataclass

import torch
from torch import nn, Tensor

from sl_vqvae.nn.encoders.transformer import PoseTransformerEncoder
from sl_vqvae.nn.decoders.transformer import PoseTransformerDecoder
from sl_vqvae.nn.quantizers.quantizer import QuantizerEMA


@dataclass
class TransformerVQVAE_Output:
    encoder_output: Tensor
    reconstructed_input: Tensor
    total_loss: Tensor
    reconstruction_loss: Tensor
    quantizer_loss: Tensor
    quantized_indices: Tensor


class TransformerVQVAE(nn.Module):
    """
    VQ-VAE over concatenated pose sequences.

    Input / output: (N, T, C_in), e.g. (16, 500, 130) for 65 landmarks x (x, y).
    Latent:         (N, T, embedding_dim), one code per frame.

    No body-part separation: the full pose vector is treated as a single stream.
    A Transformer encoder/decoder pair models the temporal dimension, and an
    EMA-updated codebook quantizes each frame's latent.
    """

    def __init__(
        self,
        c_in: int = 130,
        embedding_dim: int = 256,
        n_embeddings: int = 1000,
        max_length: int = 500,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pos_encoding: str = "rope",
        attn_mask_strategy: str | None = None,
        commitment_loss_factor: float = 0.25,
        quantizer_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings

        self.encoder = PoseTransformerEncoder(
            c_in=c_in,
            c_hidden=embedding_dim,
            max_length=max_length,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,
            attn_mask_strategy=attn_mask_strategy,
        )

        self.quantizer = QuantizerEMA(
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
            commitment_loss_factor=commitment_loss_factor,
            decay=quantizer_ema_decay,
        )

        self.decoder = PoseTransformerDecoder(
            c_out=c_in,
            c_hidden=embedding_dim,
            c_in=embedding_dim,
            max_length=max_length,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,
            attn_mask_strategy=attn_mask_strategy,
        )

    def loss_function(
        self,
        original_input: Tensor,
        reconstructed_input: Tensor,
        quantizer_loss: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Masked MSE: only valid (non-pad) frames contribute to the loss.
        mask_f = mask.unsqueeze(-1).to(reconstructed_input.dtype)  # (N, T, 1)
        squared_error = (reconstructed_input - original_input) ** 2
        denom = mask_f.sum().clamp(min=1.0) * original_input.size(-1)
        reconstruction_loss = (squared_error * mask_f).sum() / denom
        total_loss = reconstruction_loss + quantizer_loss
        return total_loss, reconstruction_loss

    def forward(self, x: Tensor, mask: Tensor) -> TransformerVQVAE_Output:
        """
        Args:
            x:    (N, T, C_in) -- pose sequence.
            mask: (N, T) -- 1=valid, 0=pad.
        """
        z_e = self.encoder(x, mask)

        quantizer_output = self.quantizer(z_e)

        x_hat = self.decoder(quantizer_output.quantized_vectors, mask)

        total_loss, recon_loss = self.loss_function(
            x, x_hat, quantizer_output.loss, mask
        )

        return TransformerVQVAE_Output(
            encoder_output=z_e,
            reconstructed_input=x_hat,
            total_loss=total_loss,
            reconstruction_loss=recon_loss,
            quantizer_loss=quantizer_output.loss,
            quantized_indices=quantizer_output.quantized_indices,
        )


if __name__ == "__main__":
    N, T, C_in = 16, 500, 130

    model = TransformerVQVAE(
        c_in=C_in, embedding_dim=256, n_embeddings=1000, max_length=T
    ).cuda()
    x = torch.randn(N, T, C_in).cuda()
    mask = torch.ones(N, T).bool().cuda()
    out = model(x, mask)

    print(out.reconstructed_input.shape)   # (16, 500, 130)
    print(out.encoder_output.shape)        # (16, 500, 256)
    print(out.quantized_indices.shape)     # (16, 500)
    print(out.total_loss)
