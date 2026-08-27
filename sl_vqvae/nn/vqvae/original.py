from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from sl_vqvae.nn.quantizers.quantizer import Quantizer, QuantizerEMA
from sl_vqvae.nn.quantizers.blocks import Encoder, Decoder


@dataclass
class VQVAE_Output:
    encoder_output: Tensor
    reconstructed_input: Tensor
    total_loss: Tensor
    reconstruction_loss: Tensor
    quantizer_loss: Tensor
    quantized_indices: Tensor


class VQVAE(nn.Module):
    """
    VQ-VAE over flat pose vectors.

    Input / output: (N, in_channels), e.g. (N, 130) for 65 landmarks x (x, y).
    Latent:         (N, embedding_dim), one code per sample.
    """

    def __init__(
        self,
        c_in: int = 130,
        embedding_dim: int = 512,
        n_embeddings: int = 1000,
        c_hidden: int = 256,
        n_residual_blocks: int = 2,
        commitment_loss_factor: float = 0.25,
        quantization_loss_factor: float = 1.0,
        use_quantizer_ema: bool = False,
        quantizer_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings

        self.encoder = Encoder(
            c_in=c_in,
            c_hidden=c_hidden,
            c_out=embedding_dim,
            n_blocks=n_residual_blocks,
        )

        if use_quantizer_ema:
            self.quantizer = QuantizerEMA(
                n_embeddings=n_embeddings,
                embedding_dim=embedding_dim,
                commitment_loss_factor=commitment_loss_factor,
                decay=quantizer_ema_decay,
            )
        else:
            self.quantizer = Quantizer(
                n_embeddings=n_embeddings,
                embedding_dim=embedding_dim,
                commitment_loss_factor=commitment_loss_factor,
                quantization_loss_factor=quantization_loss_factor,
            )

        self.decoder = Decoder(
            c_in=embedding_dim,
            c_hidden=c_hidden,
            c_out=c_in,
            n_blocks=n_residual_blocks,
        )

    def loss_function(
        self,
        original_input: Tensor,
        reconstructed_input: Tensor,
        quantizer_loss: Tensor,
    ) -> tuple[Tensor, Tensor]:
        reconstruction_loss = F.mse_loss(reconstructed_input, original_input)
        total_loss = reconstruction_loss + quantizer_loss
        return total_loss, reconstruction_loss

    def forward(self, x: Tensor) -> VQVAE_Output:
        z_e = self.encoder(x)

        quantizer_output = self.quantizer(z_e)

        x_hat = self.decoder(quantizer_output.quantized_vectors)

        # 4. Losses
        total_loss, recon_loss = self.loss_function(x, x_hat, quantizer_output.loss)

        return VQVAE_Output(
            encoder_output=z_e,
            reconstructed_input=x_hat,
            total_loss=total_loss,
            reconstruction_loss=recon_loss,
            quantizer_loss=quantizer_output.loss,
            quantized_indices=quantizer_output.quantized_indices,
        )


if __name__ == "__main__":
    N, C_in = 16, 130

    model = VQVAE(c_in=C_in, embedding_dim=512, n_embeddings=1000)
    x = torch.randn(N, C_in)
    out = model(x)

    print(out.reconstructed_input.shape)   # (16, 130)
    print(out.encoder_output.shape)        # (16, 512)
    print(out.quantized_indices.shape)     # (16,)
    print(out.total_loss)