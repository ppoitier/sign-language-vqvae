from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.nn import functional as F


@dataclass
class QuantizerOutput:
    quantized_vectors: Tensor   # shape (..., D)
    quantized_indices: Tensor   # shape (...)
    loss: Tensor


class Quantizer(nn.Module):
    """
    Shape-agnostic vector quantizer: accepts any z of shape (..., D) and returns
    the same shape. For flat pose latents this is (N, D); it also handles
    (N, L, D) or (N, T, D) unchanged if you later go per-landmark or per-frame.

    van den Oord, Vinyals & Kavukcuoglu, 2017, "Neural Discrete Representation Learning"
    """

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        commitment_loss_factor: float,
        quantization_loss_factor: float,
    ):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_loss_factor = commitment_loss_factor
        self.quantization_loss_factor = quantization_loss_factor

        self.embeddings = nn.Embedding(self.n_embeddings, self.embedding_dim)
        # NOTE: the standard uniform(-1/K, 1/K) init gets very tight for large K
        # (K=1000 -> +-0.001) while a LayerNorm'd encoder output has norm ~sqrt(D).
        # If codebook usage collapses early, try init from encoder-output
        # statistics or use QuantizerEMA below.
        self.embeddings.weight.data.uniform_(
            -1 / self.n_embeddings, 1 / self.n_embeddings
        )

    def forward(self, z: Tensor) -> QuantizerOutput:
        input_shape = z.shape
        z_flat = z.reshape(-1, self.embedding_dim)

        # --- Find nearest neighbors ---
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embeddings.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).reshape(input_shape)

        # --- Losses ---
        # Codebook loss: pulls the embeddings toward the encoder output.
        embedding_loss = F.mse_loss(quantized, z.detach())
        # Commitment loss: pulls the encoder output toward the chosen embedding.
        commitment_loss = F.mse_loss(z, quantized.detach())

        loss = (
            self.quantization_loss_factor * embedding_loss
            + self.commitment_loss_factor * commitment_loss
        )

        # --- Straight-Through Estimator ---
        # Bengio, Leonard & Courville, 2013, "Estimating or Propagating Gradients
        # Through Stochastic Neurons for Conditional Computation"
        quantized = z + (quantized - z).detach()

        return QuantizerOutput(
            quantized_vectors=quantized,
            quantized_indices=encoding_indices.reshape(input_shape[:-1]),
            loss=loss,
        )


class QuantizerEMA(nn.Module):
    """
    EMA-updated codebook variant. The codebook is not trained by gradient descent;
    only the commitment loss reaches the encoder. Codes that fall unused are
    periodically revived from batch samples (see `_revive_dead_codes`) so usage
    can't collapse permanently onto a handful of codes.
    """

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        commitment_loss_factor: float,
        decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.decay = decay
        self.eps = eps
        self.commitment_loss_factor = commitment_loss_factor
        # A code whose EMA cluster size decays below this is considered dead
        # and gets replaced (see `_revive_dead_codes`), so it never lingers
        # long enough to hit the eps floor in the smoothing below.
        self.dead_code_threshold = dead_code_threshold

        embeddings = torch.randn(n_embeddings, embedding_dim)
        self.register_buffer("embeddings", embeddings)
        self.register_buffer("cluster_size", torch.zeros(n_embeddings))
        self.register_buffer("ema_embed", embeddings.clone())

    def forward(self, z: Tensor) -> QuantizerOutput:
        input_shape = z.shape
        z_flat = z.reshape(-1, self.embedding_dim)

        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embeddings.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, self.embeddings).reshape(input_shape)

        if self.training:
            with torch.no_grad():
                one_hot_encoding = F.one_hot(
                    encoding_indices, self.n_embeddings
                ).type(z_flat.dtype)

                # Running counts and running sums of assigned vectors.
                n_i = torch.sum(one_hot_encoding, dim=0)
                dw = one_hot_encoding.t() @ z_flat.detach()

                self.cluster_size.mul_(self.decay).add_(n_i, alpha=1 - self.decay)
                self.ema_embed.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                # Laplace smoothing applied ONLY for the normalization below --
                # it must not be written back into the running cluster_size state.
                n = torch.sum(self.cluster_size)
                smoothed_cluster_size = (
                    (self.cluster_size + self.eps)
                    / (n + self.n_embeddings * self.eps)
                    * n
                )

                self.embeddings.copy_(
                    self.ema_embed / smoothed_cluster_size.unsqueeze(-1)
                )

                self._revive_dead_codes(z_flat)

        # Only the commitment loss, since the codebook is updated by EMA.
        loss = self.commitment_loss_factor * F.mse_loss(z, quantized.detach())

        # --- Straight-Through Estimator ---
        quantized = z + (quantized - z).detach()

        return QuantizerOutput(
            quantized_vectors=quantized,
            quantized_indices=encoding_indices.reshape(input_shape[:-1]),
            loss=loss,
        )

    def _revive_dead_codes(self, z_flat: Tensor) -> None:
        """Reset codes whose EMA usage has decayed away to random vectors
        sampled from this batch, instead of leaving them to drift off toward
        the data manifold via the eps floor in the smoothing above. Without
        this, a code that stops being picked never gets picked again --
        usage collapses onto a handful of codes and stays there.
        """
        dead = self.cluster_size < self.dead_code_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return

        samples = z_flat[torch.randint(0, z_flat.shape[0], (n_dead,), device=z_flat.device)]
        self.embeddings[dead] = samples
        self.ema_embed[dead] = samples
        self.cluster_size[dead] = self.dead_code_threshold


if __name__ == "__main__":
    N, D, K = 16, 512, 1000

    z = torch.randn(N, D, requires_grad=True)

    q = Quantizer(
        n_embeddings=K,
        embedding_dim=D,
        commitment_loss_factor=0.25,
        quantization_loss_factor=1.0,
    )
    out = q(z)
    print(out.quantized_vectors.shape, out.quantized_indices.shape)

    q_ema = QuantizerEMA(
        n_embeddings=K,
        embedding_dim=D,
        commitment_loss_factor=0.25,
    )
    out_ema = q_ema(z)
    print(out_ema.quantized_vectors.shape, out_ema.quantized_indices.shape)