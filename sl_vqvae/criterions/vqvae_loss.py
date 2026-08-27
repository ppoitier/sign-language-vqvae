from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from sl_vqvae.nn.vqvae.body_part_graph_tokenizer import TokenizerOutput


@dataclass
class LossOutput:
    """Loss components, kept separate so the trainer can log each one."""

    total: Tensor
    reconstruction: Tensor
    vq: Tensor


def _masked_mean(error: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``error`` (N, T, L, C) over the frames flagged by ``mask`` (N, T).

    Padded frames must not contribute, otherwise the loss rewards the model for
    reconstructing zeros.
    """
    # (N, T) -> (N, T, 1, 1) so it broadcasts over landmarks and coordinates.
    frame_mask = mask[:, :, None, None].to(error.dtype)
    denom = frame_mask.sum().clamp(min=1.0) * error.shape[-1] * error.shape[-2]
    return (error * frame_mask).sum() / denom


class VQVAELoss(nn.Module):
    """Reconstruction + vector-quantization loss for the body-part tokenizer.

    The reconstruction term is averaged over body parts so its scale does not
    depend on how many parts the model has. The VQ term is the (already
    weighted, see ``Quantizer``) commitment + codebook loss, likewise averaged
    over parts.

    Note:
        The VQ loss is produced inside each quantizer over *all* latent
        positions, including padded frames. With windowed data that is a small
        effect (padding frames are few and encode near-constant latents), but if
        you later train on heavily padded batches, consider masking the VQ term
        at its source too.
    """

    def __init__(self, reconstruction: str = "l1", vq_loss_weight: float = 1.0):
        super().__init__()
        if reconstruction not in ("l1", "l2"):
            raise ValueError(f"Unknown reconstruction loss: {reconstruction!r}")
        self.reconstruction = reconstruction
        self.vq_loss_weight = vq_loss_weight

    def _reconstruction_error(self, recon: Tensor, target: Tensor) -> Tensor:
        if self.reconstruction == "l1":
            return F.l1_loss(recon, target, reduction="none")
        return F.mse_loss(recon, target, reduction="none")

    def forward(self, output: TokenizerOutput, batch: dict) -> LossOutput:
        targets: dict[str, Tensor] = batch["poses"]
        mask: Tensor = batch["masks"]

        recon_terms = []
        for body_part, recon in output.reconstructions.items():
            error = self._reconstruction_error(recon, targets[body_part])
            recon_terms.append(_masked_mean(error, mask))
        reconstruction = torch.stack(recon_terms).mean()

        vq = torch.stack(
            [q.loss for q in output.quantizer_outputs.values()]
        ).mean()

        total = reconstruction + self.vq_loss_weight * vq
        return LossOutput(total=total, reconstruction=reconstruction, vq=vq)
