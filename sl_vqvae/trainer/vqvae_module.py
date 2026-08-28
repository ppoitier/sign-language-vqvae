from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch import Tensor

from sl_vqvae.metrics.vqvae.perplexity import CodebookPerplexity
from sl_vqvae.metrics.vqvae.reconstruction import MaskedMPJPE
from sl_vqvae.nn.vqvae.transformer import (
    TransformerVQVAE,
    TransformerVQVAE_Output,
)


def flatten_poses(poses, body_parts):
    poses = torch.cat([poses[body_part] for body_part in body_parts], dim=2)
    N, T, L, C = poses.shape
    return poses.reshape(N, T, -1).contiguous()


class VQVAETrainingModule(L.LightningModule):
    """LightningModule that trains a `TransformerVQVAE` pose tokenizer.

    The model (`TransformerVQVAE`) already knows how to reconstruct a pose
    sequence and compute its own loss (reconstruction + vector-quantization).
    This module only owns the *training policy* around it: turning a raw
    batch into model inputs, logging losses, tracking metrics, and
    configuring the optimizer.

    Metrics (see the `sl_vqvae.metrics.vqvae` sub-module for what they mean
    and why they're tracked):
        - `mpjpe`: Mean Per-Joint Position Error, the reconstruction quality
          in real coordinate units.
        - `perplexity`: effective number of codebook entries in use, a
          codebook-collapse warning signal.
    """

    def __init__(
        self,
        model: TransformerVQVAE,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        num_coordinates: int = 2,
        body_parts: tuple[str, ...] = ('upper_pose', 'left_hand', 'right_hand'),
        cache_test_outputs: bool = True,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_coordinates = num_coordinates
        self.cache_test_outputs = cache_test_outputs

        self.test_outputs = dict()
        self.save_hyperparameters(ignore=["model", "test_outputs"])

        self.body_parts = body_parts

        # Separate metric instances per stage: each accumulates over its own
        # epoch and must not mix train/val/test running state.
        self.train_perplexity = CodebookPerplexity(model.n_embeddings)
        self.val_perplexity = CodebookPerplexity(model.n_embeddings)
        self.test_perplexity = CodebookPerplexity(model.n_embeddings)
        self.train_mpjpe = MaskedMPJPE(num_coordinates)
        self.val_mpjpe = MaskedMPJPE(num_coordinates)
        self.test_mpjpe = MaskedMPJPE(num_coordinates)

    def _metrics(self, stage: str) -> tuple[CodebookPerplexity, MaskedMPJPE]:
        if stage == "training":
            return self.train_perplexity, self.train_mpjpe
        if stage == "test":
            return self.test_perplexity, self.test_mpjpe
        return self.val_perplexity, self.val_mpjpe

    def _log_losses(self, output: TransformerVQVAE_Output, stage: str, batch_size: int) -> None:
        self.log_dict(
            {
                f"{stage}/loss": output.total_loss,
                f"{stage}/reconstruction_loss": output.reconstruction_loss,
                f"{stage}/vq_loss": output.quantizer_loss,
            },
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def _update_metrics(self, output: TransformerVQVAE_Output, x: Tensor, mask: Tensor, stage: str) -> None:
        perplexity, mpjpe = self._metrics(stage)
        mpjpe(output.reconstructed_input, x, mask)
        perplexity(output.quantized_indices[mask])
        self.log(f"{stage}/mpjpe", mpjpe, on_step=False, on_epoch=True)
        self.log(f"{stage}/perplexity", perplexity, on_step=False, on_epoch=True)

    def forward_step(self, batch: dict, stage: str) -> tuple[Tensor, TransformerVQVAE_Output]:
        mask = batch["masks"]
        x = flatten_poses(batch["poses"], self.body_parts).float()
        output = self.model(x, mask)

        self._log_losses(output, stage, batch_size=mask.shape[0])
        self._update_metrics(output, x, mask, stage)

        return output.total_loss, output

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, _ = self.forward_step(batch, "training")
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, _ = self.forward_step(batch, "validation")
        return loss

    def on_test_epoch_start(self) -> None:
        if self.test_output_dir is not None:
            self.test_output_dir.mkdir(parents=True, exist_ok=True)

    def test_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, output = self.forward_step(batch, "test")

        # TODO: we have to split output into samples
        if self.cache_test_outputs:
            self.test_outputs.update({
                sample_id: {
                    'recon_poses': output.reconstructed_input,
                    'tokens': output.quantized_indices,
                }
                for sample_id, output in zip(batch['id'], output)
            })

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
