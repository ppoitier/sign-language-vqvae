import lightning as L
import torch
from torch import Tensor, nn

from sl_vqvae.metrics.vqvae.perplexity import CodebookPerplexity
from sl_vqvae.metrics.vqvae.reconstruction import MaskedMPJPE
from sl_vqvae.nn.vqvae.output import VQVAEOutput


class VQVAETrainingModule(L.LightningModule):
    """LightningModule that trains any dict-in/dict-out VQ-VAE pose tokenizer:
    `model(poses: dict[str, Tensor], mask: Tensor) -> VQVAEOutput`.

    This covers both `TransformerVQVAE` (one codebook shared by all body
    parts) and `BodyPartTransformerVQVAE` (one codebook per modality) -- the
    model tells this module how its body parts are grouped into codebooks via
    `model.modality_groups`, and metrics are tracked per group rather than
    globally, so a collapsed or poorly reconstructing codebook doesn't hide
    behind a single aggregate number (see `sl_vqvae.metrics.vqvae` for what
    each metric means).
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        num_coordinates: int = 2,
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

        self.modality_groups: dict[str, list[str]] = model.modality_groups

        # Separate metric instances per stage: each accumulates over its own
        # epoch and must not mix train/val/test running state.
        self.train_perplexity = self._build_perplexity_metrics()
        self.val_perplexity = self._build_perplexity_metrics()
        self.test_perplexity = self._build_perplexity_metrics()
        self.train_mpjpe = self._build_mpjpe_metrics()
        self.val_mpjpe = self._build_mpjpe_metrics()
        self.test_mpjpe = self._build_mpjpe_metrics()

    def _build_perplexity_metrics(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                modality: CodebookPerplexity(self.model.n_embeddings(body_parts[0]))
                for modality, body_parts in self.modality_groups.items()
            }
        )

    def _build_mpjpe_metrics(self) -> nn.ModuleDict:
        # Per body part, not per modality group: body parts in the same group
        # (e.g. left_hand/right_hand) can still have different landmark
        # counts than others, so they can't always be pooled into one metric.
        body_parts = [bp for parts in self.modality_groups.values() for bp in parts]
        return nn.ModuleDict({body_part: MaskedMPJPE(self.num_coordinates) for body_part in body_parts})

    def _metrics(self, stage: str) -> tuple[nn.ModuleDict, nn.ModuleDict]:
        if stage == "training":
            return self.train_perplexity, self.train_mpjpe
        if stage == "test":
            return self.test_perplexity, self.test_mpjpe
        return self.val_perplexity, self.val_mpjpe

    def _log_losses(self, output: VQVAEOutput, stage: str, batch_size: int) -> None:
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

    def _update_metrics(
        self, output: VQVAEOutput, poses: dict[str, Tensor], mask: Tensor, stage: str
    ) -> None:
        perplexity, mpjpe = self._metrics(stage)

        # Perplexity: one instance per codebook group. Combining indices
        # across body parts sharing a codebook is safe regardless of how
        # many landmarks each body part has (indices don't carry shape).
        for modality, body_parts in self.modality_groups.items():
            indices = torch.cat(
                [output.quantizer_outputs[bp].quantized_indices[mask] for bp in body_parts]
            )
            perplexity[modality](indices)
            self.log(
                f"{stage}/perplexity/{modality}", perplexity[modality], on_step=False, on_epoch=True
            )

        # MPJPE: one instance per body part, since it reshapes back into
        # landmarks and body parts can have different landmark counts.
        for body_part, recon in output.reconstructions.items():
            mpjpe[body_part](recon.flatten(2), poses[body_part].flatten(2), mask)
            self.log(f"{stage}/mpjpe/{body_part}", mpjpe[body_part], on_step=False, on_epoch=True)

    def forward_step(self, batch: dict, stage: str) -> tuple[Tensor, VQVAEOutput]:
        poses = {k: v.float() for k, v in batch["poses"].items()}
        mask = batch["masks"]
        output = self.model(poses, mask)

        self._log_losses(output, stage, batch_size=mask.shape[0])
        self._update_metrics(output, poses, mask, stage)

        return output.total_loss, output

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, _ = self.forward_step(batch, "training")
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, _ = self.forward_step(batch, "validation")
        return loss

    def on_test_epoch_start(self) -> None:
        self.test_outputs.clear()

    def test_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, output = self.forward_step(batch, "test")

        if self.cache_test_outputs:
            window_ids = batch["window_id"]
            lengths = batch["lengths"]
            for i, window_id in enumerate(window_ids):
                length = lengths[i].item()
                self.test_outputs[window_id] = {
                    "recon_poses": {
                        body_part: recon[i, :length].detach().cpu().half()
                        for body_part, recon in output.reconstructions.items()
                    },
                    "tokens": {
                        body_part: q.quantized_indices[i, :length].detach().cpu().to(torch.int16)
                        for body_part, q in output.quantizer_outputs.items()
                    },
                }

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
