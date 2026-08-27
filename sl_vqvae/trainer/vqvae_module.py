import lightning as L
import torch
from torch import nn

from sl_vqvae.criterions.vqvae_loss import VQVAELoss
from sl_vqvae.metrics.vqvae.perplexity import CodebookPerplexity
from sl_vqvae.metrics.vqvae.reconstruction import MaskedMPJPE
from sl_vqvae.nn.vqvae.body_part_graph_tokenizer import (
    BodyPartGraphTokenizer,
    TokenizerOutput,
)


def perplexity_metric_per_body_part() -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            body_part: CodebookPerplexity(size)
            for body_part, size in codebook_sizes.items()
        }
    )


class VQVAETrainingModule(L.LightningModule):
    """LightningModule that trains a body-part VQ-VAE tokenizer.

    The module owns only the *training policy*: it runs the model, hands the
    output to the criterion, updates the metrics and configures the optimizer.
    The model, the loss and the metrics all live in their own modules so each
    can be read, tested and swapped independently.
    """

    def __init__(
        self,
        model: BodyPartGraphTokenizer,
        criterion: VQVAELoss | None = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion or VQVAELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["model", "criterion"])

        # One perplexity metric per body part, plus a reconstruction metric, for
        # each stage. Metrics must be distinct module instances per stage so
        # their running states never mix. We keep train / val in separate
        # ModuleDicts because "train" is a reserved key on an nn.ModuleDict.
        codebook_sizes = model.codebook_sizes



        self.train_perplexity = nn.ModuleDict({

        })
        self.val_perplexity = _perplexity_metrics()
        self.train_mpjpe = MaskedMPJPE()
        self.val_mpjpe = MaskedMPJPE()

    def _metrics(self, stage: str) -> tuple[nn.ModuleDict, MaskedMPJPE]:
        if stage == "train":
            return self.train_perplexity, self.train_mpjpe
        return self.val_perplexity, self.val_mpjpe

    def forward(self, poses: dict) -> TokenizerOutput:
        return self.model(poses)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        output = self.model(batch["poses"])
        losses = self.criterion(output, batch)
        mask = batch["masks"]

        on_step = stage == "train"
        self.log_dict(
            {
                f"{stage}/loss": losses.total,
                f"{stage}/reconstruction_loss": losses.reconstruction,
                f"{stage}/vq_loss": losses.vq,
            },
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=mask.shape[0],
        )

        # Metrics: only count valid frames. Log the metric objects so Lightning
        # accumulates over the epoch and resets them for us.
        perplexity, mpjpe = self._metrics(stage)
        mpjpe(output.reconstructions, batch["poses"], mask)
        self.log(f"{stage}/mpjpe", mpjpe, on_step=False, on_epoch=True)

        for body_part, quantizer_output in output.quantizer_outputs.items():
            valid_indices = quantizer_output.quantized_indices[mask]
            metric = perplexity[body_part]
            metric(valid_indices)
            self.log(
                f"{stage}/perplexity/{body_part}",
                metric,
                on_step=False,
                on_epoch=True,
            )

        return losses.total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
