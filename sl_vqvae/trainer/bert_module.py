import lightning as L
import torch
from torch import Tensor, nn

from sl_vqvae.metrics.bert.accuracy import MaskedTokenAccuracy
from sl_vqvae.nn.bert.output import BERTOutput


class BERTTrainingModule(L.LightningModule):
    """LightningModule that pretrains a masked-pose model:
    `model(poses: dict[str, Tensor], tokens: dict[str, Tensor], mask: Tensor) -> BERTOutput`.

    Tokens are prediction targets already produced by a trained VQ-VAE
    quantizer (see `sl_vqvae.nn.vqvae`, `sl_vqvae.scripts.extract_tokens`)
    and read from `batch["targets"]["tokens"]`, as produced by the
    `sl_vqvae.targets.TokenTarget` target encoder. Loss and accuracy are
    logged per body part so a group whose predictions are stuck at chance
    doesn't hide behind an aggregate number.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"])

        self.body_parts: tuple[str, ...] = model.body_parts

        self.train_accuracy = self._build_accuracy_metrics()
        self.val_accuracy = self._build_accuracy_metrics()
        self.test_accuracy = self._build_accuracy_metrics()

    def _build_accuracy_metrics(self) -> nn.ModuleDict:
        return nn.ModuleDict({body_part: MaskedTokenAccuracy() for body_part in self.body_parts})

    def _accuracy_metrics(self, stage: str) -> nn.ModuleDict:
        if stage == "training":
            return self.train_accuracy
        if stage == "test":
            return self.test_accuracy
        return self.val_accuracy

    def _log_loss(self, output: BERTOutput, stage: str, batch_size: int) -> None:
        self.log(
            f"{stage}/loss",
            output.loss,
            on_step=(stage == "training"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def _update_metrics(self, output: BERTOutput, tokens: dict[str, Tensor], stage: str) -> None:
        accuracy = self._accuracy_metrics(stage)
        for body_part in self.body_parts:
            accuracy[body_part](output.logits[body_part], tokens[body_part], output.masks[body_part])
            self.log(f"{stage}/accuracy/{body_part}", accuracy[body_part], on_step=False, on_epoch=True)

    def forward_step(self, batch: dict, stage: str) -> Tensor:
        poses = {k: v.float() for k, v in batch["poses"].items()}
        tokens = {k: v.long() for k, v in batch["targets"]["tokens"].items()}
        mask = batch["masks"]
        output = self.model(poses, tokens, mask)

        self._log_loss(output, stage, batch_size=mask.shape[0])
        self._update_metrics(output, tokens, stage)

        return output.loss

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self.forward_step(batch, "training")

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self.forward_step(batch, "validation")

    def test_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self.forward_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
