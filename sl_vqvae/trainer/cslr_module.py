import lightning as L
import torch
from torch import Tensor, nn

from sl_vqvae.metrics.cslr.wer import SignErrorRate
from sl_vqvae.nn.cslr.decoding import ctc_greedy_decode
from sl_vqvae.nn.cslr.output import CSLROutput


class CSLRTrainingModule(L.LightningModule):
    """LightningModule that trains a CTC-based CSLR model:
    `model(poses: dict[str, Tensor], mask: Tensor, labels: Tensor, label_lengths: Tensor) -> CSLROutput`
    (see `sl_vqvae.nn.cslr.transformer.CSLRPoseTransformer`).

    Expects `batch["targets"]["labels"]`: (N, S) long, padded gloss ids, and
    `batch["targets"]["label_lengths"]`: (N,) long -- the real (unpadded)
    length of each sample's gloss sequence. CTC needs this explicitly since
    it never infers lengths from a pad value (unlike e.g. `TokenTarget`,
    which pads to -1 and relies on a separate frame mask instead). Adjust
    `forward_step` if the target encoder that produces
    `sample["targets"]["labels"]` ends up shaped or named differently.

    Sign Error Rate (edit distance between the greedy CTC decoding and the
    reference gloss sequence, see `sl_vqvae.metrics.cslr.wer.SignErrorRate`)
    is tracked alongside loss, since CTC loss alone doesn't say how close the
    decoded gloss sequence actually is to the reference.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"])

        self.blank_id: int = model.blank_id

        self.train_wer = SignErrorRate()
        self.val_wer = SignErrorRate()
        self.test_wer = SignErrorRate()

    def _wer_metric(self, stage: str) -> SignErrorRate:
        if stage == "training":
            return self.train_wer
        if stage == "test":
            return self.test_wer
        return self.val_wer

    def _log_loss(self, output: CSLROutput, stage: str, batch_size: int) -> None:
        self.log(
            f"{stage}/loss",
            output.loss,
            on_step=(stage == "training"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def _update_metrics(self, output: CSLROutput, labels: Tensor, label_lengths: Tensor, stage: str) -> None:
        predictions = ctc_greedy_decode(output.log_probs, output.input_lengths, self.blank_id)
        references = [labels[i, : label_lengths[i]].tolist() for i in range(labels.size(0))]

        wer = self._wer_metric(stage)
        wer(predictions, references)
        self.log(f"{stage}/wer", wer, on_step=False, on_epoch=True, prog_bar=True)

    def forward_step(self, batch: dict, stage: str) -> Tensor:
        poses = {k: v.float() for k, v in batch["poses"].items()}
        mask = batch["masks"]
        labels = batch["targets"]["labels"].long()
        label_lengths = batch["targets"]["label_lengths"].long()
        output = self.model(poses, mask, labels, label_lengths)

        self._log_loss(output, stage, batch_size=mask.shape[0])
        self._update_metrics(output, labels, label_lengths, stage)

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
