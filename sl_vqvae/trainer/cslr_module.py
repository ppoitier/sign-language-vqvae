import lightning as L
import torch
from torch import Tensor, nn

from sl_vqvae.metrics.cslr.wer import DeletionRate, InsertionRate, SignErrorRate, SubstitutionRate
from sl_vqvae.nn.cslr.decoding import ctc_greedy_decode
from sl_vqvae.nn.cslr.output import CSLROutput


class CSLRTrainingModule(L.LightningModule):
    """LightningModule that trains a CTC-based CSLR model:
    `model(poses: dict[str, Tensor], mask: Tensor, labels: Tensor, label_lengths: Tensor) -> CSLROutput`
    (see `sl_vqvae.nn.cslr.transformer.CSLRPoseTransformer`).

    Expects `batch["targets"]["labels"]`: (N, S) long, gloss ids padded with
    `label_pad_value` -- the shape produced by `sldl.targets.ContinuousRecognitionTarget.collate`,
    a single tensor with no separate lengths field. `label_lengths` for CTC is
    derived here as `(labels != label_pad_value).sum(-1)`, so `label_pad_value`
    must be a ID outside the real gloss vocabulary (`0 .. vocab_size - 1`) --
    it defaults to `model.blank_id` (== `vocab_size`), which is guaranteed to
    never be a real class id. Configure
    `ContinuousRecognitionTarget(pad_value=vocab_size, ...)` to match (see
    `sl_vqvae.scripts.train_cslr`). Adjust `forward_step` if the target
    encoder that produces `sample["targets"]["labels"]` ends up shaped or
    named differently.

    Sign Error Rate (edit distance between the greedy CTC decoding and the
    reference gloss sequence, see `sl_vqvae.metrics.cslr.wer.SignErrorRate`)
    is tracked alongside loss, since CTC loss alone doesn't say how close the
    decoded gloss sequence actually is to the reference. The WER is also
    broken down into substitution/insertion/deletion rates -- e.g. useful to
    tell apart early-training CTC "blank collapse" (near-all deletions, since
    greedy decoding an all-blank path yields empty predictions) from later,
    genuinely wrong predictions (substitutions).
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        label_pad_value: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"])

        self.blank_id: int = model.blank_id
        self.label_pad_value: int = self.blank_id if label_pad_value is None else label_pad_value

        self.train_wer = SignErrorRate()
        self.val_wer = SignErrorRate()
        self.test_wer = SignErrorRate()

        self.train_wer_breakdown = nn.ModuleDict(
            {"substitutions": SubstitutionRate(), "insertions": InsertionRate(), "deletions": DeletionRate()}
        )
        self.val_wer_breakdown = nn.ModuleDict(
            {"substitutions": SubstitutionRate(), "insertions": InsertionRate(), "deletions": DeletionRate()}
        )
        self.test_wer_breakdown = nn.ModuleDict(
            {"substitutions": SubstitutionRate(), "insertions": InsertionRate(), "deletions": DeletionRate()}
        )

    def _wer_metric(self, stage: str) -> SignErrorRate:
        if stage == "training":
            return self.train_wer
        if stage == "test":
            return self.test_wer
        return self.val_wer

    def _wer_breakdown(self, stage: str) -> nn.ModuleDict:
        if stage == "training":
            return self.train_wer_breakdown
        if stage == "test":
            return self.test_wer_breakdown
        return self.val_wer_breakdown

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

        breakdown = self._wer_breakdown(stage)
        for metric in breakdown.values():
            metric(predictions, references)
        self.log_dict(
            {f"{stage}/wer_{name}": metric for name, metric in breakdown.items()},
            on_step=False,
            on_epoch=True,
        )

    def _validate_ctc_inputs(self, labels: Tensor, label_lengths: Tensor, input_lengths: Tensor) -> None:
        """Check CTC preconditions eagerly on CPU, before the compiled model runs.

        The CUDA CTC kernel does not bounds-check its arguments: an out-of-range
        target id, an empty (all-padding) window, or a sample whose gloss count
        exceeds its valid-frame count all read out of bounds and surface much
        later as a bare `illegal memory access` at the first host-device sync
        (F.ctc_loss), with a traceback that points at the sync rather than the
        real cause. Failing here instead turns that into an actionable error.
        Note `zero_infinity=True` does NOT protect against target_length >
        input_length on CUDA -- the bad read happens before the clamp.
        """
        labels = labels.cpu()
        input_lengths = input_lengths.cpu()
        label_lengths = label_lengths.cpu()

        # Real (non-padding) target ids must be valid class indices [0, vocab_size).
        real = labels[labels != self.label_pad_value]
        if real.numel() and (real.min() < 0 or real.max() >= self.blank_id):
            raise RuntimeError(
                f"CTC target ids out of range [0, {self.blank_id}): "
                f"min={real.min().item()}, max={real.max().item()}."
            )

        empty = (input_lengths == 0).nonzero(as_tuple=True)[0]
        if empty.numel():
            raise RuntimeError(
                f"{empty.numel()} sample(s) have input_length == 0 (fully-padded window); "
                f"first offending batch indices: {empty[:10].tolist()}."
            )

        infeasible = (label_lengths > input_lengths).nonzero(as_tuple=True)[0]
        if infeasible.numel():
            pairs = [(i.item(), label_lengths[i].item(), input_lengths[i].item()) for i in infeasible[:10]]
            raise RuntimeError(
                f"{infeasible.numel()} sample(s) have target_length > input_length, which the CUDA "
                f"CTC kernel cannot handle even with zero_infinity=True. "
                f"(batch_idx, label_length, input_length): {pairs}."
            )

    def forward_step(self, batch: dict, stage: str) -> Tensor:
        poses = {k: v.float() for k, v in batch["poses"].items()}
        mask = batch["masks"]
        labels = batch["targets"]["labels"].long()
        label_lengths = (labels != self.label_pad_value).sum(dim=1)
        self._validate_ctc_inputs(labels, label_lengths, mask.sum(dim=1))
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
