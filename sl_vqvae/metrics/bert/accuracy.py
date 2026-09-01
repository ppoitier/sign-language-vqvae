import torch
from torch import Tensor
from torchmetrics import Metric


class MaskedTokenAccuracy(Metric):
    """Top-1 accuracy of masked-token predictions.

    Accumulates correct/total counts over masked positions only, so the
    epoch-level number reflects how well the model predicts hidden tokens
    rather than being diluted by the (trivially easy, unpredicted) unmasked
    positions.
    """

    higher_is_better = True

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits: Tensor, target: Tensor, mask: Tensor) -> None:
        """
        Args:
            logits: (N, T, K) predicted class logits.
            target: (N, T) ground-truth token ids.
            mask:   (N, T) bool -- True at positions to score (masked positions).
        """
        if not mask.any():
            return
        preds = logits[mask].argmax(dim=-1)
        targets = target[mask]
        self.correct += (preds == targets).sum()
        self.total += targets.numel()

    def compute(self) -> Tensor:
        return self.correct / self.total.clamp(min=1.0)
