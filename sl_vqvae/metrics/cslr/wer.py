import torch
from torch import Tensor
from torchmetrics import Metric


def _levenshtein(a: list[int], b: list[int]) -> int:
    """Edit distance (insert/delete/substitute, unit cost) between two id sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous_row = list(range(len(b) + 1))
    for i, a_id in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, b_id in enumerate(b, start=1):
            cost = 0 if a_id == b_id else 1
            current_row[j] = min(
                previous_row[j] + 1,  # deletion
                current_row[j - 1] + 1,  # insertion
                previous_row[j - 1] + cost,  # substitution
            )
        previous_row = current_row
    return previous_row[-1]


class SignErrorRate(Metric):
    """Word/Sign Error Rate: edit distance between greedy-decoded gloss
    predictions (see `sl_vqvae.nn.cslr.decoding.ctc_greedy_decode`) and
    reference gloss sequences.

    Accumulated as total edits / total reference length over the whole
    epoch (the standard corpus-level WER), not averaged per-sample first --
    averaging per-sample would let short sequences (where one error is a
    huge relative penalty) dominate the score.
    """

    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.add_state("edits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions: list[list[int]], references: list[list[int]]) -> None:
        for prediction, reference in zip(predictions, references):
            self.edits += _levenshtein(prediction, reference)
            self.total += len(reference)

    def compute(self) -> Tensor:
        return self.edits / self.total.clamp(min=1.0)
