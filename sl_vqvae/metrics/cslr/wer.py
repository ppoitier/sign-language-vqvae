import torch
from torch import Tensor
from torchmetrics import Metric


def _levenshtein_ops(prediction: list[int], reference: list[int]) -> tuple[int, int, int]:
    """Edit-distance DP between `prediction` and `reference`, backtraced into
    (substitutions, insertions, deletions) using standard WER terminology
    relative to the reference: insertions are prediction tokens with no
    reference counterpart, deletions are reference tokens missing from the
    prediction. Ties in the backtrace are broken in favor of alignment
    (match/substitution) over insertion/deletion.
    """
    n, m = len(prediction), len(reference)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if prediction[i - 1] == reference[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # extra prediction token -> insertion
                dp[i][j - 1] + 1,  # missing reference token -> deletion
                dp[i - 1][j - 1] + cost,  # aligned tokens -> match/substitution
            )

    substitutions = insertions = deletions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if prediction[i - 1] == reference[j - 1] else 1):
            if prediction[i - 1] != reference[j - 1]:
                substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            insertions += 1
            i -= 1
        else:
            deletions += 1
            j -= 1
    return substitutions, insertions, deletions


class SignErrorRate(Metric):
    """Word/Sign Error Rate: edit distance between greedy-decoded gloss
    predictions (see `sl_vqvae.nn.cslr.decoding.ctc_greedy_decode`) and
    reference gloss sequences.

    Accumulated as total edits / total reference length over the whole
    epoch (the standard corpus-level WER), not averaged per-sample first --
    averaging per-sample would let short sequences (where one error is a
    huge relative penalty) dominate the score.

    See `SubstitutionRate` / `InsertionRate` / `DeletionRate` for the
    per-operation breakdown of this same edit distance -- useful to tell
    apart e.g. CTC "blank collapse" (all-deletion errors) from genuinely
    wrong predictions (substitutions).
    """

    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.add_state("substitutions", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("insertions", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("deletions", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions: list[list[int]], references: list[list[int]]) -> None:
        for prediction, reference in zip(predictions, references):
            substitutions, insertions, deletions = _levenshtein_ops(prediction, reference)
            self.substitutions += substitutions
            self.insertions += insertions
            self.deletions += deletions
            self.total += len(reference)

    def compute(self) -> Tensor:
        return (self.substitutions + self.insertions + self.deletions) / self.total.clamp(min=1.0)


class _EditOperationRate(Metric):
    """Base for the WER breakdown components (see `SignErrorRate`). Each
    subclass is its own `torchmetrics.Metric` -- rather than reading
    `SignErrorRate`'s internal state -- so it logs, syncs (DDP), and resets
    correctly through Lightning's per-epoch metric lifecycle exactly like
    `SignErrorRate` itself.
    """

    higher_is_better = False
    _op_index: int  # 0=substitutions, 1=insertions, 2=deletions

    def __init__(self):
        super().__init__()
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions: list[list[int]], references: list[list[int]]) -> None:
        for prediction, reference in zip(predictions, references):
            self.count += _levenshtein_ops(prediction, reference)[self._op_index]
            self.total += len(reference)

    def compute(self) -> Tensor:
        return self.count / self.total.clamp(min=1.0)


class SubstitutionRate(_EditOperationRate):
    """Fraction of reference tokens that were substituted for a wrong id."""

    _op_index = 0


class InsertionRate(_EditOperationRate):
    """Fraction of reference tokens' worth of extra, unmatched predicted ids."""

    _op_index = 1


class DeletionRate(_EditOperationRate):
    """Fraction of reference tokens missing entirely from the prediction."""

    _op_index = 2
