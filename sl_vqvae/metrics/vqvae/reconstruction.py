import torch
from torch import Tensor
from torchmetrics import Metric


class MaskedMPJPE(Metric):
    """Mean Per-Joint Position Error over valid (non-padded) frames.

    For each joint we take the Euclidean distance between the reconstructed and
    the target coordinates, then average over all joints, frames and body parts.
    This is the standard reconstruction metric for pose data and is easier to
    read than the raw training loss (it is in coordinate units, not weighted).

    Accepts the tokenizer's per-body-part dicts directly.
    """

    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        reconstructions: dict[str, Tensor],
        targets: dict[str, Tensor],
        mask: Tensor,
    ) -> None:
        # (N, T) -> (N, T, 1) to broadcast over landmarks.
        frame_mask = mask[:, :, None].to(self.sum_error.dtype)
        for body_part, recon in reconstructions.items():
            # (N, T, L, C) -> per-joint Euclidean distance (N, T, L).
            distance = torch.linalg.norm(recon - targets[body_part], dim=-1)
            self.sum_error += (distance * frame_mask).sum()
            # One count per valid (frame, joint) pair.
            self.count += frame_mask.sum() * distance.shape[-1]

    def compute(self) -> Tensor:
        return self.sum_error / self.count.clamp(min=1.0)
