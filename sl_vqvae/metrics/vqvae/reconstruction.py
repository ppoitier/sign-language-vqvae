import torch
from torch import Tensor
from torchmetrics import Metric


class MaskedMPJPE(Metric):
    """Mean Per-Joint Position Error (MPJPE) over valid (non-padded) frames.

    MPJPE is the standard reconstruction metric for pose/keypoint models: for
    every joint (landmark), take the Euclidean distance between its
    reconstructed and target coordinates, then average that distance over all
    joints and frames. Unlike the raw training loss (e.g. masked L1/L2, which
    is summed/averaged per-coordinate and often scaled by extra loss weights),
    MPJPE stays in real coordinate units and is directly comparable across
    runs, making it the number to read when judging "how good is the
    reconstruction", as opposed to "what is the optimizer minimizing".

    Padded frames are excluded via ``mask`` so they don't dilute the score.

    Args:
        num_coordinates: Number of coordinates per landmark (2 for (x, y), 3
            for (x, y, z)). Used to unflatten the model's per-frame feature
            vector back into individual joints.
    """

    higher_is_better = False

    def __init__(self, num_coordinates: int = 2):
        super().__init__()
        self.num_coordinates = num_coordinates
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, reconstruction: Tensor, target: Tensor, mask: Tensor) -> None:
        """
        Args:
            reconstruction: (N, T, L * C) reconstructed frames.
            target:         (N, T, L * C) ground-truth frames.
            mask:           (N, T) -- 1/True for valid (non-pad) frames.
        """
        n, t, _ = reconstruction.shape
        recon = reconstruction.reshape(n, t, -1, self.num_coordinates)
        tgt = target.reshape(n, t, -1, self.num_coordinates)

        # (N, T, L, C) -> per-joint Euclidean distance (N, T, L).
        distance = torch.linalg.norm(recon - tgt, dim=-1)

        # (N, T) -> (N, T, 1) to broadcast over landmarks.
        frame_mask = mask[:, :, None].to(self.sum_error.dtype)
        self.sum_error += (distance * frame_mask).sum()
        # One count per valid (frame, joint) pair.
        self.count += frame_mask.sum() * distance.shape[-1]

    def compute(self) -> Tensor:
        return self.sum_error / self.count.clamp(min=1.0)
