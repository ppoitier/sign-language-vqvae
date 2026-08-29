import torch
from torch import Tensor
from torch.nn import functional as F


def masked_reconstruction_loss(
    poses: dict[str, Tensor],
    reconstructions: dict[str, Tensor],
    mask: Tensor,
    kind: str = "l2",
    weights: dict[str, float] | None = None,
) -> Tensor:
    """Masked per-body-part reconstruction error, weighted-averaged over body parts.

    Padded frames (mask == 0) don't contribute, otherwise the loss would
    reward the model for reconstructing zeros there. Each body part is
    averaged separately before being combined, so parts with more landmarks
    (e.g. upper_pose) don't dominate the total just by virtue of their size.

    Args:
        poses:           body_part -> (N, T, L, C) ground truth.
        reconstructions: body_part -> (N, T, L, C) reconstruction.
        mask:            (N, T) -- 1=valid, 0=pad.
        kind:            "l1" or "l2".
        weights:         body_part -> coefficient controlling how much that
            part's error contributes, e.g. {"upper_pose": 0.1, "left_hand":
            1.0, "right_hand": 1.0} to weight hands 10x more than pose. Body
            parts missing from this dict default to 1.0. The result is
            always normalized by the sum of weights, so leaving `weights`
            as `None` (or all equal) reduces to a plain mean over parts.
    """
    if kind not in ("l1", "l2"):
        raise ValueError(f"Unknown reconstruction loss: {kind!r}")
    error_fn = F.l1_loss if kind == "l1" else F.mse_loss

    frame_mask = mask[:, :, None, None]
    weighted_terms = []
    weight_sum = 0.0
    for body_part, recon in reconstructions.items():
        weight = 1.0 if weights is None else weights.get(body_part, 1.0)

        error = error_fn(recon, poses[body_part], reduction="none")
        frame_mask_f = frame_mask.to(error.dtype)
        denom = frame_mask_f.sum().clamp(min=1.0) * error.shape[-1] * error.shape[-2]
        term = (error * frame_mask_f).sum() / denom

        weighted_terms.append(weight * term)
        weight_sum += weight

    return torch.stack(weighted_terms).sum() / weight_sum
