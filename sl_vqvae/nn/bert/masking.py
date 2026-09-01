"""
Masked Unit Modeling (MUM) masking, adapted from Zhao et al. 2023 ("BEST"),
Sec. 3.3, to operate on pre-tokenized body-part groups instead of raw pose
features.

For a sequence of length T:
  1. Pick round(alpha * T) valid frame positions uniformly at random -> the
     "selected" frames.
  2. For each selected frame, independently mask each vocabulary group
     (e.g. 'hand', 'pose') with 50% probability.

Body parts sharing a vocabulary group (e.g. left_hand/right_hand sharing
'hand') are always masked together, since they share both the embedding
table and the prediction head.
"""
import torch
from torch import Tensor


def sample_bert_mask(
    valid: Tensor, alpha: float, groups: tuple[str, ...]
) -> dict[str, Tensor]:
    """
    Args:
        valid:  (N, T) bool -- True = real frame, False = padding.
        alpha:  fraction of valid frames to select as masking candidates.
        groups: vocabulary group names.
    Returns:
        group -> (N, T) bool mask, True = replace this group's token with the
        mask token at this frame.
    """
    N, T = valid.shape
    device = valid.device
    num_select = max(1, round(alpha * T))

    # Random score per position; padded positions are forced last so they are
    # never selected once masked out below.
    scores = torch.rand(N, T, device=device).masked_fill(~valid, -1.0)
    _, top_idx = scores.topk(min(num_select, T), dim=1)
    selected = torch.zeros(N, T, dtype=torch.bool, device=device)
    selected.scatter_(1, top_idx, True)
    selected &= valid

    return {group: selected & (torch.rand(N, T, device=device) < 0.5) for group in groups}


def apply_bert_mask(
    embeds: dict[str, Tensor], masks: dict[str, Tensor], mask_token: Tensor
) -> dict[str, Tensor]:
    """
    Args:
        embeds:     body_part -> (N, T, d_part) token embeddings.
        masks:      body_part -> (N, T) bool, True = mask this position.
        mask_token: (d_part,) learnable embedding shared across all parts.
    """
    out = {}
    for body_part, x in embeds.items():
        mask = masks[body_part].unsqueeze(-1)  # (N, T, 1)
        out[body_part] = torch.where(mask, mask_token, x)
    return out
