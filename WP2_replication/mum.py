"""
mum.py
------
Masked Unit Modeling (MUM) masking strategy, Sec. 3.3.

For a triplet sequence of length T:
  1. Pick alpha * T frame positions uniformly at random -> "selected" frames.
  2. For each selected frame, independently mask each of {hand-part, body-part}
     with 50% probability, using a shared learnable mask token e_mask.
     ("hand-part" here covers left+right hand together, matching the paper's
     three described cases: hand-only, body-only, hand+body masked.)

Returns boolean masks per part (M_l, M_r, M_b) so the caller can:
  - replace masked embeddings with the mask token,
  - and know which positions to compute the reconstruction loss over.
"""
import torch


def sample_mum_mask(B: int, T: int, alpha: float, device=None):
    """
    Returns three (B, T) bool tensors: mask_left, mask_right, mask_body.
    mask_left == mask_right always (both hands masked/unmasked together, since
    the paper treats "hand" as one unit when deciding hand-vs-body masking).
    """
    device = device or "cpu"
    num_select = max(1, int(round(alpha * T)))

    selected = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(T, device=device)[:num_select]
        selected[b, idx] = True

    # for each selected frame: mask hand w.p. 0.5, mask body w.p. 0.5 (independent)
    hand_coin = torch.rand(B, T, device=device) < 0.5
    body_coin = torch.rand(B, T, device=device) < 0.5

    mask_hand = selected & hand_coin
    mask_body = selected & body_coin

    return mask_hand, mask_hand.clone(), mask_body  # (M_l, M_r, M_b)


def apply_mask(f_left, f_right, f_body, mask_l, mask_r, mask_b, mask_token: torch.Tensor):
    """
    f_left/right/body: (B, T, D_part)
    mask_*: (B, T) bool
    mask_token: (D_part,) learnable embedding, broadcast to masked positions.
    """
    f_left = f_left.clone()
    f_right = f_right.clone()
    f_body = f_body.clone()
    f_left[mask_l] = mask_token
    f_right[mask_r] = mask_token
    f_body[mask_b] = mask_token
    return f_left, f_right, f_body
