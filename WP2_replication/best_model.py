"""
best_model.py
-------------
The BEST framework itself (Sec. 3.2-3.4):
    Pose Embedding Layer -> [mask] -> +temporal pos-enc -> Transformer Encoder
        -> (pre-training) softmax classifiers over (k_l, k_r, k_b)   [Eq. 5]
        -> (fine-tuning)  MLP head over gloss vocabulary
"""
import math
import torch
import torch.nn as nn

from gcn import PoseEmbeddingLayer
from mum import sample_mum_mask, apply_mask


class SinusoidalPositionalEncoding(nn.Module):
    """Standard Transformer temporal position encoding, f_temp,t (Sec 3.2)."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, dim)

    def forward(self, T: int):
        return self.pe[:T]  # (T, dim)


class BESTBackbone(nn.Module):
    """
    Shared trunk used both for MUM pre-training and downstream fine-tuning:
    pose embedding -> masking (train-time, pre-training only) -> +pos-enc
    -> Transformer encoder. Returns per-frame, per-part output features.
    """

    def __init__(self, D: int = 1536, n_heads: int = 8, n_layers: int = 4,
                 ffn_dim: int = 2048, dropout: float = 0.1, max_pe_len:int = 512):
        super().__init__()
        assert D % 3 == 0
        self.D = D
        self.d_part = D // 3

        self.pose_embed = PoseEmbeddingLayer(D=D)
        self.pos_enc = SinusoidalPositionalEncoding(D, max_len=max_pe_len)
        self.mask_token = nn.Parameter(torch.zeros(self.d_part))
        nn.init.normal_(self.mask_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, body, left_hand, right_hand, alpha: float = None):
        """
        body/left_hand/right_hand: (B, T, K, 2)
        alpha: mask ratio; if given (pre-training), applies MUM masking and
               also returns the masks so the caller can compute the MUM loss.
        Returns:
            f_out: (B, T, D) encoder output
            masks: (mask_l, mask_r, mask_b) or None if alpha is None
        """
        f_body, f_left, f_right, _ = self.pose_embed(body, left_hand, right_hand)
        B, T, _ = f_body.shape

        masks = None
        if alpha is not None:
            mask_l, mask_r, mask_b = sample_mum_mask(B, T, alpha, device=f_body.device)
            f_left, f_right, f_body = apply_mask(
                f_left, f_right, f_body, mask_l, mask_r, mask_b, self.mask_token
            )
            masks = (mask_l, mask_r, mask_b)

        f_in = torch.cat([f_body, f_left, f_right], dim=-1)     # (B, T, D)
        f_in = f_in + self.pos_enc(T).unsqueeze(0).to(f_in.device)
        f_out = self.transformer(f_in)                           # (B, T, D)
        return f_out, masks


class BESTPretrainModel(nn.Module):
    """
    Stage-2 model: backbone + softmax classifiers for MUM (Eq. 5).
    W1/b1 are SHARED between left and right hand (paper: "the corresponding
    label of each part", one hand projection W1 for both hand sub-units).
    """

    def __init__(self, D: int = 1536, num_hand_codes: int = 1000,
                 num_body_codes: int = 500, n_heads: int = 8, n_layers: int = 4, max_pe_len:int = 512):
        super().__init__()
        self.backbone = BESTBackbone(D=D, n_heads=n_heads, n_layers=n_layers, max_pe_len=max_pe_len)
        d_part = D // 3
        self.hand_head = nn.Linear(d_part, num_hand_codes)   # W1, b1
        self.body_head = nn.Linear(d_part, num_body_codes)   # W2, b2

    def forward(self, body, left_hand, right_hand, alpha: float = 0.4):
        f_out, (mask_l, mask_r, mask_b) = self.backbone(body, left_hand, right_hand, alpha=alpha)
        d_part = f_out.shape[-1] // 3
        f_body, f_left, f_right = f_out[..., :d_part], f_out[..., d_part:2 * d_part], f_out[..., 2 * d_part:]

        logits_l = self.hand_head(f_left)     # (B, T, num_hand_codes)
        logits_r = self.hand_head(f_right)
        logits_b = self.body_head(f_body)     # (B, T, num_body_codes)

        return {
            "logits_l": logits_l, "logits_r": logits_r, "logits_b": logits_b,
            "mask_l": mask_l, "mask_r": mask_r, "mask_b": mask_b,
        }


def mum_loss(out, k_l, k_r, k_b):
    """
    Eq. (6): maximize sum of log p(k_t | V_sign) over masked positions only,
    i.e. standard cross-entropy restricted to masked frames per part.
    k_l, k_r, k_b: (B, T) long tensors of pseudo-label indices from the tokenizer.
    """
    ce = nn.functional.cross_entropy

    def masked_ce(logits, targets, mask):
        if mask.sum() == 0:
            return logits.sum() * 0.0
        logits_sel = logits[mask]      # (N_masked, num_codes)
        targets_sel = targets[mask]    # (N_masked,)
        return ce(logits_sel, targets_sel)

    loss_l = masked_ce(out["logits_l"], k_l, out["mask_l"])
    loss_r = masked_ce(out["logits_r"], k_r, out["mask_r"])
    loss_b = masked_ce(out["logits_b"], k_b, out["mask_b"])
    total = loss_l + loss_r + loss_b
    return total, {"loss_l": loss_l.item(), "loss_r": loss_r.item(), "loss_b": loss_b.item()}


class BESTClassifier(nn.Module):
    """
    Stage-3 model (Sec. 3.4): pre-trained backbone (no masking at inference)
    + MLP prediction head over the temporally-pooled sequence, replacing the
    decoder. RGB late-fusion (Ours(+R)) is a separate RGB branch summed with
    this branch's softmax output at inference time and is not implemented
    here (out of scope: any RGB backbone, e.g. I3D, can be plugged in and its
    softmax added to this model's softmax).
    """

    def __init__(self, backbone: BESTBackbone, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        D = backbone.D
        self.head = nn.Sequential(
            nn.Linear(D, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, body, left_hand, right_hand):
        f_out, _ = self.backbone(body, left_hand, right_hand, alpha=None)  # no masking
        pooled = f_out.mean(dim=1)      # simple temporal average pooling
        return self.head(pooled)        # (B, num_classes)
