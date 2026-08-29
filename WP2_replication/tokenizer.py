"""
tokenizer.py
------------
Coupling tokenizer: a discrete VAE (d-VAE) that jointly discretizes a pose
triplet unit into (k_l, k_r, k_b) pseudo-labels, corresponding to Sec. 3.1
"Tokenization in Pre-Training" and Eq. (1)-(3) of the BEST paper.

Pipeline (per frame):
    (J_left, J_right, J_body) --Enc--> (z_l, z_r, z_b)
                               --Quantize (Eq.1)--> (h_kl, h_kr, d_kb) = z_q
                               --Dec--> (Ĵ_left, Ĵ_right, Ĵ_body)

Trained BEFORE the main BEST pre-training, then frozen and used only to
produce pseudo labels {k_t} for the MUM pretext task.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from gcn import PoseEmbeddingLayer


class TripletEncoder(nn.Module):
    """Enc(J_sign) -> (z_l, z_r, z_b), one latent vector per part."""

    def __init__(self, d_part: int = 512, hidden_dim: int = 128):
        super().__init__()
        # Re-use the same GCN-based part encoder used for pose embedding;
        # here D = 3 * d_part so each part embedding already has dim d_part.
        self.pose_embed = PoseEmbeddingLayer(D=3 * d_part, hidden_dim=hidden_dim)

    def forward(self, body, left_hand, right_hand):
        # squeeze a dummy time dim of 1 since the tokenizer works per-frame
        f_body, f_left, f_right, _ = self.pose_embed(
            body.unsqueeze(1), left_hand.unsqueeze(1), right_hand.unsqueeze(1)
        )
        return f_left.squeeze(1), f_right.squeeze(1), f_body.squeeze(1)  # z_l, z_r, z_b


class TripletDecoder(nn.Module):
    """Dec(z_q) -> reconstructed 2D joints for left hand, right hand, body."""

    def __init__(self, d_part: int = 512, hidden_dim: int = 256,
                 num_left_joints: int = 21, num_right_joints: int = 21,
                 num_body_joints: int = 23):
        super().__init__()

        def mlp(out_joints):
            return nn.Sequential(
                nn.Linear(d_part, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_joints * 2),
            )

        self.dec_left = mlp(num_left_joints)
        self.dec_right = mlp(num_right_joints)
        self.dec_body = mlp(num_body_joints)
        self.nl, self.nr, self.nb = num_left_joints, num_right_joints, num_body_joints

    def forward(self, zq_l, zq_r, zq_b):
        left = self.dec_left(zq_l).view(-1, self.nl, 2)
        right = self.dec_right(zq_r).view(-1, self.nr, 2)
        body = self.dec_body(zq_b).view(-1, self.nb, 2)
        return left, right, body


class VectorQuantizer(nn.Module):
    """
    Nearest-neighbour codebook lookup, Eq. (1):
        k = argmin_k || z - codebook[k] ||
    Implements the straight-through estimator so gradients flow z -> Enc
    even though the argmin itself is non-differentiable.
    """

    def __init__(self, num_codes: int, code_dim: int):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        # self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z: torch.Tensor):
        # z: (B, C) -> distances to every codeword: (B, num_codes)
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)                 # (B,)  -- this is k
        z_q = self.codebook(indices)                  # (B, C)
        # straight-through estimator: copy gradient from z_q to z
        z_q_st = z + (z_q - z).detach()
        return z_q_st, z_q, indices

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes, code_dim, decay=0.99, eps=1e-5, revive_every=50, usage_threshold=1):
        super().__init__()
        self.decay, self.eps = decay, eps
        self.revive_every, self.usage_threshold = revive_every, usage_threshold
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.requires_grad_(False)   # updated by EMA, not Adam
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", self.codebook.weight.data.clone())
        self.register_buffer("usage", torch.zeros(num_codes))
        self._step = 0

    def forward(self, z):
        dist = (z.pow(2).sum(1, keepdim=True) - 2 * z @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1))
        indices = dist.argmin(dim=1)
        z_q = self.codebook(indices)
        z_q_st = z + (z_q - z).detach()

        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.codebook.weight.shape[0]).float()
                self.cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
                self.ema_w.mul_(self.decay).add_(one_hot.t() @ z, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.eps) / (n + self.codebook.weight.shape[0] * self.eps) * n
                self.codebook.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
                self.usage += one_hot.sum(0)

                self._step += 1
                if self._step % self.revive_every == 0:
                    dead = self.usage < self.usage_threshold
                    if dead.any():
                        idx = torch.randint(0, z.shape[0], (dead.sum(),), device=z.device)
                        self.codebook.weight.data[dead] = z[idx]
                        self.cluster_size[dead] = 1.0
                        self.ema_w[dead] = z[idx]
                    self.usage.zero_()

        return z_q_st, z_q, indices


class CouplingTokenizer(nn.Module):
    """
    Full d-VAE: encoder + (hand codebook shared by left/right, body codebook)
    + decoder. "Coupling" = the same hand codebook Q_hand(.) is used for both
    left and right hand, so the two hands are tokenized in a shared space
    while still being jointly optimized alongside the body codebook.
    """

    def __init__(self, d_part: int = 512, num_hand_codes: int = 1000,
                 num_body_codes: int = 500):
        super().__init__()
        self.encoder = TripletEncoder(d_part=d_part)
        self.left_hand_quantizer = VectorQuantizerEMA(num_hand_codes, d_part)   # Q_hand(.)
        self.right_hand_quantizer = VectorQuantizerEMA(num_hand_codes, d_part)   # Q_hand(.)
        self.body_quantizer = VectorQuantizerEMA(num_body_codes, d_part)   # Q_body(.)
        self.decoder = TripletDecoder(d_part=d_part)
        self.weight_init = False

    @torch.no_grad()
    def init_tokenizer_weights(self, z_l, z_r, z_b):
        self._init_codebook_from_samples(self.left_hand_quantizer.codebook, z_l)
        self._init_codebook_from_samples(self.right_hand_quantizer.codebook, z_r)
        self._init_codebook_from_samples(self.body_quantizer.codebook, z_b)
        self.weight_init = True

    @staticmethod
    def _init_codebook_from_samples(codebook: nn.Embedding, z: torch.Tensor):
        num_codes = codebook.weight.shape[0]
        n = z.shape[0]
        if n >= num_codes:
            idx = torch.randperm(n, device=z.device)[:num_codes]
        else:
            # not enough real samples yet: sample with replacement, and warn
            idx = torch.randint(0, n, (num_codes,), device=z.device)
        codebook.weight.data.copy_(z[idx])

    def forward(self, body, left_hand, right_hand):
        if not self.weight_init:
            raise ValueError("VectorQuantizer weights were not initialized. Call init_tokenizer_weights before forward pass.")
        
        z_l, z_r, z_b = self.encoder(body, left_hand, right_hand)

        zq_l_st, zq_l, k_l = self.left_hand_quantizer(z_l)
        zq_r_st, zq_r, k_r = self.right_hand_quantizer(z_r)   # SAME codebook -> coupling
        zq_b_st, zq_b, k_b = self.body_quantizer(z_b)

        left_hat, right_hat, body_hat = self.decoder(zq_l_st, zq_r_st, zq_b_st)

        out = {
            "left_hat": left_hat, "right_hat": right_hat, "body_hat": body_hat,
            "z_l": z_l, "z_r": z_r, "z_b": z_b,
            "zq_l": zq_l, "zq_r": zq_r, "zq_b": zq_b,
            "k_l": k_l, "k_r": k_r, "k_b": k_b,
        }
        return out

    @torch.no_grad()
    def tokenize(self, body, left_hand, right_hand):
        """Inference-only: return pseudo labels (k_l, k_r, k_b) for MUM pre-training."""
        self.eval()
        out = self.forward(body, left_hand, right_hand)
        return out["k_l"], out["k_r"], out["k_b"]


def dvae_loss(out, J_left, J_right, J_body, beta1=0.1, beta2=1.0, beta3=0.9):
    """
    Eq. (3):
        L = L_hand + beta1 * L_body
            + beta2 * || sg[z]  - z_q ||^2      (commitment: move codebook to z)
            + beta3 * || sg[z_q] - z  ||^2      (commitment: move z to codebook)
    L_hand = ||Ĵ_left - J_left|| + ||Ĵ_right - J_right||   (Sec. 3.1)
    L_body = ||Ĵ_body - J_body||
    """
    L_hand = F.l1_loss(out["left_hat"], J_left) + F.l1_loss(out["right_hat"], J_right)
    L_body = F.l1_loss(out["body_hat"], J_body)

    def commitment(z, zq):
        term1 = F.mse_loss(z.detach(), zq)   # || sg[z]  - z_q ||^2
        term2 = F.mse_loss(z, zq.detach())   # || sg[z_q] - z  ||^2
        return term1, term2

    t1_l, t2_l = commitment(out["z_l"], out["zq_l"])
    t1_r, t2_r = commitment(out["z_r"], out["zq_r"])
    t1_b, t2_b = commitment(out["z_b"], out["zq_b"])

    commit_term1 = t1_l + t1_r + t1_b
    commit_term2 = t2_l + t2_r + t2_b

    loss = L_hand + beta1 * L_body + beta2 * commit_term1 + beta3 * commit_term2
    logs = {"L_hand": L_hand.item(), "L_body": L_body.item(),
            "commit1": commit_term1.item(), "commit2": commit_term2.item()}
    return loss, logs
