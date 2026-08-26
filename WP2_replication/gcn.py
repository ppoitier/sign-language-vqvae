"""
gcn.py
------
Graph-convolutional pose embedding, corresponding to Sec. 3.2 "Pose Embedding
Layer" of the BEST paper (Zhao et al., AAAI 2023).

The paper extracts, per frame, a *pose triplet unit*: an upper-body pose
(23 joints), a left-hand pose (21 joints) and a right-hand pose (21 joints),
each in 2D (x, y). It embeds each part with the ST-GCN-style graph
convolution of Cai et al. (2019), and concatenates the three part embeddings
into one D-dim vector (D_part = D/3 each).

We don't have the exact Cai et al. adjacency/architecture, so this is a
faithful *simplified* single-hop spatial GCN: it is the standard
"normalize(A) @ X @ W" graph-conv block, stacked a couple of times, followed
by a temporal-agnostic pooling over joints. This is a reasonable, commonly
used stand-in and keeps the same input/output contract (K joints x 2 ->
D_part vector per frame).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sign_language_tools.pose.mediapipe.edges import UPPER_POSE_EDGES, HAND_EDGES
NUM_BODY_JOINTS = 23
NUM_HAND_JOINTS = 21


def build_adjacency(num_joints: int, edges) -> torch.Tensor:
    """Symmetric-normalized adjacency with self-loops: D^-1/2 (A+I) D^-1/2."""
    A = torch.eye(num_joints)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = A.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


class GraphConv(nn.Module):
    """One spatial graph-convolution: X' = act(A_norm @ X @ W)."""

    def __init__(self, in_dim: int, out_dim: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("A", adjacency)          # (K, K), fixed topology
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, C_in)
        x = self.A @ x            # aggregate neighbor features: (B, K, C_in)
        x = self.linear(x)        # (B, K, C_out)
        return self.act(x)


class PartGCN(nn.Module):
    """Small stack of GraphConv blocks + joint pooling -> single part embedding."""

    def __init__(self, num_joints: int, edges, in_dim: int = 2,
                 hidden_dim: int = 128, out_dim: int = 512, num_layers: int = 2):
        super().__init__()
        A = build_adjacency(num_joints, edges)
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            GraphConv(dims[i], dims[i + 1], A) for i in range(len(dims) - 1)
        ])
        self.out_dim = out_dim

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        """joints: (B, K, 2) 2D coordinates for one part, one frame -> (B, out_dim)"""
        x = joints
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=1)  # pool over joints -> (B, out_dim)


class PoseEmbeddingLayer(nn.Module):
    """
    Embeds a full pose triplet unit (body, left hand, right hand) for one
    frame into a D-dim vector, D_part = D / 3 each, then concatenated
    (Sec. 3.2, "Pose Embedding Layer").
    """

    def __init__(self, D: int = 1536, hidden_dim: int = 128, gcn_layers: int = 2):
        super().__init__()
        assert D % 3 == 0, "D must be divisible by 3 (body, left hand, right hand)"
        self.d_part = D // 3
        self.body_gcn = PartGCN(NUM_BODY_JOINTS, UPPER_POSE_EDGES, out_dim=self.d_part,
                                 hidden_dim=hidden_dim, num_layers=gcn_layers)
        self.lhand_gcn = PartGCN(NUM_HAND_JOINTS, HAND_EDGES, out_dim=self.d_part,
                                  hidden_dim=hidden_dim, num_layers=gcn_layers)
        self.rhand_gcn = PartGCN(NUM_HAND_JOINTS, HAND_EDGES, out_dim=self.d_part,
                                  hidden_dim=hidden_dim, num_layers=gcn_layers)

    def forward(self, body, left_hand, right_hand):
        """
        body:       (B, T, 23, 2)
        left_hand:  (B, T, 21, 2)
        right_hand: (B, T, 21, 2)
        returns:
            f_body, f_left, f_right : each (B, T, D_part)
            f_concat                : (B, T, D)  concatenation of the three
        """
        B, T = body.shape[0], body.shape[1]

        def embed_part(gcn, x):
            x = x.reshape(B * T, x.shape[2], x.shape[3])   # (B*T, K, 2)
            out = gcn(x)                                    # (B*T, D_part)
            return out.reshape(B, T, -1)

        f_body = embed_part(self.body_gcn, body)
        f_left = embed_part(self.lhand_gcn, left_hand)
        f_right = embed_part(self.rhand_gcn, right_hand)
        f_concat = torch.cat([f_body, f_left, f_right], dim=-1)  # (B, T, D)
        return f_body, f_left, f_right, f_concat
