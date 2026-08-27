import torch
import torch.nn as nn
from pyarrow.lib import Tensor

from sign_language_tools.pose.mediapipe.edges import UPPER_POSE_EDGES, HAND_EDGES


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
        self.register_buffer("A", adjacency)          # (L, L), fixed topology
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, L, C_in)
        x = self.A @ x  # aggregate neighbor features
        x = self.linear(x)
        # x: (N, L, C_out)
        return self.act(x)


class BodyPartGCN(nn.Module):
    def __init__(
            self,
            n_joints: int,
            edges: list[tuple[int, int]],
            c_in: int = 2,
            c_hidden: int = 128,
            c_out: int = 512,
            n_layers: int = 2,
    ):
        super().__init__()
        A = build_adjacency(n_joints, edges)
        self.layers = nn.Sequential(
            GraphConv(c_in, c_hidden, A),
            *[GraphConv(c_hidden, c_hidden, A) for _ in range(n_layers)],
            GraphConv(c_hidden, c_out, A),
        )

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joints: tensor of shape (N, L, 2).
                2D coordinates for one body part, one frame -> (N, c_out)
        """
        out = self.layers(joints)
        return out.mean(dim=1)  # pool over joints -> (N, c_out)


class PoseEmbeddingLayer(nn.Module):
    def __init__(
            self,
            embed_dim_part = 512,
            c_hidden: int = 128,
            n_gcn_layers: int = 2,
            n_body_joints: int = 23,
            n_hand_joints: int = 21,
    ):
        super().__init__()
        body_gcn = BodyPartGCN(n_body_joints, UPPER_POSE_EDGES, c_out=embed_dim_part, c_hidden=c_hidden, n_layers=n_gcn_layers)
        hand_gcn = BodyPartGCN(n_hand_joints, HAND_EDGES, c_out=embed_dim_part, c_hidden=c_hidden, n_layers=n_gcn_layers)

        self.gcn_encoders = nn.ModuleDict({
            'upper_pose': body_gcn,
            'left_hand': hand_gcn,
            'right_hand': hand_gcn
        })

    def forward(self, pose):
        out: dict[str, Tensor] = dict()
        for body_part, x in pose.items():
            if body_part == 'left_hand':
                # Flip the x-axis
                x = torch.cat([1.0 - x[..., :1], x[..., 1:]], dim=-1)

            N, T, L, C = x.shape
            # (N, T, L, C) -> (N * T, L, C)
            x = x.view(N * T, L, C)
            z = self.gcn_encoders[body_part](x)
            out[body_part] = z.view(N, T, -1)
        return out


if __name__ == '__main__':
    N, T, L_pose, L_hand, C = 16, 500, 23, 21, 2

    x = {
        'upper_pose': torch.rand(N, T, L_pose, C),
        'left_hand': torch.rand(N, T, L_hand, C),
        'right_hand': torch.rand(N, T, L_hand, C)
    }

    model = PoseEmbeddingLayer()
    out = model(x)
    print({k: v.shape for k, v in out.items()})

