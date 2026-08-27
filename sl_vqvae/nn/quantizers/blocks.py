import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm, with a skip connection."""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """
    Input:  (N, C_in)   e.g. C_in=130 (flattened x,y over 65 landmarks)
    Output: (N, D)      D = embedding_dim fed to the VQ codebook

    Overall encoder/decoder + vector-quantization structure follows:
        van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017).
        "Neural Discrete Representation Learning." NeurIPS.
    adapted from Conv2d (images) to Linear layers (flattened pose
    coordinates), following the coupling-tokenization triplet-unit
    design in:
        Zhao, W., Hu, H., Zhou, W., Shi, J., & Li, H. (2023).
        "BEST: BERT Pre-Training for Sign Language Recognition with
        Coupling Tokenization." AAAI.
    """

    def __init__(self, c_in=130, c_out=512, c_hidden=256, n_blocks=2):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(c_in, c_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c_hidden // 2, c_hidden),
            nn.ReLU(inplace=True),
        )
        self.residual_stack = nn.Sequential(
            *[ResidualBlock(c_hidden) for _ in range(n_blocks)]
        )
        self.fc_out = nn.Linear(c_hidden, c_out)

    def forward(self, x):
        # x: (N, C_in)
        x = self.fc_in(x)
        x = self.residual_stack(x)
        z = self.fc_out(x)  # (N, embedding_dim)
        return z


class Decoder(nn.Module):
    """
    Input:  (N, D)      D = embedding_dim (quantized vectors z_q)
    Output: (N, C_out)  reconstructed flattened landmark coordinates

    Mirrors the Encoder architecture, per the symmetric encoder/decoder
    design in:
        van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017).
        "Neural Discrete Representation Learning." NeurIPS.
    """

    def __init__(self,
        c_in=512,
        c_out=130,
        c_hidden=256,
        n_blocks=2,
    ):
        super().__init__()
        self.fc_in = nn.Linear(c_in, c_hidden)
        self.residual_stack = nn.Sequential(
            *[ResidualBlock(c_hidden) for _ in range(n_blocks)]
        )
        self.fc_out = nn.Sequential(
            nn.Linear(c_hidden, c_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c_hidden // 2, c_out),
        )

    def forward(self, z_q):
        # z_q: (N, D)
        x = self.fc_in(z_q)
        x = self.residual_stack(x)
        x_hat = self.fc_out(x)  # (N, C_out)
        return x_hat


if __name__ == "__main__":
    encoder = Encoder(c_in=130, c_hidden=256, c_out=512)
    decoder = Decoder(c_in=512, c_hidden=256, c_out=130)

    N, C_in = 16, 130
    x = torch.randn(N, C_in)
    z = encoder(x)
    x_hat = decoder(z)
    print(x_hat.shape)  # torch.Size([16, 130])
