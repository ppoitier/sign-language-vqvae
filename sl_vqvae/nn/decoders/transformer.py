import torch
from torch import nn, Tensor
from torchtune.modules import RotaryPositionalEmbeddings

from sl_vqvae.nn.transformers.layers import TransformerEncoderLayer
from sl_vqvae.nn.transformers.attention_patterns import get_attn_mask_mod, padding_mask_mod, and_masks, build_block_mask
from sl_vqvae.nn.transformers.sinusoidal_pos_encoding import SinusoidalPositionalEncoding


class PoseTransformerDecoder(nn.Module):
    def __init__(
        self,
        c_out: int,
        c_hidden: int,
        max_length: int,
        c_in: int | None = None,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pos_encoding: str = "rope",
        attn_mask_strategy: str | None = None,
    ):
        super().__init__()
        c_in = c_in if c_in is not None else c_hidden
        self.n_heads = n_heads
        self.attn_mask_mod = get_attn_mask_mod(attn_mask_strategy)

        head_dim = c_hidden // n_heads

        self.proj_in = nn.Linear(c_in, c_hidden)

        # -- Positional encoding --
        rope = None
        self.additive_pe = None
        if pos_encoding == "rope":
            rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_length)
        elif pos_encoding == "sinusoidal":
            self.additive_pe = SinusoidalPositionalEncoding(c_hidden, max_seq_length=max_length)

        # -- Decoder layers (mirrors the encoder: self-attention only) --
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=c_hidden,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    rope=rope,
                )
                for _ in range(n_layers)
            ]
        )

        self.proj_out = nn.Linear(c_hidden, c_out)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x:    (N, T, C_in) -- (quantized) latent sequence.
            mask: (N, T) -- 1=valid, 0=pad.
        Returns:
            (N, T, c_out) -- reconstructed sequence.
        """
        N = x.size(0)

        padding = mask.bool()

        x = self.proj_in(x)

        if self.additive_pe is not None:
            x = self.additive_pe(x.transpose(1, 2)).transpose(1, 2)

        # -- Build attention mask --
        T = x.size(1)
        mask_mod = padding_mask_mod(padding)
        if self.attn_mask_mod is not None:
            mask_mod = and_masks(mask_mod, self.attn_mask_mod)
        block_mask = build_block_mask(
            mask_mod, B=N, H=None, Q_LEN=T, KV_LEN=T, device=x.device
        )

        for layer in self.layers:
            x = layer(x, block_mask=block_mask)

        return self.proj_out(x)


if __name__ == '__main__':
    N, T, C_hidden, C_out = 16, 500, 256, 130
    z = torch.randn(N, T, C_hidden).cuda()
    mask = torch.ones(N, T).bool().cuda()
    model = PoseTransformerDecoder(c_out=C_out, c_hidden=C_hidden, max_length=500).cuda()
    print(model(z, mask).shape)
