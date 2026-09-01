import torch
import torch.nn as nn
import math
from positionnal_encoding import PositionalEncoding


# ===CSLR Transformer =========

class CSLRTransformer(nn.Module):
    def __init__(
        self,
        input_dim=768,
        d_model=512,
        num_layers=6,
        nhead=8,
        ff_dim=2048,
        vocab_size=100,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        self.pos_encoding = PositionalEncoding(
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(
            d_model,
            vocab_size + 1  # +1 for CTC blank
        )

    def forward(self, x):

        x = self.input_proj(x)

        x = self.pos_encoding(x)

        x = self.encoder(x)

        logits = self.classifier(x)

        return logits
