import torch.nn as nn
from positionalencoding import PositionalEncoding

class MotionMaskGIT(nn.Module):

    def __init__(
        self,
        codebook_size=1024, #### AGAIN WE CAN CHANGE EVERYTHING HERE ! AND WE SHOULD. THIS IS JUST TO TEST IF EVERYTHING WAS WORKING
        d_model=768,
        nhead=12,
        num_layers=6
    ):
        super().__init__()

        self.mask_token_id = codebook_size

        self.embedding = nn.Embedding(
            codebook_size + 1,
            d_model
        )

        self.pos_encoding = PositionalEncoding(
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(
            d_model,
            codebook_size
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        x = self.pos_encoding(x)

        x = self.transformer(x)

        logits = self.classifier(x)

        return logits
