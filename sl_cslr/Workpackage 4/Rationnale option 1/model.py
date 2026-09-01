import torch
import torch.nn as nn
from CSLR_transformer import CSLRTransformer

from hyperparameters import DEVICE, VOCAB_SIZE, INPUT_DIM

#=== =Model======

model = CSLRTransformer(
    input_dim=INPUT_DIM,
    d_model=512,
    num_layers=6,
    nhead=8,
    ff_dim=2048,
    vocab_size=VOCAB_SIZE,
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

ctc_loss = nn.CTCLoss(
    blank=VOCAB_SIZE,
    zero_infinity=True
)