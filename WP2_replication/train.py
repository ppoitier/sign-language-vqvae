"""
train.py
--------
End-to-end driver for the three stages described in the paper:

  Stage 1  train_tokenizer()   Sec. 3.1  - fit the d-VAE coupling tokenizer
  Stage 2  pretrain()          Sec. 3.2-3.3 - MUM self-supervised pre-training
  Stage 3  finetune()          Sec. 3.4  - downstream isolated-SLR fine-tuning

A `DummySignDataset` stands in for a real dataset (e.g. WLASL/MSASL/NMFs-CSL/
SLR500 pose sequences extracted with MMPose, per Sec. 4.2 "Data Preparation").
Swap it for a Dataset that reads your own extracted 49-joint 2D poses,
following the same __getitem__ contract:
    body:  (T, 7, 2) float
    left:  (T, 21, 2) float
    right: (T, 21, 2) float
    label: int (only needed for fine-tuning)
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizer import CouplingTokenizer, dvae_loss
from best_model import BESTBackbone, BESTPretrainModel, BESTClassifier, mum_loss


# ---------------------------------------------------------------------------
# Config (Sec. 4.2 "Model Hyper-Parameters" / "Training Setup")
# ---------------------------------------------------------------------------
class Cfg:
    D = 1536                 # Transformer input size
    N_HEADS = 8
    N_LAYERS = 4              # not stated explicitly in the paper; pick to fit your budget
    FFN_DIM = 2048
    NUM_HAND_CODES = 1000     # |V_hand|  (M1)
    NUM_BODY_CODES = 500      # |V_body|  (M2)
    T = 32                    # frames per clip (Sec 4.2: 32 frames sampled)
    ALPHA = 0.4               # MUM mask ratio (not stated exactly; paper ablates it -- tune this)
    BATCH_SIZE = 8
    NUM_CLASSES = 100         # e.g. MSASL100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DummySignDataset(Dataset):
    """Synthetic stand-in for a pose-extracted sign language dataset."""

    def __init__(self, n_samples=64, T=Cfg.T, num_classes=Cfg.NUM_CLASSES):
        self.n = n_samples
        self.T = T
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        body = torch.randn(self.T, 7, 2)
        left = torch.randn(self.T, 21, 2)
        right = torch.randn(self.T, 21, 2)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return body, left, right, label


# ---------------------------------------------------------------------------
# Stage 1: train the coupling tokenizer (frame-level, no temporal modeling)
# ---------------------------------------------------------------------------
def train_tokenizer(epochs=5, lr=1e-3, decay_every=10, decay_factor=0.1):
    dataset = DummySignDataset()
    loader = DataLoader(dataset, batch_size=Cfg.BATCH_SIZE, shuffle=True)

    tokenizer = CouplingTokenizer(
        d_part=Cfg.D // 3,
        num_hand_codes=Cfg.NUM_HAND_CODES,
        num_body_codes=Cfg.NUM_BODY_CODES,
    ).to(Cfg.DEVICE)

    opt = torch.optim.Adam(tokenizer.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_every, gamma=decay_factor)

    tokenizer.train()
    for epoch in range(epochs):
        running = 0.0
        for body, left, right, _ in loader:
            # tokenizer operates per-frame: flatten (B, T, K, 2) -> (B*T, K, 2)
            B, T = body.shape[0], body.shape[1]
            body_f = body.reshape(B * T, *body.shape[2:]).to(Cfg.DEVICE)
            left_f = left.reshape(B * T, *left.shape[2:]).to(Cfg.DEVICE)
            right_f = right.reshape(B * T, *right.shape[2:]).to(Cfg.DEVICE)

            out = tokenizer(body_f, left_f, right_f)
            loss, logs = dvae_loss(out, left_f, right_f, body_f)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        print(f"[tokenizer] epoch {epoch+1}/{epochs}  loss={running/len(loader):.4f}  {logs}")

    torch.save(tokenizer.state_dict(), "tokenizer.pt")
    return tokenizer


# ---------------------------------------------------------------------------
# Stage 2: MUM self-supervised pre-training (Sec. 3.2-3.3)
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_pseudo_labels(tokenizer: CouplingTokenizer, body, left, right):
    """body/left/right: (B, T, K, 2) -> k_l, k_r, k_b: (B, T) long tensors."""
    B, T = body.shape[0], body.shape[1]
    body_f = body.reshape(B * T, *body.shape[2:])
    left_f = left.reshape(B * T, *left.shape[2:])
    right_f = right.reshape(B * T, *right.shape[2:])
    k_l, k_r, k_b = tokenizer.tokenize(body_f, left_f, right_f)
    return k_l.view(B, T), k_r.view(B, T), k_b.view(B, T)


def pretrain(tokenizer: CouplingTokenizer, epochs=5, lr=1e-4, warmup_epochs=6,
             weight_decay=0.01):
    dataset = DummySignDataset()
    loader = DataLoader(dataset, batch_size=Cfg.BATCH_SIZE, shuffle=True)

    tokenizer = tokenizer.to(Cfg.DEVICE)
    for p in tokenizer.parameters():
        p.requires_grad = False   # "tokenizer ... with all parameters frozen during pre-training"
    tokenizer.eval()

    model = BESTPretrainModel(
        D=Cfg.D, num_hand_codes=Cfg.NUM_HAND_CODES, num_body_codes=Cfg.NUM_BODY_CODES,
        n_heads=Cfg.N_HEADS, n_layers=Cfg.N_LAYERS,
    ).to(Cfg.DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(step, steps_per_epoch=len(loader)):
        warmup_steps = warmup_epochs * steps_per_epoch
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0  # (paper: linear decay after warmup -- add a decay schedule of your choice)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model.train()
    step = 0
    for epoch in range(epochs):
        running = 0.0
        for body, left, right, _ in loader:
            body, left, right = body.to(Cfg.DEVICE), left.to(Cfg.DEVICE), right.to(Cfg.DEVICE)
            k_l, k_r, k_b = get_pseudo_labels(tokenizer, body, left, right)

            out = model(body, left, right, alpha=Cfg.ALPHA)
            loss, logs = mum_loss(out, k_l, k_r, k_b)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            running += loss.item()
        print(f"[pretrain] epoch {epoch+1}/{epochs}  loss={running/len(loader):.4f}  {logs}")

    torch.save(model.backbone.state_dict(), "backbone_pretrained.pt")
    return model.backbone


# ---------------------------------------------------------------------------
# Stage 3: downstream fine-tuning (Sec. 3.4)
# ---------------------------------------------------------------------------
def finetune(backbone: BESTBackbone, epochs=10, lr=1e-4, decay_every=10, decay_factor=0.1):
    dataset = DummySignDataset(num_classes=Cfg.NUM_CLASSES)
    loader = DataLoader(dataset, batch_size=Cfg.BATCH_SIZE, shuffle=True)

    model = BESTClassifier(backbone, num_classes=Cfg.NUM_CLASSES).to(Cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_every, gamma=decay_factor)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running, correct, total = 0.0, 0, 0
        for body, left, right, label in loader:
            body, left, right = body.to(Cfg.DEVICE), left.to(Cfg.DEVICE), right.to(Cfg.DEVICE)
            label = label.to(Cfg.DEVICE)

            logits = model(body, left, right)
            loss = criterion(logits, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            correct += (logits.argmax(-1) == label).sum().item()
            total += label.size(0)
        sched.step()
        print(f"[finetune] epoch {epoch+1}/{epochs}  loss={running/len(loader):.4f}  "
              f"acc={correct/total:.4f}")

    torch.save(model.state_dict(), "best_finetuned.pt")
    return model


if __name__ == "__main__":
    # Stage 1
    tokenizer = train_tokenizer(epochs=2)          # use ~epochs=100s+ / real data in practice
    # Stage 2
    backbone = pretrain(tokenizer, epochs=2)
    # Stage 3
    finetune(backbone, epochs=2)
