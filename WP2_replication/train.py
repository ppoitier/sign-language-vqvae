"""
train.py
--------
End-to-end driver for the three stages described in the paper:

  Stage 1  train_tokenizer()   Sec. 3.1  - fit the d-VAE coupling tokenizer
  Stage 2  pretrain()          Sec. 3.2-3.3 - MUM self-supervised pre-training
  Stage 3  finetune()          Sec. 3.4  - downstream isolated-SLR fine-tuning

A `DummySignDataset` stands in for a real dataset (e.g. WLASL/MSASL/NMFs-CSL/
SLR500 pose sequences extracted with MMPose, per Sec. 4.2 "Data Preparation").
Swap it for a Dataset that reads your own extracted 65-joint 2D poses,
following the same __getitem__ contract:
    body:  (T, 23, 2) float
    left:  (T, 21, 2) float
    right: (T, 21, 2) float
    label: int (only needed for fine-tuning)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pathlib import Path

from tokenizer import CouplingTokenizer, dvae_loss
from best_model import BESTBackbone, BESTPretrainModel, BESTClassifier, mum_loss

from sldl import SignLanguageDataset, SignLanguageCollator
from sign_language_tools.pose.transforms import DropCoordinates, CenterOnLandmarks
from sign_language_tools.common.transforms import ApplyToAll, MapTransform, Identity, Compose

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
    T = 256                   # frames per clip (Sec 4.2: 32 frames sampled)
    ALPHA = 0.4               # MUM mask ratio (not stated exactly; paper ablates it -- tune this)
    BATCH_SIZE = 16
    NUM_CLASSES = 100         # e.g. MSASL100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ROOT = Path(__file__).resolve().parent / "data"

hand_transform = Compose([
    CenterOnLandmarks(landmark_idx=0),
])

transforms = Compose([
    ApplyToAll(DropCoordinates("z")),
    MapTransform({
        'upper_pose': Identity(),
        'left_hand': hand_transform,
        'right_hand': hand_transform,
    }),
])

TRAIN_DATASET = SignLanguageDataset(
    shards_url=(Cfg.ROOT / "shards" / "annotated" / "shard_000000.tar").as_uri(),
    use_windows=True,
    window_size=Cfg.T,
    window_stride=Cfg.T,
    max_empty_windows=0,
    pose_transform=transforms,
    show_loading_progress=True,
)

VAL_DATASET = SignLanguageDataset(
    shards_url=(Cfg.ROOT / "shards" / "annotated" / "shard_000001.tar").as_uri(),
    use_windows=True,
    window_size=Cfg.T,
    window_stride=Cfg.T,
    max_empty_windows=0,
    pose_transform=transforms,
    show_loading_progress=True,
)

@torch.no_grad()
def codebook_usage(tokenizer, loader, prep_fn):
    tokenizer.eval()
    used_hand_left, used_hand_right, used_body = set(), set(), set()
    for batch in loader:
        body_f, left_f, right_f = prep_fn(batch)
        k_l, k_r, k_b = tokenizer.tokenize(body_f, left_f, right_f)
        used_hand_left.update(k_l.tolist()); used_hand_right.update(k_r.tolist())
        used_body.update(k_b.tolist())
    tokenizer.train()
    return len(used_hand_left), len(used_hand_right), len(used_body)

# ---------------------------------------------------------------------------
# Stage 1: train the coupling tokenizer (frame-level, no temporal modeling)
# ---------------------------------------------------------------------------
def train_tokenizer(epochs=5, lr=1e-3, decay_every=10, decay_factor=0.1, eval_dataset=None, min_samples=8000, best_ckpt_path="tokenizer_best.pt"):
    train_loader = DataLoader(TRAIN_DATASET, batch_size=Cfg.BATCH_SIZE, shuffle=True, collate_fn=SignLanguageCollator())
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(eval_dataset, batch_size=Cfg.BATCH_SIZE, shuffle=False, collate_fn=SignLanguageCollator())

    tokenizer = CouplingTokenizer(
        d_part=Cfg.D // 3,
        num_hand_codes=Cfg.NUM_HAND_CODES,
        num_body_codes=Cfg.NUM_BODY_CODES,
    ).to(Cfg.DEVICE)

    opt = torch.optim.Adam(tokenizer.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_every, gamma=decay_factor)

    def _prep_batch(batch):
        body = batch["poses"]["upper_pose"].float()
        left = batch["poses"]["left_hand"].float()
        right = batch["poses"]["right_hand"].float()
        B, T = body.shape[0], body.shape[1]
        mask = batch["masks"].view(B * T, *batch["masks"].shape[2:])
        body_f = body.view(B * T, *body.shape[2:])[mask].to(Cfg.DEVICE)
        left_f = left.view(B * T, *left.shape[2:])[mask].to(Cfg.DEVICE)
        right_f = right.view(B * T, *right.shape[2:])[mask].to(Cfg.DEVICE)
        return body_f, left_f, right_f

    @torch.no_grad()
    def evaluate():
        tokenizer.eval()
        running = 0.0
        for batch in eval_loader:
            body_f, left_f, right_f = _prep_batch(batch)
            out = tokenizer(body_f, left_f, right_f)
            loss, _ = dvae_loss(out, left_f, right_f, body_f)
            running += loss.item()
        tokenizer.train()
        return running / len(eval_loader)

    best_eval_loss = float("inf")

    zs_l, zs_r, zs_b = [], [], []
    n = 0
    with torch.no_grad():
        for batch in train_loader:
            body_f, left_f, right_f = _prep_batch(batch)
            z_l, z_r, z_b = tokenizer.encoder(body_f, left_f, right_f)
            zs_l.append(z_l); zs_r.append(z_r); zs_b.append(z_b)
            n += z_l.shape[0]
            if n >= min_samples:
                break
    tokenizer.init_tokenizer_weights(torch.cat(zs_l), torch.cat(zs_r), torch.cat(zs_b))

    for epoch in range(epochs):
        print(codebook_usage(tokenizer, train_loader, _prep_batch))
        tokenizer.train()
        running = 0.0
        for batch in train_loader:
            body_f, left_f, right_f = _prep_batch(batch)

            out = tokenizer(body_f, left_f, right_f)
            loss, logs = dvae_loss(out, left_f, right_f, body_f)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        train_loss = running / len(train_loader)

        msg = f"[tokenizer] epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  {logs}"

        if eval_loader is not None:
            eval_loss = evaluate()
            msg += f"  eval_loss={eval_loss:.4f}"
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(tokenizer.state_dict(), best_ckpt_path)
                msg += "  (new best, saved)"

        torch.save(tokenizer.state_dict(), f"tokenizer_{epoch}.pt")
        print(msg)

    return tokenizer

def load_tokenizer(path:str):
    tokenizer = CouplingTokenizer(
        d_part=Cfg.D // 3,
        num_hand_codes=Cfg.NUM_HAND_CODES,
        num_body_codes=Cfg.NUM_BODY_CODES,
    )

    tokenizer.load_state_dict(torch.load(path, weights_only=True))
    return tokenizer.to(Cfg.DEVICE)


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


def pretrain(tokenizer: CouplingTokenizer, epochs=5, lr=1e-4, warmup_epochs=6, weight_decay=0.01):
    train_loader = DataLoader(TRAIN_DATASET, batch_size=Cfg.BATCH_SIZE, shuffle=True, collate_fn=SignLanguageCollator())

    tokenizer = tokenizer.to(Cfg.DEVICE)
    for p in tokenizer.parameters():
        p.requires_grad = False   # "tokenizer ... with all parameters frozen during pre-training"
    tokenizer.eval()

    model = BESTPretrainModel(
        D=Cfg.D, num_hand_codes=Cfg.NUM_HAND_CODES, num_body_codes=Cfg.NUM_BODY_CODES,
        n_heads=Cfg.N_HEADS, n_layers=Cfg.N_LAYERS, max_pe_len=Cfg.T
    ).to(Cfg.DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(step, steps_per_epoch=len(train_loader)):
        warmup_steps = warmup_epochs * steps_per_epoch
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0  # (paper: linear decay after warmup -- add a decay schedule of your choice)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model.train()
    step = 0
    for epoch in range(epochs):
        running = 0.0
        for batch in train_loader:
            body = batch["poses"]["upper_pose"].float().to(Cfg.DEVICE)
            left = batch["poses"]["left_hand"].float().to(Cfg.DEVICE)
            right = batch["poses"]["right_hand"].float().to(Cfg.DEVICE)

            k_l, k_r, k_b = get_pseudo_labels(tokenizer, body, left, right)

            out = model(body, left, right, alpha=Cfg.ALPHA)
            loss, logs = mum_loss(out, k_l, k_r, k_b)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            running += loss.item()
        print(f"[pretrain] epoch {epoch+1}/{epochs}  loss={running/len(train_loader):.4f}  {logs}")

        torch.save(model.backbone.state_dict(), f"backbone_pretrained_{epoch}.pt")
    return model.backbone


# ---------------------------------------------------------------------------
# Stage 3: downstream fine-tuning (Sec. 3.4)
# ---------------------------------------------------------------------------
def finetune(backbone: BESTBackbone, epochs=10, lr=1e-4, decay_every=10, decay_factor=0.1):
    train_loader = DataLoader(TRAIN_DATASET, batch_size=Cfg.BATCH_SIZE, shuffle=True, collate_fn=SignLanguageCollator())

    model = BESTClassifier(backbone, num_classes=Cfg.NUM_CLASSES).to(Cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_every, gamma=decay_factor)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running, correct, total = 0.0, 0, 0
        for body, left, right, label in train_loader:
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
        print(f"[finetune] epoch {epoch+1}/{epochs}  loss={running/len(train_loader):.4f}  "
              f"acc={correct/total:.4f}")

    torch.save(model.state_dict(), "best_finetuned.pt")
    return model


if __name__ == "__main__":
    print(f"Working with {Cfg.DEVICE}...")
    # Stage 1
    tokenizer = train_tokenizer(epochs=100, eval_dataset=VAL_DATASET)
    # tokenizer = load_tokenizer("tokenizer_best.pt")
    # Stage 2
    backbone = pretrain(tokenizer, epochs=100)
    # # Stage 3
    # finetune(backbone, epochs=2)
