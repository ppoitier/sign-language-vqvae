# BEST (Zhao et al., AAAI 2023) — PyTorch Reimplementation

A from-scratch reimplementation of *"BEST: BERT Pre-Training for Sign
Language Recognition with Coupling Tokenization"* (arXiv:2302.05075).
This is **not the official code** (which the authors haven't publicly
released) — it's a faithful reconstruction from the paper's equations and
architecture description, built to actually run end-to-end (verified with a
smoke test on tiny dimensions).

## Files → paper sections

| File | Paper section | What it implements |
|---|---|---|
| `gcn.py` | Sec 3.2, "Pose Embedding Layer" | GCN over 7 body / 21+21 hand joints → `D_part`-dim embedding per part per frame |
| `tokenizer.py` | Sec 3.1, Eq. (1)–(3) | d-VAE: encoder → coupled hand/body vector quantizers → decoder; `dvae_loss` implements Eq. (3) |
| `mum.py` | Sec 3.3 | MUM masking: pick `α·T` frames, mask hand/body independently per frame w.p. 0.5 each |
| `best_model.py` | Sec 3.2 (Transformer), Eq. (5)–(6), Sec 3.4 | Shared `BESTBackbone` (pose embed → mask → +pos-enc → `nn.TransformerEncoder`); `BESTPretrainModel` adds the two softmax heads (`W1` shared across hands, `W2` for body) and `mum_loss` implements Eq. (6); `BESTClassifier` swaps the decoder for an MLP head for fine-tuning |
| `train.py` | Sec 4.2 | 3-stage driver: `train_tokenizer()` → `pretrain()` → `finetune()`, with a `DummySignDataset` stand-in |

## Pipeline

```
Stage 1 (train_tokenizer):
    2D pose triplet --Enc--> z_l,z_r,z_b --argmin over codebook (Eq.1)--> k_l,k_r,k_b
                                          --Dec--> reconstructed pose, trained w/ L_d-VAE (Eq.3)
    tokenizer is then FROZEN.

Stage 2 (pretrain, self-supervised, MUM):
    pose sequence --embed--> mask α·T frames (hand/body indep., Sec 3.3)
                  --+pos-enc--> TransformerEncoder --> per-frame per-part features
                  --softmax heads--> predict frozen-tokenizer's k_l,k_r,k_b at masked
                    positions only, cross-entropy (Eq.5-6)

Stage 3 (finetune):
    pretrained backbone (no masking) --> mean-pool over T --> MLP --> gloss class
    (RGB late-fusion "(+R)" from the paper is a separate branch, not included here —
     you'd sum this model's softmax with an RGB model's, e.g. I3D, at inference)
```

## Known simplifications / where the paper under-specifies

- **GCN topology**: the paper cites Cai et al. (2019)'s GCN but doesn't give
  exact adjacency/architecture. `gcn.py` uses a standard 2-hop normalized
  adjacency spatial GCN with a plausible 49-joint (7 body + 21+21 hand)
  MMPose-style skeleton — swap `BODY_EDGES`/`HAND_EDGES` for your detector's
  actual joint layout.
- **Number of Transformer layers `N`**: not stated in the paper (only heads=8,
  D=1536, FFN=2048 are given). `Cfg.N_LAYERS` is left as a knob — tune to
  your compute budget.
- **MUM mask ratio `α`**: the paper ablates *whether* hand/body are masked
  (Table 7) but the exact `α` value used isn't stated in the excerpted text;
  `Cfg.ALPHA` is a placeholder — sweep it like Table 8's data-scale ablation
  suggests they tuned other hyperparameters.
- **Linear LR decay after warmup** (pre-training) is stubbed as constant
  post-warmup in `train.py`; swap `lr_lambda` for a proper linear decay if
  you need to match the paper exactly.
- **RGB fusion branch** ("Ours (+R)") is out of scope — it's just another
  video backbone (I3D/3D-R50/etc.) whose softmax gets summed with this
  model's softmax at inference.
- `DummySignDataset` generates random tensors purely so the training loop is
  runnable and testable; point a real `Dataset` (pose sequences extracted
  with MMPose per Sec. 4.2) at the same `(body, left_hand, right_hand, label)`
  contract to train for real.

## Quick start

```bash
pip install torch
python train.py   # runs tokenizer -> MUM pre-train -> fine-tune on dummy data
```

For real use, edit `Cfg` in `train.py` (D=1536, 32-frame clips, real codebook
sizes 1000/500, etc.) and replace `DummySignDataset` with your pose dataset.
