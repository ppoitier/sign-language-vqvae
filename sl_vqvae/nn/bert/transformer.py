import torch
from torch import nn, Tensor
from torchtune.modules import RotaryPositionalEmbeddings

from sl_vqvae.nn.bert.masking import apply_bert_mask, sample_bert_mask
from sl_vqvae.nn.bert.output import BERTOutput
from sl_vqvae.nn.encoders.gcn import PoseEmbeddingLayer
from sl_vqvae.nn.transformers.attention_patterns import and_masks, build_block_mask, get_attn_mask_mod, padding_mask_mod
from sl_vqvae.nn.transformers.layers import TransformerEncoderLayer
from sl_vqvae.nn.transformers.sinusoidal_pos_encoding import SinusoidalPositionalEncoding


class BERTPoseTransformer(nn.Module):
    """
    Masked pose pretraining following Zhao et al. 2023 ("BEST"), Sec. 3.2-3.3:
    GCN pose embedding -> mask -> +pos-enc -> Transformer -> predict the
    *precomputed* VQ-VAE token id at every masked position.

    Unlike WP2_replication's BEST, tokenization is not learned here: targets
    are token ids already produced by a trained sl_vqvae VQ-VAE quantizer
    (see `sl_vqvae.nn.vqvae`, `sl_vqvae.scripts.extract_tokens`), so this
    model only pretrains the masked-prediction Transformer, reusing this
    repo's own `PoseEmbeddingLayer` / `TransformerEncoderLayer` / attention
    pattern infra instead of duplicating them.

    Body parts that share a vocabulary group (e.g. 'left_hand'/'right_hand'
    sharing 'hand', matching `BodyPartsQuantizer`) already share one GCN
    encoder (see `PoseEmbeddingLayer`) and here also share one prediction
    head -- the "coupling" from the original paper.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        body_part_mapping: dict[str, str] | None = None,
        n_embeddings: dict[str, int] | None = None,
        n_pose_landmarks: int = 23,
        n_hand_landmarks: int = 21,
        gcn_hidden_dim: int = 128,
        gcn_layers: int = 2,
        max_length: int = 500,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pos_encoding: str = "rope",
        attn_mask_strategy: str | None = None,
        mask_ratio: float = 0.4,
    ):
        super().__init__()
        if body_part_mapping is None:
            body_part_mapping = {
                "upper_pose": "pose",
                "left_hand": "hand",
                "right_hand": "hand",
            }
        if n_embeddings is None:
            n_embeddings = {"pose": 500, "hand": 1000}

        self.body_parts = tuple(body_part_mapping.keys())
        self.body_part_mapping = dict(body_part_mapping)
        self.groups = tuple(dict.fromkeys(body_part_mapping.values()))
        self.n_embeddings = dict(n_embeddings)
        self.embedding_dim = embedding_dim
        self.mask_ratio = mask_ratio

        d_part = embedding_dim
        d_model = d_part * len(self.body_parts)
        self.d_model = d_model

        self.pose_embed = PoseEmbeddingLayer(
            embed_dim_part=d_part,
            c_hidden=gcn_hidden_dim,
            n_gcn_layers=gcn_layers,
            n_body_joints=n_pose_landmarks,
            n_hand_joints=n_hand_landmarks,
        )
        self.mask_token = nn.Parameter(torch.zeros(d_part))
        nn.init.normal_(self.mask_token, std=0.02)

        self.attn_mask_mod = get_attn_mask_mod(attn_mask_strategy)

        rope = None
        self.additive_pe = None
        if pos_encoding == "rope":
            head_dim = d_model // n_heads
            rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_length)
        elif pos_encoding == "sinusoidal":
            self.additive_pe = SinusoidalPositionalEncoding(d_model, max_seq_length=max_length)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    rope=rope,
                )
                for _ in range(n_layers)
            ]
        )

        self.heads = nn.ModuleDict({group: nn.Linear(d_part, n_embeddings[group]) for group in self.groups})

    def forward(self, poses: dict[str, Tensor], tokens: dict[str, Tensor], mask: Tensor) -> BERTOutput:
        """
        Args:
            poses:  body_part -> (N, T, L, C) raw pose sequence (model input).
            tokens: body_part -> (N, T) long token ids, precomputed by a
                    trained VQ-VAE quantizer (prediction targets).
            mask:   (N, T) bool -- True = valid frame, False = pad.
        """
        N, T = mask.shape
        d_part = self.embedding_dim

        embeds = self.pose_embed(poses)  # body_part -> (N, T, d_part)

        group_masks = sample_bert_mask(mask, self.mask_ratio, self.groups)
        body_part_masks = {bp: group_masks[self.body_part_mapping[bp]] for bp in self.body_parts}
        embeds = apply_bert_mask(embeds, body_part_masks, self.mask_token)

        x = torch.cat([embeds[bp] for bp in self.body_parts], dim=-1)  # (N, T, d_model)

        if self.additive_pe is not None:
            x = self.additive_pe(x.transpose(1, 2)).transpose(1, 2)

        mask_mod = padding_mask_mod(mask)
        if self.attn_mask_mod is not None:
            mask_mod = and_masks(mask_mod, self.attn_mask_mod)
        block_mask = build_block_mask(mask_mod, B=N, H=None, Q_LEN=T, KV_LEN=T, device=mask.device)

        for layer in self.layers:
            x = layer(x, block_mask=block_mask)

        logits = {}
        for i, body_part in enumerate(self.body_parts):
            f_bp = x[..., i * d_part : (i + 1) * d_part]
            logits[body_part] = self.heads[self.body_part_mapping[body_part]](f_bp)

        loss, accuracies = self.loss_function(logits, tokens, body_part_masks)

        return BERTOutput(logits=logits, masks=body_part_masks, loss=loss, accuracies=accuracies)

    def loss_function(
        self, logits: dict[str, Tensor], tokens: dict[str, Tensor], masks: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        losses = {}
        accuracies = {}
        for body_part in self.body_parts:
            m = masks[body_part]
            if not m.any():
                losses[body_part] = logits[body_part].sum() * 0.0
                accuracies[body_part] = torch.zeros((), device=logits[body_part].device)
                continue
            logits_sel = logits[body_part][m]
            targets_sel = tokens[body_part][m]
            losses[body_part] = nn.functional.cross_entropy(logits_sel, targets_sel)
            accuracies[body_part] = (logits_sel.argmax(-1) == targets_sel).float().mean()
        loss = torch.stack(list(losses.values())).sum()
        return loss, accuracies


if __name__ == "__main__":
    N, T, L_pose, L_hand, C = 8, 32, 23, 21, 2
    poses = {
        "upper_pose": torch.randn(N, T, L_pose, C),
        "left_hand": torch.randn(N, T, L_hand, C),
        "right_hand": torch.randn(N, T, L_hand, C),
    }
    tokens = {
        "upper_pose": torch.randint(0, 500, (N, T)),
        "left_hand": torch.randint(0, 1000, (N, T)),
        "right_hand": torch.randint(0, 1000, (N, T)),
    }
    mask = torch.ones(N, T).bool()

    model = BERTPoseTransformer(embedding_dim=32, n_heads=4, n_layers=2, max_length=T)
    out = model(poses, tokens, mask)

    print({k: v.shape for k, v in out.logits.items()})
    print({k: v.item() for k, v in out.accuracies.items()})
    print(out.loss)
