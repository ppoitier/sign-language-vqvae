import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings

from sl_vqvae.nn.cslr.output import CSLROutput
from sl_vqvae.nn.encoders.gcn import PoseEmbeddingLayer
from sl_vqvae.nn.transformers.attention_patterns import and_masks, build_block_mask, get_attn_mask_mod, padding_mask_mod
from sl_vqvae.nn.transformers.layers import TransformerEncoderLayer
from sl_vqvae.nn.transformers.sinusoidal_pos_encoding import SinusoidalPositionalEncoding


class CSLRPoseTransformer(nn.Module):
    """
    Continuous Sign Language Recognition: GCN pose embedding -> +pos-enc ->
    Transformer -> per-frame gloss logits, trained with CTC so the (much
    shorter) gloss sequence never needs frame-level alignment.

    The backbone (`pose_embed` / `additive_pe` / `layers`) is architecturally
    identical, submodule-for-submodule, to `sl_vqvae.nn.bert.BERTPoseTransformer`
    so a checkpoint from `BERTTrainingModule` can be loaded directly into it
    via `load_pretrained_bert` -- fine-tuning the masked-pretraining encoder
    for recognition instead of training one from scratch.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        body_parts: tuple[str, ...] = ("upper_pose", "left_hand", "right_hand"),
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
        vocab_size: int = 1000,
    ):
        super().__init__()
        self.body_parts = tuple(body_parts)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.blank_id = vocab_size

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

        self.classifier = nn.Linear(d_model, vocab_size + 1)  # +1 for the CTC blank

    def forward(self, poses: dict[str, Tensor], mask: Tensor, labels: Tensor, label_lengths: Tensor) -> CSLROutput:
        """
        Args:
            poses:         body_part -> (N, T, L, C) raw pose sequence.
            mask:          (N, T) bool -- True = valid frame, False = pad.
            labels:        (N, S) long gloss ids, padded (CTC never reads past label_lengths).
            label_lengths: (N,) long -- number of real gloss ids per sample.
        """
        N, T = mask.shape

        embeds = self.pose_embed(poses)  # body_part -> (N, T, d_part)
        x = torch.cat([embeds[bp] for bp in self.body_parts], dim=-1)  # (N, T, d_model)

        if self.additive_pe is not None:
            x = self.additive_pe(x.transpose(1, 2)).transpose(1, 2)

        mask_mod = padding_mask_mod(mask)
        if self.attn_mask_mod is not None:
            mask_mod = and_masks(mask_mod, self.attn_mask_mod)
        block_mask = build_block_mask(mask_mod, B=N, H=None, Q_LEN=T, KV_LEN=T, device=mask.device)

        for layer in self.layers:
            x = layer(x, block_mask=block_mask)

        logits = self.classifier(x)  # (N, T, vocab_size + 1)

        loss, log_probs, input_lengths = self.loss_function(logits, mask, labels, label_lengths)

        return CSLROutput(logits=logits, log_probs=log_probs, input_lengths=input_lengths, loss=loss)

    def loss_function(
        self, logits: Tensor, mask: Tensor, labels: Tensor, label_lengths: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        input_lengths = mask.sum(dim=1)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, N, vocab_size + 1), required by ctc_loss
        loss = F.ctc_loss(
            log_probs,
            labels,
            input_lengths,
            label_lengths,
            blank=self.blank_id,
            zero_infinity=True,
        )
        return loss, log_probs, input_lengths

    def load_pretrained_bert(self, checkpoint_path: str) -> nn.modules.module._IncompatibleKeys:
        """Initialize `pose_embed` / `additive_pe` / `layers` from a checkpoint
        saved by `BERTTrainingModule` (see `sl_vqvae.trainer.bert_module`),
        for CTC fine-tuning on top of the masked-pretraining backbone.

        The BERT mask token and per-group prediction heads have no CSLR
        counterpart, and the CTC classifier has no BERT counterpart -- both
        are naturally left out by filtering to the shared submodule names,
        so `strict=False` here is expected to report `classifier.*` missing
        and nothing unexpected.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = {key.replace("._orig_mod.", "."): value for key, value in checkpoint["state_dict"].items()}
        prefix = "model."
        backbone_names = {"pose_embed", "additive_pe", "layers"}
        backbone_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix) and key[len(prefix):].split(".", 1)[0] in backbone_names
        }
        return self.load_state_dict(backbone_state, strict=False)


if __name__ == "__main__":
    N, T, S, L_pose, L_hand, C = 8, 32, 6, 23, 21, 2
    poses = {
        "upper_pose": torch.randn(N, T, L_pose, C).cuda(),
        "left_hand": torch.randn(N, T, L_hand, C).cuda(),
        "right_hand": torch.randn(N, T, L_hand, C).cuda(),
    }
    mask = torch.ones(N, T).bool().cuda()
    label_lengths = torch.randint(1, S + 1, (N,))
    labels = torch.zeros(N, S, dtype=torch.long).cuda()
    for i, length in enumerate(label_lengths):
        labels[i, :length] = torch.randint(0, 100, (length.item(),))

    model = CSLRPoseTransformer(embedding_dim=32, n_heads=4, n_layers=2, max_length=T, vocab_size=100).cuda()
    out = model(poses, mask, labels, label_lengths)

    print(out.logits.shape)
    print(out.loss)
