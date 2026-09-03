import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

class MaskGITTrainingModule(L.LightningModule):
    """LightningModule for training MotionMaskGIT on discrete motion tokens.

    Takes discrete VQ-VAE tokens from `batch["targets"]["tokens"]`, applies
    dynamic random masking by replacing tokens with the model's mask token ID,
    and trains the model via cross-entropy loss computed strictly over the
    masked token positions.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"]) #save hyperparameters at checkpoint but not the model's weight

        self.mask_token_id = 1500 #codebook size


    def _apply_masking(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Replaces a subset of input tokens with the mask token ID (1500) based on a dynamic random ratio."""
        batch_size, seq_len = tokens.shape

        r = torch.rand(1).item()
        dynamic_mask_ratio = math.cos(r * math.pi / 2)

        #generates 500x3 random nb b 0&1 and evaluates, if < masking ration: TRUE, FALSE else.
        mask = torch.rand(batch_size, seq_len, device=tokens.device) < dynamic_mask_ratio

        #fallback mechanism de Gemini si jamais notre masque pred par hasard QUE des faux et donc rien à masquer rip
        if not mask.any():
            random_idx = torch.randint(0, seq_len, (batch_size,), device=tokens.device)
            mask[torch.arange(batch_size, device=tokens.device), random_idx] = True

        #garder nos tokens de base pour servir de label plus tard, donc clone pour dupliquer et masquer ça sans toucher à l'original
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token_id #apply le masque de vrai faux with boolean indexing

        return masked_tokens, mask


    def _compute_loss(
            self,
            logits: Tensor,
            targets: Tensor,
            mask: Tensor,
        ) -> tuple[Tensor, Tensor]:
            """Calculates cross-entropy loss and accuracy restricted strictly to masked indices."""

            #prendre en compte seulement ceux qui devaient être reconstruit, pas ceux qui ont pas bougé
            masked_logits = logits[mask]
            masked_targets = targets[mask]
            loss = F.cross_entropy(masked_logits, masked_targets)
            return loss


    def forward_step(self, batch: dict, stage: str) -> Tensor:
        tokens = {k: v.long() for k, v in batch["targets"]["tokens"].items()}
        targets = tokens.clone()
        masked_tokens, mask = self._apply_masking(tokens)
        logits = self.model(masked_tokens)
        loss = self._compute_loss(logits, targets, mask)

        batch_size = tokens.size(0)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "training"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/accuracy",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self.forward_step(batch, "training")

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self.forward_step(batch, "validation")

    def test_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self.forward_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )