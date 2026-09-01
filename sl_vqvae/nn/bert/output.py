from dataclasses import dataclass

from torch import Tensor


@dataclass
class BERTOutput:
    logits: dict[str, Tensor]  # body_part -> (N, T, n_embeddings[group])
    masks: dict[str, Tensor]  # body_part -> (N, T) bool, positions replaced by the mask token
    loss: Tensor
    accuracies: dict[str, Tensor]  # body_part -> scalar accuracy over masked positions
