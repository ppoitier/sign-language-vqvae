import torch
from torchmetrics import Metric


class CodebookPerplexity(Metric):
    """Effective number of codes actually used by the VQ codebook.

    Computed as ``exp(entropy(usage distribution))``, where the usage
    distribution is how often each of the ``n_embeddings`` codes was picked
    across the epoch. It ranges from 1 (every frame quantized to the same
    single code -- "codebook collapse") to ``n_embeddings`` (all codes used
    equally often). It says nothing about reconstruction quality by itself,
    but a perplexity stuck near 1 is a reliable early signal that the
    codebook has collapsed and the model is not really learning to
    discretize its latent space.
    """

    higher_is_better = True

    def __init__(self, n_embeddings: int):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.add_state("counts", default=torch.zeros(n_embeddings), dist_reduce_fx="sum")

    def update(self, indices):
        self.counts += torch.bincount(indices.flatten(), minlength=self.n_embeddings)

    def compute(self):
        probs = self.counts / self.counts.sum()
        return torch.exp(-(probs * torch.log(probs + 1e-10)).sum())