import torch
from torchmetrics import Metric


class CodebookPerplexity(Metric):
    def __init__(self, n_embeddings: int):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.add_state("counts", default=torch.zeros(n_embeddings), dist_reduce_fx="sum")

    def update(self, indices):
        self.counts += torch.bincount(indices.flatten(), minlength=self.n_embeddings)

    def compute(self):
        probs = self.counts / self.counts.sum()
        return torch.exp(-(probs * torch.log(probs + 1e-10)).sum())