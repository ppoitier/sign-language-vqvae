from dataclasses import dataclass

from torch import Tensor


@dataclass
class CSLROutput:
    logits: Tensor  # (N, T, vocab_size + 1), vocab_size = blank id
    log_probs: Tensor  # (T, N, vocab_size + 1) log-softmax, ready for CTCLoss / greedy decoding
    input_lengths: Tensor  # (N,) number of valid (non-pad) frames per sample
    loss: Tensor
