import numpy as np
import torch
from sldl.targets import TargetEncoder
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class TokenTarget(TargetEncoder):
    """Looks up precomputed VQ-VAE token ids for each sample's window.

    Reads the `{window_id: {body_part: (T,) token ids}}` dict produced by
    `sl_vqvae.scripts.extract_tokens` (a trained VQ-VAE run once, offline,
    over the whole dataset) and returns the tokens for each sample's window,
    keyed the same way as `poses`, for use as BERT pretraining targets (see
    `sl_vqvae.trainer.bert_module.BERTTrainingModule`).

    Args:
        tokens_path: Path to the `.npy` file saved by `extract_tokens.py`.
        pad_value: Value used to pad ragged windows when collating a batch.
            Tokens are only ever read as `tokens[body_part][mask]`, and `mask`
            is restricted to valid, non-padded frames by the MUM masking (see
            `sl_vqvae.nn.bert.masking.sample_bert_mask`), so this value is
            never read by any computation. Defaults to -1 (an id no VQ-VAE
            codebook produces) purely so padding is visibly distinguishable
            from a real code -- e.g. code 0 -- when inspecting a raw batch.
    """

    def __init__(self, tokens_path: str, pad_value: int = -1):
        self.tokens: dict = np.load(tokens_path, allow_pickle=True).item()
        self.pad_value = pad_value

    def encode(self, sample: dict) -> dict[str, Tensor]:
        window_tokens = self.tokens[sample["window_id"]]
        return {body_part: torch.as_tensor(ids, dtype=torch.long) for body_part, ids in window_tokens.items()}

    def collate(self, batch_targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        body_parts = batch_targets[0].keys()
        return {
            body_part: pad_sequence(
                [target[body_part] for target in batch_targets],
                batch_first=True,
                padding_value=self.pad_value,
            )
            for body_part in body_parts
        }
