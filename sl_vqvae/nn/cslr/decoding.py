from torch import Tensor


def ctc_greedy_decode(log_probs: Tensor, input_lengths: Tensor, blank_id: int) -> list[list[int]]:
    """Greedy (best-path) CTC decoding: take the arg-max class at every frame,
    then collapse repeated ids and drop the blank -- the standard approximation
    to full CTC beam search, cheap enough to run every validation epoch.

    Args:
        log_probs:     (T, N, C) log-probabilities, e.g. `CSLROutput.log_probs`.
        input_lengths: (N,) number of valid (non-pad) frames per sample.
        blank_id:      CTC blank class id (see `CSLRPoseTransformer.blank_id`).

    Returns:
        List of N variable-length lists of predicted gloss ids.
    """
    predicted_ids = log_probs.argmax(dim=-1).transpose(0, 1)  # (N, T)
    decoded = []
    for i, length in enumerate(input_lengths.tolist()):
        ids = predicted_ids[i, :length].tolist()
        collapsed = []
        previous = None
        for id_ in ids:
            if id_ != previous and id_ != blank_id:
                collapsed.append(id_)
            previous = id_
        decoded.append(collapsed)
    return decoded
