from sldl import SignLanguageDataset


def load_dataset(root: str, annotated: bool = True, split: str = 'training'):
    if split == 'training':
        shard_files = "shard_{000002..000007}.tar"
    elif split == 'validation':
        shard_files = "shard_000001.tar"
    elif split == 'testing':
        shard_files = "shard_000000.tar"
    else:
        raise ValueError(f"Unknown split: {split}")
    shard_folder = 'annotated' if annotated else 'unannotated'
    shards_url = f"{root}/shards/{shard_folder}/{shard_files}"

    return SignLanguageDataset(
        shards_url=shards_url,
        use_windows=True,
        window_size=3500,
        window_stride=2800,
        max_empty_windows=0,
        show_loading_progress=True,
    )
