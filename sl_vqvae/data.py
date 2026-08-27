from torch.utils.data import DataLoader
from sldl import SignLanguageDataset, SignLanguageCollator

from sign_language_tools.common.transforms import ApplyToAll, Compose
from sign_language_tools.pose.transforms import DropCoordinates


def load_dataset(root: str, annotated: bool = True, split: str = "training"):
    if split == "training":
        shard_files = "shard_{000002..000007}.tar"
    elif split == "validation":
        shard_files = "shard_000001.tar"
    elif split == "testing":
        shard_files = "shard_000000.tar"
    else:
        raise ValueError(f"Unknown split: {split}")
    shard_folder = "annotated" if annotated else "unannotated"
    shards_url = f"{root}/shards/{shard_folder}/{shard_files}"

    return SignLanguageDataset(
        shards_url=shards_url,
        pose_transform=ApplyToAll(DropCoordinates("z")),
        use_windows=True,
        window_size=500,
        window_stride=400,
        max_empty_windows=0,
        show_loading_progress=True,
    )


def load_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=SignLanguageCollator(),
    )


def load_datasets_and_dataloaders(root: str, batch_size=16, n_workers=0):
    datasets = {
        x: load_dataset(root, annotated=True, split=x)
        for x in ["training", "validation", "testing"]
    }

    data_loaders = {
        x: load_dataloader(
            datasets[x], batch_size=batch_size, shuffle=(x == "training"), num_workers=n_workers
        )
        for x in ["training", "validation", "testing"]
    }

    return datasets, data_loaders
