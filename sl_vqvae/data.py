from torch.utils.data import DataLoader
from sldl import SignLanguageDataset, SignLanguageCollator
from sldl.targets import TargetEncoder

from sign_language_tools.common.transforms import ApplyToAll, Compose, MapTransform, Identity
from sign_language_tools.pose.transforms import DropCoordinates, CenterOnLandmarks, HorizontalFlip


def load_dataset(
    root: str,
    annotated: bool = True,
    split: str = "training",
    targets: dict[str, TargetEncoder] | None = None,
):
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

    hand_transform = Compose([
        CenterOnLandmarks(landmark_idx=0),
    ])

    transforms = Compose([
        ApplyToAll(DropCoordinates("z")),
        MapTransform({
            'upper_pose': Identity(),
            'left_hand': Compose([
                HorizontalFlip(),
                CenterOnLandmarks(landmark_idx=0),
            ]),
            'right_hand': CenterOnLandmarks(landmark_idx=0),
        }),
    ])

    return SignLanguageDataset(
        shards_url=shards_url,
        pose_transform=transforms,
        use_windows=True,
        window_size=500,
        window_stride=400,
        max_empty_windows=0,
        show_loading_progress=True,
        targets=targets,
    )


def load_dataloader(dataset, batch_size, shuffle=True, num_workers=0, targets: dict[str, TargetEncoder] | None = None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=SignLanguageCollator(targets=targets),
    )


def load_datasets_and_dataloaders(
    root: str, batch_size=16, n_workers=0, annotated: bool = True, targets: dict[str, TargetEncoder] | None = None
):
    datasets = {
        x: load_dataset(root, annotated=annotated, split=x, targets=targets)
        for x in ["training", "validation", "testing"]
    }

    data_loaders = {
        x: load_dataloader(
            datasets[x],
            batch_size=batch_size,
            shuffle=(x == "training"),
            num_workers=n_workers,
            targets=targets,
        )
        for x in ["training", "validation", "testing"]
    }

    return datasets, data_loaders
