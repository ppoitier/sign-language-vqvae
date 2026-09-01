from torch.utils.data import ConcatDataset, DataLoader
from sldl import SignLanguageDataset, SignLanguageCollator
from sldl.targets import TargetEncoder

from sign_language_tools.common.transforms import ApplyToAll, Compose, MapTransform, Identity
from sign_language_tools.pose.transforms import DropCoordinates, CenterOnLandmarks, HorizontalFlip


def load_dataset(
    root: str,
    annotated: bool = True,
    split: str = "training",
    window_size: int = 500,
    window_stride: int = 400,
    max_empty_windows: int | None = 0,
    targets: dict[str, TargetEncoder] | None = None,
):
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

    def build(shard_folder: str, shard_files: str, use_annotations: bool):
        return SignLanguageDataset(
            shards_url=f"{root}/shards/{shard_folder}/{shard_files}",
            pose_transform=transforms,
            annotations=("both_hands",) if use_annotations else None,
            use_windows=True,
            window_size=window_size,
            window_stride=window_stride,
            # Empty-window filtering relies on annotations: shards with none
            # would have every window look "empty" and get dropped, so it's
            # only applied when annotations are actually requested.
            max_empty_windows=max_empty_windows if use_annotations else None,
            show_loading_progress=True,
            targets=targets,
        )

    if split == "training_with_unannotated":
        annotated_dataset = build("annotated", "shard_{000002..000007}.tar", False)
        unannotated_dataset = build("unannotated", "shard_{000000..000011}.tar", False)
        return ConcatDataset([annotated_dataset, unannotated_dataset])

    if split == "training":
        shard_files = "shard_{000002..000007}.tar"
    elif split == "validation":
        shard_files = "shard_000001.tar"
    elif split == "testing":
        shard_files = "shard_000000.tar"
    else:
        raise ValueError(f"Unknown split: {split}")
    shard_folder = "annotated" if annotated else "unannotated"
    return build(shard_folder, shard_files, annotated)


def load_dataloader(dataset, batch_size, shuffle=True, num_workers=0, targets: dict[str, TargetEncoder] | None = None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=SignLanguageCollator(targets=targets),
    )


def load_datasets_and_dataloaders(
    root: str,
    batch_size=16,
    n_workers=0,
    annotated: bool = True,
    train_split: str = "training",
    window_size: int = 500,
    window_stride: int = 400,
    max_empty_windows: int | None = 0,
    targets: dict[str, TargetEncoder] | None = None,
):
    split_for = {"training": train_split, "validation": "validation", "testing": "testing"}
    datasets = {
        x: load_dataset(
            root,
            annotated=annotated,
            split=split_for[x],
            window_size=window_size,
            window_stride=window_stride,
            max_empty_windows=max_empty_windows,
            targets=targets,
        )
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
