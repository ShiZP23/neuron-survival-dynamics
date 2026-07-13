from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - import guard for environments without torchvision
    datasets = None
    transforms = None
    TORCHVISION_IMPORT_ERROR = exc
else:
    TORCHVISION_IMPORT_ERROR = None


IMAGE_DATASETS = ["mnist", "fashion_mnist", "cifar10"]

DATASET_SPECS: Dict[str, Dict[str, object]] = {
    "mnist": {
        "dataset_cls": "MNIST",
        "channels": 1,
        "image_size": (28, 28),
        "num_classes": 10,
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "fashion_mnist": {
        "dataset_cls": "FashionMNIST",
        "channels": 1,
        "image_size": (28, 28),
        "num_classes": 10,
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    "cifar10": {
        "dataset_cls": "CIFAR10",
        "channels": 3,
        "image_size": (32, 32),
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
}


@dataclass
class ImageDatasetBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    dataset_name: str
    channels: int
    image_size: Tuple[int, int]
    num_classes: int


def _require_torchvision() -> None:
    if datasets is None or transforms is None:
        raise ImportError(
            "torchvision is required for the init-only image study. "
            "Install it with `python3 -m pip install torchvision`."
        ) from TORCHVISION_IMPORT_ERROR


def _build_transform(dataset_name: str):
    spec = DATASET_SPECS[dataset_name]
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(spec["mean"], spec["std"]),
        ]
    )


def _dataset_class(dataset_name: str):
    _require_torchvision()
    class_name = DATASET_SPECS[dataset_name]["dataset_cls"]
    return getattr(datasets, class_name)


def _fixed_subset(dataset: Dataset, max_samples: Optional[int], subset_seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(subset_seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def load_image_dataset(
    dataset_name: str,
    data_root: str,
    val_size: int = 5000,
    split_seed: int = 0,
    download: bool = True,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    subset_seed: int = 0,
) -> ImageDatasetBundle:
    if dataset_name not in IMAGE_DATASETS:
        raise ValueError(f"dataset_name must be one of {IMAGE_DATASETS}")

    dataset_cls = _dataset_class(dataset_name)
    transform = _build_transform(dataset_name)
    root = str(Path(data_root).expanduser())

    full_train = dataset_cls(root=root, train=True, transform=transform, download=download)
    test_dataset = dataset_cls(root=root, train=False, transform=transform, download=download)

    if not 0 < val_size < len(full_train):
        raise ValueError(f"val_size must be between 1 and {len(full_train) - 1}")

    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(split_seed)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    train_dataset = _fixed_subset(train_dataset, max_train_samples, subset_seed + 101)
    val_dataset = _fixed_subset(val_dataset, max_val_samples, subset_seed + 202)
    test_dataset = _fixed_subset(test_dataset, max_test_samples, subset_seed + 303)

    spec = DATASET_SPECS[dataset_name]
    return ImageDatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        dataset_name=dataset_name,
        channels=spec["channels"],
        image_size=spec["image_size"],
        num_classes=spec["num_classes"],
    )


def create_data_loaders(
    bundle: ImageDatasetBundle,
    batch_size: int,
    train_order_seed: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_generator = torch.Generator().manual_seed(train_order_seed)
    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        bundle.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
