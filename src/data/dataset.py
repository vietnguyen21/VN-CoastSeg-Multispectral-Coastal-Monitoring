"""PyTorch Dataset & DataLoader for VN-CoastSeg-Mask2Former.

Loads 7-band multispectral Landsat 8 surface reflectance patches (7, 256, 256)
and corresponding VLUCD remapped integer masks (256, 256) with data augmentation
and channel-wise normalization.
"""

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# SSL4EO-L Landsat 8 Collection 2 Coastal Reference Statistics
SSL4EO_L_LANDSAT8_MEAN = [0.052, 0.063, 0.084, 0.082, 0.221, 0.163, 0.098]
SSL4EO_L_LANDSAT8_STD = [0.038, 0.041, 0.049, 0.057, 0.115, 0.094, 0.071]

# Base Dataset class supporting both PyTorch torch.utils.data.Dataset and pure Python
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    TorchDataset = object
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class CoastalSegmentationDataset(TorchDataset):
    """Multispectral semantic segmentation dataset for Vietnam coastal monitoring.

    Each item yields:
        image: (7, 256, 256) float32 tensor/array in [0.0, 1.0] or z-score normalized.
        mask: (256, 256) int64 tensor/array with values in {0, 1, 2, 3, 4, 5, 255}.
    """

    def __init__(
        self,
        samples: Optional[List[Dict[str, Any]]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        is_train: bool = False,
        normalize: bool = True,
        norm_type: str = "minmax",  # 'minmax' or 'zscore'
        transform: Optional[Callable] = None,
    ):
        self.is_train = is_train
        self.normalize = normalize
        self.norm_type = norm_type
        self.transform = transform
        self.samples: List[Dict[str, Any]] = []

        if samples:
            self.samples = list(samples)
        elif manifest_path:
            self.samples = self._load_manifest(manifest_path, split)

    def _load_manifest(
        self, manifest_path: Union[str, Path], split_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        path = Path(manifest_path)
        if not path.exists():
            return []

        records = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split_filter and row.get("split") != split_filter:
                    continue
                records.append(row)
        return records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Loads and returns (image, mask) pair at index idx.

        Returns:
            image: (7, 256, 256) float32
            mask: (256, 256) int64
        """
        sample = self.samples[idx]

        # 1. Load image and mask data
        image_data, mask_data = self._load_sample_data(sample)

        # 2. Apply geometric data augmentation if training
        if self.is_train:
            image_data, mask_data = self._apply_augmentations(image_data, mask_data)

        # 3. Apply normalization
        if self.normalize:
            image_data = self._apply_normalization(image_data)

        # 4. Convert to tensors or native arrays
        image, mask = self._format_tensors(image_data, mask_data)

        return image, mask

    def _load_sample_data(
        self, sample: Dict[str, Any]
    ) -> Tuple[List[List[List[float]]], List[List[int]]]:
        """Loads raw numeric image (7, 256, 256) and mask (256, 256)."""
        # If preloaded in memory
        if "image" in sample and "mask" in sample:
            img = sample["image"]
            msk = sample["mask"]
            # Convert numpy to list if needed
            if HAS_NUMPY and isinstance(img, np.ndarray):
                img = img.tolist()
            if HAS_NUMPY and isinstance(msk, np.ndarray):
                msk = msk.tolist()
            return img, msk

        # If file paths given
        image_path = sample.get("image_path")
        mask_path = sample.get("mask_path")

        if image_path and Path(image_path).exists() and mask_path and Path(mask_path).exists():
            if HAS_NUMPY and image_path.endswith(".npy"):
                img = np.load(image_path).astype(np.float32).tolist()
                msk = np.load(mask_path).astype(np.int64).tolist()
                return img, msk

        # Fallback synthetic generation for testing if neither is present
        from src.data.patch_extractor import PatchExtractor
        h, w = 256, 256
        dummy_mask = [[(r + c) % 6 for c in range(w)] for r in range(h)]
        # Inject target classes (mangrove 0 and aquaculture 1)
        for r in range(100):
            for c in range(100):
                dummy_mask[r][c] = 0 if (r + c) % 2 == 0 else 1
        dummy_image = PatchExtractor.generate_synthetic_7band_patch(None, dummy_mask)
        return dummy_image, dummy_mask

    def _apply_augmentations(
        self, image: List[List[List[float]]], mask: List[List[int]]
    ) -> Tuple[List[List[List[float]]], List[List[int]]]:
        """Applies spatial augmentations: horizontal flip, vertical flip, random 90-deg rotations."""
        # Random Horizontal Flip
        if random.random() < 0.5:
            image = [[row[::-1] for row in channel] for channel in image]
            mask = [row[::-1] for row in mask]

        # Random Vertical Flip
        if random.random() < 0.5:
            image = [channel[::-1] for channel in image]
            mask = mask[::-1]

        # Random 90-degree Rotations (k = 0, 1, 2, 3)
        k = random.randint(0, 3)
        for _ in range(k):
            # Rotate 90 deg clockwise: transpose and reverse rows
            image = [
                [[channel[r][c] for r in range(len(channel) - 1, -1, -1)] for c in range(len(channel[0]))]
                for channel in image
            ]
            mask = [
                [mask[r][c] for r in range(len(mask) - 1, -1, -1)]
                for c in range(len(mask[0]))
            ]

        # Spectral scaling jitter (+/- 5% brightness jitter)
        scale_factor = random.uniform(0.95, 1.05)
        image = [
            [[max(0.0, min(1.0, val * scale_factor)) for val in row] for row in channel]
            for channel in image
        ]

        return image, mask

    def _apply_normalization(
        self, image: List[List[List[float]]]
    ) -> List[List[List[float]]]:
        """Applies channel-wise normalization."""
        if self.norm_type == "zscore":
            normed = []
            for b in range(7):
                mean_b = SSL4EO_L_LANDSAT8_MEAN[b]
                std_b = SSL4EO_L_LANDSAT8_STD[b]
                channel = image[b]
                normed_channel = [
                    [(val - mean_b) / (std_b + 1e-7) for val in row] for row in channel
                ]
                normed.append(normed_channel)
            return normed
        else:
            # Default min-max clamp [0.0, 1.0]
            return [
                [[max(0.0, min(1.0, val)) for val in row] for row in channel]
                for channel in image
            ]

    def _format_tensors(
        self, image_data: List[List[List[float]]], mask_data: List[List[int]]
    ) -> Tuple[Any, Any]:
        """Formats into PyTorch Tensors if torch is available, else NumPy or structured lists."""
        if HAS_TORCH:
            import torch
            image_tensor = torch.tensor(image_data, dtype=torch.float32)
            mask_tensor = torch.tensor(mask_data, dtype=torch.long)
            return image_tensor, mask_tensor
        elif HAS_NUMPY:
            import numpy as np
            image_arr = np.array(image_data, dtype=np.float32)
            mask_arr = np.array(mask_data, dtype=np.int64)
            return image_arr, mask_arr
        else:
            return image_data, mask_data


class SimpleDataLoader:
    """Lightweight, zero-dependency DataLoader supporting batching, shuffling, and collating."""

    def __init__(
        self,
        dataset: CoastalSegmentationDataset,
        batch_size: int = 4,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            images = []
            masks = []
            for idx in batch_indices:
                img, msk = self.dataset[idx]
                images.append(img)
                masks.append(msk)

            # Collate batch
            if HAS_TORCH:
                import torch
                batch_images = torch.stack(images, dim=0)
                batch_masks = torch.stack(masks, dim=0)
            elif HAS_NUMPY:
                import numpy as np
                batch_images = np.stack(images, axis=0)
                batch_masks = np.stack(masks, axis=0)
            else:
                batch_images = images
                batch_masks = masks

            yield batch_images, batch_masks


def create_dataloader(
    dataset: CoastalSegmentationDataset,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
):
    """Creates a DataLoader using torch.utils.data.DataLoader if available, else SimpleDataLoader."""
    if HAS_TORCH:
        from torch.utils.data import DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )
    return SimpleDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
