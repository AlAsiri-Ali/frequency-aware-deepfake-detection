from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


# Normalize spatial inputs using ImageNet statistics to match the pretrained backbone.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass
# Store each manifest row as a structured sample record.
class SampleRecord:
    image_path: str
    source_video_path: str
    label: int
    video_id: str
    dataset: str
    compression: str
    split: str
    width: int
    height: int
    frame_idx: int


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    # Convert a PIL image to a tensor without external transform libraries.
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    # Convert a tensor back to PIL for augmentations that are easier in image form.
    tensor = tensor.detach().clamp(0.0, 1.0).cpu()
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def resize_tensor(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    # Resize tensors with PyTorch to keep the pipeline fully tensor-based.
    return F.interpolate(
        tensor.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    ).squeeze(0)


def read_manifest(manifest_path: str | Path) -> List[SampleRecord]:
    # Read sample paths and labels from the manifest.
    manifest_path = Path(manifest_path)
    records: List[SampleRecord] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                SampleRecord(
                    image_path=row["image_path"],
                    source_video_path=row.get("source_video_path", ""),
                    label=int(row["label"]),
                    video_id=row.get("video_id", "unknown"),
                    dataset=row.get("dataset", "unknown"),
                    compression=row.get("compression", "raw"),
                    split=row.get("split", "unknown"),
                    width=int(row.get("width", 0)),
                    height=int(row.get("height", 0)),
                    frame_idx=int(row.get("frame_idx", -1)),
                )
            )
    if not records:
        raise ValueError(f"No samples found in manifest: {manifest_path}")
    return records


class ComposeTensorTransforms:
    # Apply custom tensor transforms sequentially.
    def __init__(self, transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor
        for transform in self.transforms:
            out = transform(out)
        return out


class RandomHorizontalFlipTensor:
    # Apply random horizontal flipping.
    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return torch.flip(tensor, dims=[2])
        return tensor


class RandomBrightnessContrastTensor:
    # Apply random brightness and contrast jitter.
    def __init__(self, brightness: float = 0.10, contrast: float = 0.10) -> None:
        self.brightness = max(0.0, float(brightness))
        self.contrast = max(0.0, float(contrast))

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor
        if self.brightness > 0:
            factor = 1.0 + torch.empty(1).uniform_(-self.brightness, self.brightness).item()
            out = out * factor
        if self.contrast > 0:
            factor = 1.0 + torch.empty(1).uniform_(-self.contrast, self.contrast).item()
            mean = out.mean(dim=(1, 2), keepdim=True)
            out = (out - mean) * factor + mean
        return out.clamp(0.0, 1.0)


class RandomJPEGCompressionTensor:
    # Simulate JPEG compression to expose the model to compression artifacts.
    def __init__(self, p: float = 0.5, quality_min: int = 35, quality_max: int = 95) -> None:
        self.p = float(p)
        self.quality_min = int(quality_min)
        self.quality_max = int(quality_max)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return tensor

        quality = int(torch.randint(self.quality_min, self.quality_max + 1, (1,)).item())
        image = tensor_to_pil(tensor)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        return pil_to_tensor(compressed)


class RandomDownUpResizeTensor:
    # Simulate resolution loss by downsampling and upsampling the image.
    def __init__(self, p: float = 0.3, scale_min: float = 0.5, scale_max: float = 0.9) -> None:
        self.p = float(p)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return tensor

        _, h, w = tensor.shape
        scale = float(torch.empty(1).uniform_(self.scale_min, self.scale_max).item())
        down_h = max(32, int(round(h * scale)))
        down_w = max(32, int(round(w * scale)))

        x = tensor.unsqueeze(0)
        x = F.interpolate(x, size=(down_h, down_w), mode="bilinear", align_corners=False, antialias=False)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=False)
        return x.squeeze(0).clamp(0.0, 1.0)


class RandomGaussianBlurTensor:
    # Add mild blur so the spatial branch is less sensitive to overly sharp crops.
    def __init__(self, p: float = 0.2, radius_min: float = 0.2, radius_max: float = 1.2) -> None:
        self.p = float(p)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return tensor

        radius = float(torch.empty(1).uniform_(self.radius_min, self.radius_max).item())
        image = tensor_to_pil(tensor)
        blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return pil_to_tensor(blurred)


def build_train_transforms(
    flip_p: float = 0.5,
    brightness: float = 0.10,
    contrast: float = 0.10,
    jpeg_p: float = 0.5,
    jpeg_quality_min: int = 35,
    jpeg_quality_max: int = 95,
    resize_p: float = 0.3,
    resize_scale_min: float = 0.5,
    resize_scale_max: float = 0.9,
    blur_p: float = 0.2,
    blur_radius_min: float = 0.2,
    blur_radius_max: float = 1.2,
) -> ComposeTensorTransforms:
    # Apply augmentations only to the spatial branch.
    # Keep the raw branch untouched because the frequency module uses the original crop values.
    return ComposeTensorTransforms(
        [
            RandomJPEGCompressionTensor(
                p=jpeg_p,
                quality_min=jpeg_quality_min,
                quality_max=jpeg_quality_max,
            ),
            RandomGaussianBlurTensor(
                p=blur_p,
                radius_min=blur_radius_min,
                radius_max=blur_radius_max,
            ),
            RandomBrightnessContrastTensor(
                brightness=brightness,
                contrast=contrast,
            ),
            RandomHorizontalFlipTensor(p=flip_p),
            RandomDownUpResizeTensor(
                p=resize_p,
                scale_min=resize_scale_min,
                scale_max=resize_scale_max,
            ),
        ]
    )


class DeepfakeDataset(Dataset):
    """
    Unified dataset for all supported model types:
    - spatial: uses only "spatial"
    - dual / dual_nomask: uses "spatial" + "raw" + "raw_sizes"
    """

    def __init__(
        self,
        manifest_path: str | Path,
        spatial_size: int = 299,
        normalize_spatial: bool = True,
        spatial_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        # Read the manifest once during initialization for efficient indexed access.
        self.records = read_manifest(manifest_path)
        self.spatial_size = int(spatial_size)
        self.normalize_spatial = bool(normalize_spatial)
        self.spatial_transform = spatial_transform

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, image_path: str) -> Image.Image:
        # Convert all images to RGB to keep the channel layout consistent.
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, index: int) -> Dict[str, object]:
        # Return both the normalized spatial view and the raw crop.
        record = self.records[index]
        image = self._load_rgb(record.image_path)
        raw_tensor = pil_to_tensor(image)

        # Start from the raw crop, then apply optional spatial augmentation.
        spatial_tensor = raw_tensor.clone()
        if self.spatial_transform is not None:
            spatial_tensor = self.spatial_transform(spatial_tensor)

        spatial_tensor = resize_tensor(spatial_tensor, (self.spatial_size, self.spatial_size)).clamp(0.0, 1.0)
        # Normalize only the spatial input because the frequency branch uses raw values.
        if self.normalize_spatial:
            spatial_tensor = (spatial_tensor - IMAGENET_MEAN) / IMAGENET_STD

        _, raw_h, raw_w = raw_tensor.shape
        return {
            "spatial": spatial_tensor,
            "raw": raw_tensor,
            "raw_size": (int(raw_h), int(raw_w)),
            "label": int(record.label),
            "image_path": record.image_path,
            "source_video_path": record.source_video_path,
            "video_id": record.video_id,
            "dataset": record.dataset,
            "compression": record.compression,
            "split": record.split,
            "frame_idx": int(record.frame_idx),
        }


def collate_with_padding(batch: List[Dict[str, object]]) -> Dict[str, object]:
    # Pad raw crops to the largest shape in the batch.
    spatial = torch.stack([item["spatial"] for item in batch], dim=0)
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)

    raw_sizes: List[Tuple[int, int]] = [item["raw_size"] for item in batch]
    max_h = max(h for h, _ in raw_sizes)
    max_w = max(w for _, w in raw_sizes)
    raw_batch = torch.zeros((len(batch), 3, max_h, max_w), dtype=torch.float32)

    # Copy each raw crop into the top-left region of the padded tensor.
    for idx, item in enumerate(batch):
        raw = item["raw"]
        h, w = item["raw_size"]
        raw_batch[idx, :, :h, :w] = raw

    return {
        "spatial": spatial,
        "raw": raw_batch,
        "raw_sizes": raw_sizes,
        "label": labels,
        "image_path": [item["image_path"] for item in batch],
        "source_video_path": [item["source_video_path"] for item in batch],
        "video_id": [item["video_id"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "compression": [item["compression"] for item in batch],
        "split": [item["split"] for item in batch],
        "frame_idx": [item["frame_idx"] for item in batch],
    }

SpatialDeepfakeDataset = DeepfakeDataset
