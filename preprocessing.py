from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


LOGGER = logging.getLogger("preprocessing")

# Keep preprocessing logs consistent and easier to follow during long runs.


@dataclass
# Store the minimum information needed to process one video.
class VideoItem:
    video_path: Path
    label: int
    dataset: str
    compression: str
    split: str


@dataclass
# Group the main extraction settings in one place.
class ExtractionConfig:
    padding: int = 40
    min_face_size: int = 50
    image_ext: str = ".png"
    max_real_faces: int = 20
    max_fake_faces: int = 5
    min_conf: float = 0.80


class FaceExtractor:
    # Keep face detection separate from the rest of the preprocessing pipeline.
    def __init__(self, device: str = "cpu") -> None:
        # Use MTCNN to detect faces before cropping each frame.
        try:
            from facenet_pytorch import MTCNN
        except ImportError as exc:
            raise ImportError(
                "facenet-pytorch is required for preprocessing. Install it via `pip install facenet-pytorch`."
            ) from exc

        self.detector = MTCNN(keep_all=True, device=device)

    def detect_biggest_face(
        self,
        image_rgb: np.ndarray,
        min_conf: float = 0.80,
    ) -> Optional[Tuple[int, int, int, int]]:
        # Keep the largest confident face because each crop is expected to focus on the main subject.
        pil_image = Image.fromarray(image_rgb)
        boxes, probs = self.detector.detect(pil_image)

        if boxes is None or len(boxes) == 0 or probs is None:
            return None

        valid_indices = []
        for i, p in enumerate(probs):
            if p is not None and p >= min_conf:
                valid_indices.append(i)

        if len(valid_indices) == 0:
            return None

        filtered_boxes = boxes[valid_indices]
        areas = (filtered_boxes[:, 2] - filtered_boxes[:, 0]) * (
            filtered_boxes[:, 3] - filtered_boxes[:, 1]
        )
        best_idx = int(np.argmax(areas))
        x1, y1, x2, y2 = filtered_boxes[best_idx]

        return int(x1), int(y1), int(x2), int(y2)


def setup_logging() -> None:
    # Use a simple logging format for command-line preprocessing runs.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_labels_from_metadata(metadata_csv: Path, input_root: Path) -> dict:
    # Read external labels when provided instead of inferring them from folder names.
    mapping = {}
    with metadata_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"video_path", "label"}
        if reader.fieldnames is None or not required_columns.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Metadata CSV must contain columns: {sorted(required_columns)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            # Resolve relative paths against the input root so the manifest remains portable.
            raw_path = row["video_path"].strip()
            path_obj = Path(raw_path)
            if not path_obj.is_absolute():
                path_obj = (input_root / path_obj).resolve()

            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Invalid label {label} for video {raw_path}. Expected 0 or 1.")

            mapping[path_obj.as_posix()] = label

    return mapping


def load_split_from_csv(split_csv: Path, target_split: str, input_root: Path) -> set:
    # Build the set of videos that belong to the requested split only.
    allowed = set()
    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"video_path", "split"}
        if reader.fieldnames is None or not required_columns.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Split CSV must contain columns: {sorted(required_columns)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            split_name = row["split"].strip().lower()
            if split_name != target_split.lower():
                continue

            raw_path = row["video_path"].strip()
            path_obj = Path(raw_path)
            if not path_obj.is_absolute():
                path_obj = (input_root / path_obj).resolve()

            allowed.add(path_obj.as_posix())

    return allowed


def infer_label_from_path(path: Path) -> int:
    # Infer labels from common dataset naming patterns when metadata is not available.
    lower_parts = {part.lower() for part in path.parts}
    name = path.as_posix().lower()

    # Infer real samples from common directory names.
    if (
        "youtube" in lower_parts
        or "youtube-real" in lower_parts
        or "celeb-real" in lower_parts
        or "real" in lower_parts
        or "real" in name
    ):
        return 0

    # Infer fake samples from common manipulation directory names.
    if (
        "fake" in lower_parts
        or "deepfakes" in lower_parts
        or "faceswap" in lower_parts
        or "face2face" in lower_parts
        or "neuraltextures" in lower_parts
        or "celeb-synthesis" in lower_parts
        or "synthesis" in name
        or "manipulated" in lower_parts
    ):
        return 1

    raise ValueError(
        f"Could not infer label from path: {path}. "
        f"Please provide --metadata-csv or organize folders so real/fake can be inferred."
    )


def discover_videos(
    input_root: Path,
    dataset: str,
    compression: str,
    split: str,
    label_map: Optional[dict] = None,
    allowed_videos: Optional[set] = None,
) -> List[VideoItem]:
    # Collect supported video files and then filter them with the selected dataset rules.
    video_paths = sorted(
        list(input_root.rglob("*.mp4"))
        + list(input_root.rglob("*.avi"))
        + list(input_root.rglob("*.mov"))
    )

    if not video_paths:
        raise FileNotFoundError(f"No videos found under: {input_root}")

    items: List[VideoItem] = []

    allowed_manipulations = {"deepfakes", "face2face", "faceswap", "neuraltextures"}
    allowed_originals = {"youtube"}

    skipped_not_in_split = 0
    skipped_not_matching_ffpp_structure = 0
    skipped_not_in_metadata = 0

    for path in video_paths:
        path = path.resolve()
        path_parts = {part.lower() for part in path.parts}

        # FF++ has a known folder structure, so filter by both source type and compression level.
        if dataset == "ffpp":
            is_allowed_original = any(orig in path_parts for orig in allowed_originals)
            is_allowed_fake = any(fake in path_parts for fake in allowed_manipulations)

            if not (is_allowed_original or is_allowed_fake):
                skipped_not_matching_ffpp_structure += 1
                continue

            if compression.lower() not in path_parts:
                skipped_not_matching_ffpp_structure += 1
                continue

        if allowed_videos is not None and path.as_posix() not in allowed_videos:
            skipped_not_in_split += 1
            continue

        # Prefer metadata labels when available because they are more reliable than path-based inference.
        if label_map is not None:
            key = path.as_posix()
            if key not in label_map:
                skipped_not_in_metadata += 1
                LOGGER.warning("Skipping video not found in metadata: %s", path)
                continue
            label = label_map[key]
        else:
            label = infer_label_from_path(path)

        items.append(
            VideoItem(
                video_path=path,
                label=label,
                dataset=dataset,
                compression=compression,
                split=split,
            )
        )

    LOGGER.info("Skipped videos not in requested split: %d", skipped_not_in_split)
    LOGGER.info("Skipped videos not matching FF++ filters: %d", skipped_not_matching_ffpp_structure)
    LOGGER.info("Skipped videos missing from metadata: %d", skipped_not_in_metadata)

    return items


def uniform_frame_indices(frame_count: int, max_samples: int) -> List[int]:
    # Sample frames uniformly to cover the video without evaluating every frame.
    if frame_count <= 0:
        return []

    if frame_count <= max_samples:
        return list(range(frame_count))

    indices = np.linspace(0, frame_count - 1, num=max_samples, dtype=np.int32)
    return sorted(set(int(x) for x in indices.tolist()))


def crop_with_padding(
    image_rgb: np.ndarray,
    box: Tuple[int, int, int, int],
    padding: int,
) -> np.ndarray:
    # Convert the detected face box into a square crop for more consistent context.
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    side = max(bw, bh) + 2 * padding

    new_x1 = int(cx - side / 2)
    new_y1 = int(cy - side / 2)
    new_x2 = int(cx + side / 2)
    new_y2 = int(cy + side / 2)

    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(w, new_x2)
    new_y2 = min(h, new_y2)

    crop = image_rgb[new_y1:new_y2, new_x1:new_x2]
    return crop


def sanitize_video_id(video_path: Path, dataset: str) -> str:
    # Combine dataset information, source type, and a short hash to avoid ID collisions.
    base_id = video_path.stem.replace(" ", "_")
    parent_parts = {p.lower() for p in video_path.parts}

    source = "unknown"
    for candidate in [
        "youtube",
        "deepfakes",
        "face2face",
        "faceswap",
        "neuraltextures",
        "celeb-real",
        "youtube-real",
        "celeb-synthesis",
    ]:
        if any(candidate in p for p in parent_parts):
            source = candidate.replace("-", "_")
            break

    path_hash = hashlib.md5(video_path.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{dataset}_{source}_{base_id}_{path_hash}"


def save_crop(crop_rgb: np.ndarray, output_dir: Path, image_name: str) -> Path:
    # Create output folders on demand so each video can save its crops independently.
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / image_name
    Image.fromarray(crop_rgb).save(out_path)
    return out_path


def extract_faces_from_video(
    item: VideoItem,
    extractor: FaceExtractor,
    output_root: Path,
    config: ExtractionConfig,
) -> List[dict]:
    # Main extraction loop for one video.
    cap = cv2.VideoCapture(str(item.video_path))
    if not cap.isOpened():
        LOGGER.warning("Could not open video: %s", item.video_path)
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Use a smaller cap for fake videos to keep the dataset more balanced during preprocessing.
    target_faces = config.max_fake_faces if item.label == 1 else config.max_real_faces
    candidate_indices = uniform_frame_indices(frame_count, max(target_faces * 4, target_faces))
    candidate_set = set(candidate_indices)

    saved_rows: List[dict] = []
    saved_count = 0
    video_id = sanitize_video_id(item.video_path, item.dataset)
    output_dir = (
        output_root
        / item.dataset
        / item.compression
        / item.split
        / ("fake" if item.label == 1 else "real")
        / video_id
    )

    current_frame_idx = 0
    skipped_no_face = 0
    skipped_small_crop = 0
    skipped_empty_crop = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break

        # Check only sampled frame indices for faces to reduce unnecessary computation.
        if current_frame_idx not in candidate_set:
            current_frame_idx += 1
            continue

        if saved_count >= target_faces:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        box = extractor.detect_biggest_face(frame_rgb, min_conf=config.min_conf)

        # Skip frames without a confident face instead of forcing a low-quality crop.
        if box is None:
            skipped_no_face += 1
            current_frame_idx += 1
            continue

        crop_rgb = crop_with_padding(frame_rgb, box, padding=config.padding)
        if crop_rgb.size == 0:
            skipped_empty_crop += 1
            current_frame_idx += 1
            continue

        crop_h, crop_w = crop_rgb.shape[:2]
        # Ignore very small crops because they usually contain too little facial detail.
        if crop_h < config.min_face_size or crop_w < config.min_face_size:
            skipped_small_crop += 1
            current_frame_idx += 1
            continue

        image_name = f"{video_id}_frame{current_frame_idx:05d}{config.image_ext}"
        crop_path = save_crop(crop_rgb, output_dir, image_name)

        saved_rows.append(
            {
                "image_path": str(crop_path.resolve()),
                "source_video_path": str(item.video_path.resolve()),
                "label": item.label,
                "video_id": video_id,
                "dataset": item.dataset,
                "compression": item.compression,
                "split": item.split,
                "width": crop_w,
                "height": crop_h,
                "frame_idx": current_frame_idx,
            }
        )

        saved_count += 1
        current_frame_idx += 1

    cap.release()

    LOGGER.info(
        "Processed video=%s | saved=%d | skipped_no_face=%d | skipped_empty=%d | skipped_small=%d",
        video_id,
        len(saved_rows),
        skipped_no_face,
        skipped_empty_crop,
        skipped_small_crop,
    )

    return saved_rows


def write_manifest(rows: Sequence[dict], manifest_path: Path) -> None:
    # Link every saved crop back to its source video and metadata in the manifest.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "source_video_path",
        "label",
        "video_id",
        "dataset",
        "compression",
        "split",
        "width",
        "height",
        "frame_idx",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    # Expose the main preprocessing options through command-line arguments.
    parser = argparse.ArgumentParser(description="Extract raw face crops for deepfake detection.")
    parser.add_argument("--dataset", type=str, required=True, choices=["ffpp", "celebdf", "custom"])
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--compression", type=str, default="raw")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--padding", type=int, default=40)
    parser.add_argument("--min-face-size", type=int, default=50)
    parser.add_argument("--min-conf", type=float, default=0.80)
    parser.add_argument("--max-real-faces", type=int, default=20)
    parser.add_argument("--max-fake-faces", type=int, default=5)
    parser.add_argument("--manifest-name", type=str, default="manifest.csv")
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--split-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    # Prepare the configuration, run extraction, and write the final manifest.
    setup_logging()
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    label_map = None
    if args.metadata_csv is not None:
        label_map = load_labels_from_metadata(Path(args.metadata_csv), input_root)
        LOGGER.info("Loaded %d labels from metadata CSV", len(label_map))

    allowed_videos = None
    if args.split_csv is not None:
        allowed_videos = load_split_from_csv(Path(args.split_csv), args.split, input_root)
        LOGGER.info("Loaded %d allowed videos for split '%s'", len(allowed_videos), args.split)

    # Celeb-DF usually uses a slightly larger number of real samples per video.
    if args.dataset == "celebdf":
        max_real = 30
    else:
        max_real = args.max_real_faces

    config = ExtractionConfig(
        padding=args.padding,
        min_face_size=args.min_face_size,
        max_real_faces=max_real,
        max_fake_faces=args.max_fake_faces,
        min_conf=args.min_conf,
    )

    extractor = FaceExtractor(device=args.device)

    items = discover_videos(
        input_root=input_root,
        dataset=args.dataset,
        compression=args.compression,
        split=args.split,
        label_map=label_map,
        allowed_videos=allowed_videos,
    )

    LOGGER.info("Discovered %d videos under %s", len(items), input_root)

    all_rows: List[dict] = []
    videos_with_crops = 0
    videos_with_no_crops = 0

    # Process videos independently so one failure does not stop the full run.
    for item in tqdm(items, desc="Extracting faces"):
        rows = extract_faces_from_video(
            item=item,
            extractor=extractor,
            output_root=output_root,
            config=config,
        )
        all_rows.extend(rows)

        if len(rows) == 0:
            videos_with_no_crops += 1
            LOGGER.warning("No valid crops extracted from video: %s", item.video_path)
        else:
            videos_with_crops += 1

    manifest_path = output_root / args.dataset / args.compression / args.split / args.manifest_name
    write_manifest(all_rows, manifest_path)

    LOGGER.info("Saved %d face crops", len(all_rows))
    LOGGER.info("Manifest written to %s", manifest_path)
    LOGGER.info("Videos with crops: %d", videos_with_crops)
    LOGGER.info("Videos with no crops: %d", videos_with_no_crops)


if __name__ == "__main__":
    main()