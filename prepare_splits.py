
"""
prepare_splits.py

Creates the project video-level split file:
    data/dataset/splits.csv

Purpose:
- Split FF++ c23 videos into train / val / test.
- Keep the split at the video-identity level to reduce leakage.
- Include only FF++ c40 videos that match the FF++ c23 test identities.
- Include all Celeb-DF videos as external test data.

This script should be run before preprocessing.py.
The preprocessing script can then read data/dataset/splits.csv and create
the processed manifests used for training and evaluation.
"""

from pathlib import Path
import csv
import random


# Configuration

DATA_ROOT = Path("data/dataset")
OUT_CSV = DATA_ROOT / "splits.csv"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


# Path helpers

def rel(path: Path) -> str:
    """Return a path relative to DATA_ROOT using forward slashes."""
    return path.relative_to(DATA_ROOT).as_posix()


def list_mp4(folder: Path) -> list[Path]:
    """Return sorted .mp4 files from a folder. If the folder is missing, return an empty list."""
    return sorted(folder.glob("*.mp4")) if folder.exists() else []


# Dataset loaders

def ffpp_real(compression: str) -> list[Path]:
    """Load FF++ original YouTube videos for a compression level such as c23 or c40."""
    return list_mp4(DATA_ROOT / f"FF_Data/original_sequences/youtube/{compression}/videos")


def ffpp_fake(compression: str) -> list[Path]:
    """Load FF++ manipulated videos for all four manipulation methods."""
    files: list[Path] = []

    for method in FFPP_METHODS:
        folder = DATA_ROOT / f"FF_Data/manipulated_sequences/{method}/{compression}/videos"
        files.extend(list_mp4(folder))

    return sorted(files)


def celebdf_videos() -> list[Path]:
    """Load all Celeb-DF videos. Celeb-DF is used as external test data only."""
    base = DATA_ROOT / "Celeb-DF-v2"
    folders = ["Celeb-real", "YouTube-real", "Celeb-synthesis"]

    files: list[Path] = []
    for folder in folders:
        files.extend(list_mp4(base / folder))

    return sorted(files)

# Split logic

def fake_pair(path: Path) -> tuple[str, str] | None:
    """
    Extract source and target IDs from an FF++ fake filename.

    Expected FF++ fake filename format:
        020_344.mp4

    The fake video is assigned to a split only if both IDs are in the same split.
    This avoids putting related identities across train/val/test.
    """
    parts = path.stem.split("_")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def split_real_ids(real_files: list[Path]) -> dict[str, set[str]]:
    """Split FF++ real video IDs into train / val / test using a fixed seed."""
    ids = sorted(video.stem for video in real_files)

    if not ids:
        raise RuntimeError(
            "No FF++ c23 real videos found. Check that data/dataset contains the FF++ folders."
        )

    random.Random(SEED).shuffle(ids)

    n = len(ids)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    return {
        "train": set(ids[:train_end]),
        "val": set(ids[train_end:val_end]),
        "test": set(ids[val_end:]),
    }


def add_ffpp_c23(rows: list[list[str]], real_files: list[Path], fake_files: list[Path], ids_by_split: dict[str, set[str]]) -> None:
    """
    Add FF++ c23 videos to rows.

    Real videos:
        assigned based on their own video ID.

    Fake videos:
        assigned only when both source and target IDs belong to the same split.
        Mixed-split fake videos are intentionally skipped.
    """
    for split_name, split_ids in ids_by_split.items():
        for video in real_files:
            if video.stem in split_ids:
                rows.append([rel(video), split_name])

        for video in fake_files:
            pair = fake_pair(video)
            if pair is None:
                continue

            src_id, tgt_id = pair
            if src_id in split_ids and tgt_id in split_ids:
                rows.append([rel(video), split_name])


def add_ffpp_c40_matched_test(rows: list[list[str]], real_files: list[Path], fake_files: list[Path], c23_test_ids: set[str]) -> None:
    """
    Add only matched FF++ c40 videos as test data.

    This keeps c40 evaluation aligned with the FF++ c23 test split:
    - c40 real video is included if its ID is in c23 test IDs.
    - c40 fake video is included if both source and target IDs are in c23 test IDs.
    """
    for video in real_files:
        if video.stem in c23_test_ids:
            rows.append([rel(video), "test"])

    for video in fake_files:
        pair = fake_pair(video)
        if pair is None:
            continue

        src_id, tgt_id = pair
        if src_id in c23_test_ids and tgt_id in c23_test_ids:
            rows.append([rel(video), "test"])


def add_celebdf_test(rows: list[list[str]], files: list[Path]) -> None:
    """Add all Celeb-DF videos as test data only."""
    for video in files:
        rows.append([rel(video), "test"])

# Main

def main() -> None:
    rows: list[list[str]] = []

    # Load source videos.
    c23_real = ffpp_real("c23")
    c23_fake = ffpp_fake("c23")
    c40_real = ffpp_real("c40")
    c40_fake = ffpp_fake("c40")
    celebdf = celebdf_videos()

    # Create the FF++ c23 train/val/test split.
    ids_by_split = split_real_ids(c23_real)

    # Add c23 training/validation/testing data.
    add_ffpp_c23(rows, c23_real, c23_fake, ids_by_split)

    # Add matched c40 test data only.
    add_ffpp_c40_matched_test(rows, c40_real, c40_fake, ids_by_split["test"])

    # Add Celeb-DF as external test data.
    add_celebdf_test(rows, celebdf)

    # Write splits.csv.
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "split"])
        writer.writerows(rows)

    print(f"Saved: {OUT_CSV}")
    print(f"Total videos written: {len(rows)}")
    print(
        "FF++ c23 real IDs train/val/test: "
        f"{len(ids_by_split['train'])}/{len(ids_by_split['val'])}/{len(ids_by_split['test'])}"
    )
    print("FF++ c40 test videos are matched to the FF++ c23 test IDs.")
    print("Celeb-DF videos are assigned to test only.")


if __name__ == "__main__":
    main()
