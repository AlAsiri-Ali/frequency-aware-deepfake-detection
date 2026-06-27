from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DeepfakeDataset, collate_with_padding
from metrics import classification_metrics
from model import build_model


def forward_batch(model_type: str, model: torch.nn.Module, batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    # Keep the evaluation loop clean by hiding model-specific input handling.
    spatial = batch["spatial"].to(device, non_blocking=True)
    if model_type == "spatial":
        return model(spatial=spatial)

    raw = batch["raw"].to(device, non_blocking=True)
    raw_sizes = batch["raw_sizes"]
    return model(spatial=spatial, raw=raw, raw_sizes=raw_sizes)


@torch.no_grad()
def evaluate_model(
    model_type: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame, pd.DataFrame]:
    # Disable gradients because this stage only measures checkpoint performance.
    model.eval()
    rows: List[dict] = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        labels = batch["label"].to(device, non_blocking=True)
        logits = forward_batch(model_type=model_type, model=model, batch=batch, device=device)

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # Store frame-level outputs first, then aggregate them at the video level.
        for idx in range(len(labels_np)):
            frame_idx = batch["frame_idx"][idx]
            rows.append(
                {
                    "image_path": batch["image_path"][idx],
                    "source_video_path": batch["source_video_path"][idx],
                    "video_id": batch["video_id"][idx],
                    "dataset": batch["dataset"][idx],
                    "compression": batch["compression"][idx],
                    "split": batch["split"][idx],
                    "frame_idx": int(frame_idx),
                    "label": int(labels_np[idx]),
                    "prob_fake": float(probs[idx]),
                    "pred": int(probs[idx] >= threshold),
                }
            )

    frame_df = pd.DataFrame(rows)
    frame_metrics = classification_metrics(
        frame_df["label"].to_numpy(),
        frame_df["prob_fake"].to_numpy(),
        threshold=threshold,
    )

    # Compute video scores by averaging frame probabilities within each video.
    video_df = (
        frame_df.groupby(["video_id", "dataset", "compression"], as_index=False)
        .agg(label=("label", "max"), prob_fake=("prob_fake", "mean"))
    )
    video_df["pred"] = (video_df["prob_fake"] >= threshold).astype(int)

    video_metrics = classification_metrics(
        video_df["label"].to_numpy(),
        video_df["prob_fake"].to_numpy(),
        threshold=threshold,
    )
    return frame_metrics, video_metrics, frame_df, video_df


# Rebuild the model from the checkpoint arguments used during training.
def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    requested_model_type: str | None = None,
):
    # Allow the requested model type to override the checkpoint value when needed.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})

    checkpoint_model_type = checkpoint.get("model_type")
    model_type = requested_model_type or checkpoint_model_type or args.get("model_type", "spatial")
    if model_type not in {"spatial", "dual", "dual_nomask"}:
        raise ValueError(f"Unsupported model type in checkpoint: {model_type}")

    model = build_model(
        model_type=model_type,
        backbone_name=args.get("backbone_name", "xception41"),
        pretrained=False,
        hidden_dim=args.get("hidden_dim", 512),
        dropout=args.get("dropout", 0.2),
        fusion_embed_dim=args.get("fusion_embed_dim", 512),
        fusion_heads=args.get("fusion_heads", 8),
        fusion_token_grid_size=args.get("fusion_token_grid_size", 14),
        frequency_artifact_size=args.get("frequency_artifact_size", 256),
        frequency_mask_size=args.get("frequency_mask_size", 64),
        frequency_out_channels=args.get("frequency_out_channels", 256),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint, model_type


def parse_args() -> argparse.Namespace:
    # Evaluation mainly requires the checkpoint, the test manifest, and optional output paths.
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector models (spatial or dual).")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="test")
    # Add dual_nomask so the ablation can be evaluated with the same script.
    parser.add_argument("--model-type", type=str, default=None, choices=["spatial", "dual", "dual_nomask"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--predictions-csv", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    # Rebuild the model, run inference, and optionally save predictions.
    args = parse_args()
    device = torch.device(args.device)

    dataset = DeepfakeDataset(args.test_manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_with_padding,
    )

    model, checkpoint, model_type = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        requested_model_type=args.model_type,
    )
    # Reuse the validation threshold unless one is provided manually.
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", 0.5))

    frame_metrics, video_metrics, frame_df, video_df = evaluate_model(
        model_type=model_type,
        model=model,
        loader=loader,
        device=device,
        threshold=threshold,
    )

    result = {
        "dataset": args.dataset_name,
        "model_type": model_type,
        "threshold": float(threshold),
        "frame_metrics": frame_metrics,
        "video_metrics": video_metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Save both frame-level and video-level CSV files for later analysis.
    if args.predictions_csv:
        out_path = Path(args.predictions_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame_df.to_csv(out_path, index=False)
        video_out_path = out_path.with_name(out_path.stem + "_video.csv")
        video_df.to_csv(video_out_path, index=False)
        print(f"Frame predictions saved to: {out_path}")
        print(f"Video predictions saved to: {video_out_path}")


if __name__ == "__main__":
    main()
