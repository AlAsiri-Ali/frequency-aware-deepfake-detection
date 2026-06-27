from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DeepfakeDataset, build_train_transforms, collate_with_padding
from metrics import classification_metrics, find_best_threshold
from model import build_model


def seed_everything(seed: int) -> None:
    # Set all random seeds to improve reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_class_weights(dataset: DeepfakeDataset, device: torch.device) -> torch.Tensor:
    # Compute class weights to reduce the effect of label imbalance.
    labels = np.array([record.label for record in dataset.records], dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def forward_batch(model_type: str, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    # Keep batch forwarding consistent across training and validation.
    spatial = batch["spatial"].to(device, non_blocking=True)
    if model_type == "spatial":
        return model(spatial=spatial)

    raw = batch["raw"].to(device, non_blocking=True)
    raw_sizes = batch["raw_sizes"]
    return model(
        spatial=spatial,
        raw=raw,
        raw_sizes=raw_sizes,
    )


@torch.no_grad()
def evaluate_epoch(
    model_type: str,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    # Run validation without gradients because only metrics are needed.
    model.eval()

    total_loss = 0.0
    frame_rows: List[dict] = []

    for batch in tqdm(loader, desc="Validation", leave=False):
        labels = batch["label"].to(device, non_blocking=True)
        logits = forward_batch(model_type=model_type, model=model, batch=batch, device=device)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        video_ids = batch["video_id"]

        total_loss += float(loss.item()) * labels.size(0)
        for idx in range(len(labels_np)):
            frame_rows.append(
                {
                    "video_id": video_ids[idx],
                    "label": int(labels_np[idx]),
                    "prob_fake": float(probs[idx]),
                }
            )

    frame_y_true = np.array([row["label"] for row in frame_rows], dtype=np.int64)
    frame_y_prob = np.array([row["prob_fake"] for row in frame_rows], dtype=np.float32)

    # Average frame predictions per video because model selection uses video-level performance.
    grouped_probs: Dict[str, List[float]] = {}
    grouped_labels: Dict[str, int] = {}
    for row in frame_rows:
        grouped_probs.setdefault(row["video_id"], []).append(row["prob_fake"])
        grouped_labels[row["video_id"]] = row["label"]

    video_y_true = np.array([grouped_labels[vid] for vid in grouped_probs.keys()], dtype=np.int64)
    video_y_prob = np.array([float(np.mean(grouped_probs[vid])) for vid in grouped_probs.keys()], dtype=np.float32)

    # Select the threshold on validation videos and reuse it for both frame-level and video-level metrics.
    best_threshold, _ = find_best_threshold(video_y_true, video_y_prob, metric="balanced_accuracy")
    frame_metrics = classification_metrics(frame_y_true, frame_y_prob, threshold=best_threshold)
    video_metrics = classification_metrics(video_y_true, video_y_prob, threshold=best_threshold)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    frame_metrics["loss"] = avg_loss
    video_metrics["loss"] = avg_loss
    frame_metrics["num_samples"] = int(len(frame_y_true))
    video_metrics["num_samples"] = int(len(video_y_true))

    return {
        "threshold": float(best_threshold),
        "frame": frame_metrics,
        "video": video_metrics,
    }


def train_one_epoch(
    model_type: str,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float | None = 1.0,
) -> Dict[str, float]:
    # Run one full pass over the training loader.
    model.train()

    total_loss = 0.0
    y_true: List[int] = []
    y_prob_fake: List[float] = []
    # Enable mixed precision only when CUDA and a scaler are available.
    use_amp = scaler is not None and device.type == "cuda"

    for batch in tqdm(loader, desc="Training", leave=False):
        labels = batch["label"].to(device, non_blocking=True)
        # Clear gradients before each optimization step.
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            # Automatic mixed precision reduces memory use during CUDA training.
            with torch.amp.autocast(device_type="cuda"):
                logits = forward_batch(model_type=model_type, model=model, batch=batch, device=device)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Apply gradient clipping as a safeguard against unstable updates.
            # The same rule is used in full-precision training.
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = forward_batch(model_type=model_type, model=model, batch=batch, device=device)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1]
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob_fake.extend(probs.detach().cpu().numpy().tolist())

    metrics = classification_metrics(np.array(y_true), np.array(y_prob_fake), threshold=0.5)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


class EarlyStopping:
    # Base early stopping on the selected validation score.
    def __init__(self, patience: int = 5, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best_value = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        # Return True when training should stop.
        if self.best_value is None:
            self.best_value = value
            return False

        improved = value > self.best_value if self.mode == "max" else value < self.best_value
        if improved:
            self.best_value = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def load_state_dict(self, state: dict) -> None:
        self.patience = state.get("patience", self.patience)
        self.mode = state.get("mode", self.mode)
        self.best_value = state.get("best_value", self.best_value)
        self.bad_epochs = state.get("bad_epochs", self.bad_epochs)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_score: float,
    history: list,
    model_type: str,
    early_stopping: EarlyStopping,
    args: argparse.Namespace,
    extra: Dict[str, Any] | None = None,
) -> None:
    # Store model weights, optimizer state, scheduler state, and training history.
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score,
        "history": history,
        "model_type": model_type,
        "early_stopping": {
            "patience": early_stopping.patience,
            "mode": early_stopping.mode,
            "best_value": early_stopping.best_value,
            "bad_epochs": early_stopping.bad_epochs,
        },
        "args": vars(args),
    }
    if extra is not None:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    # Expose the main experiment settings through the CLI for reproducibility.
    parser = argparse.ArgumentParser(description="Train deepfake detector models (spatial or dual).")

    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    # Add dual_nomask as a separate ablation mode without changing the existing options.
    parser.add_argument("--model-type", type=str, default="spatial", choices=["spatial", "dual", "dual_nomask"])

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--backbone-name", type=str, default="xception41")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--fusion-embed-dim", type=int, default=512)
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--fusion-token-grid-size", type=int, default=14)
    parser.add_argument("--frequency-artifact-size", type=int, default=256)
    parser.add_argument("--frequency-mask-size", type=int, default=64)
    parser.add_argument("--frequency-out-channels", type=int, default=256)

    parser.add_argument("--disable-aug", action="store_true")
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--brightness-jitter", type=float, default=0.10)
    parser.add_argument("--contrast-jitter", type=float, default=0.10)
    parser.add_argument("--jpeg-prob", type=float, default=0.5)
    parser.add_argument("--jpeg-quality-min", type=int, default=35)
    parser.add_argument("--jpeg-quality-max", type=int, default=95)
    parser.add_argument("--resize-prob", type=float, default=0.3)
    parser.add_argument("--resize-scale-min", type=float, default=0.5)
    parser.add_argument("--resize-scale-max", type=float, default=0.9)
    parser.add_argument("--blur-prob", type=float, default=0.2)
    parser.add_argument("--blur-radius-min", type=float, default=0.2)
    parser.add_argument("--blur-radius-max", type=float, default=1.2)

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="video_auc",
        choices=["video_auc", "video_balanced_accuracy"],
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-last-every-epoch", action="store_true")
    return parser.parse_args()


def main() -> None:
    # Prepare data, build the model, and run the training loop.
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Allow augmentation to be disabled when a cleaner baseline is needed.
    train_transform = None
    if not args.disable_aug:
        train_transform = build_train_transforms(
            flip_p=args.flip_prob,
            brightness=args.brightness_jitter,
            contrast=args.contrast_jitter,
            jpeg_p=args.jpeg_prob,
            jpeg_quality_min=args.jpeg_quality_min,
            jpeg_quality_max=args.jpeg_quality_max,
            resize_p=args.resize_prob,
            resize_scale_min=args.resize_scale_min,
            resize_scale_max=args.resize_scale_max,
            blur_p=args.blur_prob,
            blur_radius_min=args.blur_radius_min,
            blur_radius_max=args.blur_radius_max,
        )

    train_dataset = DeepfakeDataset(args.train_manifest, spatial_transform=train_transform)
    val_dataset = DeepfakeDataset(args.val_manifest)

    # Use a custom collate function so variable-sized raw crops remain usable in a batch.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_with_padding,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_with_padding,
    )

    # Delegate model creation to a shared factory so training and evaluation stay aligned.
    model = build_model(
        model_type=args.model_type,
        backbone_name=args.backbone_name,
        pretrained=not args.no_pretrained,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        fusion_embed_dim=args.fusion_embed_dim,
        fusion_heads=args.fusion_heads,
        fusion_token_grid_size=args.fusion_token_grid_size,
        frequency_artifact_size=args.frequency_artifact_size,
        frequency_mask_size=args.frequency_mask_size,
        frequency_out_channels=args.frequency_out_channels,
    ).to(device)

    # Use class weights computed from the training manifest only.
    class_weights = build_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    history = []
    best_score = -float("inf")

    best_ckpt_path = output_dir / "best.pt"
    last_ckpt_path = output_dir / "last.pt"
    history_path = output_dir / "history.json"
    start_epoch = 1

    # Resuming restores the full training state, not just the model weights.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_model_type = checkpoint.get("model_type")
        if checkpoint_model_type is not None and checkpoint_model_type != args.model_type:
            raise ValueError(
                f"Checkpoint model_type={checkpoint_model_type} does not match requested {args.model_type}."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        best_score = checkpoint.get("best_score", -float("inf"))
        history = checkpoint.get("history", [])
        start_epoch = checkpoint.get("epoch", 0) + 1
        if checkpoint.get("early_stopping") is not None:
            early_stopping.load_state_dict(checkpoint["early_stopping"])

        print(f"[INFO] Resumed from {args.resume} at epoch {start_epoch}")

    # Each epoch runs training, validation, and then checkpoint selection.
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model_type=args.model_type,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            grad_clip=args.grad_clip,
        )
        val_summary = evaluate_epoch(
            model_type=args.model_type,
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Select the model by either video AUC or video balanced accuracy.
        if args.selection_metric == "video_balanced_accuracy":
            selection_score = val_summary["video"]["balanced_accuracy"]
            selection_metric = "video_balanced_accuracy"
            if np.isnan(selection_score):
                selection_score = val_summary["video"]["auc"]
                selection_metric = "video_auc_fallback"
        else:
            selection_score = val_summary["video"]["auc"]
            selection_metric = "video_auc"
            if np.isnan(selection_score):
                selection_score = val_summary["video"]["balanced_accuracy"]
                selection_metric = "video_balanced_accuracy_fallback"

        # Update the learning rate scheduler with the same score used for checkpoint selection.
        scheduler.step(selection_score)

        epoch_summary = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_summary,
            "selection_metric": selection_metric,
            "selection_score": float(selection_score),
            "lr_groups": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False))

        checkpoint_extra = {
            "threshold": float(val_summary["threshold"]),
            "selection_metric": selection_metric,
            "selection_score": float(selection_score),
            "class_weights": class_weights.detach().cpu().tolist(),
        }

        # Store only the best validation checkpoint as best.pt.
        if selection_score > best_score:
            best_score = float(selection_score)
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                history=history,
                model_type=args.model_type,
                early_stopping=early_stopping,
                args=args,
                extra=checkpoint_extra,
            )
            print(f"[INFO] Saved best checkpoint to {best_ckpt_path}")

        # Update last.pt either every epoch or only at the final epoch, depending on the selected flag.
        if args.save_last_every_epoch or epoch == args.epochs:
            save_checkpoint(
                path=last_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                history=history,
                model_type=args.model_type,
                early_stopping=early_stopping,
                args=args,
                extra=checkpoint_extra,
            )

        # Also save the training history as JSON for lightweight run inspection.
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        if early_stopping.step(float(selection_score)):
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break


if __name__ == "__main__":
    main()
