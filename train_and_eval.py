"""Train and evaluate privacy-preserving MRI anonymization models.

Pipeline
--------
1. Download (or reuse) the Kaggle brain-MRI dataset and build stratified
   train / validation / test splits.
2. Train ``BaselineAutoencoder`` with a uniform MSE loss and
   ``ProposedSharpCAE`` with the ROI-weighted ``SharpROILoss``.
3. Report test-set ROI PSNR and ROI SSIM for both anonymizers.
4. Train ``MedicalResNet18`` on clean scans, then benchmark downstream tumour
   classification on clean, baseline-anonymized and proposed-anonymized scans.
5. Save loss curves, ROI error heatmaps and a classification bar chart to
   ``results/`` alongside a ``metrics.json`` summary.

Run ``python train_and_eval.py --help`` for the available flags.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as skimage_ssim
from torch.utils.data import DataLoader, Subset

from dataset import (
    PrivacyMRIDataset,
    assert_no_split_leakage,
    download_brain_mri_dataset,
    index_dataset,
)
from models import (
    BaselineAutoencoder,
    MedicalResNet18,
    ProposedSharpCAE,
    SharpROILoss,
)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(preferred: str = "auto") -> torch.device:
    """Pick the best available accelerator: CUDA, then Apple MPS, then CPU."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_for_classifier(x: torch.Tensor) -> torch.Tensor:
    """Map ``[0, 1]`` MRI intensities to the ``[-1, 1]`` range ResNet expects."""
    return (x - 0.5) / 0.5


def crop_to_roi(
    x: torch.Tensor,
    roi_bbox: tuple[int, int, int, int],
    out_size: int = 128,
) -> torch.Tensor:
    """Crop the diagnostic ROI and resize it to a fixed classifier input size."""
    y0, y1, x0, x1 = roi_bbox
    return F.interpolate(
        x[..., y0:y1, x0:x1],
        size=(out_size, out_size),
        mode="bilinear",
        align_corners=False,
    )


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:d}m {secs:02d}s"


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def build_splits(
    samples: Sequence,
    seed: int,
    test_fraction: float = 0.2,
    val_fraction: float = 0.15,
) -> Dict[str, List[int]]:
    """Stratified train / val / test index lists over ``samples``."""
    labels = np.array([label for _, label in samples])
    indices = np.arange(len(samples))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_fraction,
        stratify=labels[indices],
        random_state=seed,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_fraction,
        stratify=labels[train_idx],
        random_state=seed,
    )
    return {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
        "test": sorted(int(i) for i in test_idx),
    }


# --------------------------------------------------------------------------- #
# Autoencoder training
# --------------------------------------------------------------------------- #
def train_autoencoder(
    name: str,
    model: torch.nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    loaders: Dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_path: Path,
) -> Dict[str, List[float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {"train": [], "val": []}
    best_val = math.inf
    best_state = copy.deepcopy(model.state_dict())
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for batch in loaders["train"]:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            masks = batch["roi_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = criterion(predictions, targets, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / max(seen, 1)
        model.eval()
        val_loss, val_seen = 0.0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                images = batch["image"].to(device)
                targets = batch["target"].to(device)
                masks = batch["roi_mask"].to(device)
                loss = criterion(model(images), targets, masks)
                val_loss += loss.item() * images.size(0)
                val_seen += images.size(0)

        val_loss /= max(val_seen, 1)
        scheduler.step()
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            elapsed = format_seconds(time.time() - started)
            print(
                f"[{name}] epoch {epoch:3d}/{epochs} "
                f"train {train_loss:.5f}  val {val_loss:.5f}  elapsed {elapsed}"
            )

    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": name,
            "state_dict": best_state,
            "history": history,
            "best_val_loss": best_val,
        },
        checkpoint_path,
    )
    print(
        f"[{name}] best val loss {best_val:.5f}  "
        f"checkpoint -> {checkpoint_path}"
    )
    return history


# --------------------------------------------------------------------------- #
# Reconstruction metrics
# --------------------------------------------------------------------------- #
def roi_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.5)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def masked_psnr(sum_squared_error: float, pixel_count: int) -> float:
    mse = sum_squared_error / max(pixel_count, 1)
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def evaluate_reconstruction(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    roi_bbox: tuple[int, int, int, int],
) -> Dict[str, float]:
    """ROI and full-image PSNR / SSIM against the anonymized target."""
    model.eval()
    y0, y1, x0, x1 = roi_bbox

    roi_sq_error, roi_pixels = 0.0, 0
    full_sq_error, full_pixels = 0.0, 0
    roi_ssim_scores: List[float] = []
    full_ssim_scores: List[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            masks = batch["roi_mask"].to(device)
            predictions = model(images)

            diff_sq = (predictions - targets) ** 2
            roi_sq_error += (diff_sq * masks).sum().item()
            roi_pixels += int(masks.sum().item())
            full_sq_error += diff_sq.sum().item()
            full_pixels += diff_sq.numel()

            predictions_np = predictions.detach().cpu().numpy()[:, 0]
            targets_np = targets.detach().cpu().numpy()[:, 0]
            for pred, target in zip(predictions_np, targets_np):
                pred = np.clip(pred, 0.0, 1.0)
                roi_ssim_scores.append(
                    skimage_ssim(
                        pred[y0:y1, x0:x1],
                        target[y0:y1, x0:x1],
                        data_range=1.0,
                    )
                )
                full_ssim_scores.append(
                    skimage_ssim(pred, target, data_range=1.0)
                )

    return {
        "roi_psnr": masked_psnr(roi_sq_error, roi_pixels),
        "roi_ssim": float(np.mean(roi_ssim_scores)),
        "full_psnr": masked_psnr(full_sq_error, full_pixels),
        "full_ssim": float(np.mean(full_ssim_scores)),
    }


# --------------------------------------------------------------------------- #
# Downstream classifier
# --------------------------------------------------------------------------- #
def train_classifier(
    loaders: Dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_path: Path,
    roi_bbox: tuple[int, int, int, int],
) -> MedicalResNet18:
    try:
        model = MedicalResNet18(pretrained=True)
    except Exception as exc:  # offline / weight download failure
        print(f"[classifier] pretrained weights unavailable ({exc}); training from scratch.")
        model = MedicalResNet18(pretrained=False)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_f1, best_state = -1.0, copy.deepcopy(model.state_dict())
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for batch in loaders["train"]:
            roi = crop_to_roi(batch["clean"].to(device), roi_bbox)
            inputs = normalize_for_classifier(roi)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                roi = crop_to_roi(batch["clean"].to(device), roi_bbox)
                inputs = normalize_for_classifier(roi)
                logits = model(inputs)
                val_preds.append(logits.argmax(dim=1).cpu())
                val_labels.append(batch["label"])

        val_preds_np = torch.cat(val_preds).numpy()
        val_labels_np = torch.cat(val_labels).numpy()
        val_f1 = f1_score(val_labels_np, val_preds_np, average="binary", zero_division=0)
        val_acc = accuracy_score(val_labels_np, val_preds_np)

        scheduler.step()
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            elapsed = format_seconds(time.time() - started)
            print(
                f"[classifier] epoch {epoch:3d}/{epochs} "
                f"train {running_loss / max(seen, 1):.5f}  "
                f"val acc {val_acc:.4f}  val F1 {val_f1:.4f}  elapsed {elapsed}"
            )

    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": "MedicalResNet18", "state_dict": best_state}, checkpoint_path)
    print(f"[classifier] best val F1 {best_f1:.4f}  checkpoint -> {checkpoint_path}")
    return model


def benchmark_classification(
    classifier: MedicalResNet18,
    baseline: BaselineAutoencoder,
    proposed: ProposedSharpCAE,
    loader: DataLoader,
    device: torch.device,
    roi_bbox: tuple[int, int, int, int],
) -> Dict[str, Dict[str, float]]:
    """Accuracy / F1 on clean, baseline-anonymized and proposed-anonymized ROI scans."""
    classifier.eval()
    baseline.eval()
    proposed.eval()

    modes = ("clean", "baseline", "proposed")
    predictions: Dict[str, List[torch.Tensor]] = {mode: [] for mode in modes}
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"]
            clean = batch["clean"].to(device)
            phi_input = batch["image"].to(device)

            candidates = {
                "clean": clean,
                "baseline": baseline(phi_input),
                "proposed": proposed(phi_input),
            }
            all_labels.append(labels)
            for mode, inputs in candidates.items():
                roi = crop_to_roi(inputs, roi_bbox)
                logits = classifier(normalize_for_classifier(roi))
                predictions[mode].append(logits.argmax(dim=1).cpu())

    labels_np = torch.cat(all_labels).numpy()
    results: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        preds_np = torch.cat(predictions[mode]).numpy()
        results[mode] = {
            "accuracy": float(accuracy_score(labels_np, preds_np)),
            "f1": float(f1_score(labels_np, preds_np, average="binary", zero_division=0)),
        }
    return results


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_loss_curves(
    histories: Dict[str, Dict[str, List[float]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, (name, history) in zip(axes, histories.items()):
        epochs = np.arange(1, len(history["train"]) + 1)
        ax.plot(epochs, history["train"], label="Train", linewidth=2)
        ax.plot(epochs, history["val"], label="Validation", linewidth=2)
        ax.set_title(f"{name} loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Anonymization training loss", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_heatmaps(
    dataset: PrivacyMRIDataset,
    indices: Sequence[int],
    baseline: BaselineAutoencoder,
    proposed: ProposedSharpCAE,
    device: torch.device,
    output_path: Path,
) -> None:
    samples = [dataset[i] for i in indices]
    if not samples:
        return

    images = torch.stack([s["image"] for s in samples]).to(device)
    targets = torch.stack([s["target"] for s in samples]).to(device)
    masks = torch.stack([s["roi_mask"] for s in samples]).to(device)

    baseline.eval()
    proposed.eval()
    with torch.no_grad():
        baseline_out = baseline(images)
        proposed_out = proposed(images)

    baseline_error = ((baseline_out - targets).abs() * masks).cpu().numpy()[:, 0]
    proposed_error = ((proposed_out - targets).abs() * masks).cpu().numpy()[:, 0]
    global_vmax = max(float(baseline_error.max()), float(proposed_error.max()), 1e-3)

    to_numpy = lambda t: t.detach().cpu().numpy()[:, 0]

    fig, axes = plt.subplots(
        len(samples),
        6,
        figsize=(20, 3.2 * len(samples)),
        squeeze=False,
    )
    titles = (
        "Input (+ simulated PHI)",
        "Target (anonymized)",
        "Baseline reconstruction",
        "Proposed reconstruction",
        "Baseline ROI |error|",
        "Proposed ROI |error|",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=11)

    for row in range(len(samples)):
        clean_panels = [
            to_numpy(images)[row],
            to_numpy(targets)[row],
            to_numpy(baseline_out)[row],
            to_numpy(proposed_out)[row],
        ]
        for column, panel in enumerate(clean_panels):
            axes[row, column].imshow(panel, cmap="gray", vmin=0, vmax=1)

        for column, error in ((4, baseline_error), (5, proposed_error)):
            artist = axes[row, column].imshow(
                error[row], cmap="inferno", vmin=0.0, vmax=global_vmax
            )
            fig.colorbar(artist, ax=axes[row, column], fraction=0.046, pad=0.02)

        for column in range(6):
            axes[row, column].axis("off")

    fig.suptitle(
        "Anonymization comparison (errors restricted to the diagnostic ROI)",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_classification_results(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    modes = list(results.keys())
    accuracy = [results[mode]["accuracy"] for mode in modes]
    f1 = [results[mode]["f1"] for mode in modes]

    positions = np.arange(len(modes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(positions - width / 2, accuracy, width, label="Accuracy")
    ax.bar(positions + width / 2, f1, width, label="F1-Score")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Clean scans", "Baseline anonymized", "Proposed anonymized"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Downstream tumour classification")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for pos, (acc, score) in enumerate(zip(accuracy, f1)):
        ax.text(pos - width / 2, acc + 0.02, f"{acc:.3f}", ha="center", fontsize=9)
        ax.text(pos + width / 2, score + 0.02, f"{score:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_summary(
    reconstruction: Dict[str, Dict[str, float]],
    classification: Dict[str, Dict[str, float]],
) -> None:
    print("\n" + "=" * 66)
    print("Anonymization quality (test set)")
    print("-" * 66)
    print(f"{'Model':<12}{'ROI PSNR':>12}{'ROI SSIM':>12}{'Full PSNR':>12}{'Full SSIM':>12}")
    for name, metrics in reconstruction.items():
        print(
            f"{name:<12}"
            f"{metrics['roi_psnr']:>12.3f}"
            f"{metrics['roi_ssim']:>12.4f}"
            f"{metrics['full_psnr']:>12.3f}"
            f"{metrics['full_ssim']:>12.4f}"
        )

    print("\n" + "=" * 66)
    print("Downstream tumour classification (test set)")
    print("-" * 66)
    print(f"{'Input':<24}{'Accuracy':>14}{'F1-Score':>14}")
    labels = {
        "clean": "Clean scans",
        "baseline": "Baseline anonymized",
        "proposed": "Proposed anonymized",
    }
    for mode, metrics in classification.items():
        print(
            f"{labels.get(mode, mode):<24}"
            f"{metrics['accuracy']:>14.4f}"
            f"{metrics['f1']:>14.4f}"
        )
    print("=" * 66 + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Existing dataset folder; skips the kagglehub download.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Autoencoder training epochs.")
    parser.add_argument("--classifier-epochs", type=int, default=12,
                        help="ResNet-18 classifier training epochs.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Autoencoder learning rate.")
    parser.add_argument("--classifier-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Optional cap on total samples for quick runs.")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--heatmap-samples", type=int, default=4,
                        help="Number of test scans in the error-heatmap figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {device}")

    root = args.data_root if args.data_root is not None else download_brain_mri_dataset()
    print(f"Dataset root: {root}")

    samples = index_dataset(root)
    if args.num_samples is not None and args.num_samples < len(samples):
        keep = train_test_split(
            np.arange(len(samples)),
            train_size=args.num_samples,
            stratify=[label for _, label in samples],
            random_state=args.seed,
        )[0]
        samples = [samples[i] for i in sorted(keep)]
    print(f"Indexed {len(samples)} MRI scans.")

    split_indices = build_splits(samples, seed=args.seed)
    assert_no_split_leakage(samples, split_indices)
    print("Leakage check passed: no scan appears in more than one split.")
    print(
        "Split sizes -> train: {train}, val: {val}, test: {test}".format(
            **{name: len(indices) for name, indices in split_indices.items()}
        )
    )

    base_dataset = PrivacyMRIDataset(samples, seed=args.seed)
    loaders = {
        name: make_loader(
            Subset(base_dataset, indices),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            device=device,
            seed=args.seed,
        )
        for name, indices in split_indices.items()
    }
    roi_bbox = roi_bounding_box(base_dataset.roi_mask)

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # --- Train anonymizers -------------------------------------------------- #
    baseline = BaselineAutoencoder()
    proposed = ProposedSharpCAE()
    print(
        f"BaselineAutoencoder parameters: {sum(p.numel() for p in baseline.parameters()):,}"
    )
    print(
        f"ProposedSharpCAE parameters:    {sum(p.numel() for p in proposed.parameters()):,}"
    )

    baseline_history = train_autoencoder(
        name="baseline",
        model=baseline,
        criterion=lambda pred, target, mask: F.mse_loss(pred, target),
        loaders=loaders,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=args.checkpoints_dir / "baseline_autoencoder.pt",
    )
    proposed_history = train_autoencoder(
        name="proposed",
        model=proposed,
        criterion=SharpROILoss(roi_weight=25.0, periphery_weight=2.0),
        loaders=loaders,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=args.checkpoints_dir / "proposed_sharp_cae.pt",
    )

    # --- Reconstruction quality -------------------------------------------- #
    reconstruction = {
        "baseline": evaluate_reconstruction(baseline, loaders["test"], device, roi_bbox),
        "proposed": evaluate_reconstruction(proposed, loaders["test"], device, roi_bbox),
    }
    print(
        "Test ROI PSNR -> baseline {baseline_roi_psnr:.3f} dB, "
        "proposed {proposed_roi_psnr:.3f} dB".format(
            baseline_roi_psnr=reconstruction["baseline"]["roi_psnr"],
            proposed_roi_psnr=reconstruction["proposed"]["roi_psnr"],
        )
    )

    # --- Downstream classifier --------------------------------------------- #
    classifier = train_classifier(
        loaders=loaders,
        device=device,
        epochs=args.classifier_epochs,
        lr=args.classifier_lr,
        checkpoint_path=args.checkpoints_dir / "medical_resnet18.pt",
        roi_bbox=roi_bbox,
    )
    classification = benchmark_classification(
        classifier, baseline, proposed, loaders["test"], device, roi_bbox
    )

    # --- Plots and metrics -------------------------------------------------- #
    plot_loss_curves(
        {"Baseline (uniform MSE)": baseline_history,
         "Proposed (ROI-weighted)": proposed_history},
        args.results_dir / "loss_curve.png",
    )

    heatmap_indices = split_indices["test"][: max(1, args.heatmap_samples)]
    plot_error_heatmaps(
        base_dataset,
        heatmap_indices,
        baseline,
        proposed,
        device,
        args.results_dir / "roi_error_heatmaps.png",
    )
    plot_classification_results(
        classification, args.results_dir / "classification_comparison.png"
    )

    metrics = {
        "device": str(device),
        "epochs": args.epochs,
        "classifier_epochs": args.classifier_epochs,
        "splits": {name: len(indices) for name, indices in split_indices.items()},
        "reconstruction": reconstruction,
        "classification": classification,
    }
    with open(args.results_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print_summary(reconstruction, classification)
    print(f"Plots and metrics saved to {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
