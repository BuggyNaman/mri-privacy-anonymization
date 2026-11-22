"""Presentation-ready Streamlit demo for privacy-preserving MRI anonymization.

Launch with::

    .venv/bin/streamlit run app.py

The app streams one scan through both anonymizers and the downstream classifier,
reusing the same simulation primitives (``dataset.py``) and preprocessing
(``train_and_eval.py``) that produced the trained checkpoints, so the live
numbers match the benchmark methodology.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from skimage.metrics import structural_similarity

from dataset import (
    decode_gray_bytes,
    download_brain_mri_dataset,
    index_dataset,
    inject_phi,
    load_gray_image,
    make_anonymized_target,
    make_band_mask,
    make_roi_mask,
)
from models import BaselineAutoencoder, MedicalResNet18, ProposedSharpCAE
from train_and_eval import (
    crop_to_roi,
    masked_psnr,
    normalize_for_classifier,
    roi_bounding_box,
    select_device,
)

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

CHECKPOINTS = {
    "baseline": CHECKPOINT_DIR / "baseline_autoencoder.pt",
    "proposed": CHECKPOINT_DIR / "proposed_sharp_cae.pt",
    "classifier": CHECKPOINT_DIR / "medical_resnet18.pt",
}
LABEL_NAMES = {0: "Normal", 1: "Tumor"}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading checkpoints ...")
def load_models():
    """Load all three checkpoints once and keep them resident on the device."""
    missing = [path.name for path in CHECKPOINTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint(s): " + ", ".join(missing) +
            ". Run `python train_and_eval.py` first to produce them."
        )

    def read_state(path: Path) -> dict:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:  # older torch / rich checkpoint payloads
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    device = select_device("auto")
    baseline = BaselineAutoencoder()
    proposed = ProposedSharpCAE()
    classifier = MedicalResNet18(pretrained=False)

    baseline.load_state_dict(read_state(CHECKPOINTS["baseline"]))
    proposed.load_state_dict(read_state(CHECKPOINTS["proposed"]))
    classifier.load_state_dict(read_state(CHECKPOINTS["classifier"]))

    for model in (baseline, proposed, classifier):
        model.to(device).eval()

    return device, baseline, proposed, classifier


@st.cache_resource(show_spinner="Indexing dataset ...")
def load_catalog():
    """Download (cached) and index the dataset for the sample dropdown."""
    return index_dataset(download_brain_mri_dataset())


# --------------------------------------------------------------------------- #
# Imaging helpers
# --------------------------------------------------------------------------- #
def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """Render a ``[0, 1]`` grayscale array as RGB so Streamlit shows grayscale."""
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if array.ndim == 3:
        array = array[..., 0]
    return cv2.cvtColor((array * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def heatmap_rgb(error: np.ndarray, vmax: float) -> np.ndarray:
    """Map an absolute-error map to an inferno RGB image with a shared scale."""
    normalized = np.clip(error / max(vmax, 1e-6), 0.0, 1.0)
    rgba = plt.get_cmap("inferno")(normalized)
    return (rgba[..., :3] * 255).astype(np.uint8)


def as_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(
        np.ascontiguousarray(array, dtype=np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)


def roi_scores(
    prediction: np.ndarray,
    target: np.ndarray,
    roi_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float]:
    """ROI-restricted PSNR (over mask pixels) and SSIM (over the bbox crop)."""
    squared_error = (prediction - target) ** 2
    psnr = masked_psnr(float((squared_error * roi_mask).sum()), int(roi_mask.sum()))
    y0, y1, x0, x1 = bbox
    ssim = float(
        structural_similarity(
            target[y0:y1, x0:x1],
            prediction[y0:y1, x0:x1],
            data_range=1.0,
        )
    )
    return psnr, ssim


def predict_label(
    roi_source: torch.Tensor,
    classifier: MedicalResNet18,
    device: torch.device,
    bbox: tuple[int, int, int, int],
) -> tuple[str, float]:
    """Run the downstream classifier on an ROI crop; return (label, confidence)."""
    with torch.no_grad():
        logits = classifier(normalize_for_classifier(crop_to_roi(roi_source.to(device), bbox)))
        probabilities = torch.softmax(logits, dim=1)[0]
    confidence, index = torch.max(probabilities, dim=0)
    return LABEL_NAMES[int(index)], float(confidence)


# --------------------------------------------------------------------------- #
# Data selection
# --------------------------------------------------------------------------- #
def curate_samples(catalog, per_class: int = 6):
    """Pick a few tumour and normal scans with readable dropdown labels."""
    buckets: dict[int, list[Path]] = {0: [], 1: []}
    for path, label in catalog:
        if len(buckets[label]) < per_class:
            buckets[label].append(Path(path))
    options = []
    for label in (1, 0):  # tumours first
        for path in buckets[label]:
            options.append((f"{LABEL_NAMES[label]} · {path.name}", path, label))
    return options


def get_clean_scan(mode: str) -> tuple[np.ndarray, int | None, str]:
    """Return ``(clean scan, ground-truth label or None, description)``."""
    if mode.startswith("Select Sample"):
        try:
            catalog = load_catalog()
        except Exception as exc:
            st.error(f"Could not load the dataset: {exc}")
            st.info("Switch to 'Upload Custom MRI Image' to run without the dataset.")
            st.stop()

        options = curate_samples(catalog)
        if not options:
            st.error("The dataset contains no images to choose from.")
            st.stop()

        choice = st.sidebar.selectbox(
            "Choose a scan",
            range(len(options)),
            format_func=lambda index: options[index][0],
        )
        _, path, label = options[choice]
        return load_gray_image(path), label, path.name

    upload = st.sidebar.file_uploader(
        "Upload an MRI image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )
    if upload is None:
        st.info("⬆️ Upload a grayscale MRI image from the sidebar to begin.")
        st.stop()
    return decode_gray_bytes(upload.getvalue()), None, upload.name


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def render_comparison(
    phi_image: np.ndarray,
    baseline_out: np.ndarray,
    proposed_out: np.ndarray,
    baseline_error: np.ndarray,
    proposed_error: np.ndarray,
    shared_vmax: float,
) -> None:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(gray_to_rgb(phi_image), caption="1 · Input scan + PHI",
                 width="stretch")
    with col2:
        st.image(gray_to_rgb(baseline_out), caption="2 · Baseline anonymized",
                 width="stretch")
    with col3:
        st.image(gray_to_rgb(proposed_out), caption="3 · Proposed ROI-weighted",
                 width="stretch")
    with col4:
        st.image(heatmap_rgb(baseline_error, shared_vmax),
                 caption="4a · Baseline ROI |error|", width="stretch")
        st.image(heatmap_rgb(proposed_error, shared_vmax),
                 caption="4b · Proposed ROI |error|", width="stretch")


def render_metrics(
    baseline_scores: tuple[float, float],
    proposed_scores: tuple[float, float],
) -> None:
    base_psnr, base_ssim = baseline_scores
    prop_psnr, prop_ssim = proposed_scores

    st.subheader("Live ROI reconstruction metrics")
    st.caption("Measured against the synthetic anonymized target, restricted to the diagnostic ROI.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline ROI PSNR", f"{base_psnr:.2f} dB")
    m2.metric("Proposed ROI PSNR", f"{prop_psnr:.2f} dB",
              delta=f"{prop_psnr - base_psnr:+.2f} dB")
    m3.metric("Baseline ROI SSIM", f"{base_ssim:.4f}")
    m4.metric("Proposed ROI SSIM", f"{prop_ssim:.4f}",
              delta=f"{prop_ssim - base_ssim:+.4f}")


def render_clinical_utility(predictions: dict[str, tuple[str, float]]) -> None:
    st.subheader("Downstream clinical utility")
    st.caption("MedicalResNet-18 tumour prediction per input, ROI-cropped exactly as in training.")
    columns = st.columns(3)
    titles = {
        "clean": "Clean ROI",
        "baseline": "Baseline anonymized ROI",
        "proposed": "Proposed anonymized ROI",
    }
    for column, key in zip(columns, ("clean", "baseline", "proposed")):
        label, confidence = predictions[key]
        column.metric(
            label=titles[key],
            value=label,
            delta=f"{confidence * 100:.1f}% confidence",
            delta_color="off",
        )


def render_benchmarks() -> None:
    charts = [
        ("loss_curve.png", "30-epoch training loss"),
        ("roi_error_heatmaps.png", "ROI error heatmaps (test set)"),
        ("classification_comparison.png", "Downstream classification"),
    ]
    available = [(RESULTS_DIR / name, caption) for name, caption in charts
                 if (RESULTS_DIR / name).exists()]
    if not available:
        st.info("No benchmark charts found under results/. Run training to generate them.")
        return

    with st.expander("Overall 30-epoch benchmark results (test set)"):
        for path, caption in available:
            st.image(str(path), caption=caption, width="stretch")
        metrics_path = RESULTS_DIR / "metrics.json"
        if metrics_path.exists():
            st.json(metrics_path.read_text(encoding="utf-8"), expanded=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="Privacy-Preserving MRI Anonymization",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Privacy-Preserving MRI Anonymization")
    st.caption(
        "ROI-weighted convolutional autoencoder demo — keep the diagnostic region sharp, "
        "blur the periphery, and erase burned-in PHI."
    )

    try:
        device, baseline, proposed, classifier = load_models()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.header("Demo controls")
    st.sidebar.caption(f"Inference device: **{device.type.upper()}**")
    mode = st.sidebar.radio(
        "Input source",
        ["Select Sample MRI from Dataset", "Upload Custom MRI Image"],
    )
    top_text = st.sidebar.text_input("Top PHI text", "PAT-ID: 2026")
    bottom_text = st.sidebar.text_input("Bottom PHI text", "NSUT-CLINIC")

    clean, true_label, source_name = get_clean_scan(mode)
    st.sidebar.success(f"Loaded: {source_name}")
    if true_label is not None:
        st.sidebar.caption(f"Ground-truth label: **{LABEL_NAMES[true_label]}**")

    # --- Simulate PHI + anonymized target, run both anonymizers ------------- #
    rng = np.random.default_rng(42)
    roi_mask = make_roi_mask()
    band_mask = make_band_mask()
    phi_image = inject_phi(clean, top_text, bottom_text, rng)
    target = make_anonymized_target(clean, roi_mask, band_mask)

    with torch.no_grad():
        phi_tensor = as_tensor(phi_image, device)
        baseline_tensor = baseline(phi_tensor)
        proposed_tensor = proposed(phi_tensor)

    baseline_out = baseline_tensor[0, 0].cpu().numpy()
    proposed_out = proposed_tensor[0, 0].cpu().numpy()

    bbox = roi_bounding_box(roi_mask)
    baseline_error = np.abs(baseline_out - target) * roi_mask
    proposed_error = np.abs(proposed_out - target) * roi_mask
    shared_vmax = max(float(baseline_error.max()), float(proposed_error.max()), 1e-3)

    # --- Visual comparison -------------------------------------------------- #
    render_comparison(
        phi_image, baseline_out, proposed_out,
        baseline_error, proposed_error, shared_vmax,
    )

    st.divider()

    # --- Live ROI metrics --------------------------------------------------- #
    baseline_scores = roi_scores(baseline_out, target, roi_mask, bbox)
    proposed_scores = roi_scores(proposed_out, target, roi_mask, bbox)
    render_metrics(baseline_scores, proposed_scores)

    st.divider()

    # --- Downstream utility ------------------------------------------------- #
    predictions = {
        "clean": predict_label(as_tensor(clean, device), classifier, device, bbox),
        "baseline": predict_label(baseline_tensor, classifier, device, bbox),
        "proposed": predict_label(proposed_tensor, classifier, device, bbox),
    }
    render_clinical_utility(predictions)

    st.divider()

    # --- Past the classifier, offer the aggregate benchmark ------------- #
    render_benchmarks()

    st.caption(
        "Simulated PHI and synthetic anonymization targets for coursework; "
        "not a validated de-identification pipeline."
    )


if __name__ == "__main__":
    main()
