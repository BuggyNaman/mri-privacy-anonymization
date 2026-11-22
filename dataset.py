"""Simulated privacy-preserving MRI dataset.

The dataset wraps the public 2D brain-MRI tumour dataset from Kaggle and turns
every scan into an (input, target) anonymization pair:

    input  : the scan with simulated burned-in DICOM PHI text on the top and
             bottom periphery ("PAT-ID:xxxx" / "NSUT-CLINIC").
    target : the same scan where the central elliptical *diagnostic ROI* stays
             sharp, the outer skull/periphery is blurred, and the PHI text bands
             are erased completely.

A binary elliptical ROI mask is returned alongside each pair so that training
can apply a ROI-weighted loss and evaluation can measure ROI-only PSNR / SSIM.

Nothing in this module touches the network; it is pure data plumbing.
"""

from __future__ import annotations

import zlib
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_SIZE = 128
PHI_BAND = 18  # height in pixels of the top / bottom burned-in text band
KAGGLE_HANDLE = "navoneel/brain-mri-images-for-brain-tumor-detection"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

Sample = Tuple[Path, int]  # (image path, binary tumour label)


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def download_brain_mri_dataset() -> Path:
    """Download the Kaggle brain-MRI dataset and return its local root folder."""
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "kagglehub is required to download the dataset. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    path = kagglehub.dataset_download(KAGGLE_HANDLE)
    return Path(path)


def index_dataset(root: Path | str, expected_count: int | None = 253) -> List[Sample]:
    """Return ``[(path, label), ...]`` from the **top-level** ``yes/`` and ``no/``.

    ``yes`` (tumour present) maps to label 1 and ``no`` maps to label 0.

    Only the direct children ``root/yes`` and ``root/no`` are read. The Kaggle
    archive ships a full duplicate tree under ``brain_tumor_dataset/``; scanning
    recursively (``rglob``) would index every scan twice and leak images across
    splits, so nested copies are deliberately ignored.

    Raises ``ValueError`` if duplicate filenames exist or the scan count differs
    from ``expected_count`` (pass ``None`` to skip the count check).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    samples: List[Sample] = []
    for folder_name, label in (("yes", 1), ("no", 0)):
        folder = root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Expected a top-level '{folder_name}/' folder directly under {root}."
            )
        for file in sorted(folder.iterdir()):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                samples.append((file.resolve(), label))

    if not samples:
        raise FileNotFoundError(
            f"No MRI images found in {root}/yes or {root}/no."
        )

    basenames = [path.name.lower() for path, _ in samples]
    counts = Counter(basenames)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Found {len(duplicates)} duplicate filename(s), e.g. {duplicates[:5]}. "
            "Filename collisions would leak scans across train/val/test splits."
        )

    if expected_count is not None and len(samples) != expected_count:
        raise ValueError(
            f"Expected {expected_count} unique scans from {root}/yes and {root}/no, "
            f"but found {len(samples)}. Check the dataset layout for duplicates."
        )

    return sorted(samples, key=lambda sample: str(sample[0]))


def assert_no_split_leakage(samples: Sequence[Sample], splits: dict) -> None:
    """Raise if any filename or sample index appears in more than one split."""
    owner: dict[str, str] = {}
    collisions: List[tuple[str, str, str]] = []
    for split_name, indices in splits.items():
        for index in indices:
            name = Path(samples[index][0]).name.lower()
            previous = owner.get(name)
            if previous is not None and previous != split_name:
                collisions.append((name, previous, split_name))
            else:
                owner[name] = split_name

    if collisions:
        preview = ", ".join(f"{n} ({a} & {b})" for n, a, b in collisions[:5])
        raise ValueError(f"Filename leakage across splits: {preview}")

    all_indices = [index for indices in splits.values() for index in indices]
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Sample indices are shared between splits.")


# --------------------------------------------------------------------------- #
# Simulation primitives (shared with the Streamlit demo)
# --------------------------------------------------------------------------- #
def make_roi_mask(size: int = IMG_SIZE, phi_band: int = PHI_BAND) -> np.ndarray:
    """Filled central ellipse, minus the top / bottom PHI bands."""
    mask = np.zeros((size, size), dtype=np.uint8)
    axes = (int(0.40 * size), int(0.41 * size))
    cv2.ellipse(mask, (size // 2, size // 2), axes, 0, 0, 360, 255, -1)
    mask[:phi_band] = 0
    mask[-phi_band:] = 0
    return mask.astype(np.float32) / 255.0


def make_band_mask(size: int = IMG_SIZE, phi_band: int = PHI_BAND) -> np.ndarray:
    """1 over the top / bottom periphery bands that carry burned-in PHI."""
    band = np.zeros((size, size), dtype=np.float32)
    band[:phi_band] = 1.0
    band[-phi_band:] = 1.0
    return band


def _to_unit_range(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    low, high = float(image.min()), float(image.max())
    return np.clip((image - low) / (high - low + 1e-8), 0.0, 1.0)


def load_gray_image(path: Path | str, size: int = IMG_SIZE) -> np.ndarray:
    """Read an image file as a normalized ``[0, 1]`` grayscale array of ``size``."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"Could not read image: {path}")
    return resize_gray(image, size)


def decode_gray_bytes(data: bytes, size: int = IMG_SIZE) -> np.ndarray:
    """Decode an uploaded image (raw bytes) into a normalized grayscale array."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not decode the uploaded image. Try a PNG/JPG file.")
    return resize_gray(image, size)


def resize_gray(image: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Center-resize a grayscale image to ``size`` and normalize to ``[0, 1]``."""
    shrinking = image.shape[0] > size or image.shape[1] > size
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    resized = cv2.resize(image, (size, size), interpolation=interpolation)
    return _to_unit_range(resized)


def inject_phi(
    image: np.ndarray,
    top_text: str,
    bottom_text: str,
    rng: np.random.Generator,
    phi_band: int = PHI_BAND,
) -> np.ndarray:
    """Burn ``top_text`` / ``bottom_text`` into the top / bottom periphery."""
    size = image.shape[0]
    top_x = int(rng.integers(3, 12))
    top_y = int(rng.integers(11, max(12, phi_band - 2)))
    bottom_x = int(rng.integers(8, 24))
    bottom_y = int(rng.integers(size - 8, size - 2))

    layer = np.zeros((size, size), dtype=np.uint8)
    cv2.putText(layer, top_text, (top_x, top_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, 255, 1, cv2.LINE_AA)
    cv2.putText(layer, bottom_text, (bottom_x, bottom_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, 255, 1, cv2.LINE_AA)

    # ``maximum`` keeps the text crisp without pushing the brain brighter.
    return np.maximum(image, layer.astype(np.float32) / 255.0)


def make_anonymized_target(
    clean: np.ndarray,
    roi_mask: np.ndarray,
    band_mask: np.ndarray,
) -> np.ndarray:
    """Blend sharp ROI with blurred periphery, then erase the PHI bands."""
    blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=3.0)
    feather = np.clip(cv2.GaussianBlur(roi_mask, (0, 0), sigmaX=1.5), 0.0, 1.0)
    target = feather * clean + (1.0 - feather) * blurred
    target *= 1.0 - band_mask  # PHI bands are fully erased
    return np.clip(target, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class PrivacyMRIDataset(Dataset):
    """Grayscale 128x128 MRI scans with simulated PHI and anonymized targets.

    Each item is a dict with:

    ==============  =====================  ====================================
    key             shape                  meaning
    ==============  =====================  ====================================
    ``image``       ``(1, 128, 128)``      scan with burned-in PHI (network input)
    ``target``      ``(1, 128, 128)``      sharp ROI, blurred periphery, no PHI
    ``clean``       ``(1, 128, 128)``      original scan without PHI
    ``roi_mask``    ``(1, 128, 128)``      1 inside the diagnostic ellipse
    ``label``       scalar ``long``        1 = tumour, 0 = healthy
    ``path``        ``str``                source file
    ``phi_text``    ``str``                simulated patient identifier
    ==============  =====================  ====================================

    PHI jitter is derived from a CRC32 of the file path, so a given image always
    receives the same text placement regardless of which split it lands in.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: int = IMG_SIZE,
        phi_band: int = PHI_BAND,
        seed: int = 42,
    ) -> None:
        if not samples:
            raise ValueError("PrivacyMRIDataset received an empty sample list.")
        self.samples: List[Sample] = list(samples)
        self.image_size = int(image_size)
        self.phi_band = int(phi_band)
        self.seed = int(seed)

        self.roi_mask = make_roi_mask(self.image_size, self.phi_band)    # (S, S)
        self.band_mask = make_band_mask(self.image_size, self.phi_band)  # (S, S)

    # -------------------------- construction helpers ----------------------- #
    def _load_gray(self, path: Path) -> np.ndarray:
        return load_gray_image(path, self.image_size)

    def _inject_phi(self, image: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, str]:
        """Burn simulated DICOM PHI text into the top and bottom periphery."""
        patient_id = f"PAT-ID:{int(rng.integers(0, 10_000)):04d}"
        return inject_phi(image, patient_id, "NSUT-CLINIC", rng, self.phi_band), patient_id

    def _make_target(self, clean: np.ndarray) -> np.ndarray:
        return make_anonymized_target(clean, self.roi_mask, self.band_mask)

    # ------------------------------- dataset API --------------------------- #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path, label = self.samples[index]

        clean = self._load_gray(path)
        stable_seed = (
            self.seed * 1_000_003 + zlib.crc32(str(path).encode("utf-8"))
        ) % (2**32)
        rng = np.random.default_rng(stable_seed)

        image, patient_id = self._inject_phi(clean, rng)
        target = self._make_target(clean)

        def to_tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).unsqueeze(0)

        return {
            "image": to_tensor(image),
            "target": to_tensor(target),
            "clean": to_tensor(clean),
            "roi_mask": to_tensor(self.roi_mask),
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path),
            "phi_text": patient_id,
        }
