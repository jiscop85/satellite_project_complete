from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin

from .config import ProjectConfig
from .io_raster import write_geotiff


@dataclass
class SyntheticSceneSpec:
    width: int = 512
    height: int = 512
    bands: int = 12
    pixel_size: float = 10.0
    seed: int = 42


CLASS_IDS = {
    "background": 0,
    "vegetation": 1,
    "water": 2,
    "built_up": 3,
    "bare_soil": 4,
}
# Base reflectance signatures in arbitrary units that mimic satellite responses.
# The order corresponds to: [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12] if bands=12
SPECTRAL_SIGNATURES = {
    0: np.array([0.08, 0.08, 0.08, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10], dtype=np.float32),
    1: np.array([0.05, 0.06, 0.07, 0.10, 0.20, 0.28, 0.30, 0.42, 0.40, 0.12, 0.18, 0.15], dtype=np.float32),
    2: np.array([0.03, 0.03, 0.04, 0.04, 0.02, 0.02, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01], dtype=np.float32),
    3: np.array([0.12, 0.15, 0.18, 0.22, 0.24, 0.28, 0.30, 0.26, 0.25, 0.22, 0.31, 0.34], dtype=np.float32),
    4: np.array([0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.28, 0.24, 0.23, 0.20, 0.22, 0.20], dtype=np.float32),
}


def _ellipse_mask(h: int, w: int, cx: float, cy: float, rx: float, ry: float, angle: float) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    x = xx - cx
    y = yy - cy
    ca = math.cos(angle)
    sa = math.sin(angle)
    xr = x * ca + y * sa
    yr = -x * sa + y * ca
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def _smooth_noise(rng: np.random.Generator, h: int, w: int, scale: int = 16) -> np.ndarray:
    small_h = max(2, h // scale)
    small_w = max(2, w // scale)
    coarse = rng.normal(0, 1, size=(small_h, small_w)).astype(np.float32)
    noise = np.kron(coarse, np.ones((math.ceil(h / small_h), math.ceil(w / small_w)), dtype=np.float32))
    noise = noise[:h, :w]
    noise -= noise.min()
    noise /= max(noise.max(), 1e-6)
    return noise


def generate_label_map(spec: SyntheticSceneSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    h, w = spec.height, spec.width
    labels = np.zeros((h, w), dtype=np.uint8)

    # Water body: a river-like curve and a lake
    yy, xx = np.mgrid[:h, :w]
    river_center = w * 0.55 + 25 * np.sin(np.linspace(0, 4 * np.pi, h))
    river_width = 18 + 5 * np.sin(np.linspace(0, 3 * np.pi, h))
    for y in range(h):
        x0 = river_center[y]
        rw = river_width[y]
        river = np.abs(xx[y] - x0) < rw
        labels[y, river] = 2

    lake1 = _ellipse_mask(h, w, w * 0.20, h * 0.25, w * 0.10, h * 0.08, angle=0.2)
    lake2 = _ellipse_mask(h, w, w * 0.78, h * 0.72, w * 0.08, h * 0.05, angle=-0.5)
    labels[lake1 | lake2] = 2

    # Vegetation clusters
    for _ in range(8):
        cx = rng.uniform(0.05, 0.95) * w
        cy = rng.uniform(0.05, 0.95) * h
        rx = rng.uniform(w * 0.05, w * 0.18)
        ry = rng.uniform(h * 0.04, h * 0.16)
        ang = rng.uniform(0, np.pi)
        mask = _ellipse_mask(h, w, cx, cy, rx, ry, ang)
        labels[(mask) & (labels == 0)] = 1

    # Urban clusters
    for _ in range(6):
        cx = rng.uniform(0.15, 0.90) * w
        cy = rng.uniform(0.10, 0.90) * h
        rx = rng.uniform(w * 0.05, w * 0.12)
        ry = rng.uniform(h * 0.04, h * 0.10)
        ang = rng.uniform(0, np.pi)
        mask = _ellipse_mask(h, w, cx, cy, rx, ry, ang)
        labels[(mask) & (labels == 0)] = 3

    # Bare soil patches
    for _ in range(5):
        cx = rng.uniform(0.10, 0.90) * w
        cy = rng.uniform(0.10, 0.90) * h
        rx = rng.uniform(w * 0.04, w * 0.14)
        ry = rng.uniform(h * 0.04, h * 0.12)
        ang = rng.uniform(0, np.pi)
        mask = _ellipse_mask(h, w, cx, cy, rx, ry, ang)
        labels[(mask) & (labels == 0)] = 4

    # Add some natural background variation
    bg_noise = _smooth_noise(rng, h, w, scale=24)
    labels[(labels == 0) & (bg_noise > 0.78)] = 1
    labels[(labels == 0) & (bg_noise < 0.08)] = 4

    return labels


def generate_multiband_image(labels: np.ndarray, spec: SyntheticSceneSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed + 101)
    h, w = labels.shape
    bands = np.zeros((spec.bands, h, w), dtype=np.float32)

    gradient_x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    gradient_y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    seasonal = 0.03 * np.sin(2 * np.pi * gradient_x) + 0.02 * np.cos(2 * np.pi * gradient_y)

    for cls in SPECTRAL_SIGNATURES:
        base = SPECTRAL_SIGNATURES[cls][: spec.bands].copy()
        mask = labels == cls
        for b in range(spec.bands):
            noise = rng.normal(0, 0.015 + 0.005 * b, size=(h, w)).astype(np.float32)
            spatial = seasonal + 0.01 * b * gradient_x + 0.008 * b * gradient_y
            band = base[b] + spatial + noise
            # Class-specific adjustments to make classes separable but realistic.
            if cls == 1:
                band += 0.05 * np.clip(gradient_y, 0, 1)
            elif cls == 2:
                band -= 0.02 * np.clip(gradient_x, 0, 1)
            elif cls == 3:
                band += 0.04 * np.clip(gradient_x + gradient_y, 0, 2)
            elif cls == 4:
                band += 0.02 * np.sin(4 * np.pi * gradient_x)
            bands[b][mask] = band[mask]


    # Fill background areas with smoother mixed values.
    bg = labels == 0
    for b in range(spec.bands):
        bands[b][bg] = 0.5 * bands[b][bg] + 0.5 * (0.08 + 0.03 * b + seasonal[bg])

    # Normalize and keep in a plausible reflectance range.
    bands = np.clip(bands, 0.0, 1.0)
    return bands.astype(np.float32)


def generate_scene(spec: SyntheticSceneSpec):
    labels = generate_label_map(spec)
    image = generate_multiband_image(labels, spec)
    transform = from_origin(0, spec.height * spec.pixel_size, spec.pixel_size, spec.pixel_size)
    meta = {
        "driver": "GTiff",
        "height": spec.height,
        "width": spec.width,
        "count": spec.bands,
        "dtype": "float32",
        "crs": "EPSG:32633",
        "transform": transform,
    }
    label_meta = meta.copy()
    label_meta.update(count=1, dtype="uint8")
    return image, labels, meta, label_meta


def generate_before_after_scene(spec: SyntheticSceneSpec, change_strength: float = 0.15):
    base_labels = generate_label_map(spec)
    before = generate_multiband_image(base_labels, spec)

    # Create an "after" map with plausible change: urban expansion, vegetation loss, water shift.
    after_labels = base_labels.copy()
    rng = np.random.default_rng(spec.seed + 999)
    h, w = after_labels.shape

    # Convert some vegetation to built-up
    veg = np.argwhere(after_labels == 1)
    if len(veg) > 0:
        idx = rng.choice(len(veg), size=max(1, int(len(veg) * change_strength * 0.35)), replace=False)
        after_labels[tuple(veg[idx].T)] = 3

    # Convert some bare soil to vegetation
    soil = np.argwhere(after_labels == 4)
    if len(soil) > 0:
        idx = rng.choice(len(soil), size=max(1, int(len(soil) * change_strength * 0.25)), replace=False)
        after_labels[tuple(soil[idx].T)] = 1

    # Shift some water borders to bare soil
    water = np.argwhere(after_labels == 2)
    if len(water) > 0:
        idx = rng.choice(len(water), size=max(1, int(len(water) * change_strength * 0.18)), replace=False)
        after_labels[tuple(water[idx].T)] = 4

    after = generate_multiband_image(after_labels, SyntheticSceneSpec(
        width=spec.width, height=spec.height, bands=spec.bands, pixel_size=spec.pixel_size, seed=spec.seed + 123
    ))

    return (before, base_labels), (after, after_labels)


def save_scene(image: np.ndarray, labels: np.ndarray, image_path: str | Path, labels_path: str | Path, meta: dict, label_meta: dict):
    from .io_raster import write_geotiff

    write_geotiff(image_path, image, meta, dtype="float32", nodata=-9999.0)
    write_geotiff(labels_path, labels.astype(np.uint8), label_meta, dtype="uint8", nodata=255)


def create_demo_dataset(base_dir: str | Path = "data/demo", spec: SyntheticSceneSpec | None = None):
    base_dir = Path(base_dir)
    spec = spec or SyntheticSceneSpec()

    train_dir = base_dir / "train"
    infer_dir = base_dir / "infer"
    change_dir = base_dir / "change"
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    change_dir.mkdir(parents=True, exist_ok=True)

    image, labels, meta, label_meta = generate_scene(spec)
    save_scene(image, labels, train_dir / "image.tif", train_dir / "labels.tif", meta, label_meta)

    # Inference image: similar scene but not identical
    spec_infer = SyntheticSceneSpec(width=spec.width, height=spec.height, bands=spec.bands, pixel_size=spec.pixel_size, seed=spec.seed + 7)
    infer_img, infer_lbl, infer_meta, infer_label_meta = generate_scene(spec_infer)
    save_scene(infer_img, infer_lbl, infer_dir / "image.tif", infer_dir / "labels.tif", infer_meta, infer_label_meta)

    (before_img, before_lbl), (after_img, after_lbl) = generate_before_after_scene(spec)
    save_scene(before_img, before_lbl, change_dir / "before.tif", change_dir / "before_labels.tif", meta, label_meta)
    save_scene(after_img, after_lbl, change_dir / "after.tif", change_dir / "after_labels.tif", meta, label_meta)

    return {
        "train": {"image": str(train_dir / "image.tif"), "labels": str(train_dir / "labels.tif")},
        "infer": {"image": str(infer_dir / "image.tif"), "labels": str(infer_dir / "labels.tif")},
        "change": {"before": str(change_dir / "before.tif"), "after": str(change_dir / "after.tif")},
    }

