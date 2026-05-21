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

 

