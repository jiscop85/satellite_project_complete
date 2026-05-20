from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import sobel, uniform_filter

from .config import ProjectConfig
from .indices import band_at, compute_indices, safe_div, band_at


def texture_features(source: np.ndarray) -> Dict[str, np.ndarray]:
    source = source.astype(np.float32)
    mean = uniform_filter(source, size=5)
    mean2 = uniform_filter(source ** 2, size=5)
    std = np.sqrt(np.maximum(mean2 - mean ** 2, 0.0))
    gx = sobel(source, axis=1)
    gy = sobel(source, axis=0)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "grad": grad.astype(np.float32)}


def validate_sensor(img: np.ndarray, cfg: ProjectConfig):
    from .config import preset_for

    p = preset_for(cfg.sensor)
    need = max(p.red, p.green, p.blue, p.nir, p.swir1, p.swir2)
    if img.shape[0] < need:
        raise ValueError(f"Image has {img.shape[0]} bands but preset '{p.name}' requires at least {need}")


def build_feature_stack(img: np.ndarray, cfg: ProjectConfig) -> Tuple[np.ndarray, List[str]]:
    validate_sensor(img, cfg)
    base_bands = [band_at(img, i) for i in range(1, img.shape[0] + 1)]
    band_names = [f"band_{i+1}" for i in range(img.shape[0])]

    indices = compute_indices(img, cfg)
    index_names = list(indices.keys())

    from .config import preset_for
    p = preset_for(cfg.sensor)
    nir = band_at(img, p.nir)

    tex_nir = texture_features(nir)
    tex_ndvi = texture_features(indices["ndvi"])

    arrays = base_bands + [indices[name] for name in index_names] + [
        tex_nir["mean"], tex_nir["std"], tex_nir["grad"],
        tex_ndvi["mean"], tex_ndvi["std"], tex_ndvi["grad"],
    ]
    names = band_names + index_names + [
        "nir_tex_mean", "nir_tex_std", "nir_tex_grad",
        "ndvi_tex_mean", "ndvi_tex_std", "ndvi_tex_grad",
    ]
    return np.stack(arrays, axis=0).astype(np.float32), names
