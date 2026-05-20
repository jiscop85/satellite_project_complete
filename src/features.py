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


def flatten_features(feature_stack: np.ndarray) -> np.ndarray:
    f, h, w = feature_stack.shape
    return feature_stack.reshape(f, h * w).T


def flatten_labels(label_map: np.ndarray) -> np.ndarray:
    return label_map.reshape(-1)


def preprocess_image(img: np.ndarray, cfg: ProjectConfig) -> np.ndarray:
    img = img.astype(np.float32).copy()
    img[~np.isfinite(img)] = np.nan
    if cfg.nodata_value is not None:
        img[img == cfg.nodata_value] = np.nan

    cleaned = np.empty_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        band = img[i]
        if np.all(np.isnan(band)):
            cleaned[i] = np.zeros_like(band, dtype=np.float32)
            continue
        fill = np.nanmedian(band)
        filled = np.where(np.isfinite(band), band, fill)
        cleaned[i] = filled.astype(np.float32)
    return cleaned


def colorized_rgb(image: np.ndarray, cfg: ProjectConfig) -> np.ndarray:
    from .config import preset_for

    p = preset_for(cfg.sensor)
    validate_sensor(image, cfg)

    def robust_scale(arr: np.ndarray) -> np.ndarray:
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros_like(arr, dtype=np.float32)
        lo = np.percentile(arr[finite], 2)
        hi = np.percentile(arr[finite], 98)
        if np.isclose(lo, hi):
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    rgb = np.dstack([
        robust_scale(band_at(image, p.red)),
        robust_scale(band_at(image, p.green)),
        robust_scale(band_at(image, p.blue)),
    ])
    return rgb
