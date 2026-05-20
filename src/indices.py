from __future__ import annotations

from typing import Dict

import numpy as np

from .config import ProjectConfig, preset_for


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return a / (b + eps)


def band_at(img: np.ndarray, index_1based: int) -> np.ndarray:
    idx = index_1based - 1
    if idx < 0 or idx >= img.shape[0]:
        raise IndexError(f"Band index {index_1based} out of range for image with {img.shape[0]} bands")
    return img[idx].astype(np.float32)


def compute_indices(img: np.ndarray, cfg: ProjectConfig) -> Dict[str, np.ndarray]:
    p = preset_for(cfg.sensor)
    red = band_at(img, p.red)
    green = band_at(img, p.green)
    blue = band_at(img, p.blue)
    nir = band_at(img, p.nir)
    swir1 = band_at(img, p.swir1)
    swir2 = band_at(img, p.swir2)

    ndvi = safe_div(nir - red, nir + red)
    ndwi = safe_div(green - nir, green + nir)
    mndwi = safe_div(green - swir1, green + swir1)
    ndbi = safe_div(swir1 - nir, swir1 + nir)
    bsi = safe_div((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue))
    nbr = safe_div(nir - swir2, nir + swir2)
    evi = 2.5 * safe_div(nir - red, nir + 6.0 * red - 7.5 * blue + 1.0)
    savi = 1.5 * safe_div(nir - red, nir + red + 0.5)
    msavi2 = (2 * nir + 1 - np.sqrt(np.maximum((2 * nir + 1) ** 2 - 8 * (nir - red), 0.0))) / 2.0
    gndvi = safe_div(nir - green, nir + green)
    gci = safe_div(nir, green) - 1.0

    return {
        "ndvi": ndvi.astype(np.float32),
        "ndwi": ndwi.astype(np.float32),
        "mndwi": mndwi.astype(np.float32),
        "ndbi": ndbi.astype(np.float32),
        "bsi": bsi.astype(np.float32),
        "nbr": nbr.astype(np.float32),
        "evi": evi.astype(np.float32),
        "savi": savi.astype(np.float32),
        "msavi2": msavi2.astype(np.float32),
        "gndvi": gndvi.astype(np.float32),
        "gci": gci.astype(np.float32),
    }
