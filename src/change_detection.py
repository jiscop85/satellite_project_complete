from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter

from .config import ProjectConfig
from .features import build_feature_stack, preprocess_image
from .io_raster import read_raster, write_geotiff
from .inference import predict_tiled


def spectral_change_detection(
    before_path: str | Path,
    after_path: str | Path,
    out_path: str | Path,
    cfg: Optional[ProjectConfig] = None,
):
    cfg = cfg or ProjectConfig()
    before, meta = read_raster(before_path)
    after, _ = read_raster(after_path, resample_to=(meta["height"], meta["width"]))

    before = preprocess_image(before, cfg)
    after = preprocess_image(after, cfg)

    feat_b, _ = build_feature_stack(before, cfg)
    feat_a, _ = build_feature_stack(after, cfg)

    idx_names = ["ndvi", "mndwi", "ndbi", "bsi", "nbr"]
    all_names = [
        *(f"band_{i+1}" for i in range(before.shape[0])),
        "ndvi", "ndwi", "mndwi", "ndbi", "bsi", "nbr", "evi", "savi", "msavi2", "gndvi", "gci",
        "nir_tex_mean", "nir_tex_std", "nir_tex_grad", "ndvi_tex_mean", "ndvi_tex_std", "ndvi_tex_grad",
    ]
    diffs = [np.abs(feat_a[all_names.index(name)] - feat_b[all_names.index(name)]) for name in idx_names]
    score = np.mean(np.stack(diffs, axis=0), axis=0)
    threshold = np.nanpercentile(score, 85)
    change = (score >= threshold).astype(np.uint8)
    change = median_filter(change, size=3).astype(np.uint8)

    write_geotiff(out_path, change, meta, dtype="uint8", nodata=0)
    return change, meta

