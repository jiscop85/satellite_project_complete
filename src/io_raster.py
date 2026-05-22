from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

from .config import ProjectConfig


def read_raster(path: str | Path, resample_to: Optional[Tuple[int, int]] = None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        meta = src.meta.copy()
        if resample_to is None:
            arr = src.read(out_dtype=np.float32)
        else:
            height, width = resample_to
            arr = src.read(
                out_shape=(src.count, height, width),
                resampling=Resampling.bilinear,
                out_dtype=np.float32,
            )
            meta.update(
                {
                    "height": height,
                    "width": width,
                    "transform": src.transform * Affine.scale(src.width / width, src.height / height),
                }
            )
    return arr.astype(np.float32), meta

