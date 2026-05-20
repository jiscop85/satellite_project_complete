from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import median_filter

from .config import ProjectConfig
from .features import build_feature_stack, flatten_features, preprocess_image
from .io_raster import write_geotiff
from .training import load_bundle


def raster_windows(width: int, height: int, tile_size: int):
    for row_off in range(0, height, tile_size):
        for col_off in range(0, width, tile_size):
            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=min(tile_size, width - col_off),
                height=min(tile_size, height - row_off),
            )


def read_window_with_halo(src: rasterio.io.DatasetReader, window: Window, halo: int):
    full = Window(
        col_off=max(0, int(window.col_off) - halo),
        row_off=max(0, int(window.row_off) - halo),
        width=min(src.width - max(0, int(window.col_off) - halo), int(window.width) + 2 * halo),
        height=min(src.height - max(0, int(window.row_off) - halo), int(window.height) + 2 * halo),
    )
    return src.read(window=full, out_dtype=np.float32), full


def crop_to_window(full_arr: np.ndarray, full_window: Window, target_window: Window) -> np.ndarray:
    rs = int(target_window.row_off - full_window.row_off)
    cs = int(target_window.col_off - full_window.col_off)
    re = rs + int(target_window.height)
    ce = cs + int(target_window.width)
    return full_arr[:, rs:re, cs:ce]


def predict_tiled(
    image_path: str | Path,
    model_path: str | Path,
    out_path: str | Path,
    proba_out: Optional[str | Path] = None,
    apply_smoothing: bool = True,
):
  