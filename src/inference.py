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
    bundle = load_bundle(model_path)
    cfg = ProjectConfig(**bundle.config)

    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        out_meta = meta.copy()
        out_meta.update(count=1, dtype="uint8", nodata=255)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        proba_handle = None
        try:
            with rasterio.open(out_path, "w", **out_meta) as dst:
                if proba_out is not None:
                    prob_meta = out_meta.copy()
                    prob_meta.update(dtype="float32", nodata=np.nan)
                    Path(proba_out).parent.mkdir(parents=True, exist_ok=True)
                    proba_handle = rasterio.open(proba_out, "w", **prob_meta)

                for w in raster_windows(src.width, src.height, cfg.tile_size):
                    raw, full_window = read_window_with_halo(src, w, cfg.halo)
                    raw = preprocess_image(raw, cfg)
                    feat_stack, _ = build_feature_stack(raw, cfg)
                    feat_stack = crop_to_window(feat_stack, full_window, w)

                    X = flatten_features(feat_stack)
                    pred = np.full(X.shape[0], 255, dtype=np.uint8)
                    valid = np.isfinite(X).all(axis=1)
                    if np.any(valid):
                        pred[valid] = bundle.pipeline.predict(X[valid]).astype(np.uint8)

                    pred_map = pred.reshape(int(w.height), int(w.width))
                    if apply_smoothing:
                        pred_map = median_filter(pred_map, size=cfg.smoothing_size).astype(np.uint8)

                    dst.write(pred_map, 1, window=w)

                    if proba_handle is not None and hasattr(bundle.pipeline[-1], "predict_proba"):
                        conf = np.full(X.shape[0], np.nan, dtype=np.float32)
                        if np.any(valid):
                            proba = bundle.pipeline.predict_proba(X[valid])
                            conf[valid] = np.max(proba, axis=1).astype(np.float32)
                        proba_handle.write(conf.reshape(int(w.height), int(w.width)), 1, window=w)
        finally:
            if proba_handle is not None:
                proba_handle.close()

    with rasterio.open(out_path) as ds:
        return ds.read(1), ds.meta.copy()


def predict_full(image_path: str | Path, model_path: str | Path, apply_smoothing: bool = True):
    tmp_out = Path(image_path).with_suffix(".prediction.tif")
    return predict_tiled(image_path, model_path, tmp_out, proba_out=None, apply_smoothing=apply_smoothing)
