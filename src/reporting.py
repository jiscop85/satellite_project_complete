from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio

from .config import ProjectConfig


def summarize_prediction_map(prediction_path: str | Path, out_csv: str | Path, cfg: Optional[ProjectConfig] = None) -> List[Dict[str, object]]:
    cfg = cfg or ProjectConfig()
    with rasterio.open(prediction_path) as src:
        pred = src.read(1)
        area = abs(src.transform.a * src.transform.e)

    vals, counts = np.unique(pred.astype(np.int32), return_counts=True)
    rows = []
    for cid, count in zip(vals.tolist(), counts.tolist()):
        if cid == 255:
            continue
        rows.append(
            {
                "class_id": int(cid),
                "class_name": cfg.class_names.get(int(cid), f"class_{cid}"),
                "pixel_count": int(count),
                "area_m2": float(count * area),
                "area_ha": float(count * area / 10000.0),
            }
        )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "pixel_count", "area_m2", "area_ha"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def save_json(obj: Dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_label(path: str | Path, resample_to: Optional[Tuple[int, int]] = None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label not found: {path}")

    with rasterio.open(path) as src:
        meta = src.meta.copy()
        if resample_to is None:
            arr = src.read(1, out_dtype=np.int32)
        else:
            height, width = resample_to
            arr = src.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.nearest,
                out_dtype=np.int32,
            )
            meta.update(
                {
                    "height": height,
                    "width": width,
                    "transform": src.transform * Affine.scale(src.width / width, src.height / height),
                }
            )
    return arr.astype(np.int32), meta


def write_geotiff(path: str | Path, array: np.ndarray, meta: dict, dtype: Optional[str] = None, nodata: Optional[float] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = array if array.ndim == 3 else array[np.newaxis, ...]
    out_meta = meta.copy()
    out_meta.update(count=arr.shape[0], dtype=dtype or str(arr.dtype), nodata=nodata)
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(arr.astype(out_meta["dtype"]))


def write_class_map(path: str | Path, class_map: np.ndarray, meta: dict, cfg: ProjectConfig):
    write_geotiff(path, class_map.astype(np.uint8), meta, dtype="uint8", nodata=255)
    with rasterio.open(path, "r+") as dst:
        cmap = {cid: tuple(rgb) for cid, rgb in cfg.color_map.items()}
        cmap[255] = (255, 255, 255)
        try:
            dst.write_colormap(1, cmap)
        except Exception:
            pass


def valid_nodata_mask(img: np.ndarray, nodata_value: float) -> np.ndarray:
    if np.isnan(nodata_value):
        return np.isfinite(img).all(axis=0)
    return np.isfinite(img).all(axis=0) & np.all(img != nodata_value, axis=0)
