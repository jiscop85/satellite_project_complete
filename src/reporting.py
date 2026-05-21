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
