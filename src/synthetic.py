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


