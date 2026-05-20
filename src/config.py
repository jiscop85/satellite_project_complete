from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class SensorPreset:
    name: str
    red: int
    green: int
    blue: int
    nir: int
    swir1: int
    swir2: int
    extra_bands: Tuple[int, ...] = ()


SENSOR_PRESETS: Dict[str, SensorPreset] = {
    "sentinel2": SensorPreset("sentinel2", red=4, green=3, blue=2, nir=8, swir1=11, swir2=12),
    "landsat8": SensorPreset("landsat8", red=4, green=3, blue=2, nir=5, swir1=6, swir2=7),
    "landsat9": SensorPreset("landsat9", red=4, green=3, blue=2, nir=5, swir1=6, swir2=7),
    "generic": SensorPreset("generic", red=4, green=3, blue=2, nir=5, swir1=6, swir2=7),
}


