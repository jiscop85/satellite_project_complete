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

@dataclass
class ProjectConfig:
    sensor: str = "sentinel2"
    nodata_value: float = -9999.0
    random_state: int = 42

    test_size: float = 0.2
    cv_folds: int = 5

    max_samples_total: int = 300_000
    max_samples_per_class: int = 60_000
    class_balance: bool = True

    n_estimators: int = 600
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    n_jobs: int = -1

    tile_size: int = 1024
    halo: int = 24
    smoothing_size: int = 3

    cloud_mask_enabled: bool = False

    class_names: Dict[int, str] = None
    color_map: Dict[int, Tuple[int, int, int]] = None

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = {
                0: "background",
                1: "vegetation",
                2: "water",
                3: "built_up",
                4: "bare_soil",
            }
        if self.color_map is None:
            self.color_map = {
                0: (0, 0, 0),
                1: (0, 160, 0),
                2: (0, 90, 200),
                3: (220, 60, 60),
                4: (190, 170, 80),
            }

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


def preset_for(sensor: str) -> SensorPreset:
    key = sensor.lower().strip()
    if key not in SENSOR_PRESETS:
        raise ValueError(f"Unknown sensor preset: {sensor}")
    return SENSOR_PRESETS[key]

