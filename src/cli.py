from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .app_streamlit import run_app
from .change_detection import semantic_change_detection, spectral_change_detection
from .config import ProjectConfig
from .inference import predict_tiled
from .reporting import summarize_prediction_map
from .synthetic import SyntheticSceneSpec, create_demo_dataset
from .training import train_model

LOGGER = logging.getLogger("satellite_project")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Satellite Intelligence Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")
