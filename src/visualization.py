from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .config import ProjectConfig
from .features import colorized_rgb


def plot_rgb(image: np.ndarray, cfg: ProjectConfig, title: str = "RGB"):
    rgb = colorized_rgb(image, cfg)
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_class_map(class_map: np.ndarray, title: str = "Class Map"):
    plt.figure(figsize=(10, 8))
    plt.imshow(class_map, cmap="tab20")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_change_map(change_map: np.ndarray, title: str = "Change Map"):
    plt.figure(figsize=(10, 8))
    plt.imshow(change_map, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
