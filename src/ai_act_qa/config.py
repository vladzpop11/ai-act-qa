"""Project settings (configs/config.yaml) and paths."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def project_path(relative: str) -> Path:
    return ROOT / relative


def load_config() -> dict:
    return yaml.safe_load(project_path("configs/config.yaml").read_text())
