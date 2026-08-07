"""Shared fixtures for deterministic, independent CV pipeline tests."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CI and sandbox homes can be read-only; Matplotlib only needs this cache for
# report figures, not for any scientific calculation.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/sic-wafer-counter-mpl")


@pytest.fixture()
def default_config() -> dict[str, Any]:
    """Return a mutable test configuration without costly crop exports."""

    config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    config["output"]["save_candidate_crops"] = False
    config["output"]["generate_heatmap"] = False
    return config


@pytest.fixture()
def generate_synthetic(tmp_path: Path):
    """Create fresh labelled images; detector code never reads those labels."""

    from scripts.generate_synthetic_wafer import generate_synthetic_wafer

    def factory(kind: str, seed: int = 20230112) -> dict[str, Any]:
        return generate_synthetic_wafer(kind, tmp_path / "synthetic", seed)

    return factory


@pytest.fixture()
def config_copy(default_config: dict[str, Any]):
    """Make independent config copies for tests that change one parameter."""

    return lambda: copy.deepcopy(default_config)

