"""General utilities used by the SiC wafer analysis pipeline.

The helpers in this module deliberately avoid importing any of the computer-
vision modules.  This keeps configuration loading, logging and report writing
usable from small command-line tools as well as from the main pipeline.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import logging
import os
import platform
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import yaml


LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")


class ConfigurationError(ValueError):
    """Raised when a YAML configuration is missing or structurally invalid."""


def deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a recursive, non-mutating merge of two mappings.

    Nested mappings are merged recursively.  Lists and scalar values from
    ``override`` replace the corresponding value in ``base``.  Inputs are
    deep-copied so callers may safely modify the returned configuration.
    """

    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML mapping with explicit, user-facing error messages."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            f"Could not read YAML configuration {config_path}: {exc}"
        ) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigurationError(
            f"Top-level YAML value must be a mapping, got {type(loaded).__name__}"
        )
    return dict(loaded)


def load_config(
    path: str | os.PathLike[str] | None,
    *,
    defaults: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and recursively combine default, file and runtime configuration.

    Merge precedence is ``defaults < YAML file < overrides``.  ``path`` may be
    ``None`` when a caller only wants defaults and runtime overrides.
    """

    config: dict[str, Any] = copy.deepcopy(dict(defaults or {}))
    if path is not None:
        config = deep_merge(config, load_yaml(path))
    if overrides:
        config = deep_merge(config, overrides)
    return config


def _coerce_log_level(level: str | int, verbose: bool) -> int:
    if verbose:
        return logging.DEBUG
    if isinstance(level, int):
        return level
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown logging level: {level!r}")
    return numeric


def setup_logging(
    output_dir: str | os.PathLike[str] | None = None,
    *,
    level: str | int = "INFO",
    verbose: bool = False,
    log_filename: str = "run.log",
    logger_name: str | None = None,
) -> logging.Logger:
    """Configure deterministic console and optional UTF-8 file logging.

    Existing handlers installed by an earlier call from this package are
    removed first, preventing duplicate lines in repeated test/CLI runs.
    Third-party root handlers are not modified when ``logger_name`` is given.
    """

    numeric_level = _coerce_log_level(level, verbose)
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_sic_wafer_handler", False):
            logger.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    setattr(stream_handler, "_sic_wafer_handler", True)
    logger.addHandler(stream_handler)

    if output_dir is not None:
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            directory / log_filename, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        setattr(file_handler, "_sic_wafer_handler", True)
        logger.addHandler(file_handler)

    return logger


# A familiar alias for callers that prefer the verb "configure".
configure_logging = setup_logging


@dataclasses.dataclass(slots=True)
class ElapsedTimer:
    """Simple monotonic timer suitable for reports and log messages."""

    label: str = "operation"
    started_at: float = dataclasses.field(default_factory=time.perf_counter)
    ended_at: float | None = None

    @property
    def elapsed_seconds(self) -> float:
        """Return elapsed monotonic seconds without stopping the timer."""

        end = self.ended_at if self.ended_at is not None else time.perf_counter()
        return max(0.0, end - self.started_at)

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds."""

        if self.ended_at is None:
            self.ended_at = time.perf_counter()
        return self.elapsed_seconds


@contextlib.contextmanager
def timed(
    label: str,
    logger: logging.Logger | None = None,
    *,
    level: int = logging.INFO,
) -> Iterator[ElapsedTimer]:
    """Context manager that records and optionally logs elapsed time."""

    timer = ElapsedTimer(label=label)
    try:
        yield timer
    finally:
        elapsed = timer.stop()
        if logger is not None:
            logger.log(level, "%s completed in %.3f s", label, elapsed)


def _json_default(value: Any) -> Any:
    """Convert common scientific-Python values into JSON-compatible objects."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_text(
    path: str | os.PathLike[str], text: str, *, encoding: str = "utf-8"
) -> Path:
    """Atomically replace a text file using a temporary sibling file."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except OSError as exc:
        if temporary_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary_name)
        raise OSError(f"Could not atomically write {destination}: {exc}") from exc
    return destination


def atomic_write_json(
    path: str | os.PathLike[str],
    data: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> Path:
    """Serialize JSON and atomically replace ``path``.

    NaN and infinity are rejected because they are not valid interoperable
    JSON values and can conceal numerical failures in scientific reports.
    """

    try:
        payload = json.dumps(
            data,
            ensure_ascii=False,
            indent=indent,
            sort_keys=sort_keys,
            allow_nan=False,
            default=_json_default,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not serialize JSON for {path}: {exc}") from exc
    return atomic_write_text(path, payload + "\n")


def atomic_write_yaml(path: str | os.PathLike[str], data: Any) -> Path:
    """Serialize configuration as safe YAML and atomically replace ``path``."""

    try:
        payload = yaml.safe_dump(
            data, allow_unicode=True, sort_keys=False, default_flow_style=False
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not serialize YAML for {path}: {exc}") from exc
    return atomic_write_text(path, payload)


def ensure_odd(value: int, *, minimum: int = 1, name: str = "value") -> int:
    """Validate that an integer kernel size is positive and odd."""

    integer = int(value)
    if integer < minimum or integer % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= {minimum}, got {value}")
    return integer


def validate_percentiles(low: float, high: float) -> tuple[float, float]:
    """Validate and return a percentile range in ascending order."""

    low_value, high_value = float(low), float(high)
    if not (0.0 <= low_value < high_value <= 100.0):
        raise ValueError(
            "Normalization percentiles must satisfy "
            f"0 <= low < high <= 100, got ({low}, {high})"
        )
    return low_value, high_value


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat()


def software_versions() -> dict[str, str]:
    """Return lightweight runtime version information for reproducibility."""

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for module_name in ("scipy", "skimage", "cv2", "tifffile"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = "not installed"
    return versions


def require_mapping(value: Any, *, name: str) -> MutableMapping[str, Any]:
    """Return ``value`` as a mutable mapping or raise a clear configuration error."""

    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)
