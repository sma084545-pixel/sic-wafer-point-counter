"""Fresh macOS checkout bootstrap and launcher regression coverage."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = PROJECT_ROOT / "scripts" / "bootstrap_local_web_workbench.py"


def _load_bootstrap() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sic_local_bootstrap", BOOTSTRAP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_check_finds_a_compatible_runtime_without_installing() -> None:
    completed = subprocess.run(
        [sys.executable, str(BOOTSTRAP), "--check"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith(("ready\t", "compatible\t"))


def test_apple_python39_can_execute_bootstrap_when_available() -> None:
    """The screenshot failure path can locate, but never run on, Python 3.9."""

    apple_python = Path("/usr/bin/python3")
    if not apple_python.is_file():
        pytest.skip("Apple system Python is not installed on this host")
    version = subprocess.run(
        [str(apple_python), "-c", "import sys; print(*sys.version_info[:2])"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tuple(int(part) for part in version.split()) >= (3, 10):
        pytest.skip("System Python no longer reproduces the legacy bootstrap case")

    completed = subprocess.run(
        [str(apple_python), str(BOOTSTRAP), "--check"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith(("ready\t", "compatible\t"))
    assert "\t3.9." not in completed.stdout


def test_candidates_include_explicit_and_user_local_versioned_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_bootstrap()
    home = tmp_path / "home"
    explicit = tmp_path / "explicit-python"
    local = home / ".local" / "bin" / "python3.10"
    for path in (explicit, local):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    monkeypatch.setattr(module.shutil, "which", lambda _name, path=None: None)

    candidates = module.candidate_interpreters(
        tmp_path / "project",
        environ={
            "HOME": str(home),
            "PATH": "",
            "SIC_WAFER_PYTHON": str(explicit),
        },
    )
    assert candidates[:2] == [explicit, local]


def test_incompatible_dot_venv_is_preserved_for_isolated_runtime(
    tmp_path: Path,
) -> None:
    module = _load_bootstrap()
    project = tmp_path / "project"
    (project / ".venv").mkdir(parents=True)
    target = module._environment_target(project, Path(sys.executable))
    assert target == project / ".venv-sic"
    assert (project / ".venv").is_dir()
