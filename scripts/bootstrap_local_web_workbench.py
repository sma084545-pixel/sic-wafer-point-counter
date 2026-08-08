#!/usr/bin/env python3
"""Prepare and open the macOS local workbench from a fresh checkout.

This bootstrap intentionally remains compatible with Apple's Python 3.9.  It
uses that interpreter only to locate Python 3.10+, create an isolated virtual
environment, and install the project.  The scientific application itself is
never run on an unsupported Python version.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple


MINIMUM_PYTHON = (3, 10)
VERSIONED_NAMES = (
    "python3.14",
    "python3.13",
    "python3.12",
    "python3.11",
    "python3.10",
)
REQUIRED_IMPORTS = (
    "flask",
    "numpy",
    "scipy",
    "skimage",
    "cv2",
    "tifffile",
    "PIL",
    "yaml",
    "pandas",
    "matplotlib",
    "jinja2",
    "sic_wafer_counter",
)


def _venv_python(environment_dir: Path) -> Path:
    """Return the expected Python executable inside a virtual environment."""

    if os.name == "nt":
        return environment_dir / "Scripts" / "python.exe"
    return environment_dir / "bin" / "python"


def _unique_existing(paths: Iterable[Optional[Path]]) -> List[Path]:
    """Return existing executable paths once, preserving priority order."""

    result = []  # type: List[Path]
    seen = set()
    for candidate in paths:
        if candidate is None:
            continue
        expanded = candidate.expanduser()
        try:
            key = str(expanded.resolve())
        except OSError:
            key = str(expanded)
        if key in seen or not expanded.is_file() or not os.access(expanded, os.X_OK):
            continue
        seen.add(key)
        result.append(expanded)
    return result


def candidate_interpreters(
    project_dir: Path,
    environ: Optional[Mapping[str, str]] = None,
) -> List[Path]:
    """Find likely Python 3.10+ executables without assuming shell aliases."""

    values = os.environ if environ is None else environ
    home = Path(values.get("HOME", str(Path.home())))
    candidates = []  # type: List[Optional[Path]]
    override = values.get("SIC_WAFER_PYTHON")
    if override:
        candidates.append(Path(override))

    # Ready project environments take precedence over system installations.
    candidates.extend(
        (
            _venv_python(project_dir / ".venv"),
            _venv_python(project_dir / ".venv-sic"),
        )
    )
    candidates.extend(sorted(project_dir.glob(".venv-sic-py*/bin/python"), reverse=True))

    for name in VERSIONED_NAMES:
        found = shutil.which(name, path=values.get("PATH"))
        candidates.append(Path(found) if found else None)
    for root in (home / ".local" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin")):
        for name in VERSIONED_NAMES:
            candidates.append(root / name)
        candidates.append(root / "python3")

    framework = Path("/Library/Frameworks/Python.framework/Versions")
    if framework.is_dir():
        candidates.extend(sorted(framework.glob("*/bin/python3"), reverse=True))
    return _unique_existing(candidates)


def python_version(python: Path) -> Optional[Tuple[int, int, int]]:
    """Read an interpreter version without importing project dependencies."""

    try:
        completed = subprocess.run(
            [str(python), "-c", "import sys; print(*sys.version_info[:3])"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        parts = completed.stdout.strip().split()
        if completed.returncode != 0 or len(parts) != 3:
            return None
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def is_compatible_python(python: Path) -> bool:
    """Return whether ``python`` satisfies the scientific runtime minimum."""

    version = python_version(python)
    return version is not None and version[:2] >= MINIMUM_PYTHON


def runtime_is_ready(python: Path, project_dir: Path) -> bool:
    """Check the interpreter version and every core local-workbench import."""

    if not is_compatible_python(python):
        return False
    source_dir = str(project_dir / "src")
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_dir if not old_pythonpath else source_dir + os.pathsep + old_pythonpath
    )
    statement = "; ".join("import " + name for name in REQUIRED_IMPORTS)
    try:
        completed = subprocess.run(
            [str(python), "-c", statement],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            # A first import can build Matplotlib's font cache and load
            # OpenCV/scikit-image.  On a fresh macOS environment this can take
            # longer than one minute even though the installation is healthy.
            timeout=180,
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _version_label(python: Path) -> str:
    version = python_version(python)
    if version is None:
        return "unknown"
    return ".".join(str(part) for part in version)


def _existing_environment_interpreters(project_dir: Path) -> List[Path]:
    paths = [
        _venv_python(project_dir / ".venv"),
        _venv_python(project_dir / ".venv-sic"),
    ]
    paths.extend(sorted(project_dir.glob(".venv-sic-py*/bin/python"), reverse=True))
    return _unique_existing(paths)


def _install_project(runtime: Path, project_dir: Path) -> None:
    print("正在安装本机工作台依赖；首次安装可能需要数分钟……", flush=True)
    subprocess.run(
        [str(runtime), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=str(project_dir),
        check=True,
    )
    subprocess.run(
        [str(runtime), "-m", "pip", "install", "-e", str(project_dir)],
        cwd=str(project_dir),
        check=True,
    )


def _environment_target(project_dir: Path, base_python: Path) -> Path:
    primary = project_dir / ".venv"
    if not primary.exists():
        return primary
    isolated = project_dir / ".venv-sic"
    if not isolated.exists():
        print("现有 .venv 不兼容；将保留它并创建独立的 .venv-sic。", flush=True)
        return isolated
    version = python_version(base_python)
    suffix = "unknown" if version is None else "{}{}".format(version[0], version[1])
    versioned = project_dir / (".venv-sic-py" + suffix)
    if versioned.exists() and not is_compatible_python(_venv_python(versioned)):
        raise RuntimeError(
            "现有虚拟环境均不兼容。请将 .venv-sic-py{} 改名后重试。".format(suffix)
        )
    return versioned


def prepare_runtime(project_dir: Path) -> Path:
    """Return a ready Python 3.10+ runtime, preparing one when necessary."""

    candidates = candidate_interpreters(project_dir)
    for python in candidates:
        if runtime_is_ready(python, project_dir):
            print(
                "本机运行环境已就绪：{} (Python {})".format(
                    python, _version_label(python)
                ),
                flush=True,
            )
            return python

    # Reuse a compatible project environment that merely lacks dependencies.
    for python in _existing_environment_interpreters(project_dir):
        if is_compatible_python(python):
            _install_project(python, project_dir)
            if not runtime_is_ready(python, project_dir):
                raise RuntimeError("依赖安装结束，但核心模块导入检查失败。")
            return python

    base_python = next(
        (python for python in candidates if is_compatible_python(python)),
        None,
    )
    if base_python is None:
        raise RuntimeError(
            "未找到 Python 3.10 或更新版本。\n"
            "请先从 https://www.python.org/downloads/macos/ 安装 Python 3.12 或更新版本，\n"
            "或在已安装 Homebrew 时运行：brew install python@3.12\n"
            "安装完成后重新双击启动文件。"
        )

    target = _environment_target(project_dir, base_python)
    runtime = _venv_python(target)
    if not runtime.is_file():
        print(
            "检测到 {} (Python {})；正在创建 {}……".format(
                base_python, _version_label(base_python), target.name
            ),
            flush=True,
        )
        subprocess.run(
            [str(base_python), "-m", "venv", str(target)],
            cwd=str(project_dir),
            check=True,
        )
    if not is_compatible_python(runtime):
        raise RuntimeError("创建的虚拟环境不是 Python 3.10 或更新版本。")
    _install_project(runtime, project_dir)
    if not runtime_is_ready(runtime, project_dir):
        raise RuntimeError("依赖安装结束，但核心模块导入检查失败。")
    return runtime


def check_runtime(project_dir: Path) -> int:
    """Report whether bootstrap can proceed, without installing anything."""

    candidates = candidate_interpreters(project_dir)
    for python in candidates:
        if runtime_is_ready(python, project_dir):
            print("ready\t{}\t{}".format(python, _version_label(python)))
            return 0
    for python in candidates:
        if is_compatible_python(python):
            print("compatible\t{}\t{}".format(python, _version_label(python)))
            return 0
    print("missing\tPython 3.10+", file=sys.stderr)
    return 2


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只检查，不安装或启动")
    parser.add_argument(
        "--prepare-only", action="store_true", help="准备环境，但不启动页面"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_dir = Path(__file__).resolve().parents[1]
    if args.check:
        return check_runtime(project_dir)

    try:
        runtime = prepare_runtime(project_dir)
        if args.prepare_only:
            print("本机工作台安装完成。")
            return 0
        environment = os.environ.copy()
        environment["SIC_WAFER_PYTHON"] = str(runtime)
        subprocess.run(
            ["/bin/bash", str(project_dir / "scripts" / "run_local_web_workbench.sh"), "--open"],
            cwd=str(project_dir),
            env=environment,
            check=True,
        )
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("\n本机工作台准备失败：{}".format(exc), file=sys.stderr)
        print("项目目录：{}".format(project_dir), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
