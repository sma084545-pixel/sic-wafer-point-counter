#!/usr/bin/env python3
"""Install or remove a per-user macOS LaunchAgent for the local workbench."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import plistlib
import subprocess
import sys


LABEL = "org.sic-wafer-counter.local-workbench"


def _run_launchctl(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *arguments], check=check, text=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the local SiC browser workbench service on macOS.")
    parser.add_argument("--uninstall", action="store_true", help="remove the per-user LaunchAgent")
    parser.add_argument("--no-open", action="store_true", help="do not open the local browser page after installation")
    args = parser.parse_args()

    if sys.platform != "darwin":
        parser.error("This installer is only for macOS.")

    project_dir = Path(__file__).resolve().parents[1]
    launcher = project_dir / "scripts" / "run_local_web_workbench.sh"
    if not launcher.is_file():
        parser.error(f"launcher missing: {launcher}")

    uid = os.getuid()
    domain = f"gui/{uid}"
    destination = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    log_path = project_dir / "results" / "web_workbench_launchd.log"

    _run_launchctl(["bootout", domain, str(destination)], check=False)
    if args.uninstall:
        destination.unlink(missing_ok=True)
        print("Local workbench service removed.")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Label": LABEL,
        "ProgramArguments": ["/bin/bash", str(launcher), "--serve"],
        "WorkingDirectory": str(project_dir),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
        "ProcessType": "Background",
    }
    destination.write_bytes(plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=False))
    _run_launchctl(["bootstrap", domain, str(destination)])
    _run_launchctl(["kickstart", "-k", f"{domain}/{LABEL}"])
    print("Local workbench service installed: http://127.0.0.1:8765/")
    if not args.no_open:
        subprocess.run(["open", "http://127.0.0.1:8765/"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
