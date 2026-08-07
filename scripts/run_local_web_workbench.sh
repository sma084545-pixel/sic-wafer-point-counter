#!/bin/bash
# Start the local SiC browser workbench from this checkout.
# The script deliberately imports src/ first, so it cannot fall back to an
# older globally installed copy of sic_wafer_counter.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---serve}"
PORT="${SIC_WAFER_WEB_PORT:-8765}"
HOST="127.0.0.1"
LOG_DIR="$PROJECT_DIR/results"
LOG_FILE="$LOG_DIR/web_workbench_server.log"

mkdir -p "$LOG_DIR"

choose_python() {
  local candidate
  local candidates=(
    "$PROJECT_DIR/.venv/bin/python"
    "${SIC_WAFER_PYTHON:-}"
    "$(command -v python3 || true)"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]] && "$candidate" -c 'import flask' >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(choose_python)"; then
  echo "无法找到项目运行环境。请在项目目录运行：python3.10 -m venv .venv && .venv/bin/python -m pip install -e ." >&2
  exit 1
fi

is_running() {
  "$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.25)
    raise SystemExit(0 if sock.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0 else 1)
PY
}

serve() {
  export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  exec "$PYTHON_BIN" -m sic_wafer_counter.cli web \
    --workspace "$PROJECT_DIR" --host "$HOST" --port "$PORT"
}

case "$MODE" in
  --serve)
    serve
    ;;
  --open)
    if ! is_running; then
      echo "正在启动本机工作台（首次启动可能需要约一分钟）…"
      nohup "$0" --serve >>"$LOG_FILE" 2>&1 < /dev/null &
      # Importing OpenCV/scikit-image can take tens of seconds on a cold
      # macOS start. Do not open a browser tab until the HTTP server listens.
      for _ in $(seq 1 240); do
        is_running && break
        sleep 0.25
      done
    fi
    if ! is_running; then
      echo "本机工作台未能启动；请查看：$LOG_FILE" >&2
      exit 1
    fi
    /usr/bin/open "http://$HOST:$PORT/"
    ;;
  *)
    echo "用法：$0 [--serve|--open]" >&2
    exit 2
    ;;
esac
