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
    "${SIC_WAFER_PYTHON:-}"
    "$PROJECT_DIR/.venv/bin/python"
    "$PROJECT_DIR/.venv-sic/bin/python"
    "$PROJECT_DIR/.venv-sic-py314/bin/python"
    "$PROJECT_DIR/.venv-sic-py313/bin/python"
    "$PROJECT_DIR/.venv-sic-py312/bin/python"
    "$PROJECT_DIR/.venv-sic-py311/bin/python"
    "$PROJECT_DIR/.venv-sic-py310/bin/python"
    "$(command -v python3 || true)"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]] && \
      PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$candidate" -c \
      'import sys; assert sys.version_info >= (3, 10); import flask, sic_wafer_counter' \
      >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(choose_python)"; then
  echo "无法找到已准备好的 Python 3.10+ 项目环境。" >&2
  echo "请运行：python3 scripts/bootstrap_local_web_workbench.py" >&2
  echo "引导程序会保留不兼容的旧环境，并自动寻找 python3.10–python3.14。" >&2
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
      # A fresh environment may also build Matplotlib's font cache.  Allow up
      # to three minutes and do not open a dead browser tab while it starts.
      for _ in $(seq 1 720); do
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
