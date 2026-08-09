#!/bin/bash
# Start the local SiC browser workbench from this checkout.
# The script deliberately imports src/ first, so it cannot fall back to an
# older globally installed copy of sic_wafer_counter.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---serve}"
BASE_PORT="${SIC_WAFER_WEB_PORT:-8765}"
MAX_PORT="${SIC_WAFER_WEB_PORT_MAX:-$((BASE_PORT + 20))}"
PORT="$BASE_PORT"
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

if [[ ! "$BASE_PORT" =~ ^[0-9]+$ || ! "$MAX_PORT" =~ ^[0-9]+$ ]] || \
  (( BASE_PORT < 1 || MAX_PORT > 65535 || MAX_PORT < BASE_PORT )); then
  echo "本机工作台端口范围无效：$BASE_PORT-$MAX_PORT" >&2
  exit 2
fi

EXPECTED_VERSION="$(
  PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c \
    'import sic_wafer_counter; print(sic_wafer_counter.__version__)'
)"
EXPECTED_WORKSPACE_ID="$(
  "$PYTHON_BIN" - "$PROJECT_DIR" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

print(sha256(str(Path(sys.argv[1]).resolve()).encode("utf-8")).hexdigest())
PY
)"

is_running() {
  local port="${1:-$PORT}"
  "$PYTHON_BIN" - "$HOST" "$port" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.25)
    raise SystemExit(0 if sock.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0 else 1)
PY
}

server_matches_project() {
  local port="${1:-$PORT}"
  "$PYTHON_BIN" - "$HOST" "$port" "$EXPECTED_VERSION" "$EXPECTED_WORKSPACE_ID" <<'PY'
import json
import sys
from urllib.request import urlopen

host, port, expected_version, expected_workspace_id = sys.argv[1:]
try:
    with urlopen("http://{}:{}/api/health".format(host, port), timeout=0.75) as response:
        payload = json.load(response)
except Exception:
    raise SystemExit(1)
matches = (
    payload.get("application") == "sic-wafer-point-counter"
    and payload.get("software_version") == expected_version
    and payload.get("workspace_id") == expected_workspace_id
    and payload.get("status") == "ready"
)
raise SystemExit(0 if matches else 1)
PY
}

resolve_open_port() {
  local candidate
  for candidate in $(seq "$BASE_PORT" "$MAX_PORT"); do
    if server_matches_project "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
    if ! is_running "$candidate"; then
      if (( candidate != BASE_PORT )); then
        echo "端口 ${BASE_PORT} 已由旧版本或其他程序占用；当前 v${EXPECTED_VERSION} 将使用端口 ${candidate}。" >&2
      fi
      printf '%s' "$candidate"
      return 0
    fi
  done
  echo "端口 $BASE_PORT-$MAX_PORT 均被占用，无法安全启动本机工作台。" >&2
  return 1
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
  --resolve-port)
    resolve_open_port
    printf '\n'
    ;;
  --open)
    PORT="$(resolve_open_port)"
    export SIC_WAFER_WEB_PORT="$PORT"
    if ! server_matches_project "$PORT"; then
      if is_running "$PORT"; then
        echo "端口 $PORT 在检查后被其他程序占用；未打开不明服务。" >&2
        exit 1
      fi
      echo "正在启动本机工作台（首次启动可能需要约一分钟）…"
      nohup "$0" --serve >>"$LOG_FILE" 2>&1 < /dev/null &
      # A fresh environment may also build Matplotlib's font cache.  Allow up
      # to three minutes and do not open a dead browser tab while it starts.
      for _ in $(seq 1 720); do
        server_matches_project "$PORT" && break
        sleep 0.25
      done
    fi
    if ! server_matches_project "$PORT"; then
      echo "本机工作台未能启动；请查看：$LOG_FILE" >&2
      exit 1
    fi
    /usr/bin/open "http://$HOST:$PORT/"
    ;;
  *)
    echo "用法：$0 [--serve|--open|--resolve-port]" >&2
    exit 2
    ;;
esac
