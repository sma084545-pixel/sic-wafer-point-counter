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
from http.client import HTTPConnection
import sys

host, port, expected_version, expected_workspace_id = sys.argv[1:]
connection = HTTPConnection(host, int(port), timeout=0.75)
try:
    connection.request("GET", "/api/health", headers={"Connection": "close"})
    response = connection.getresponse()
    if response.status != 200:
        raise SystemExit(1)
    payload = json.loads(response.read())
except Exception:
    raise SystemExit(1)
finally:
    connection.close()
matches = (
    payload.get("application") == "sic-wafer-point-counter"
    and payload.get("software_version") == expected_version
    and payload.get("workspace_id") == expected_workspace_id
    and payload.get("status") == "ready"
)
raise SystemExit(0 if matches else 1)
PY
}

wait_for_matching_server() {
  local port="${1:-$PORT}"
  local deadline=$((SECONDS + 180))
  local next_notice=$((SECONDS + 10))
  while (( SECONDS < deadline )); do
    if server_matches_project "$port"; then
      return 0
    fi
    if (( SECONDS >= next_notice )); then
      echo "仍在等待本机工作台就绪（端口 $port）……" >&2
      next_notice=$((SECONDS + 10))
    fi
    sleep 0.25
  done
  return 1
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
      # A fresh environment may also build Matplotlib's font cache.  Wait for
      # this exact checkout, with a real three-minute wall-clock deadline.
      wait_for_matching_server "$PORT" || true
    fi
    if ! server_matches_project "$PORT"; then
      echo "本机工作台未能启动；请查看：$LOG_FILE" >&2
      exit 1
    fi
    echo "本机工作台已启动：http://$HOST:$PORT/（v$EXPECTED_VERSION）"
    /usr/bin/open "http://$HOST:$PORT/"
    echo "浏览器已打开；这个终端窗口现在可以关闭。"
    ;;
  *)
    echo "用法：$0 [--serve|--open|--resolve-port]" >&2
    exit 2
    ;;
esac
