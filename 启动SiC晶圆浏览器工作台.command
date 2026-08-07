#!/bin/bash
# 双击此文件即可启动本机分析页面并在默认浏览器中打开。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /bin/bash "$PROJECT_DIR/scripts/run_local_web_workbench.sh" --open
