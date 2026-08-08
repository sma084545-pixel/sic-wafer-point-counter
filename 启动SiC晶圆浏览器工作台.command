#!/bin/bash
# 双击此文件即可准备运行环境、启动本机分析页面并打开默认浏览器。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP="$PROJECT_DIR/scripts/bootstrap_local_web_workbench.py"

for name in python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
  if command -v "$name" >/dev/null 2>&1; then
    exec "$(command -v "$name")" "$BOOTSTRAP"
  fi
done

echo "没有找到可用于安装引导的 python3。" >&2
echo "请从 https://www.python.org/downloads/macos/ 安装 Python 3.12 或更新版本，然后重新双击本文件。" >&2
exit 1
