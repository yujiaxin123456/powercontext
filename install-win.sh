#!/usr/bin/env bash
# PowerContext 2026 竞赛 —— Windows 本地自测环境安装器
#
# 官方 requirements.txt 里每个包都带 sys_platform == 'linux' 标记，在 Windows 上会被
# 全部跳过；pylibseekdb 也只发布 Linux/macOS 轮子。这里用剥掉标记的 requirements.win.txt
# 建环境，只用于本地跑 SQLite 路径的验收测试（retrieval-02 混合检索必须有 seekdb，跑不了）。
#
# 用法： ./install-win.sh
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W 2>/dev/null || pwd)"
COMP="$ROOT/competition"
VENV="$COMP/.venv"
REQ="$ROOT/requirements.win.txt"
PY="$VENV/Scripts/python.exe"

[ -f "$REQ" ] || { echo "缺少 $REQ"; exit 1; }

if command -v uv >/dev/null 2>&1; then
  echo "==> 用 uv 创建 Python 3.12 虚拟环境：$VENV"
  uv venv --python 3.12 "$VENV"
  echo "==> 安装 Windows 依赖清单"
  uv pip install --python "$PY" -r "$REQ"
else
  echo "==> 未找到 uv，回退到 python -m venv（需要本机已装 Python 3.12）"
  python -m venv "$VENV"
  "$VENV/Scripts/python.exe" -m pip install --upgrade pip
  "$VENV/Scripts/python.exe" -m pip install -r "$REQ"
fi

# 源码导入期就需要这三个：OceanBase 向量列 / 原生连接驱动
echo "==> 补装 OceanBase 连接依赖"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PY" pyobvector pymysql aiomysql
else
  "$VENV/Scripts/python.exe" -m pip install pyobvector pymysql aiomysql
fi

echo
echo "==> 自检"
"$PY" -c "import powercontext, sys; print('powercontext ok on', sys.version.split()[0])" 2>/dev/null \
  || PYTHONPATH="$ROOT/src;$COMP" "$PY" -c "import powercontext, sys; print('powercontext ok on', sys.version.split()[0])"

echo
echo "完成。接着跑： ./run-selftest.sh"
echo "注意：retrieval-02-hybrid-search 需要 pylibseekdb（仅 Linux/macOS），Windows 上无法验收。"
