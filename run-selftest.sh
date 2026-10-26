#!/usr/bin/env bash
# PowerContext 2026 竞赛 —— Windows 本地自测运行器
#
# 官方要求 Linux x86_64，这里做了 Windows 适配：
#   1. 依赖清单里的 platform 标记被剥离（仓库根目录 requirements.win.txt）
#   2. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 在 Windows 上会让 pytest 静默退出，故不使用
#   3. OPENBLAS_NUM_THREADS=1 / OMP_NUM_THREADS=1 必须设置，否则 OpenBLAS 内存分配失败导致静默崩溃
#
# 用法：
#   ./run-selftest.sh              # 跑全部
#   ./run-selftest.sh csv          # 跑单题
#   可选: csv pi pdf rerank incremental count-trigger embedding lifecycle evidence hybrid

set -u

# Git Bash 的 pwd 返回 /d/... 形式，Windows 原生 Python 只认 D:/...，故用 pwd -W
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W 2>/dev/null || pwd)"
COMP="$ROOT/competition"

export POWERCONTEXT_CANDIDATE="$ROOT"
export POWERCONTEXT_PUBLIC="$COMP"
export PYTHONPATH="$ROOT/src;$COMP"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONUTF8=1
export TZ=UTC
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PY="$COMP/.venv/Scripts/python.exe"
[ -x "$PY" ] || { echo "缺少虚拟环境：先跑 ./install-win.sh"; exit 1; }

declare -A DIRS=(
  [csv]=source-01-table-evidence
  [pi]=source-02-pi-history
  [pdf]=source-03-pdf-retrieval
  [rerank]=retrieval-01-reranking
  [hybrid]=retrieval-02-hybrid-search
  [evidence]=context-01-evidence-selection
  [incremental]=context-02-incremental-pack
  [count-trigger]=processing-01-count-trigger
  [embedding]=memory-02-embedding-batches
  [lifecycle]=memory-01-source-lifecycle
)

run_one() {
  local key="$1" dir="${DIRS[$1]}"
  local out
  out="$(cd "$COMP" && "$PY" -m pytest -q \
      -o addopts="--import-mode=importlib" \
      "challenges/$dir/acceptance" 2>&1)"
  local code=$?
  local line
  line="$(printf '%s\n' "$out" | grep -E '^[0-9]+ (passed|failed)|passed|failed|error' | tail -1)"
  if [ $code -eq 0 ]; then
    printf '  \033[32m✔\033[0m %-15s %s\n' "$key" "${line:-通过}"
  else
    printf '  \033[31m✘\033[0m %-15s %s\n' "$key" "${line:-失败}"
  fi
  printf '%s\n' "$out" > "/tmp/selftest-$key.log"
  return $code
}

if [ $# -gt 0 ]; then
  # 支持一次给多个题：./run-selftest.sh csv pi pdf
  fail=0
  for key in "$@"; do
    if [ -z "${DIRS[$key]:-}" ]; then
      printf '  \033[33m?\033[0m %-15s 未知题名（可选: %s）\n' "$key" "${!DIRS[*]}"
      fail=$((fail + 1))
      continue
    fi
    run_one "$key" || fail=$((fail + 1))
  done
  exit $fail
fi

echo "PowerContext 2026 自测（Windows 适配环境）"
echo "----------------------------------------"
fail=0
for key in csv pi pdf rerank hybrid evidence incremental count-trigger embedding lifecycle; do
  run_one "$key" || fail=$((fail + 1))
done
echo "----------------------------------------"
echo "失败 $fail 题；单题详细日志见 /tmp/selftest-<题>.log"
exit $fail
