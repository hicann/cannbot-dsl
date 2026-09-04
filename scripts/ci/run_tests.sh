#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
LOG_DIR=""
MODE=""
OPS_LIST=""

# torch_npu 依赖 libhccl 等动态库，需先加载 CANN 环境
source_cann_env() {
    local candidates=(
        "${ASCEND_TOOLKIT_HOME:-}/set_env.sh"
        "$HOME/Ascend/ascend-toolkit/set_env.sh"
        "/usr/local/Ascend/cann-9.2.0/set_env.sh"
        "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    )
    for env_sh in "${candidates[@]}"; do
        if [[ -f "$env_sh" ]]; then
            set +u
            # shellcheck disable=SC1090
            source "$env_sh"
            set -u
            return 0
        fi
    done
    echo "[WARN] ascend-toolkit set_env.sh not found; torch_npu may fail to import"
}

source_cann_env

# torch 2.12 auto-loads torch_npu through an entry point.  Loading it before
# CANN is initialized can fail on libhccl, and repeated collection can register
# the same torch library twice.  NPU tests import torch_npu explicitly later.
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"

usage() {
    cat <<EOF
Usage: bash scripts/ci/run_tests.sh [options]
Options:
    --ops=op1,op2   Operators to test (comma separated). Default: all.
    --mode=cpu|npu|all
                    cpu: run tests not marked 'npu' (no device needed)
                    npu: run only 'npu' marked tests (device needed)
                    all: run everything
                    Default: auto-detect via NPU device count.
    --log-dir=DIR   Directory for pytest log and junit.xml. Default: \${REPO_ROOT}/log
    -h|--help       Print this help.
EOF
}

parse_args() {
    for arg in "$@"; do
        case "$arg" in
            --ops=*)
                OPS_LIST="${arg#*=}"
                ;;
            --mode=*)
                MODE="${arg#*=}"
                ;;
            --log-dir=*)
                LOG_DIR="${arg#*=}"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "[ERROR] unknown argument: $arg" >&2
                usage
                exit 1
                ;;
        esac
    done
}

detect_npu_count() {
    "$PYTHON" - <<'EOF' 2>/dev/null || echo 0
try:
    import torch_npu
    print(torch_npu.npu.device_count())
except Exception:
    print(0)
EOF
}

build_test_targets() {
    local ops_list="$1"
    TEST_TARGETS=()
    if [[ -z "$ops_list" || "$ops_list" == "all" ]]; then
        TEST_TARGETS=("test")
        return
    fi
    IFS=',' read -ra ops <<< "$ops_list"
    for op in "${ops[@]}"; do
        TEST_TARGETS+=("test/${op}")
    done
}

print_summary() {
    local junit="$1"
    "$PYTHON" - "$junit" <<'EOF'
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
try:
    root = ET.parse(path).getroot()
except Exception:
    print("[SUMMARY] junit.xml unavailable")
    sys.exit(0)

# pytest writes <testsuites><testsuite .../></testsuites>; pick the suite carrying the counts
suite = root
if suite.tag == "testsuites" and len(suite) > 0:
    suite = suite[0]

tests = int(suite.get("tests", 0))
failures = int(suite.get("failures", 0))
errors = int(suite.get("errors", 0))
skipped = int(suite.get("skipped", 0))
passed = tests - failures - errors - skipped
rate = passed / tests * 100 if tests else 0.0
print(f"[SUMMARY] TOTAL: {tests} | PASSED: {passed} | FAILED: {failures} | "
      f"ERRORS: {errors} | SKIPPED: {skipped} | RATE: {rate:.2f}%")
if failures or errors:
    sys.exit(1)
EOF
}

parse_args "$@"
[[ -n "$LOG_DIR" ]] || LOG_DIR="${REPO_ROOT}/log"
mkdir -p "$LOG_DIR"

if [[ -z "$MODE" ]]; then
    if [[ "$(detect_npu_count)" -gt 0 ]]; then
        MODE="npu"
    else
        MODE="cpu"
    fi
    echo "[INFO] auto-detected mode: ${MODE}"
fi

case "$MODE" in
    cpu)
        MARKERS=(-m "not npu")
        ;;
    npu)
        MARKERS=(-m "npu")
        ;;
    all)
        MARKERS=()
        ;;
    *)
        echo "[ERROR] unsupported mode: ${MODE}. Supported: cpu, npu, all" >&2
        exit 1
        ;;
esac

build_test_targets "$OPS_LIST"
echo "[INFO] mode: ${MODE}, targets: ${TEST_TARGETS[*]}"
echo "[INFO] running pytest ..."

JUNIT="${LOG_DIR}/junit.xml"
cd "$REPO_ROOT"
set +e
"$PYTHON" -m pytest "${TEST_TARGETS[@]}" "${MARKERS[@]}" \
    --junitxml="$JUNIT" 2>&1 | tee "${LOG_DIR}/pytest.log"
rc=${PIPESTATUS[0]}
set -e

print_summary "$JUNIT"
if [[ "$rc" -eq 5 ]]; then
    echo "[INFO] no tests collected in this mode, treated as pass"
    exit 0
fi
exit "$rc"
