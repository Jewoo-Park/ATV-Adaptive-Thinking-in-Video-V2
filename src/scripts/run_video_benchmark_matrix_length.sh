#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export REASONING_TASK_TYPE="length"
bash src/scripts/run_video_benchmark_matrix.sh
