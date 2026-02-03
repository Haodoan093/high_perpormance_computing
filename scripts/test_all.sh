#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Build binaries"
bash scripts/build.sh

echo "[2/4] Correctness: unified (LF/LW/MC) serial vs MPI"
# Same IC / BC / dt across schemes for fairness.
GRID_LIST="80x60 120x80" NP_LIST="1 2 4" STEPS=50 DT=0.01 EPS=1e-12 bash scripts/unified_correctness.sh

echo "[4/4] Benchmark: quick scaling run"
# Choose NX divisible by max NP (here 4) to avoid early-exit.
NX=240 NY=240 STEPS=120 DT=0.005 NP_LIST="1 2 4" BENCH_LW=1 BENCH_MC=1 bash scripts/bench.sh >/dev/null

echo "OK: tests passed. Check outputs/ and results/ for artifacts."