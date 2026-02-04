#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${GRID_LIST_1:=40x30 80x60 120x80}"
: "${GRID_LIST_2:=160x120 240x180}"
: "${NP_LIST_CORRECTNESS:=1 2 4}"
: "${STEPS_1:=50}"
: "${DT_1:=0.01}"
: "${STEPS_2:=80}"
: "${DT_2:=0.005}"
: "${EPS:=1e-12}"

echo "[1/4] Build binaries"
bash scripts/build.sh

echo "[2/4] Correctness: unified (LF/LW/MC) serial vs MPI (config 1)"
# Same IC / BC / dt across schemes for fairness.
GRID_LIST="$GRID_LIST_1" NP_LIST="$NP_LIST_CORRECTNESS" STEPS="$STEPS_1" DT="$DT_1" EPS="$EPS" bash scripts/unified_correctness.sh

echo "[3/4] Correctness: unified (LF/LW/MC) serial vs MPI (config 2)"
GRID_LIST="$GRID_LIST_2" NP_LIST="$NP_LIST_CORRECTNESS" STEPS="$STEPS_2" DT="$DT_2" EPS="$EPS" bash scripts/unified_correctness.sh

echo "[4/4] Benchmark: quick scaling run"
# Choose NX divisible by max NP (here 4) to avoid early-exit.
NX=240 NY=240 STEPS=120 DT=0.005 NP_LIST="1 2 4" BENCH_LW=1 BENCH_MC=1 bash scripts/bench.sh >/dev/null

echo "OK: tests passed. Check outputs/ and results/ for artifacts."