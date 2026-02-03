#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/build.sh >/dev/null

: "${DT:=0.005}"
: "${DX:=1.0}"
: "${DY:=1.0}"
: "${G:=9.81}"
: "${STEPS:=200}"

: "${NP_LIST:=1 2 4}"
: "${BENCH_LW:=1}"
: "${BENCH_MC:=1}"
: "${NO_OUTPUT:=1}"

TS="$(date +%Y%m%d_%H%M%S)"
RES_DIR="results/sweep_${TS}"
mkdir -p "$RES_DIR"

OUT_LONG="$RES_DIR/bench_long.csv"
first=1

for n in $(seq 200 200 1800); do
  tmp="$RES_DIR/bench_${n}x${n}.csv"
  NX="$n" NY="$n" STEPS="$STEPS" DT="$DT" DX="$DX" DY="$DY" G="$G" \
    NP_LIST="$NP_LIST" BENCH_LW="$BENCH_LW" BENCH_MC="$BENCH_MC" NO_OUTPUT="$NO_OUTPUT" \
    BENCH_OUT="$tmp" bash scripts/bench.sh >/dev/null

  if (( first == 1 )); then
    cat "$tmp" >"$OUT_LONG"
    first=0
  else
    tail -n +2 "$tmp" >>"$OUT_LONG"
  fi

done

# Generate pivot tables for plotting/report convenience
python3 scripts/pivot_bench_table.py "$OUT_LONG" --scheme lf --nps "$(echo "$NP_LIST" | tr ' ' ',')" --out "$RES_DIR/table_lf.csv" || true
python3 scripts/pivot_bench_table.py "$OUT_LONG" --scheme lw --nps "$(echo "$NP_LIST" | tr ' ' ',')" --out "$RES_DIR/table_lw.csv" || true
python3 scripts/pivot_bench_table.py "$OUT_LONG" --scheme mc --nps "$(echo "$NP_LIST" | tr ' ' ',')" --out "$RES_DIR/table_mc.csv" || true

echo "Wrote $OUT_LONG" >&2
