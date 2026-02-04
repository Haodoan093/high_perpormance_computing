#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Build if needed
if [[ ! -x ./sw_serial || ! -x ./sw_mpi ]]; then
  bash scripts/build.sh >/dev/null
fi

: "${DT:=0.01}"
: "${DX:=1.0}"
: "${DY:=1.0}"
: "${G:=9.81}"
: "${STEPS:=60}"

# Default covers the common MPI counts.
: "${NP_LIST:=1 2 4}"

# Exactly 10 grids by default; ensure NX divisible by all np in NP_LIST (default includes 4).
# Format: "NXxNY" (space-separated)
: "${GRID_LIST:=80x60 100x80 120x80 160x120 200x150 240x180 280x200 320x240 360x260 480x360}"

: "${EPS:=1e-12}"
: "${MPIRUN:=mpirun}"
: "${MPIRUN_ARGS:=}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/correctness_lf_only_${TS}"
mkdir -p "$OUT_DIR"

RESULTS_CSV="$OUT_DIR/results.csv"
echo "nx,ny,np,steps,dt,dx,dy,g,eps,serial_time_s,mpi_time_s,speedup,max_abs_diff,mean_abs_diff,max_rel_diff,n,serial_csv,mpi_csv,pass" >"$RESULTS_CSV"

echo "[correctness_lf_only] out=$OUT_DIR" >&2

total=0
skipped=0

for grid in $GRID_LIST; do
  nx="${grid%x*}"
  ny="${grid#*x}"

  for np in $NP_LIST; do
    if (( nx % np != 0 )); then
      echo "[correctness_lf_only] skip grid=$grid np=$np (nx not divisible)" >&2
      skipped=$((skipped+1))
      continue
    fi

    out_serial="$OUT_DIR/serial_${nx}x${ny}.csv"
    out_mpi="$OUT_DIR/mpi_np${np}_${nx}x${ny}.csv"

    serial_log="$(./sw_serial \
      --nx "$nx" --ny "$ny" --steps "$STEPS" \
      --dt "$DT" --dx "$DX" --dy "$DY" --g "$G" \
      --out "$out_serial" 2>&1)"

    serial_time_s="$(echo "$serial_log" | sed -n 's/.*finished in \([0-9.][0-9.]*\) s.*/\1/p' | head -n 1)"

    mpi_log="$($MPIRUN $MPIRUN_ARGS -np "$np" ./sw_mpi \
      --nx "$nx" --ny "$ny" --steps "$STEPS" \
      --dt "$DT" --dx "$DX" --dy "$DY" --g "$G" \
      --out "$out_mpi" 2>&1)"

    mpi_time_s="$(echo "$mpi_log" | sed -n 's/.*finished in \([0-9.][0-9.]*\) s.*/\1/p' | head -n 1)"

    speedup="$(awk -v s="$serial_time_s" -v m="$mpi_time_s" 'BEGIN{ if(s==""||m==""||m==0){print ""} else {printf "%.6f", s/m} }')"

    echo "[correctness_lf_only] grid=$grid np=$np" >&2
    python3 scripts/check_csv_invariants.py "$out_serial" >/dev/null
    python3 scripts/check_csv_invariants.py "$out_mpi" >/dev/null

    set +e
    cmp_out="$(EPS="$EPS" python3 scripts/compare_csvs.py "$out_serial" "$out_mpi" 2>&1)"
    cmp_rc=$?
    set -e

    # compare_csvs.py prints one line like:
    # max_abs_diff=... mean_abs_diff=... max_rel_diff=... n=... EPS=...
    max_abs="$(echo "$cmp_out" | sed -n 's/.*max_abs_diff=\([^ ]*\).*/\1/p')"
    mean_abs="$(echo "$cmp_out" | sed -n 's/.*mean_abs_diff=\([^ ]*\).*/\1/p')"
    max_rel="$(echo "$cmp_out" | sed -n 's/.*max_rel_diff=\([^ ]*\).*/\1/p')"
    ncount="$(echo "$cmp_out" | sed -n 's/.* n=\([^ ]*\).*/\1/p')"

    pass=0
    if [[ $cmp_rc -eq 0 ]]; then pass=1; fi

    echo "${nx},${ny},${np},${STEPS},${DT},${DX},${DY},${G},${EPS},${serial_time_s},${mpi_time_s},${speedup},${max_abs},${mean_abs},${max_rel},${ncount},${out_serial},${out_mpi},${pass}" >>"$RESULTS_CSV"

    echo "$cmp_out"

    if [[ $cmp_rc -ne 0 ]]; then
      echo "FAIL: compare_csvs exited $cmp_rc (see $RESULTS_CSV)" >&2
      exit $cmp_rc
    fi

    total=$((total+1))
  done
done

echo "OK: lf-only correctness passed (ran=$total skipped=$skipped) out=$OUT_DIR results=$RESULTS_CSV" >&2
