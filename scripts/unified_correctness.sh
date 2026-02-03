#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/build.sh >/dev/null

: "${DT:=0.01}"
: "${DX:=1.0}"
: "${DY:=1.0}"
: "${G:=9.81}"
: "${STEPS:=60}"

: "${NP_LIST:=1 2 4}"
# Space-separated list of grids like: "80x60 120x80"
: "${GRID_LIST:=80x60 120x80}"

: "${EPS:=1e-12}"
: "${MPIRUN:=mpirun}"
: "${MPIRUN_ARGS:=}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/correctness_${TS}"
mkdir -p "$OUT_DIR"

echo "[correctness] out=$OUT_DIR" >&2

to_exe() {
  local scheme="$1"
  if [[ "$scheme" == "lf" ]]; then echo "./sw_mpi"; return; fi
  if [[ "$scheme" == "lw" ]]; then echo "./sw_mpi_lw"; return; fi
  echo "./sw_mpi_mc"
}

for grid in $GRID_LIST; do
  nx="${grid%x*}"
  ny="${grid#*x}"

  for scheme in lf lw mc; do
    scheme_dir="$OUT_DIR/$scheme/${nx}x${ny}"
    mkdir -p "$scheme_dir"

    out_serial="$scheme_dir/serial.csv"
    ./sw_serial_schemes --scheme "$scheme" \
      --nx "$nx" --ny "$ny" --steps "$STEPS" \
      --dt "$DT" --dx "$DX" --dy "$DY" --g "$G" \
      --out "$out_serial" >/dev/null

    for np in $NP_LIST; do
      if (( nx % np != 0 )); then
        echo "[correctness] skip $scheme grid=$grid np=$np (nx not divisible)" >&2
        continue
      fi

      out_mpi="$scheme_dir/mpi_np${np}.csv"
      exe="$(to_exe "$scheme")"

      $MPIRUN $MPIRUN_ARGS -np "$np" "$exe" \
        --nx "$nx" --ny "$ny" --steps "$STEPS" \
        --dt "$DT" --dx "$DX" --dy "$DY" --g "$G" \
        --out "$out_mpi" >/dev/null

      echo "[correctness] scheme=$scheme grid=$grid np=$np" >&2
      python3 scripts/check_csv_invariants.py "$out_serial" >/dev/null
      python3 scripts/check_csv_invariants.py "$out_mpi" >/dev/null
      EPS="$EPS" python3 scripts/compare_csvs.py "$out_serial" "$out_mpi"
    done
  done
done

echo "OK: correctness suite passed ($OUT_DIR)" >&2
