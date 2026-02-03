#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ./sw_serial || ! -x ./sw_mpi ]]; then
  bash scripts/build.sh >/dev/null
fi

: "${NP:=4}"
: "${NX:=120}"
: "${NY:=120}"
: "${STEPS:=50}"
: "${DT:=0.01}"
: "${DX:=1.0}"
: "${DY:=1.0}"
: "${G:=9.81}"
: "${EPS:=1e-12}"

: "${OUT_SERIAL:=outputs/verify_serial.csv}"
: "${OUT_MPI:=outputs/verify_mpi.csv}"

: "${MPIRUN:=mpirun}"
: "${MPIRUN_ARGS:=}"

mkdir -p "$(dirname "$OUT_SERIAL")" "$(dirname "$OUT_MPI")" outputs results

if (( NX % NP != 0 )); then
  echo "[verify] SKIP: NX=$NX not divisible by NP=$NP" >&2
  exit 0
fi

rm -f "$OUT_SERIAL" "$OUT_MPI"

./sw_serial \
  --nx "$NX" --ny "$NY" --steps "$STEPS" \
  --dt "$DT" --dx "$DX" --dy "$DY" --g "$G" \
  --out "$OUT_SERIAL" >/dev/null

$MPIRUN $MPIRUN_ARGS -np "$NP" ./sw_mpi \
  --nx "$NX" --ny "$NY" --steps "$STEPS" \
  --dt "$DT" --dx "$DX" --dy "$DY" --g "$G" \
  --out "$OUT_MPI" >/dev/null

python3 scripts/check_csv_invariants.py "$OUT_SERIAL"
python3 scripts/check_csv_invariants.py "$OUT_MPI"

EPS="$EPS" python3 scripts/compare_csvs.py "$OUT_SERIAL" "$OUT_MPI"
