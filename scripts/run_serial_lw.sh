#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ./sw_serial_lw ]]; then
  bash scripts/build.sh
fi

: "${NX:=200}"
: "${NY:=200}"
: "${STEPS:=400}"
: "${DT:=0.01}"
: "${DX:=1.0}"
: "${DY:=1.0}"
: "${G:=9.81}"
: "${OUT:=outputs/h_final_serial_lw.csv}"
: "${NO_OUTPUT:=0}"

args=(--nx "$NX" --ny "$NY" --steps "$STEPS" --dt "$DT" --dx "$DX" --dy "$DY" --g "$G")
if [[ "$NO_OUTPUT" == "1" ]]; then
  args+=(--no-output)
else
  args+=(--out "$OUT")
fi

./sw_serial_lw "${args[@]}"
