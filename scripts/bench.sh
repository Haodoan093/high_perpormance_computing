#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ./sw_serial_schemes || ! -x ./sw_mpi || ! -x ./sw_mpi_lw || ! -x ./sw_mpi_mc ]]; then
  bash scripts/build.sh >/dev/null
fi

mkdir -p results outputs

: "${NX:=400}"
: "${NY:=400}"
: "${STEPS:=200}"
: "${DT:=0.005}"
: "${DX:=1.0}"
: "${DY:=1.0}"
: "${G:=9.81}"

# Example: NP_LIST="1 2 4 8"
: "${NP_LIST:=1 2 4}"

# If set to 1, also benchmark LW / MC (default off to keep runtime small)
: "${BENCH_LW:=0}"
: "${BENCH_MC:=0}"

# When 1 (default), benchmark compute only (skip CSV gather/write)
: "${NO_OUTPUT:=1}"

: "${MPIRUN:=mpirun}"
: "${MPIRUN_ARGS:=}"

TS="$(date +%Y%m%d_%H%M%S)"
: "${BENCH_OUT:=results/bench_${TS}.csv}"
OUT_CSV="$BENCH_OUT"

extract_time_s() {
  python3 -c 'import re,sys; s=sys.stdin.read(); m=re.search(r"finished in\s+([0-9eE+\-.]+)\s+s", s);\
print(m.group(1) if m else ""); sys.exit(0 if m else 2)'
}

extract_hmin_hmax() {
  python3 -c 'import re,sys; s=sys.stdin.read(); m=re.search(r"h min/max\s*=\s*([0-9eE+\-.]+)\s*/\s*([0-9eE+\-.]+)", s);\
print((f"{m.group(1)},{m.group(2)}") if m else ","); sys.exit(0 if m else 2)'
}

calc_speedup_eff() {
  python3 - "$1" "$2" "$3" <<'PY'
import sys
# args: t_seq, t_mpi, np
try:
    t_seq=float(sys.argv[1]); t_mpi=float(sys.argv[2]); np=int(sys.argv[3])
except Exception:
    print(",")
    sys.exit(2)
if t_mpi<=0 or np<=0:
    print(",")
    sys.exit(0)
speedup=t_seq/t_mpi
eta=t_seq/(t_mpi*np)
print(f"{speedup},{eta}")
PY
}

schemes=("lf")
if [[ "$BENCH_LW" == "1" ]]; then schemes+=("lw"); fi
if [[ "$BENCH_MC" == "1" ]]; then schemes+=("mc"); fi

{
  echo "scheme,impl,np,nx,ny,steps,dt,dx,dy,g,time_s,t_seq_s,speedup,efficiency,hmin,hmax,no_output"

  for scheme in "${schemes[@]}"; do
    # Serial baseline
    serial_cmd=(./sw_serial_schemes --scheme "$scheme" --nx "$NX" --ny "$NY" --steps "$STEPS" --dt "$DT" --dx "$DX" --dy "$DY" --g "$G")
    if [[ "$NO_OUTPUT" == "1" ]]; then serial_cmd+=(--no-output); fi

    serial_out="$(${serial_cmd[@]} 2>&1)"
    t_seq="$(printf "%s" "$serial_out" | extract_time_s)"
    hpair="$(printf "%s" "$serial_out" | extract_hmin_hmax)"
    hmin="${hpair%,*}"; hmax="${hpair#*,}"

    echo "$scheme,serial,1,$NX,$NY,$STEPS,$DT,$DX,$DY,$G,$t_seq,$t_seq,1,1,$hmin,$hmax,$NO_OUTPUT"

    # MPI runs
    for np in $NP_LIST; do
      if (( np < 1 )); then continue; fi
      if (( NX % np != 0 )); then
        echo "[bench] skip scheme=$scheme np=$np (NX=$NX not divisible)" >&2
        continue
      fi

      if [[ "$scheme" == "lf" ]]; then exe=./sw_mpi
      elif [[ "$scheme" == "lw" ]]; then exe=./sw_mpi_lw
      else exe=./sw_mpi_mc
      fi

      mpi_cmd=($MPIRUN $MPIRUN_ARGS -np "$np" "$exe" --nx "$NX" --ny "$NY" --steps "$STEPS" --dt "$DT" --dx "$DX" --dy "$DY" --g "$G")
      if [[ "$NO_OUTPUT" == "1" ]]; then mpi_cmd+=(--no-output); fi

      mpi_out="$(${mpi_cmd[@]} 2>&1)"
      t_mpi="$(printf "%s" "$mpi_out" | extract_time_s)"
      hpair2="$(printf "%s" "$mpi_out" | extract_hmin_hmax)"
      hmin2="${hpair2%,*}"; hmax2="${hpair2#*,}"

      sp_eff="$(calc_speedup_eff "$t_seq" "$t_mpi" "$np")"
      speedup="${sp_eff%,*}"; eff="${sp_eff#*,}"

      echo "$scheme,mpi,$np,$NX,$NY,$STEPS,$DT,$DX,$DY,$G,$t_mpi,$t_seq,$speedup,$eff,$hmin2,$hmax2,$NO_OUTPUT"
    done
  done
} | tee "$OUT_CSV"

echo "Wrote $OUT_CSV" >&2
