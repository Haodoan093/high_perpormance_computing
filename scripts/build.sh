#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

CC_SERIAL=${CC_SERIAL:-gcc}
CC_MPI=${CC_MPI:-mpicc}
CFLAGS=${CFLAGS:-"-O3 -std=c11 -Wall -Wextra -Wshadow -Wconversion -Wno-unused-parameter"}

echo "[build] Building serial (LF)..."
$CC_SERIAL $CFLAGS -o sw_serial shallow_water_conservative_serial.c -lm

echo "[build] Building serial (LW)..."
$CC_SERIAL $CFLAGS -o sw_serial_lw shallow_water_conservative_serial_lw.c -lm

echo "[build] Building serial (MC)..."
$CC_SERIAL $CFLAGS -o sw_serial_mc shallow_water_conservative_serial_mc.c -lm

echo "[build] Building serial (LF/LW/MC unified)..."
$CC_SERIAL $CFLAGS -o sw_serial_schemes shallow_water_conservative_serial_schemes.c -lm

echo "[build] Building MPI (LF)..."
$CC_MPI $CFLAGS -o sw_mpi shallow_water_conservative_mpi.c -lm

echo "[build] Building MPI (LW)..."
$CC_MPI $CFLAGS -o sw_mpi_lw shallow_water_conservative_mpi_lw.c -lm

echo "[build] Building MPI (MC)..."
$CC_MPI $CFLAGS -o sw_mpi_mc shallow_water_conservative_mpi_mc.c -lm

echo "Built:"
echo "  $ROOT_DIR/sw_serial"
echo "  $ROOT_DIR/sw_serial_lw"
echo "  $ROOT_DIR/sw_serial_mc"
echo "  $ROOT_DIR/sw_serial_schemes"
echo "  $ROOT_DIR/sw_mpi"
echo "  $ROOT_DIR/sw_mpi_lw"
echo "  $ROOT_DIR/sw_mpi_mc"