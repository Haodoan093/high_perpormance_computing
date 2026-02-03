#!/usr/bin/env python3
"""Compute basic sanity metrics from a SWE output CSV.

No third-party dependencies.

Outputs:
- shape
- min/max
- sum (proxy for mass if dx*dy constant)
"""

import csv
import math
import sys
from typing import Tuple


def read_csv(path: str) -> Tuple[int, int, float, float, float]:
    nrows = 0
    ncols = None
    vmin = float("inf")
    vmax = float("-inf")
    vsum = 0.0

    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            nrows += 1
            if ncols is None:
                ncols = len(row)
            elif len(row) != ncols:
                raise ValueError(f"Non-rectangular CSV: expected {ncols} cols, got {len(row)}")

            for x in row:
                v = float(x)
                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v
                vsum += v

    if ncols is None:
        ncols = 0

    return nrows, ncols, vmin, vmax, vsum


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>")
        return 2

    path = sys.argv[1]
    nrows, ncols, vmin, vmax, vsum = read_csv(path)
    print(f"shape={nrows}x{ncols} min={vmin:.6e} max={vmax:.6e} sum={vsum:.10e}")

    # Basic sanity checks for water depth h
    if nrows == 0 or ncols == 0:
        print("FAIL: empty CSV")
        return 3
    if not math.isfinite(vmin) or not math.isfinite(vmax) or not math.isfinite(vsum):
        print("FAIL: non-finite values")
        return 4
    if vmin <= 0.0:
        print("FAIL: non-positive depth detected")
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
