#!/usr/bin/env python3
"""Compare two SWE output CSVs (same shape) and report error metrics.

Prints:
- max_abs_diff
- mean_abs_diff
- max_rel_diff   (relative to max(1,|a|) to avoid div-by-zero)

Exit status:
- 0 if max_abs_diff < EPS
- 1 if max_abs_diff >= EPS
- 2 for shape mismatch

EPS can be set via env var EPS (default 1e-12).

No third-party dependencies.
"""

import csv
import os
import sys
from typing import List


def read_csv(path: str) -> List[List[float]]:
    with open(path, newline="") as f:
        return [[float(x) for x in row] for row in csv.reader(f) if row]


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <a.csv> <b.csv>")
        return 2

    a_path, b_path = sys.argv[1], sys.argv[2]
    a = read_csv(a_path)
    b = read_csv(b_path)

    if len(a) != len(b) or any(len(ar) != len(br) for ar, br in zip(a, b)):
        print("Shape mismatch")
        return 2

    max_abs = 0.0
    max_rel = 0.0
    sum_abs = 0.0
    count = 0

    for ar, br in zip(a, b):
        for av, bv in zip(ar, br):
            d = abs(av - bv)
            if d > max_abs:
                max_abs = d
            denom = max(1.0, abs(av))
            r = d / denom
            if r > max_rel:
                max_rel = r
            sum_abs += d
            count += 1

    mean_abs = sum_abs / max(1, count)
    eps = float(os.environ.get("EPS", "1e-12"))

    print(
        f"max_abs_diff={max_abs:.3e} mean_abs_diff={mean_abs:.3e} "
        f"max_rel_diff={max_rel:.3e} n={count} EPS={eps:.1e}"
    )

    return 0 if max_abs < eps else 1


if __name__ == "__main__":
    raise SystemExit(main())
