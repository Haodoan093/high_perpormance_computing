#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser(description="Pivot bench CSV (long format) into a wide table like: grid, seq_time, mpi_time(np)(eff).")
    ap.add_argument("bench_csv", help="Input CSV produced by scripts/bench.sh")
    ap.add_argument("--scheme", default="lf", choices=["lf", "lw", "mc"], help="Scheme to pivot")
    ap.add_argument("--nps", default="1,2,4", help="Comma-separated NP values to include")
    ap.add_argument("--out", default="-", help="Output CSV path (default: stdout)")
    return ap.parse_args()

def main():
    args = parse_args()
    wanted_nps = [int(x) for x in args.nps.split(",") if x.strip()]

    # key: (nx, ny) -> dict
    seq_time = {}
    mpi = defaultdict(dict)  # (nx,ny)[np] -> (time, eff)

    with open(args.bench_csv, newline="") as f:
        r = csv.DictReader(f)
        required = {"scheme", "impl", "np", "nx", "ny", "time_s", "efficiency"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns in input CSV: {sorted(missing)}")

        for row in r:
            if row["scheme"] != args.scheme:
                continue
            nx = int(row["nx"])
            ny = int(row["ny"])
            impl = row["impl"]
            np = int(float(row["np"]))
            t = float(row["time_s"])
            eff = float(row["efficiency"]) if row["efficiency"] not in ("", None) else float("nan")

            if impl == "serial":
                seq_time[(nx, ny)] = t
            elif impl == "mpi":
                mpi[(nx, ny)][np] = (t, eff)

    out_fields = ["grid", "nx", "ny", "seq_time_s"]
    for np in wanted_nps:
        out_fields += [f"mpi_np{np}_time_s", f"mpi_np{np}_efficiency"]

    rows = []
    for (nx, ny), t0 in sorted(seq_time.items()):
        row = {
            "grid": f"{nx}x{ny}",
            "nx": nx,
            "ny": ny,
            "seq_time_s": f"{t0:.6f}",
        }
        for np in wanted_nps:
            t_eff = mpi.get((nx, ny), {}).get(np)
            if t_eff is None:
                row[f"mpi_np{np}_time_s"] = ""
                row[f"mpi_np{np}_efficiency"] = ""
            else:
                t, eff = t_eff
                row[f"mpi_np{np}_time_s"] = f"{t:.6f}"
                row[f"mpi_np{np}_efficiency"] = f"{eff:.6f}"
        rows.append(row)

    if args.out == "-":
        w = csv.DictWriter(sys.stdout, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)
    else:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=out_fields)
            w.writeheader()
            w.writerows(rows)

if __name__ == "__main__":
    import sys
    main()
