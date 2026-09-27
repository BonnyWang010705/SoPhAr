"""Collect the optimized shift of every flight at the three altitudes.

Reads results/optimization_1/optimized_flight_shifts[tag]_R1.csv for 9,100,
12,100 and 15,100 m (1_schedule_optimization.py) and writes
results/optimization_1/optimized_shift_flight_merged_R1.csv: the flight table
with one shift column per altitude (time_shift_seconds_12100m, _9100m,
_15100m; empty when the flight crosses no qualified farm at that altitude).
Fig. 5d-f are drawn from this table (code/4_figures/fig5def.py).
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "2_coverage_and_savings"))

import model    # noqa: E402

OUT = os.path.join(model.OPT1, "optimized_shift_flight_merged_R1.csv")


def per_flight(altitude):
    path = model.shift_file(altitude)
    if not os.path.exists(path):
        raise SystemExit("missing %s\nrun: python code/3_optimization/"
                         "1_schedule_optimization.py --altitude %d" % (path, altitude))
    shifts = pd.read_csv(path)[["Trip_ID", "time_shift_seconds"]]
    # a flight has one shift, repeated on each of its crossings
    return shifts.drop_duplicates(subset=["Trip_ID"])


def main():
    merged = pd.merge(per_flight(12100), per_flight(9100), on="Trip_ID", how="outer",
                      suffixes=("_12100m", "_9100m"))
    merged = pd.merge(merged, per_flight(15100), on="Trip_ID", how="outer")
    merged = merged.rename(columns={"time_shift_seconds": "time_shift_seconds_15100m"})

    flights = pd.merge(model.load_flights(), merged, on="Trip_ID", how="left")
    flights.to_csv(OUT, index=False)
    print("flights with a shift: 12,100 m %s, 9,100 m %s, 15,100 m %s"
          % tuple(format(flights["time_shift_seconds_%dm" % h].notna().sum(), ",")
                  for h in (12100, 9100, 15100)))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
