"""Flight schedule optimization at one cruise altitude (Fig. 5).

All qualified farms and all flights take part. Each flight may be shifted by
up to +/-1,800 s; the shifts minimise the time during which flights crossing
the same farm compete for its capacity (solvers.py). The power model is then
rerun on the shifted crossings.

    python 1_schedule_optimization.py                          # 12,100 m, Adam
    python 1_schedule_optimization.py --altitude 9100
    python 1_schedule_optimization.py --altitude 15100
    python 1_schedule_optimization.py --solver gurobi          # needs gurobipy

Reads results/baseline/flight_solar_overlap_pairs[tag]_R1.csv and writes to
results/optimization_1/:

    optimized_flight_shifts[tag]_R1.csv                        shift of every crossing
    Optimization_1_solar_farm_analysis_results[tag]_merged_R1.csv
    Optimization_1_flight_analysis_results[tag]_merged_R1.csv

The paper's shifts were computed with Adam on a GPU. The learning rate and
number of epochs of each altitude are the defaults below; a CPU run, or
another GPU, can end with shifts that differ slightly.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "2_coverage_and_savings"))

import model      # noqa: E402
import solvers    # noqa: E402

# Adam settings of the R1 run: altitude -> (epochs, learning rate)
ADAM = {9100: (10000, 70.0), 12100: (9000, 160.0), 15100: (10000, 180.0)}
SEED = 2026


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--altitude", type=int, default=model.REFERENCE_ALTITUDE,
                    choices=model.ALTITUDES, help="cruise altitude in m (default 12100)")
    ap.add_argument("--solver", choices=("adam", "gurobi"), default="adam",
                    help="adam (default, as in the paper) or gurobi (MILP, rolling horizon)")
    ap.add_argument("--epochs", type=int, help="Adam epochs (default: the R1 setting)")
    ap.add_argument("--lr", type=float, help="Adam learning rate (default: the R1 setting)")
    ap.add_argument("--device", help="torch device, e.g. cpu or cuda (default: cuda if available)")
    ap.add_argument("--window-hours", type=float, default=2,
                    help="Gurobi: length of each rolling-horizon window (default 2)")
    ap.add_argument("--time-limit", type=float,
                    help="Gurobi: time limit per window in seconds (default none)")
    ap.add_argument("--threads", type=int, default=4, help="Gurobi: threads (default 4)")
    a = ap.parse_args()
    started = time.time()
    altitude_km = a.altitude / 1000

    gdf_solar, gdf_solar_proj, farm_centers, farm_details = model.load_solar_farms()
    df_flights = model.load_flights()
    df_results = model.load_overlaps(a.altitude, gdf_solar_proj["FID"].unique(),
                                     gdf_solar=gdf_solar)

    # crossings in whole seconds from the first entry
    df = df_results[["Trip_ID", "Solar_Farm_ID", "Entry_Time", "Exit_Time"]].copy()
    df["Entry_Time"] = pd.to_datetime(df["Entry_Time"]).astype("int64") // 10**9
    df["Exit_Time"] = pd.to_datetime(df["Exit_Time"]).astype("int64") // 10**9
    timeline_start = df["Entry_Time"].min()
    df["Entry_Time"] -= timeline_start
    df["Exit_Time"] -= timeline_start

    df_pairs, safe_flights = solvers.isolate_conflicts(df)
    if a.solver == "adam":
        solvers.set_seed(SEED)
        epochs, lr = ADAM[a.altitude]
        shifts, _ = solvers.shifts_adam(df_pairs, safe_flights, max_shift=solvers.MAX_SHIFT,
                                        epochs=a.epochs or epochs, lr=a.lr or lr,
                                        device=a.device)
    else:
        shifts, _ = solvers.shifts_gurobi(df_pairs, safe_flights, max_shift=solvers.MAX_SHIFT,
                                          window_hours=a.window_hours,
                                          time_limit=a.time_limit, threads=a.threads)

    df_final = df.merge(shifts, left_on="Trip_ID", right_on="flight_id", how="left")
    df_final["optimized_entry_time"] = df_final["Entry_Time"] + df_final["time_shift_seconds"]
    df_final["optimized_exit_time"] = df_final["Exit_Time"] + df_final["time_shift_seconds"]
    before = solvers.total_overlap(df, "Entry_Time", "Exit_Time")
    after = solvers.total_overlap(df_final, "optimized_entry_time", "optimized_exit_time")
    print("total pairwise overlap: %s s before, %s s after"
          % (format(before, ",.0f"), format(after, ",.0f")), flush=True)

    os.makedirs(model.OPT1, exist_ok=True)
    df_shifts = df_final[["Trip_ID", "Solar_Farm_ID", "time_shift_seconds"]]
    df_shifts.to_csv(model.shift_file(a.altitude), index=False)

    # rerun the power model on the shifted crossings (rows are aligned: the
    # merge above keeps the order of df_results)
    if not (np.array_equal(df_shifts["Trip_ID"].values, df_results["Trip_ID"].values)
            and np.array_equal(df_shifts["Solar_Farm_ID"].values,
                               df_results["Solar_Farm_ID"].values)):
        raise ValueError("shift table and overlap table are not aligned")
    shift = pd.to_timedelta(df_shifts["time_shift_seconds"].values, unit="s")
    df_results["Time_Shift"] = df_shifts["time_shift_seconds"].values
    df_results["Original_Entry_Time"] = df_results["Entry_Time"]
    df_results["Original_Exit_Time"] = df_results["Exit_Time"]
    df_results["Entry_Time"] = df_results["Entry_Time"] + shift
    df_results["Exit_Time"] = df_results["Exit_Time"] + shift

    df_farms, df_flight_results = model.run_power_model(df_results, farm_centers, farm_details,
                                                        altitude_km, fuel_basis="energy")
    farms = model.merge_farm_results(gdf_solar, df_farms)
    flights = model.merge_flight_results(df_flights, df_flight_results, fuel_basis="energy")
    prefix = "Optimization_1_"
    farms.to_csv(model.farm_file(a.altitude, prefix=prefix, folder=model.OPT1), index=False)
    flights.to_csv(model.flight_file(a.altitude, prefix=prefix, folder=model.OPT1), index=False)

    fuel, co2 = farms["Money_Cost_Saving"].sum(), farms["CO2_Emissions_Reduction"].sum()
    print("altitude %d m, optimized schedule:" % a.altitude)
    print("  energy supplied  : %s MWh" % format(farms["Total_Energy_Supplied_MWh"].sum(), ",.0f"))
    print("  fuel cost saving : $%s" % format(fuel, ",.0f"))
    print("  CO2 reduction    : %s t" % format(co2 / 1000, ",.0f"))
    print("  total cost saving: $%s  (fuel + CO2 at $84/t)" % format(fuel + 0.084 * co2, ",.0f"))
    print("wrote results to %s  (%.0f s)" % (model.OPT1, time.time() - started))


if __name__ == "__main__":
    main()
