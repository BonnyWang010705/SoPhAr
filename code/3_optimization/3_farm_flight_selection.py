"""Farm-and-flight choice optimization over the penetration grid (Fig. 6).

For each pair of penetration rates (solar farms rho_F, flights rho_I, each
10%, 20%, ..., 100%: 100 scenarios) the solver picks which farms get a SoPhAr
system and which flights get a receiving antenna (solvers.py). The power model
is then run on the crossings of the selected farms by the selected flights,
at the reference altitude of 12,100 m and on fixed schedules.

    python 3_farm_flight_selection.py                          # greedy search
    python 3_farm_flight_selection.py --solver gurobi          # needs gurobipy
    python 3_farm_flight_selection.py --farm-rates 0.2 0.5 --flight-rates 0.5
    python 3_farm_flight_selection.py --compare                # greedy vs Gurobi, 10%/10%

Reads results/baseline/flight_solar_overlap_pairs_R1.csv and writes to
results/optimization_2/:

    Optimization_2_Results_R1/      solar_results_<F>farm_<I>flight.csv and
                                    flight_results_<F>farm_<I>flight.csv of
                                    every scenario
    Optimization_2_summary_R1.csv   totals of every scenario

Both solvers write the same files; a run over some of the scenarios replaces
only those scenarios in the folder and the summary.
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

ALTITUDE = 12100
RATES = [round(0.1 * k, 1) for k in range(1, 11)]
RESULTS = os.path.join(model.OPT2, "Optimization_2_Results_R1")
SUMMARY = os.path.join(model.OPT2, "Optimization_2_summary_R1.csv")
SUMMARY_COLS = ["Total_Energy_Supplied_MWh", "Total_Flight_Duration_Supported_Hours",
                "Production_Beaming_Cost", "CO2_Emissions_Beaming",
                "Original_Fuel_Cost_Baseline", "Original_CO2_Emissions_Baseline",
                "CO2_Emissions_Reduction", "Money_Cost_Saving"]


def load():
    gdf_solar, gdf_solar_proj, farm_centers, farm_details = model.load_solar_farms()
    df_flights = model.load_flights()
    df_results = model.load_overlaps(ALTITUDE, gdf_solar_proj["FID"].unique(),
                                     gdf_solar=gdf_solar)
    # the R1 run held these columns in reduced precision to save memory; kept
    # here so that the results are reproduced
    for col in ["Entry_X", "Entry_Y", "Exit_X", "Exit_Y"]:
        df_results[col] = df_results[col].astype(np.float32)
    df_results["Solar_Farm_ID"] = df_results["Solar_Farm_ID"].astype(np.int32)
    df_results["Trip_ID"] = df_results["Trip_ID"].astype(np.int32)
    print("farms: %d, flights: %s, crossings: %s"
          % (df_results["Solar_Farm_ID"].nunique(), format(df_results["Trip_ID"].nunique(), ","),
             format(len(df_results), ",")), flush=True)
    return gdf_solar, farm_centers, farm_details, df_flights, df_results


def solve(df_results, farm_p, flight_p, a):
    if a.solver == "greedy":
        return solvers.select_greedy(df_results, farm_p, flight_p)
    return solvers.select_gurobi(df_results, farm_p, flight_p, time_limit=a.time_limit,
                                 mip_gap=a.mip_gap, threads=a.threads, verbose=a.verbose)


def compare(df_results, a):
    started = time.time()
    _, _, greedy = solvers.select_greedy(df_results, 0.1, 0.1)
    t_greedy = time.time() - started
    started = time.time()
    _, _, exact = solvers.select_gurobi(df_results, 0.1, 0.1, time_limit=a.time_limit,
                                        mip_gap=a.mip_gap, threads=a.threads,
                                        verbose=a.verbose)
    t_exact = time.time() - started
    print("10%% farms / 10%% flights: greedy %.6g (%.0f s), Gurobi %.6g (%.0f s)"
          % (greedy, t_greedy, exact, t_exact))
    print("greedy objective is %.2f%% below Gurobi" % (100 * (exact - greedy) / exact))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solver", choices=("greedy", "gurobi"), default="greedy",
                    help="greedy (default, as in the paper) or gurobi (integer program)")
    ap.add_argument("--farm-rates", type=float, nargs="+", default=RATES,
                    help="solar farm penetration rates (default 0.1 ... 1.0)")
    ap.add_argument("--flight-rates", type=float, nargs="+", default=RATES,
                    help="flight penetration rates (default 0.1 ... 1.0)")
    ap.add_argument("--compare", action="store_true",
                    help="only compare greedy and Gurobi objectives at 10%%/10%%")
    ap.add_argument("--time-limit", type=float, help="Gurobi: time limit per scenario (s)")
    ap.add_argument("--mip-gap", type=float, help="Gurobi: relative MIP gap (default 1e-4)")
    ap.add_argument("--threads", type=int, help="Gurobi: threads (default all)")
    ap.add_argument("--verbose", action="store_true", help="Gurobi: show the solver log")
    a = ap.parse_args()
    started = time.time()

    gdf_solar, farm_centers, farm_details, df_flights, df_results = load()
    if a.compare:
        compare(df_results, a)
        return

    os.makedirs(RESULTS, exist_ok=True)
    summary_rows = []
    for farm_p in [round(r, 1) for r in a.farm_rates]:
        for flight_p in [round(r, 1) for r in a.flight_rates]:
            t0 = time.time()
            farms_sel, flights_sel, objective = solve(df_results, farm_p, flight_p, a)
            if farms_sel is None:
                print("  %.1f farm / %.1f flight: no solution, skipped" % (farm_p, flight_p))
                continue
            selected = df_results.loc[df_results["Solar_Farm_ID"].isin(farms_sel)
                                      & df_results["Trip_ID"].isin(flights_sel)]
            df_farms, df_flight_results = model.run_power_model(
                selected, farm_centers, farm_details, ALTITUDE / 1000, fuel_basis="energy")
            farms = model.merge_farm_results(gdf_solar, df_farms)
            flights = model.merge_flight_results(df_flights, df_flight_results,
                                                 fuel_basis="energy")
            for kind, table in (("solar", farms), ("flight", flights)):
                table.to_csv(os.path.join(RESULTS, "%s_results_%.1ffarm_%.1fflight.csv"
                                          % (kind, farm_p, flight_p)), index=False)
            row = {"p_solar_farm": farm_p, "p_flight": flight_p}
            row.update(farms[SUMMARY_COLS].sum().to_dict())
            summary_rows.append(row)
            total = row["Money_Cost_Saving"] + 0.084 * row["CO2_Emissions_Reduction"]
            print("  %.1f farm / %.1f flight: %d farms, %s flights, %s MWh, $%s  (%.0f s)"
                  % (farm_p, flight_p, len(farms_sel), format(len(flights_sel), ","),
                     format(row["Total_Energy_Supplied_MWh"], ",.0f"),
                     format(total, ",.0f"), time.time() - t0), flush=True)

    summary = pd.DataFrame(summary_rows)
    if os.path.exists(SUMMARY):
        old = pd.read_csv(SUMMARY)
        done = set(zip(summary["p_solar_farm"], summary["p_flight"]))
        old = old[[(f, i) not in done for f, i in zip(old["p_solar_farm"].round(1),
                                                      old["p_flight"].round(1))]]
        summary = (pd.concat([old, summary], ignore_index=True)
                     .sort_values(["p_solar_farm", "p_flight"]).reset_index(drop=True))
    summary.to_csv(SUMMARY, index=False)
    print("solver: %s; wrote %s and %s  (%.0f s)" % (a.solver, RESULTS, SUMMARY,
                                                     time.time() - started))


if __name__ == "__main__":
    main()
