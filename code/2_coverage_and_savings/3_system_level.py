"""Energy, duration, cost and CO2 of beaming at one cruise altitude.

For every qualified farm (safety capacity >= 27.7 MW; 438 farms serve at least
one flight) the flights passing through its energy boundary are sampled every
0.2 s and the farm capacity is shared among them nearest first (model.py).
Results are summed by farm and by flight.

    python 3_system_level.py                     # 12,100 m (reference)
    python 3_system_level.py --altitude 9100
    python 3_system_level.py --altitude 15100

Reads results/baseline/flight_solar_overlap_pairs[tag]_R1.csv and writes to
results/baseline/:

    solar_farm_analysis_results[tag]_R1.csv          by farm
    flight_analysis_results[tag]_R1.csv              by flight
    solar_farm_analysis_results[tag]_merged_R1.csv   by farm, with farm attributes
    flight_analysis_results[tag]_merged_R1.csv       by flight, with flight records

The merged tables carry two fuel bases: *_by_Duration (fuel priced per hour
of supported flight, first submission) and *_by_Energy (fuel priced per
delivered MWh), which the R1 results, figures and tables use.
"""
import argparse
import os
import time

import model


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--altitude", type=int, default=model.REFERENCE_ALTITUDE,
                    choices=model.ALTITUDES, help="cruise altitude in m (default 12100)")
    a = ap.parse_args()
    started = time.time()
    altitude_km = a.altitude / 1000

    gdf_solar, gdf_solar_proj, farm_centers, farm_details = model.load_solar_farms()
    df_results = model.load_overlaps(a.altitude, gdf_solar_proj["FID"].unique())
    print("qualified farms: %d, overlap events: %s"
          % (len(gdf_solar), format(len(df_results), ",")), flush=True)

    df_farms, df_flight_results = model.run_power_model(
        df_results, farm_centers, farm_details, altitude_km, fuel_basis="duration",
        duration_resolution="us", verbose=True)

    os.makedirs(model.BASELINE, exist_ok=True)
    df_farms.to_csv(model.farm_file(a.altitude, merged=False), index=False)
    df_flight_results.to_csv(model.flight_file(a.altitude, merged=False), index=False)

    farms = model.add_energy_basis(model.merge_farm_results(gdf_solar, df_farms))
    flights = model.add_energy_basis(model.merge_flight_results(
        model.load_flights(), df_flight_results, fuel_basis="duration"))
    farms.to_csv(model.farm_file(a.altitude), index=False)
    flights.to_csv(model.flight_file(a.altitude), index=False)

    energy = farms["Total_Energy_Supplied_MWh"].sum()
    fuel = farms["Fuel_Cost_Savings_by_Energy"].sum()
    co2 = farms["CO2_Emissions_Reduction_by_Energy"].sum()
    print("altitude %d m: %d farms, %s flights" % (a.altitude, len(farms),
                                                 format(len(flights), ",")))
    print("  energy supplied   : %s MWh" % format(energy, ",.0f"))
    print("  duration supported: %s h" % format(
        farms["Total_Flight_Duration_Supported_Hours"].sum(), ",.0f"))
    print("  fuel cost saving  : $%s" % format(fuel, ",.0f"))
    print("  CO2 reduction     : %s t" % format(co2 / 1000, ",.0f"))
    print("  total cost saving : $%s  (fuel + CO2 at $84/t)"
          % format(fuel + co2 * 0.084, ",.0f"))
    print("wrote %s and %s  (%.0f s)" % (model.farm_file(a.altitude),
                                        model.flight_file(a.altitude),
                                        time.time() - started))


if __name__ == "__main__":
    main()
