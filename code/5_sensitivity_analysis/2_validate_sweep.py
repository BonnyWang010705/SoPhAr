"""Check the sensitivity method of 1_parameter_sweep.py.

    python 2_validate_sweep.py

Three checks:
  - synthetic contention cases (simultaneous flights, ties, very weak and
    very strong farms, exact breakpoints, zero power, capacity used up)
    against a direct nearest-first allocation;
  - the daytime/nighttime clock rule across the spring and autumn
    daylight-saving transitions;
  - two real farms (FID 4701 and 4662) at every key point, against the
    sample-by-sample allocation of the baseline model (reference functions
    below, as in 3_system_level.py's source notebook).

Writes results/sensitivity/engine_validation.json.
"""
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sweep = importlib.import_module("1_parameter_sweep")

REAL_FARMS = (4701, 4662)


# ----------------------------------------------------------------------------
# Reference: the sample-by-sample model of the baseline run
# ----------------------------------------------------------------------------
def interpolate_flight_points(farm_flights, farm_center, frequency="200ms"):
    """Samples of every flight of a farm on the farm's shared time grid."""
    flight_points = []
    farm_center_x, farm_center_y = farm_center
    min_time = farm_flights["Entry_Time"].min().floor("1s")
    max_time = farm_flights["Exit_Time"].max().ceil("1s")
    farm_time_grid = pd.date_range(start=min_time, end=max_time, freq=frequency)
    for _, row in farm_flights.iterrows():
        left = farm_time_grid.searchsorted(row["Entry_Time"], side="left")
        right = farm_time_grid.searchsorted(row["Exit_Time"], side="right")
        ts = farm_time_grid[left:right]
        if len(ts) == 0:
            continue
        total_sec = (row["Exit_Time"] - row["Entry_Time"]).total_seconds()
        if total_sec <= 0:
            continue
        ratios = (ts - row["Entry_Time"]).total_seconds() / total_sec
        cur_x = row["Entry_X"] + ratios * (row["Exit_X"] - row["Entry_X"])
        cur_y = row["Entry_Y"] + ratios * (row["Exit_Y"] - row["Entry_Y"])
        dist_m = np.sqrt((cur_x - farm_center_x) ** 2 + (cur_y - farm_center_y) ** 2)
        flight_points.append(pd.DataFrame({"Time_Stamp": ts, "Trip_ID": row["Trip_ID"],
                                           "Distance_m": dist_m}))
    return pd.concat(flight_points) if flight_points else pd.DataFrame()


def calculate_waterfall_power(all_points, p_cap_safe_total, max_flight_power, efficiency,
                              cfg):
    """Nearest-first allocation of the farm capacity, sample by sample."""
    all_points = all_points.sort_values(by=["Time_Stamp", "Distance_m"])
    dist_km = all_points["Distance_m"] / 1000.0
    denominator = np.pi * cfg.wavelength_parameter * np.sqrt(dist_km ** 2 + cfg.altitude_km ** 2)
    all_points["Potential"] = ((p_cap_safe_total / denominator) * efficiency
                               * cfg.aperture_m2 / 1000)
    all_points["Needed"] = np.where(all_points["Potential"] <= max_flight_power, 1.0,
                                    max_flight_power / all_points["Potential"])
    all_points["Cum"] = all_points.groupby("Time_Stamp")["Needed"].cumsum()
    available = (1.0 - (all_points["Cum"] - all_points["Needed"])).clip(lower=0.0)
    all_points["Power"] = np.minimum(all_points["Needed"], available) * all_points["Potential"]
    return all_points[all_points["Power"] > 0].copy()


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_synthetic():
    rng = np.random.default_rng(729)
    groups = [np.array([20., 10., 5.]), np.array([2., 2.]), np.array([.01]),
              np.array([1000., 999.]), np.sort(rng.uniform(.1, 100, 17))[::-1]]
    t = np.repeat(np.arange(len(groups), dtype=np.int64), [len(g) for g in groups])
    b = np.concatenate(groups)
    day = t % 2 == 0
    ratios = np.unique(np.r_[0., .0001, .1, 1., 10.3, 100., 10000.,
                             *[1 / np.cumsum(1 / g) for g in groups]])
    s, c = sweep.response_coefficients(t, b, day, ratios)
    actual = sweep.evaluate_coefficients(s, c, ratios, sweep.Config()) * 3600 / .2
    expected = np.zeros_like(actual)
    for k, cap in enumerate(ratios):
        for i, group in enumerate(groups):
            left = 1.
            for potential in group:
                frac = min(left, cap / potential)
                expected[0 if i % 2 == 0 else 1, k] += frac * potential
                left -= frac
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-7)
    print("synthetic contention and boundary cases: passed", flush=True)


def check_clock_rule():
    for start, end in [("2025-03-08", "2025-03-11"), ("2025-11-01", "2025-11-04"),
                       ("2025-04-07", "2025-04-15")]:
        t = pd.date_range(start, end, freq="10min", tz="UTC")
        local = t.tz_convert("America/Chicago")
        np.testing.assert_array_equal(sweep.daylight_mask(t.asi8, "America/Chicago"),
                                      (local.hour >= 6) & (local.hour < 18))
    print("clock boundaries and daylight-saving transitions: passed", flush=True)


def check_real():
    cfg = sweep.Config()
    farms, flights = sweep.load_inputs(cfg)
    cases = sweep.parameter_cases(cfg)
    cases = cases[cases.is_key_point | cases.is_baseline | (cases.value == 0) | (cases.value == 1)]
    positive = cases.scale > 0
    ratios = np.unique(np.r_[0., (cases.loc[positive, "cap_mw"]
                                  / cases.loc[positive, "scale"]).to_numpy()])
    checked, max_abs = 0, 0.
    for fid in REAL_FARMS:
        farm = farms.loc[fid]
        flights_one = flights[flights.Solar_Farm_ID == fid]
        t, b = sweep.prepare_farm(flights_one, farm, cfg)
        points = interpolate_flight_points(flights_one, (farm.x, farm.y))
        points = points.sort_values(["Time_Stamp", "Distance_m"])
        np.testing.assert_array_equal(t, points.Time_Stamp.to_numpy().astype(np.int64))
        # an arbitrary, reproducible split exercises both accumulation channels
        day = (t // (3600 * 10**9)) % 2 == 0
        slope, intercept = sweep.response_coefficients(t, b, day, ratios)
        response = sweep.evaluate_coefficients(slope, intercept, ratios, cfg)
        for case in cases.itertuples(index=False):
            active = calculate_waterfall_power(points.copy(), farm.capacity, case.cap_mw,
                                               cfg.efficiency * case.scale, cfg)
            stamp = active.Time_Stamp.to_numpy().astype(np.int64)
            mask = (stamp // (3600 * 10**9)) % 2 == 0
            expected = np.array([active.loc[mask, "Power"].sum(),
                                 active.loc[~mask, "Power"].sum()]) * .2 / 3600
            actual = (np.zeros(2) if case.scale == 0 else
                      response[:, np.searchsorted(ratios, case.cap_mw / case.scale)] * case.scale)
            np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-6)
            max_abs = max(max_abs, float(np.abs(actual - expected).max()))
            checked += 1
        print("farm %d: all selected cases match the direct allocation" % fid, flush=True)
    return checked, max_abs


def main():
    check_synthetic()
    check_clock_rule()
    checked, max_abs = check_real()
    report = dict(real_farm_case_comparisons=checked, max_energy_difference_mwh=max_abs,
                  synthetic_boundary_cases="passed", interpolation_timestamps="exact match",
                  clock_boundaries_and_dst="passed")
    sweep.OUTPUT.mkdir(parents=True, exist_ok=True)
    (sweep.OUTPUT / "engine_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
