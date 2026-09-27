"""Parameter sensitivity of the reference case (Supplementary Notes 6-10).

Reruns the farm-by-farm power model of 3_system_level.py at 12,100 m for
    - the maximum power received per flight, 0-10.3 MW,
    - the receiving aperture area, 0-261.6 m2,
    - the wireless power transfer efficiency, 0-1,
one parameter at a time, and splits the delivered energy into daytime
(06:00-18:00 America/Chicago) and nighttime. The two electricity cost
scenarios reprice the daytime and nighttime energy of the reference case.

    python 1_parameter_sweep.py            # reuses saved results when inputs are unchanged
    python 1_parameter_sweep.py --force    # recomputes

Reads results/baseline/flight_solar_overlap_pairs_R1.csv and the solar farm
file, and writes to results/sensitivity/:

    physical_sensitivity_full.csv          every evaluated parameter value
    <analysis>_key_points.csv              key points of the five analyses
    baseline_by_farm.csv, farm_timezones.csv, manifest.json, validation.json

Method. The positions and distances are generated once per farm on the shared
0.2 s grid. The nearest-first allocation is a piecewise-linear function of the
ratio of per-flight power cap to the aperture x efficiency multiplier, so all
parameter values are obtained from one pass. For nearest-first full-capacity
powers b_j, H_j = sum_(i<=j) 1/b_i and b_(n+1) = 0:

    F(r) = n*r + sum_j H_j*(b_(j+1) - b_j)*max(r - 1/H_j, 0)
    total received power = multiplier * F(cap / multiplier)

The farm energies at the baseline parameters are checked against
results/baseline/solar_farm_analysis_results_R1.csv; 2_validate_sweep.py
compares the method with the direct allocation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from numba import njit
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "sensitivity"
OVERLAPS = ROOT / "results" / "baseline" / "flight_solar_overlap_pairs_R1.csv"
SOLAR_FARMS = ROOT / "data" / "solar_farms" / "uspvdb_v3_0_20250430.geojson"
BASELINE_FARMS = ROOT / "results" / "baseline" / "solar_farm_analysis_results_R1.csv"
ENGINE_VERSION = 2


@dataclass(frozen=True)
class Config:
    dt_s: float = 0.2
    max_flight_power_mw: float = 10.3
    aperture_m2: float = 261.6
    efficiency: float = 0.4478
    altitude_km: float = 12.1
    wavelength_parameter: float = 0.05
    capacity_threshold_mw: float = 27.7
    cost_per_mwh: float = 58.0
    emissions_kg_per_mwh: float = 48.0
    fuel_cost_per_mwh: float = 1540.0 / 10.3
    fuel_emissions_kg_per_mwh: float = 6952.0 / 10.3
    daytime_start_hour: int = 6
    daytime_end_hour: int = 18
    time_basis: str = "America/Chicago"


KEYS = {
    "electric_propulsion": [1, 2.5, 5, 7.5, 9, 10.3],
    "receiving_aperture": [65.4, 130.8, 196.2, 261.6],
    "transfer_efficiency": [0.15, 0.30, 0.45, 0.54, 0.60, 0.76],
}
SPECS = {
    "electric_propulsion": ("MAX_FLIGHT_POWER", "MW", 0, 10.3, 0.1),
    "receiving_aperture": ("RECEIVING_APERTURE", "m2", 0, 261.6, 2.616),
    "transfer_efficiency": ("SYSTEM_EFFICIENCY", "fraction", 0, 1, 0.01),
}
# Electricity cost grids ($/MWh): (daytime, nighttime) pairs
COST_PAIRS = {
    "cost_scenario_1": [(d, n) for d in [38, 58, 78] for n in [50, 70, 90, 110, 131]],
    "cost_scenario_2": [(d, n) for d in [50, 70, 90, 110, 131] for n in [50, 70, 90, 110, 131]],
}


def parameter_cases(cfg=Config()):
    records = []
    for name, (parameter, unit, lower, upper, step) in SPECS.items():
        base = {"electric_propulsion": cfg.max_flight_power_mw,
                "receiving_aperture": cfg.aperture_m2,
                "transfer_efficiency": cfg.efficiency}[name]
        grid = np.unique(np.round(np.r_[np.arange(lower, upper + step/2, step),
                                       KEYS[name], base, lower, upper], 10))
        for x in grid:
            scale = (x / cfg.aperture_m2 if name == "receiving_aperture" else
                     x / cfg.efficiency if name == "transfer_efficiency" else 1.0)
            cap = x if name == "electric_propulsion" else cfg.max_flight_power_mw
            label = ""
            if name == "transfer_efficiency":
                label = {0.45: "Ours (rounded)",
                         0.54: "Experimental maximum",
                         0.76: "Theoretical maximum"}.get(x, "")
            records.append(dict(analysis=name, parameter=parameter, value=float(x),
                                unit=unit, is_key_point=bool(np.any(np.isclose(x, KEYS[name], atol=1e-10, rtol=0))),
                                is_baseline=bool(np.isclose(x, base, atol=1e-10, rtol=0)),
                                dotted_region=bool((name == "electric_propulsion" and x < 1) or
                                                   (name == "transfer_efficiency" and x > .76)),
                                label=label, scale=float(scale), cap_mw=float(cap)))
    return pd.DataFrame(records)


def input_signature(cfg):
    files = [OVERLAPS, SOLAR_FARMS]
    payload = dict(engine=ENGINE_VERSION, config=asdict(cfg),
                   cases=parameter_cases(cfg).to_dict("records"),
                   inputs=[dict(path=str(p), size=p.stat().st_size,
                                mtime_ns=p.stat().st_mtime_ns) for p in files])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_inputs(cfg):
    for path in (OVERLAPS, SOLAR_FARMS):
        if not path.exists():
            raise SystemExit(f"missing {path}\nrun code/2_coverage_and_savings first, or python download_data.py")
    solar = gpd.read_file(SOLAR_FARMS)
    solar["p_cap_safe"] = np.minimum(solar.p_cap_dc, 20 * solar.p_area / 1e6)
    solar = solar[solar.p_cap_safe >= cfg.capacity_threshold_mw].to_crs("EPSG:5070")
    centers = solar.geometry.centroid
    lonlat = gpd.GeoSeries(centers, crs=solar.crs).to_crs("EPSG:4326")
    farms = pd.DataFrame(dict(farm_id=solar.FID.to_numpy(), capacity=solar.p_cap_safe.to_numpy(),
                             x=centers.x.to_numpy(), y=centers.y.to_numpy(),
                             longitude=lonlat.x.to_numpy(), latitude=lonlat.y.to_numpy())).set_index("farm_id")
    assert farms.index.is_unique
    cols = ["Trip_ID", "Solar_Farm_ID", "Entry_Time", "Exit_Time", "Entry_Longitude",
            "Entry_Latitude", "Exit_Longitude", "Exit_Latitude"]
    pieces = []
    for chunk in pd.read_csv(OVERLAPS, usecols=cols, chunksize=250_000):
        pieces.append(chunk[chunk.Solar_Farm_ID.isin(farms.index)])
    flights = pd.concat(pieces, ignore_index=True)
    for col in ("Entry_Time", "Exit_Time"):
        # The generating notebook derives these values from UTC departures.
        flights[col] = pd.to_datetime(flights[col], utc=True).dt.tz_localize(None)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    for prefix in ("Entry", "Exit"):
        flights[prefix + "_X"], flights[prefix + "_Y"] = tr.transform(
            flights[prefix + "_Longitude"].to_numpy(), flights[prefix + "_Latitude"].to_numpy())
    if flights.isna().any().any():
        raise ValueError("Missing required flight inputs; inspect before running sensitivity.")
    if not np.isfinite(flights.select_dtypes("number").to_numpy()).all():
        raise ValueError("Nonfinite flight inputs.")
    return farms, flights


@njit(cache=True)
def _interpolate(entry, exit_, coords, origin, dt_ns, cx, cy, capacity, eta, aperture, altitude, lam):
    n = len(entry)
    lengths = np.zeros(n, dtype=np.int64)
    first = np.zeros(n, dtype=np.int64)
    for i in range(n):
        # Equivalent to inclusive searchsorted slicing of the shared farm grid.
        first[i] = (entry[i] - origin + dt_ns - 1) // dt_ns
        last = (exit_[i] - origin) // dt_ns
        if exit_[i] > entry[i] and (exit_[i] - entry[i]) // 1000 > 0:
            lengths[i] = max(0, last - first[i] + 1)
    timestamps = np.empty(lengths.sum(), dtype=np.int64)
    distance = np.empty(len(timestamps), dtype=np.float64)
    power = np.empty(len(timestamps), dtype=np.float64)
    k = 0
    for i in range(n):
        # Preserve the original scalar Timedelta.total_seconds() microsecond truncation.
        duration = ((exit_[i] - entry[i]) // 1000) / 1e6
        for j in range(lengths[i]):
            stamp = origin + (first[i] + j) * dt_ns
            ratio = ((stamp - entry[i]) / 1e9) / duration
            x = coords[i, 0] + ratio * (coords[i, 2] - coords[i, 0])
            y = coords[i, 1] + ratio * (coords[i, 3] - coords[i, 1])
            d = np.sqrt((x-cx)**2 + (y-cy)**2)
            denom = np.pi * lam * np.sqrt((d/1000)**2 + altitude**2)
            timestamps[k] = stamp
            distance[k] = d
            power[k] = (capacity/denom) * eta * aperture / 1000
            k += 1
    return timestamps, distance, power


def prepare_farm(flights, farm, cfg):
    entry = flights.Entry_Time.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    exit_ = flights.Exit_Time.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    coords = flights[["Entry_X", "Entry_Y", "Exit_X", "Exit_Y"]].to_numpy(dtype=float)
    origin = (entry.min() // 1_000_000_000) * 1_000_000_000
    t, distance, b = _interpolate(entry, exit_, coords, origin, round(cfg.dt_s*1e9),
                                  farm.x, farm.y, farm.capacity, cfg.efficiency,
                                  cfg.aperture_m2, cfg.altitude_km, cfg.wavelength_parameter)
    order = np.lexsort((distance, t))
    return t[order], b[order]


def daylight_mask(timestamps, timezone, start_hour=6, end_hour=18):
    """Local-clock daytime [06:00,18:00); handles timezone and DST conversions."""
    if not len(timestamps):
        return np.zeros(0, dtype=bool)
    endpoints = pd.to_datetime([timestamps[0], timestamps[-1]], utc=True).tz_convert(timezone).tz_localize(None)
    dates = pd.date_range(endpoints[0].normalize()-pd.Timedelta(days=1),
                          endpoints[-1].normalize()+pd.Timedelta(days=1), freq="D")
    boundaries = np.sort(np.r_[
        (dates+pd.Timedelta(hours=start_hour)).tz_localize(timezone).tz_convert("UTC").asi8,
        (dates+pd.Timedelta(hours=end_hour)).tz_localize(timezone).tz_convert("UTC").asi8])
    return np.searchsorted(boundaries, timestamps, side="right") % 2 == 1


def farm_timezones(farms, cfg):
    return pd.Series([cfg.time_basis] * len(farms), index=farms.index, name="timezone")


@njit(cache=True)
def response_coefficients(t, b, day, ratios):
    """Aggregate exact piecewise-linear waterfall responses without rerunning sweeps.

    For full-capacity powers b_j in nearest-first order, H_j=sum(1/b_i).
    F(r)=n*r + sum_j H_j*(b_(j+1)-b_j)*max(r-1/H_j,0), b_(n+1)=0.
    At power multiplier s and flight cap c, total power is s*F(c/s).
    Bucket slope/intercept changes on the requested r grid, separately by day/night.
    """
    slope = np.zeros((2, len(ratios)), dtype=np.float64)
    intercept = np.zeros_like(slope)
    i = 0
    while i < len(t):
        end = i+1
        while end < len(t) and t[end] == t[i]:
            end += 1
        category = 0 if day[i] else 1
        slope[category, 0] += end-i
        h = 0.0
        for j in range(i, end):
            h += 1/b[j]
            nxt = b[j+1] if j+1 < end else 0.0
            drop = b[j]-nxt
            if drop > 0:
                k = np.searchsorted(ratios, 1/h)
                if k < len(ratios):
                    slope[category, k] -= h*drop
                    intercept[category, k] += drop
        i = end
    return slope, intercept


def evaluate_coefficients(slope, intercept, ratios, cfg):
    values = (np.cumsum(slope, axis=1)*ratios + np.cumsum(intercept, axis=1))*cfg.dt_s/3600
    if values.min() < -1e-6:
        raise AssertionError("Unexpected negative response energy")
    return np.maximum(values, 0)


def add_metrics(day_mwh, night_mwh, cfg, day_cost=None, night_cost=None):
    day_cost = cfg.cost_per_mwh if day_cost is None else day_cost
    night_cost = cfg.cost_per_mwh if night_cost is None else night_cost
    e = day_mwh + night_mwh
    cost = day_mwh*day_cost + night_mwh*night_cost
    return dict(energy_mwh=e, daytime_energy_mwh=day_mwh, nighttime_energy_mwh=night_mwh,
                weighted_cost_per_mwh=cost/e if e else np.nan,
                production_beaming_cost_usd=cost,
                displaced_fuel_cost_usd=e*cfg.fuel_cost_per_mwh,
                net_cost_saving_usd=e*cfg.fuel_cost_per_mwh-cost,
                beaming_emissions_t=e*cfg.emissions_kg_per_mwh/1000,
                displaced_fuel_emissions_t=e*cfg.fuel_emissions_kg_per_mwh/1000,
                net_co2_reduction_t=e*(cfg.fuel_emissions_kg_per_mwh-cfg.emissions_kg_per_mwh)/1000,
                equivalent_full_power_hours=e/10.3)


def calculate(cfg=Config(), force=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    signature = input_signature(cfg)
    manifest_path = OUTPUT / "manifest.json"
    if not force and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("signature") == signature and (OUTPUT / "physical_sensitivity_full.csv").exists():
            return pd.read_csv(OUTPUT / "physical_sensitivity_full.csv"), pd.read_csv(OUTPUT / "baseline_by_farm.csv"), manifest
    start = time.perf_counter()
    farms, flights = load_inputs(cfg)
    zones = farm_timezones(farms, cfg)
    farms.join(zones).to_csv(OUTPUT / "farm_timezones.csv")
    print(f"Loaded {len(flights):,} overlaps across {flights.Solar_Farm_ID.nunique()} farms.", flush=True)
    cases = parameter_cases(cfg)
    positive = cases.scale > 0
    ratios = np.unique(np.r_[0.0, (cases.loc[positive, "cap_mw"]/cases.loc[positive, "scale"]).to_numpy()])
    total_response = np.zeros((2, len(ratios)))
    baseline_index = np.searchsorted(ratios, cfg.max_flight_power_mw)
    farm_rows = []
    point_count = 0
    for i, (fid, group) in enumerate(flights.groupby("Solar_Farm_ID", sort=False)):
        farm = farms.loc[fid]
        t, b = prepare_farm(group, farm, cfg)
        day = daylight_mask(t, zones.loc[fid], cfg.daytime_start_hour, cfg.daytime_end_hour)
        slope, intercept = response_coefficients(t, b, day, ratios)
        response = evaluate_coefficients(slope, intercept, ratios, cfg)
        total_response += response
        farm_rows.append(dict(Solar_Farm_ID=int(fid), overlap_rows=len(group), point_count=len(t),
                              **add_metrics(response[0, baseline_index], response[1, baseline_index], cfg)))
        point_count += len(t)
        if i % 25 == 0 or i+1 == flights.Solar_Farm_ID.nunique():
            print(f"Farm {i+1}/{flights.Solar_Farm_ID.nunique()} | {point_count:,} points | {time.perf_counter()-start:.1f} s", flush=True)
    rows = []
    for row in cases.to_dict("records"):
        if row["scale"] == 0 or row["cap_mw"] == 0:
            e_day, e_night = 0., 0.
        else:
            k = np.searchsorted(ratios, row["cap_mw"]/row["scale"])
            e_day, e_night = total_response[:, k]*row["scale"]
        rows.append(dict(**row, **add_metrics(e_day, e_night, cfg)))
    result = pd.DataFrame(rows)
    baseline = result[(result.analysis == "electric_propulsion") & result.is_baseline].iloc[0]
    result["energy_change_vs_baseline_pct"] = 100*(result.energy_mwh/baseline.energy_mwh-1)
    baseline_farms = pd.DataFrame(farm_rows)
    manifest = dict(signature=signature, engine_version=ENGINE_VERSION, config=asdict(cfg),
                    overlap_rows=len(flights), farms=len(farm_rows), interpolated_points=point_count,
                    first_entry_utc=str(flights.Entry_Time.min()), last_exit_utc=str(flights.Exit_Time.max()),
                    elapsed_seconds=time.perf_counter()-start,
                    daytime_definition=f"{cfg.daytime_start_hour:02d}:00 inclusive to {cfg.daytime_end_hour:02d}:00 exclusive; time basis {cfg.time_basis}",
                    timezone_conversion="pandas IANA timezone conversion with DST; common time basis for every farm")
    result.to_csv(OUTPUT / "physical_sensitivity_full.csv", index=False)
    baseline_farms.to_csv(OUTPUT / "baseline_by_farm.csv", index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result, baseline_farms, manifest


def cost_tables(baseline, cfg):
    tables = {}
    for name, pairs in COST_PAIRS.items():
        rows = []
        for i, (d, n) in enumerate(pairs, 1):
            rows.append(dict(case_id=f"{'S1' if name.endswith('1') else 'S2'}-{i:02d}",
                             daytime_cost_per_mwh=d, nighttime_cost_per_mwh=n,
                             **add_metrics(baseline.daytime_energy_mwh, baseline.nighttime_energy_mwh, cfg, d, n)))
        tables[name] = pd.DataFrame(rows)
    return tables


def validate_results(result, farm_results, cfg):
    existing = pd.read_csv(BASELINE_FARMS)
    merged = farm_results.merge(existing[["Solar_Farm_ID", "Total_Energy_Supplied_MWh"]], on="Solar_Farm_ID", validate="one_to_one")
    assert len(merged) == len(farm_results) == len(existing), "Farm coverage differs from existing baseline"
    np.testing.assert_allclose(merged.energy_mwh, merged.Total_Energy_Supplied_MWh, rtol=1e-8, atol=1e-6)
    for name, group in result.groupby("analysis", sort=False):
        assert (np.diff(group.sort_values("value").energy_mwh) >= -1e-6).all(), name
        assert group.loc[group.value == 0, "energy_mwh"].iloc[0] == 0
        assert int(group.is_key_point.sum()) == len(KEYS[name])
    np.testing.assert_allclose(result.energy_mwh, result.daytime_energy_mwh+result.nighttime_energy_mwh)
    baseline = result[result.is_baseline]
    np.testing.assert_allclose(baseline.energy_mwh, baseline.energy_mwh.iloc[0], rtol=1e-10)
    report = dict(baseline_farms_compared=len(merged),
                  max_farm_energy_absolute_difference_mwh=float(np.max(np.abs(merged.energy_mwh-merged.Total_Energy_Supplied_MWh))),
                  baseline_total_mwh=float(baseline.energy_mwh.iloc[0]),
                  monotonicity="passed", zero_endpoints="passed", daytime_nighttime_reconciliation="passed")
    (OUTPUT / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_analysis(force=False, cfg=Config()):
    result, farms, manifest = calculate(cfg, force=force)
    validation = validate_results(result, farms, cfg)
    baseline = result[(result.analysis == "electric_propulsion") & result.is_baseline].iloc[0]
    tables = {name: result[(result.analysis == name) & result.is_key_point].copy() for name in KEYS}
    tables.update(cost_tables(baseline, cfg))
    for name, table in tables.items():
        table.to_csv(OUTPUT / f"{name}_key_points.csv", index=False)
    return result, tables, baseline, manifest, validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recalculate the physical model")
    args = parser.parse_args()
    full, tables, baseline, manifest, validation = run_analysis(force=args.force)
    print(json.dumps(validation, indent=2), flush=True)
    print(f"Results: {OUTPUT}", flush=True)
