"""Shared model of the SoPhAr coverage and savings analysis.

Everything the coverage, optimization and sensitivity scripts have in common
lives here: file locations, the system parameters (Methods, Section 3.3), the
loaders for the solar farm, flight and flight-farm overlap tables, and the
farm-by-farm power model.

Power model
-----------
Each flight that enters the beaming range of a farm is sampled every 0.2 s on
a time grid shared by all the flights of that farm. At each sample the power
the farm can put into the aircraft at full capacity is

    P_pot = P_farm * eta_sys * A_r / (pi * lambda * z),   z = sqrt(d^2 + h^2)

with d the ground distance between aircraft and farm centroid and h the cruise
altitude. The farm serves the flights in the air at the same instant nearest
first: each flight takes the fraction of the farm capacity it needs to reach
the 10.3 MW cruise demand, until the capacity is used up.

Savings
-------
Delivered energy displaces jet fuel at the cruise fuel flow (2,200 kg/h at
$0.7/kg, i.e. $1,540/h and 6,952 kg CO2/h for 10.3 MW). The R1 results price
the displaced fuel per delivered MWh (`*_by_Energy`); the baseline tables keep
the duration-based numbers of the first submission as well (`*_by_Duration`).
"""
import os

import numpy as np
import pandas as pd
from numba import njit

# ----------------------------------------------------------------------------
# Locations
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

DATA = os.path.join(ROOT, "data")
SOLAR_FARMS = os.path.join(DATA, "solar_farms", "uspvdb_v3_0_20250430.geojson")
AIRPORTS = os.path.join(DATA, "airports", "airport_data.geojson")
FLIGHTS_2025 = os.path.join(DATA, "flights", "processed", "flights_2025.parquet")
FLIGHT_OD = os.path.join(DATA, "flights", "processed", "flight_with_od_coor.csv")

RESULTS = os.path.join(ROOT, "results")
BASELINE = os.path.join(RESULTS, "baseline")
OPT1 = os.path.join(RESULTS, "optimization_1")
OPT2 = os.path.join(RESULTS, "optimization_2")

# Cruise altitudes (m). 12,100 m is the A320 reference; its files carry no
# altitude tag, the other two are tagged "_9100m" and "_15100m".
ALTITUDES = (9100, 12100, 15100)
REFERENCE_ALTITUDE = 12100


def tag(altitude):
    """File-name tag of a cruise altitude."""
    altitude = int(altitude)
    if altitude not in ALTITUDES:
        raise ValueError("altitude must be one of %s" % (ALTITUDES,))
    return "" if altitude == REFERENCE_ALTITUDE else "_%dm" % altitude


def overlap_file(altitude):
    return os.path.join(BASELINE, "flight_solar_overlap_pairs%s_R1.csv" % tag(altitude))


def farm_file(altitude, merged=True, prefix="", folder=BASELINE):
    return os.path.join(folder, "%ssolar_farm_analysis_results%s%s_R1.csv"
                        % (prefix, tag(altitude), "_merged" if merged else ""))


def flight_file(altitude, merged=True, prefix="", folder=BASELINE):
    return os.path.join(folder, "%sflight_analysis_results%s%s_R1.csv"
                        % (prefix, tag(altitude), "_merged" if merged else ""))


def shift_file(altitude):
    """Optimized schedule shifts of one altitude (3_optimization)."""
    return os.path.join(OPT1, "optimized_flight_shifts%s_R1.csv" % tag(altitude))


# ----------------------------------------------------------------------------
# Parameters (Supplementary Tables 16-17)
# ----------------------------------------------------------------------------
METRIC_CRS = "EPSG:5070"        # CONUS Albers equal area, metres
WGS84_CRS = "EPSG:4326"

FREQUENCY = "200ms"             # sampling step of the trajectories
TIME_INTERVAL = 0.2             # the same step in seconds
STEP_NS = pd.to_timedelta(FREQUENCY).value

SAFETY_POWER_DENSITY = 20       # W/m2, ground exposure limit S_ab
CAPACITY_THRESHOLD = 27.7       # MW, minimum safety capacity of a qualified farm

MAX_FLIGHT_POWER = 10.3         # MW, cruise power demand of the A320
RECEIVING_APERTURE = 261.6      # m2, A_r
SYSTEM_EFFICIENCY = 0.4478      # eta_sys = 68.87% x 95% x 78.67% x 87%
LAMBDA = 0.05                   # m, wavelength at 6 GHz

COST_PER_MWH = 58.0             # $/MWh, solar electricity and beaming
EMISSION_PER_MWH = 48.0         # kg CO2/MWh, solar electricity
FUEL_COST_PER_HR = 1540.0       # $/h, 2,200 kg/h at $0.7/kg
FUEL_EMISSION_PER_HR = 6952.0   # kg CO2/h, 2,200 kg/h at 3.16 kg CO2/kg
FUEL_COST_PER_MWH = 1540.0 / 10.3        # $ of fuel displaced per MWh
FUEL_EMISSION_PER_MWH = 6952.0 / 10.3    # kg CO2 displaced per MWh


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_solar_farms(qualified=True):
    """Solar farms with their safety capacity.

    Returns (gdf_solar, gdf_solar_proj, farm_centers, farm_details):
    the farms in WGS84 and in the metric CRS, a {FID: (x, y)} dictionary of
    farm centroids in metres, and a table of `p_cap_safe` indexed by FID.
    With `qualified=True` only the farms whose safety capacity reaches
    CAPACITY_THRESHOLD are kept.
    """
    import geopandas as gpd

    gdf_solar = gpd.read_file(SOLAR_FARMS)
    # effective capacity: min(DC capacity, 20 W/m2 x farm area), in MW
    gdf_solar["p_cap_safe"] = np.minimum(
        gdf_solar["p_cap_dc"], (SAFETY_POWER_DENSITY * gdf_solar["p_area"] / 1e6))
    if qualified:
        gdf_solar = gdf_solar[gdf_solar["p_cap_safe"] >= CAPACITY_THRESHOLD]
    gdf_solar_proj = gdf_solar.to_crs(METRIC_CRS)

    centroids = gdf_solar_proj.geometry.centroid
    farm_details = gdf_solar_proj[["FID", "p_cap_safe"]].set_index("FID")
    farm_centers = {fid: (pt.x, pt.y) for fid, pt in zip(gdf_solar_proj["FID"], centroids)}
    return gdf_solar, gdf_solar_proj, farm_centers, farm_details


def load_flights(path=FLIGHT_OD):
    """The analysed flights with airport coordinates (1_flight_od.py)."""
    if not os.path.exists(path):
        raise SystemExit("missing %s\nrun: python code/2_coverage_and_savings/1_flight_od.py"
                         % path)
    df = pd.read_csv(path)
    df["Wheels-off time (UTC)"] = pd.to_datetime(df["Wheels-off time (UTC)"])
    df["Landing time (UTC)"] = pd.to_datetime(df["Landing time (UTC)"])
    df["Trip_Duration_Sec"] = (df["Landing time (UTC)"]
                               - df["Wheels-off time (UTC)"]).dt.total_seconds()
    return df


def load_overlaps(altitude, valid_farms, gdf_solar=None, chunksize=1_000_000):
    """Flight-farm overlap events of one altitude (2_overlap_pairs.py).

    Only events of `valid_farms` are kept. Entry and exit points are projected
    to the metric CRS (Entry_X/Entry_Y/Exit_X/Exit_Y). When `gdf_solar` is
    given, the farm safety capacity is merged in as `p_cap_safe`.
    """
    from pyproj import Transformer

    path = overlap_file(altitude)
    if not os.path.exists(path):
        raise SystemExit("missing %s\nrun: python code/2_coverage_and_savings/"
                         "2_overlap_pairs.py --altitude %d" % (path, altitude))
    valid_farms = set(valid_farms)
    # read in chunks: the files hold every farm (up to 8 million events), but
    # only the qualified farms are needed
    parts = [chunk[chunk["Solar_Farm_ID"].isin(valid_farms)]
             for chunk in pd.read_csv(path, chunksize=chunksize)]
    df = pd.concat(parts)

    df["Entry_Time"] = pd.to_datetime(df["Entry_Time"])
    df["Exit_Time"] = pd.to_datetime(df["Exit_Time"])

    transformer = Transformer.from_crs(WGS84_CRS, METRIC_CRS, always_xy=True)
    entry_x, entry_y = transformer.transform(df["Entry_Longitude"].values,
                                             df["Entry_Latitude"].values)
    exit_x, exit_y = transformer.transform(df["Exit_Longitude"].values,
                                           df["Exit_Latitude"].values)
    df["Entry_X"], df["Entry_Y"] = entry_x, entry_y
    df["Exit_X"], df["Exit_Y"] = exit_x, exit_y

    if gdf_solar is not None:
        df = pd.merge(df, gdf_solar[["FID", "p_cap_safe"]], left_on="Solar_Farm_ID",
                      right_on="FID").drop(columns=["FID"])
    return df


# ----------------------------------------------------------------------------
# Power model (numba kernels)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _expand_flights_kernel(entry_ns, total_sec, start_idx, end_idx, entry_x, entry_y,
                           exit_x, exit_y, trip_id, origin_ns, step_ns,
                           farm_center_x, farm_center_y):
    """Sample every flight of a farm on the farm's shared 0.2 s grid.

    Returns the grid index of each sample, its trip ID and the ground distance
    (m) between the aircraft and the farm centroid.
    """
    n_pts = end_idx - start_idx + 1
    total_points = 0
    for i in range(len(n_pts)):
        if n_pts[i] > 0:
            total_points += n_pts[i]

    ts_idx_all = np.empty(total_points, dtype=np.int64)
    trip_all = np.empty(total_points, dtype=trip_id.dtype)
    dist_all = np.empty(total_points, dtype=np.float64)

    pos = 0
    for i in range(len(n_pts)):
        if n_pts[i] <= 0:
            continue
        if total_sec[i] <= 0:
            continue
        dx = exit_x[i] - entry_x[i]
        dy = exit_y[i] - entry_y[i]
        for j in range(start_idx[i], end_idx[i] + 1):
            ts_ns = origin_ns + j * step_ns
            ratio = ((ts_ns - entry_ns[i]) / 1e9) / total_sec[i]
            cur_x = entry_x[i] + ratio * dx
            cur_y = entry_y[i] + ratio * dy
            ddx = cur_x - farm_center_x
            ddy = cur_y - farm_center_y
            ts_idx_all[pos] = j
            trip_all[pos] = trip_id[i]
            dist_all[pos] = np.sqrt(ddx * ddx + ddy * ddy)
            pos += 1
    return ts_idx_all[:pos], trip_all[:pos], dist_all[:pos]


@njit(cache=True)
def _waterfall_power_kernel(ts_idx, potential_full, max_flight_power):
    """Nearest-first allocation of the farm capacity.

    The samples must be sorted by (grid index, distance). At each instant the
    nearest flight takes the fraction of capacity it needs to reach
    `max_flight_power`, the next flight takes what it needs from the rest, and
    so on until the capacity is used up.
    """
    n = len(ts_idx)
    power = np.empty(n, dtype=np.float64)
    current_ts = -1
    used_fraction = 0.0
    for i in range(n):
        ts = ts_idx[i]
        if i == 0 or ts != current_ts:
            current_ts = ts
            used_fraction = 0.0
        pf = potential_full[i]
        if pf <= 0.0:
            power[i] = 0.0
            continue
        frac_need = max_flight_power / pf
        if frac_need > 1.0:
            frac_need = 1.0
        available = 1.0 - used_fraction
        if available <= 0.0:
            actual_frac = 0.0
        elif frac_need < available:
            actual_frac = frac_need
        else:
            actual_frac = available
        power[i] = actual_frac * pf
        used_fraction += actual_frac
    return power


def expand_flights(farm_flights, farm_center, step_ns=STEP_NS, duration_resolution="ns"):
    """Sample the flights of one farm; None when no sample falls in range.

    The position at each sample is interpolated between the entry and exit
    points in proportion to the time elapsed. `duration_resolution` sets the
    resolution of the crossing duration in that ratio: "us" reproduces the
    baseline run (scalar Timedelta.total_seconds(), microseconds), "ns" the
    optimization runs.
    """
    if farm_flights.empty:
        return None
    farm_center_x, farm_center_y = farm_center

    entry_ns = farm_flights["Entry_Time"].values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = farm_flights["Exit_Time"].values.astype("datetime64[ns]").astype(np.int64)
    # the farm grid starts at the first entry, rounded down to the second
    origin_ns = farm_flights["Entry_Time"].min().floor("1s").value

    start_idx = (entry_ns - origin_ns + step_ns - 1) // step_ns   # first tick >= entry
    end_idx = (exit_ns - origin_ns) // step_ns                    # last tick <= exit

    valid = end_idx >= start_idx
    if not np.any(valid):
        return None
    ff = farm_flights.loc[valid]
    entry_ns, exit_ns = entry_ns[valid], exit_ns[valid]
    start_idx, end_idx = start_idx[valid], end_idx[valid]
    entry_x = ff["Entry_X"].to_numpy(dtype=np.float64)
    entry_y = ff["Entry_Y"].to_numpy(dtype=np.float64)
    exit_x = ff["Exit_X"].to_numpy(dtype=np.float64)
    exit_y = ff["Exit_Y"].to_numpy(dtype=np.float64)
    trip_id = ff["Trip_ID"].to_numpy()

    keep = exit_ns > entry_ns
    if not np.any(keep):
        return None
    if duration_resolution == "us":
        total_sec = ((exit_ns - entry_ns) // 1000) / 1e6
    else:
        total_sec = (exit_ns - entry_ns) / 1e9
    ts_idx_all, trip_all, dist_all = _expand_flights_kernel(
        entry_ns[keep], total_sec[keep], start_idx[keep].astype(np.int64),
        end_idx[keep].astype(np.int64), entry_x[keep], entry_y[keep], exit_x[keep],
        exit_y[keep], trip_id[keep], origin_ns, step_ns,
        float(farm_center_x), float(farm_center_y))
    if len(ts_idx_all) == 0:
        return None
    return ts_idx_all, trip_all, dist_all, origin_ns


def potential_power(dist_m, p_cap_safe, altitude_km, efficiency=SYSTEM_EFFICIENCY,
                    aperture=RECEIVING_APERTURE, lam=LAMBDA):
    """Received power (MW) at full farm capacity, Eq. (11) at slant range z."""
    dist_km = dist_m / 1000.0
    denominator = np.pi * lam * np.sqrt(dist_km ** 2 + altitude_km ** 2)
    return (p_cap_safe / denominator) * efficiency * aperture / 1000.0


def allocate_power(ts_idx_all, trip_all, dist_all, p_cap_safe, altitude_km,
                   max_flight_power=MAX_FLIGHT_POWER):
    """Received power of every active sample of one farm, nearest first."""
    if len(ts_idx_all) == 0:
        return None
    order = np.lexsort((dist_all, ts_idx_all))
    ts_idx, trip_id, dist_m = ts_idx_all[order], trip_all[order], dist_all[order]
    potential = potential_power(dist_m, p_cap_safe, altitude_km)
    power = _waterfall_power_kernel(ts_idx, potential, max_flight_power)
    active = power > 0.0
    if not np.any(active):
        return None
    return ts_idx[active], trip_id[active], power[active]


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def farm_metrics(power, farm_id, fuel_basis="energy"):
    """Energy, duration, cost and emissions of one farm from its sample powers.

    fuel_basis="energy" prices the displaced fuel per delivered MWh (R1);
    "duration" prices it per hour of supported flight (first submission).
    """
    energy_mwh = power.sum() * TIME_INTERVAL / 3600.0
    duration_h = len(power) * TIME_INTERVAL / 3600.0
    if fuel_basis == "energy":
        fuel_cost = energy_mwh * FUEL_COST_PER_MWH
        fuel_emis = energy_mwh * FUEL_EMISSION_PER_MWH
    else:
        fuel_cost = duration_h * FUEL_COST_PER_HR
        fuel_emis = duration_h * FUEL_EMISSION_PER_HR
    return {
        "Solar_Farm_ID": farm_id,
        "Total_Energy_Supplied_MWh": energy_mwh,
        "Total_Flight_Duration_Supported_Hours": duration_h,
        "Production_Beaming_Cost": energy_mwh * COST_PER_MWH,
        "CO2_Emissions_Beaming": energy_mwh * EMISSION_PER_MWH,
        "Original_Fuel_Cost_Baseline": fuel_cost,
        "Original_CO2_Emissions_Baseline": fuel_emis,
    }


def run_power_model(df_results, farm_centers, farm_details, altitude_km,
                    fuel_basis="energy", duration_resolution="ns", verbose=False):
    """Run the power model over every farm of an overlap table.

    Returns (farm results, flight results). A flight's energy and supported
    duration are summed over all the farms that serve it.
    """
    farm_rows, trips, energy, duration = [], [], [], []
    grouped = df_results.groupby("Solar_Farm_ID", sort=False)
    for i, (farm_id, farm_flights) in enumerate(grouped, start=1):
        if verbose and i % 50 == 0:
            print("  farm %d/%d" % (i, len(grouped)), flush=True)
        if farm_id not in farm_centers or farm_id not in farm_details.index:
            continue
        expanded = expand_flights(farm_flights, farm_centers[farm_id],
                                  duration_resolution=duration_resolution)
        if expanded is None:
            continue
        ts_idx_all, trip_all, dist_all, _ = expanded
        active = allocate_power(ts_idx_all, trip_all, dist_all,
                                farm_details.loc[farm_id, "p_cap_safe"], altitude_km)
        if active is None:
            continue
        _, trip_active, power_active = active
        farm_rows.append(farm_metrics(power_active, farm_id, fuel_basis))

        unique_trips, inv = np.unique(trip_active, return_inverse=True)
        trips.append(unique_trips)
        energy.append(np.bincount(inv, weights=power_active) * TIME_INTERVAL / 3600.0)
        duration.append(np.bincount(inv) * TIME_INTERVAL / 60.0)

    df_farms = pd.DataFrame(farm_rows)
    if trips:
        all_trips, inv = np.unique(np.concatenate(trips), return_inverse=True)
        df_flights = pd.DataFrame({
            "Trip_ID": all_trips,
            "Total_Power_Received_MWh": np.bincount(inv, weights=np.concatenate(energy)),
            "Duration_Supported_Min": np.bincount(inv, weights=np.concatenate(duration)),
        })
    else:
        df_flights = pd.DataFrame(columns=["Trip_ID", "Total_Power_Received_MWh",
                                           "Duration_Supported_Min"])
    return df_farms, df_flights


def merge_farm_results(gdf_solar, df_farms):
    """Attach the farm attributes and the net reduction and saving columns."""
    merged = pd.merge(gdf_solar, df_farms, left_on="FID", right_on="Solar_Farm_ID",
                      how="inner")
    merged["CO2_Emissions_Reduction"] = (merged["Original_CO2_Emissions_Baseline"]
                                         - merged["CO2_Emissions_Beaming"])
    merged["Money_Cost_Saving"] = (merged["Original_Fuel_Cost_Baseline"]
                                   - merged["Production_Beaming_Cost"])
    return merged


def merge_flight_results(df_flights_all, df_flight_results, fuel_basis="energy"):
    """Attach the flight records and the per-flight cost and emissions."""
    merged = pd.merge(df_flights_all, df_flight_results, on="Trip_ID", how="inner")
    energy = merged["Total_Power_Received_MWh"]
    merged["Production_Beaming_Cost"] = energy * COST_PER_MWH
    merged["CO2_Emissions_Beaming"] = energy * EMISSION_PER_MWH
    if fuel_basis == "energy":
        merged["Original_Fuel_Cost_Baseline"] = energy * FUEL_COST_PER_MWH
        merged["Original_CO2_Emissions_Baseline"] = energy * FUEL_EMISSION_PER_MWH
    else:
        hours = merged["Duration_Supported_Min"] / 60
        merged["Original_Fuel_Cost_Baseline"] = hours * FUEL_COST_PER_HR
        merged["Original_CO2_Emissions_Baseline"] = hours * FUEL_EMISSION_PER_HR
    merged["CO2_Emissions_Reduction"] = (merged["Original_CO2_Emissions_Baseline"]
                                         - merged["CO2_Emissions_Beaming"])
    merged["Money_Cost_Saving"] = (merged["Original_Fuel_Cost_Baseline"]
                                   - merged["Production_Beaming_Cost"])
    return merged


def add_energy_basis(results):
    """Keep the duration-based columns under *_by_Duration names and add the
    delivered-energy columns *_by_Energy used in the R1 analysis."""
    df = results.rename(columns={
        "Original_Fuel_Cost_Baseline": "Original_Fuel_Cost_by_Duration",
        "Original_CO2_Emissions_Baseline": "Original_CO2_Emissions_by_Duration",
        "CO2_Emissions_Reduction": "CO2_Emissions_Reduction_by_Duration",
        "Money_Cost_Saving": "Fuel_Cost_Savings_by_Duration"})
    energy = (df["Total_Energy_Supplied_MWh"] if "Total_Energy_Supplied_MWh" in df.columns
              else df["Total_Power_Received_MWh"])
    df["Original_Fuel_Cost_by_Energy"] = energy * FUEL_COST_PER_MWH
    df["Original_CO2_Emissions_by_Energy"] = energy * FUEL_EMISSION_PER_MWH
    df["CO2_Emissions_Reduction_by_Energy"] = (df["Original_CO2_Emissions_by_Energy"]
                                               - df["CO2_Emissions_Beaming"])
    df["Fuel_Cost_Savings_by_Energy"] = (df["Original_Fuel_Cost_by_Energy"]
                                         - df["Production_Beaming_Cost"])
    return df
