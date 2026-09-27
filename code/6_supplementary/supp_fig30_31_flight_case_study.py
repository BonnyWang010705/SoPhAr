"""Supplementary Figs. 30-31: flight case study.

Flight AA 1459 from Austin (AUS) to Los Angeles (LAX), 9 April 2025,
00:37-04:10 UTC, at 12,100 m:
  Flight_22663_Map.png          the route and the beaming ranges of the
                                qualified farms it crosses
  Flight_22663_Power_Curve.png  power received along the flight (1 s means),
                                each farm serving the flight alone, the total
                                capped at the 10.3 MW cruise demand

    python supp_fig30_31_flight_case_study.py

This case study was drawn from the flight data of the first submission, in
which the flight is Trip_ID 22663. The flight record and its crossings are
read from data/case_study/flight_22663_record.csv and
flight_22663_crossings.csv. Writes to
figures/supplementary/fig30_31_flight_case_study/. The map downloads
OpenStreetMap tiles (internet access needed).
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates        # noqa: E402
import matplotlib.lines as mlines        # noqa: E402
import matplotlib.patches as mpatches    # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
import contextily as ctx                 # noqa: E402
import geopandas as gpd                  # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import seaborn as sns                    # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from pyproj import Transformer           # noqa: E402
from shapely.geometry import LineString  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "code", "2_coverage_and_savings"))
import model  # noqa: E402

CASE_FLIGHT = 22663
CASE_DIR = os.path.join(ROOT, "data", "case_study")
RECORD = os.path.join(CASE_DIR, "flight_%d_record.csv" % CASE_FLIGHT)
CROSSINGS = os.path.join(CASE_DIR, "flight_%d_crossings.csv" % CASE_FLIGHT)
OUT = os.path.join(ROOT, "figures", "supplementary", "fig30_31_flight_case_study")
ALTITUDE = 12100
FREQUENCY = "200ms"


def load():
    for path in (RECORD, CROSSINGS):
        if not os.path.exists(path):
            sys.exit("missing %s\nrun: python download_data.py" % path)
    gdf_solar, gdf_solar_proj, _, _ = model.load_solar_farms()
    buffers = gdf_solar_proj.copy()
    buffers["geometry"] = buffers.geometry.buffer(ALTITUDE)

    record = pd.read_csv(RECORD)
    for col in ("Wheels-off time (UTC)", "Landing time (UTC)"):
        record[col] = pd.to_datetime(record[col])
    line = LineString([(record["Longitude_Origin"].iloc[0], record["Latitude_Origin"].iloc[0]),
                       (record["Longitude_Destination"].iloc[0],
                        record["Latitude_Destination"].iloc[0])])
    flight = gpd.GeoDataFrame(record, geometry=[line], crs=model.WGS84_CRS)

    # crossings of the qualified farms, with metric coordinates and capacity
    df_case = pd.read_csv(CROSSINGS)
    df_case = df_case[df_case["Solar_Farm_ID"].isin(set(gdf_solar_proj["FID"]))]
    df_case["Entry_Time"] = pd.to_datetime(df_case["Entry_Time"])
    df_case["Exit_Time"] = pd.to_datetime(df_case["Exit_Time"])
    to_metric = Transformer.from_crs(model.WGS84_CRS, model.METRIC_CRS, always_xy=True)
    df_case["Entry_X"], df_case["Entry_Y"] = to_metric.transform(
        df_case["Entry_Longitude"].values, df_case["Entry_Latitude"].values)
    df_case["Exit_X"], df_case["Exit_Y"] = to_metric.transform(
        df_case["Exit_Longitude"].values, df_case["Exit_Latitude"].values)
    df_case = pd.merge(df_case, gdf_solar[["FID", "p_cap_safe"]], left_on="Solar_Farm_ID",
                       right_on="FID").drop(columns=["FID"])
    centroids = gdf_solar_proj.set_index("FID").geometry.centroid
    df_case["centroid_x"] = centroids.loc[df_case["Solar_Farm_ID"]].x.values
    df_case["centroid_y"] = centroids.loc[df_case["Solar_Farm_ID"]].y.values
    return gdf_solar_proj, buffers, flight, df_case


def route_map(gdf_solar_proj, buffers, flight, df_case, path):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    flight.to_crs(model.METRIC_CRS).plot(ax=ax, linewidth=0.7, color="k")
    crossed = df_case["Solar_Farm_ID"]
    buffers[buffers["FID"].isin(crossed)].plot(ax=ax, facecolor="tab:blue", alpha=0.8,
                                               linewidth=0.8)
    gdf_solar_proj[gdf_solar_proj["FID"].isin(crossed)].centroid.plot(
        ax=ax, markersize=0.5, facecolor="orange")
    ctx.add_basemap(ax=ax, crs=model.METRIC_CRS, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_ylim(6.5e5, 1.7e6)
    ax.set_axis_off()
    blue_patch = mpatches.Patch(color="tab:blue", alpha=0.8, linewidth=0,
                                label="Power Transfer Range (12.1 km)")
    red_patch = mpatches.Patch(color="orange", label="Solar Farm")
    flight_line = mlines.Line2D([], [], color="k", linewidth=1.2, label="Flight Route")
    ax.legend(handles=[blue_patch, red_patch, flight_line], fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def power_curve(flight, df_case):
    """Received power at 1 s resolution, farms serving the flight alone."""
    min_time = df_case["Entry_Time"].min().floor("1s")
    max_time = df_case["Exit_Time"].max().ceil("1s")
    grid = pd.date_range(start=min_time, end=max_time, freq=FREQUENCY)
    points = []
    for _, row in df_case.iterrows():
        ts = grid[(grid >= row["Entry_Time"]) & (grid <= row["Exit_Time"])]
        if len(ts) == 0:
            continue
        total = (row["Exit_Time"] - row["Entry_Time"]).total_seconds()
        if total <= 0:
            continue
        ratios = (ts - row["Entry_Time"]).total_seconds() / total
        x = row["Entry_X"] + ratios * (row["Exit_X"] - row["Entry_X"])
        y = row["Entry_Y"] + ratios * (row["Exit_Y"] - row["Entry_Y"])
        points.append(pd.DataFrame({
            "Time_Stamp": ts, "Solar_Farm_ID": row["Solar_Farm_ID"],
            "Distance_m": np.sqrt((x - row["centroid_x"]) ** 2 + (y - row["centroid_y"]) ** 2),
            "p_cap_safe": row["p_cap_safe"]}))
    pts = pd.concat(points).sort_values(by=["Time_Stamp", "Distance_m"])
    potential = model.potential_power(pts["Distance_m"], pts["p_cap_safe"], ALTITUDE / 1000)
    needed = np.where(potential <= model.MAX_FLIGHT_POWER, 1.0,
                      model.MAX_FLIGHT_POWER / potential)
    pts["Power_Calculated"] = needed * potential
    curve = (pts.groupby("Time_Stamp")[["Power_Calculated"]].sum()
             .clip(upper=model.MAX_FLIGHT_POWER))

    # the whole flight, wheels-off to landing
    index = pd.date_range(start=flight["Wheels-off time (UTC)"].iloc[0],
                          end=flight["Landing time (UTC)"].iloc[0], freq=FREQUENCY)
    curve = curve.reindex(index)
    curve["Power_Calculated"] = curve["Power_Calculated"].fillna(0)
    out = curve.resample("1s").mean().reset_index()
    out["Time_Stamp"] = pd.to_datetime(out["index"], utc=True).dt.tz_convert("US/Central")
    return out


def plot_power(df, path):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(df["Time_Stamp"], df["Power_Calculated"], color="#d18b1b", linewidth=2,
            label="Power Received")
    ax.set_xlabel("Time", fontsize=12, labelpad=0)
    ax.set_ylabel("Power (MW)", fontsize=12, labelpad=1)
    my_tz = df["Time_Stamp"].dt.tz

    def custom_date_formatter(x, pos):
        return mdates.num2date(x).astimezone(my_tz).strftime("%H:%M:%S\n")

    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 20, 40], tz=my_tz))
    ax.xaxis.set_major_formatter(FuncFormatter(custom_date_formatter))
    plt.xticks(rotation=0)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-map", action="store_true", help="skip the map (no tiles)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    gdf_solar_proj, buffers, flight, df_case = load()
    stem = os.path.join(OUT, "Flight_%d" % CASE_FLIGHT)
    if not a.no_map:
        route_map(gdf_solar_proj, buffers, flight, df_case, stem + "_Map.png")
    curve = power_curve(flight, df_case)
    plot_power(curve, stem + "_Power_Curve.png")
    print("flight %d: %d qualified farms crossed, %.2f MWh received (no contention)"
          % (CASE_FLIGHT, df_case["Solar_Farm_ID"].nunique(),
             curve["Power_Calculated"].sum() / 3600))
    print("wrote %s_*.png" % stem)


if __name__ == "__main__":
    main()
