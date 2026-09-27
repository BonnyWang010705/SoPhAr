"""Supplementary Figs. 24-29: solar farm case studies.

Two farms at the reference altitude of 12,100 m:
  FID 230   Desert Sunlight 300 (CA), second in delivered energy;
  FID 4746  the largest farm by area among the farms that serve flights (TX).
For each farm three figures:
  <farm>_Energy_Curve.png             hourly energy delivered over the week
  <farm>_Map_Location.png             the farm and its 12.1 km beaming range
  <farm>_Map_Flight_Intersection.png  the routes that cross the range

    python supp_fig24_29_farm_case_studies.py               # both farms
    python supp_fig24_29_farm_case_studies.py --farm 230

Reads the overlap events and farm results of code/2_coverage_and_savings and
writes to figures/supplementary/fig24_29_case_studies/, with a text file of
each farm's attributes and results. The maps download OpenStreetMap tiles
(internet access needed).
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
import pandas as pd                      # noqa: E402
import seaborn as sns                    # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "code", "2_coverage_and_savings"))
import model  # noqa: E402

OUT = os.path.join(ROOT, "figures", "supplementary", "fig24_29_case_studies")
ALTITUDE = 12100
# farm -> local time zone of the energy curve
FARMS = {230: "America/Los_Angeles", 4746: "US/Central"}
OSM_HEADERS = {"User-Agent": "SoPhAr-case-study-maps/1.0"}


def load():
    gdf_solar, gdf_solar_proj, farm_centers, farm_details = model.load_solar_farms()
    gdf_solar_proj_buffer = gdf_solar_proj.copy()
    gdf_solar_proj_buffer["geometry"] = gdf_solar_proj_buffer.geometry.buffer(ALTITUDE)
    df_results = model.load_overlaps(ALTITUDE, gdf_solar_proj["FID"].unique(),
                                     gdf_solar=gdf_solar)
    flights = model.load_flights()
    lines = [LineString([(lon_o, lat_o), (lon_d, lat_d)]) for lon_o, lat_o, lon_d, lat_d in
             zip(flights["Longitude_Origin"], flights["Latitude_Origin"],
                 flights["Longitude_Destination"], flights["Latitude_Destination"])]
    gdf_flights = gpd.GeoDataFrame(flights, geometry=lines, crs=model.WGS84_CRS)
    return dict(solar=gdf_solar, proj=gdf_solar_proj, buffer=gdf_solar_proj_buffer,
                centers=farm_centers, details=farm_details, overlaps=df_results,
                flights=gdf_flights)


def hourly_energy(d, farm):
    """Energy delivered by the farm in each UTC hour (MWh)."""
    df_case = d["overlaps"][d["overlaps"]["Solar_Farm_ID"] == farm]
    ts_idx, trip, dist, origin_ns = model.expand_flights(df_case, d["centers"][farm],
                                                         duration_resolution="us")
    ts_idx, _, power = model.allocate_power(ts_idx, trip, dist,
                                            d["details"].loc[farm, "p_cap_safe"],
                                            ALTITUDE / 1000)
    stamps = pd.to_datetime(origin_ns + ts_idx * model.STEP_NS)
    out = pd.DataFrame({"hour": stamps.floor("h"), "power": power})
    curve = (out.groupby("hour")["power"].sum() * model.TIME_INTERVAL / 3600).reset_index()
    curve.columns = ["Time_Stamp_Hour", "Total_Power_Received_MWh"]
    curve["Time_Stamp_Hour"] = pd.to_datetime(curve["Time_Stamp_Hour"], utc=True)
    return df_case, curve


def energy_curve(curve, tz, path):
    curve = curve.copy()
    curve["Time_Stamp_Hour"] = curve["Time_Stamp_Hour"].dt.tz_convert(tz)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(curve["Time_Stamp_Hour"], curve["Total_Power_Received_MWh"], color="#1f77b4",
            linewidth=2, marker="o", markersize=3, label="Energy Supplied")
    ax.set_ylabel("Energy (MWh)", fontsize=12, labelpad=10)
    my_tz = curve["Time_Stamp_Hour"].dt.tz

    def custom_date_formatter(x, pos):
        dt = mdates.num2date(x).astimezone(my_tz)
        # date under the noon tick, hour only under the midnight tick
        if dt.hour == 12:
            return dt.strftime("%H:%M\n%Y-%m-%d")
        return dt.strftime("%H:%M\n")

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12, tz=my_tz))
    ax.xaxis.set_major_formatter(FuncFormatter(custom_date_formatter))
    plt.xticks(rotation=0)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def map_location(d, farm, path):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    d["proj"][d["proj"]["FID"] == farm].plot(ax=ax, color="orange", zorder=5, linewidth=0)
    d["buffer"][d["buffer"]["FID"] == farm].plot(ax=ax, color="blue", alpha=0.3, zorder=1,
                                                 linewidth=0)
    d_x, d_y = 280000, 200000
    c_x, c_y = d["centers"][farm]
    ax.set_xlim(c_x - d_x, c_x + d_x)
    ax.set_ylim(c_y - d_y, c_y + d_y)
    blue_patch = mpatches.Patch(color="blue", alpha=0.3, label="Power Transfer Range (12.1 km)")
    red_patch = mpatches.Patch(color="orange", label="Solar Farm")
    ctx.add_basemap(ax=ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=model.METRIC_CRS,
                    headers=OSM_HEADERS)
    ax.set_axis_off()
    ax.legend(handles=[blue_patch, red_patch])
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def map_intersection(d, farm, df_case, path):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    d["proj"][d["proj"]["FID"] == farm].plot(ax=ax, color="orange", zorder=5, linewidth=0)
    d["buffer"][d["buffer"]["FID"] == farm].plot(ax=ax, color="blue", alpha=0.3, linewidth=0,
                                                 zorder=1)
    routes = (d["flights"][d["flights"]["Trip_ID"].isin(df_case["Trip_ID"])]
              .drop_duplicates(subset=["Origin Airport", "Destination Airport"]))
    routes.to_crs(model.METRIC_CRS).plot(ax=ax, color="k", linewidth=0.5, alpha=0.7)
    c_x, c_y = d["centers"][farm]
    r_lim = ALTITUDE * 5
    ax.set_xlim(c_x - r_lim - ALTITUDE, c_x + r_lim + ALTITUDE)
    ax.set_ylim(c_y - r_lim, c_y + r_lim)
    blue_patch = mpatches.Patch(color="blue", alpha=0.3, label="Power Transfer Range (12.1 km)")
    red_patch = mpatches.Patch(color="orange", label="Solar Farm")
    flight_line = mlines.Line2D([], [], color="k", linewidth=1.2, label="Flight Route")
    ctx.add_basemap(ax=ax, crs=model.METRIC_CRS, source=ctx.providers.OpenStreetMap.Mapnik,
                    headers=OSM_HEADERS)
    ax.set_axis_off()
    ax.legend(handles=[blue_patch, red_patch, flight_line])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--farm", type=int, nargs="+", default=list(FARMS), choices=list(FARMS),
                    help="farm FIDs (default: 230 4746)")
    ap.add_argument("--no-maps", action="store_true", help="skip the two maps (no tiles)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = load()
    results = pd.read_csv(model.farm_file(ALTITUDE))
    for farm in a.farm:
        stem = os.path.join(OUT, "Solar_Farm_ID%d" % farm)
        info = results[results["FID"] == farm].iloc[0]
        with open(stem + "_Info.txt", "w") as f:
            f.write(info.to_string() + "\n")
        df_case, curve = hourly_energy(d, farm)
        energy_curve(curve, FARMS[farm], stem + "_Energy_Curve.png")
        print("farm %d: %s MWh over %d hours with delivery"
              % (farm, format(curve["Total_Power_Received_MWh"].sum(), ",.1f"), len(curve)))
        if not a.no_maps:
            map_location(d, farm, stem + "_Map_Location.png")
            map_intersection(d, farm, df_case, stem + "_Map_Flight_Intersection.png")
        print("wrote %s_*.png" % stem)


if __name__ == "__main__":
    main()
