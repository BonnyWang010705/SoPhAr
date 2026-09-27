"""Route maps of Fig. 3b-d: flight connections by range class.

Each map draws the unique origin-destination routes of the analysed week in
one range class (short < 1,500 km, medium 1,500-4,000 km, long > 4,000 km) as
arcs over a dark basemap, with the airports as white points. Flights to or
from AK, HI, PR, GU and VI are left out.

    python route_maps_fig3bd.py

Reads data/flights/processed/flight_with_od_coor.csv and writes
results/baseline/route_maps/R1_Flight_<Class>_Range_<n>_Unique_Routes.png,
which assemble.py places in Fig. 3. The basemap (CARTO dark, OpenStreetMap
data) is downloaded, so internet access is needed.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import contextily as ctx                 # noqa: E402
import geopandas as gpd                  # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from shapely.geometry import LineString, Point  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "code", "2_coverage_and_savings"))
import model  # noqa: E402

OUT = os.path.join(model.BASELINE, "route_maps")
EXCLUDED_STATES = ["PR", "GU", "HI", "AK", "VI"]
BASEMAP = ctx.providers.CartoDB.DarkMatter
ATTRIBUTION = "© OpenStreetMap contributors © CARTO"


def create_bezier_arc(x1, y1, x2, y2, bend_factor=0.15, num_points=100):
    """A curved line between two points of a projected plane (quadratic Bezier)."""
    if x1 == x2 and y1 == y2:
        return LineString([(x1, y1), (x2, y2)])
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    # control point pushed out perpendicular to the chord
    ctrl_x, ctrl_y = mid_x - dy * bend_factor, mid_y + dx * bend_factor
    t = np.linspace(0, 1, num_points)
    arc_x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * ctrl_x + t ** 2 * x2
    arc_y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * ctrl_y + t ** 2 * y2
    return LineString(zip(arc_x, arc_y))


def routes_and_airports(flights):
    """Unique routes as arcs and their airports, both in EPSG:5070."""
    flights = flights[~flights["STATE_Origin"].isin(EXCLUDED_STATES)
                      & ~flights["STATE_Destination"].isin(EXCLUDED_STATES)]
    unique = flights.drop_duplicates(subset=["Origin Airport", "Destination Airport"]).copy()

    origins = unique[["Longitude_Origin", "Latitude_Origin"]].rename(
        columns={"Longitude_Origin": "lon", "Latitude_Origin": "lat"})
    destinations = unique[["Longitude_Destination", "Latitude_Destination"]].rename(
        columns={"Longitude_Destination": "lon", "Latitude_Destination": "lat"})
    ports = pd.concat([origins, destinations]).drop_duplicates()
    airports = gpd.GeoDataFrame(ports, geometry=[Point(xy) for xy in zip(ports.lon, ports.lat)],
                                crs=model.WGS84_CRS).to_crs(epsg=5070)

    orig = gpd.GeoSeries([Point(x, y) for x, y in zip(unique["Longitude_Origin"],
                                                      unique["Latitude_Origin"])],
                         crs=model.WGS84_CRS).to_crs(epsg=5070)
    dest = gpd.GeoSeries([Point(x, y) for x, y in zip(unique["Longitude_Destination"],
                                                      unique["Latitude_Destination"])],
                         crs=model.WGS84_CRS).to_crs(epsg=5070)
    arcs = [create_bezier_arc(x1, y1, x2, y2, bend_factor=0.15)
            for x1, y1, x2, y2 in zip(orig.x.values, orig.y.values, dest.x.values, dest.y.values)]
    unique = gpd.GeoDataFrame(unique.drop(columns="geometry", errors="ignore"),
                              geometry=arcs, crs="EPSG:5070")
    return unique, airports


def main():
    flights = model.load_flights()
    unique, airports = routes_and_airports(flights)
    km = unique["Distance_km"]
    categories = [
        ("Short", "Short Range (<1500km)", km < 1500, "#00d2ff"),
        ("Medium", "Medium Range (1500-4000km)", (km >= 1500) & (km <= 4000), "#ffd700"),
        ("Long", "Long Range (>4000km)", km > 4000, "#ff5e5e"),
    ]
    os.makedirs(OUT, exist_ok=True)
    for name, label, mask, color in categories:
        subset = unique[mask]
        fig, ax = plt.subplots(figsize=(20, 15))
        if not subset.empty:
            subset.plot(ax=ax, color=color, linewidth=0.8, alpha=0.4, label=label, zorder=5)
        airports.plot(ax=ax, color="white", markersize=15, alpha=0.7, label="Airports",
                      zorder=10)
        ctx.add_basemap(ax, crs=5070, source=BASEMAP, attribution=ATTRIBUTION)
        ax.legend(loc="upper right", facecolor="black", labelcolor="white", fontsize=12)
        ax.set_axis_off()
        path = os.path.join(OUT, "R1_Flight_%s_Range_%d_Unique_Routes.png" % (name, len(subset)))
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("%s range: %d airports, %d unique routes -> %s"
              % (name, len(airports), len(subset), path))


if __name__ == "__main__":
    main()
