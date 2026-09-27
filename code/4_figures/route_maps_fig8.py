"""Route maps of the flights selected in each penetration scenario.

For every farm-and-flight choice scenario (3_optimization/
3_farm_flight_selection.py) the unique routes of the selected flights are
drawn as arcs coloured by range class over a dark basemap. Fig. 8a (arXiv
version) and Supplementary Figs. 14-23 place these maps.

    python route_maps_fig8.py                       # all 100 scenarios
    python route_maps_fig8.py --farm-rates 0.2 0.5 0.8 --flight-rates 0.2 0.5 0.8

Reads results/optimization_2/Optimization_2_Results_R1/ and writes
results/optimization_2/route_maps/flight_map_<F>farm_<I>flight.png. The basemap
(CARTO dark, OpenStreetMap data) is downloaded, so internet access is needed.
"""
import argparse
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import contextily as ctx                 # noqa: E402
import pandas as pd                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "2_coverage_and_savings"))
import model                                              # noqa: E402
from route_maps_fig3bd import BASEMAP, routes_and_airports  # noqa: E402

RATES = [round(0.1 * k, 1) for k in range(1, 11)]
RESULTS = os.path.join(model.OPT2, "Optimization_2_Results_R1")
OUT = os.path.join(model.OPT2, "route_maps")


def flight_map(selected):
    unique, airports = routes_and_airports(selected)
    km = unique["Distance_km"]
    categories = [
        ("Short Range (<1500km)", km < 1500, "#00d2ff", 6, 0.8),
        ("Medium Range (1500-4000km)", (km >= 1500) & (km <= 4000), "#ffd700", 5, 1),
        ("Long Range (>4000km)", km > 4000, "#ff5e5e", 7, 2),
    ]
    fig, ax = plt.subplots(figsize=(20, 15), dpi=150)
    for label, mask, color, zorder, linewidth in categories:
        subset = unique[mask]
        if not subset.empty:
            subset.plot(ax=ax, color=color, linewidth=linewidth, alpha=0.4, label=label,
                        zorder=zorder)
    airports.plot(ax=ax, color="white", markersize=15, alpha=0.7, label="Airports", zorder=10)
    ctx.add_basemap(ax, crs=5070, source=BASEMAP)
    ax.legend(loc="upper right", facecolor="black", labelcolor="white", fontsize=12)
    ax.set_xlim(-2.5e6, 2.3e6)
    ax.set_ylim(0.2e6, 3.3e6)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--farm-rates", type=float, nargs="+", default=RATES)
    ap.add_argument("--flight-rates", type=float, nargs="+", default=RATES)
    a = ap.parse_args()
    if not os.path.isdir(RESULTS):
        sys.exit("missing %s\nrun: python code/3_optimization/3_farm_flight_selection.py"
                 % RESULTS)
    os.makedirs(OUT, exist_ok=True)
    for farm_p in [round(r, 1) for r in a.farm_rates]:
        for flight_p in [round(r, 1) for r in a.flight_rates]:
            selected = pd.read_csv(os.path.join(RESULTS, "flight_results_%.1ffarm_%.1fflight.csv"
                                                % (farm_p, flight_p)), low_memory=False)
            name = "flight_map_%sfarm_%sflight.png" % (farm_p, flight_p)
            with open(os.path.join(OUT, name), "wb") as f:
                f.write(flight_map(selected))
            print("  %s" % name, flush=True)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
