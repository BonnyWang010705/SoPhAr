"""Fig. 8 of the arXiv version: flights selected by the
farm-and-flight choice optimization, redrawn in the style of the main Fig. 3.

a, route maps of the selected flights in the nine scenarios of Fig. 7, placed as in
Fig. 3b-d; b, state-level total cost reduction of the selected flights by
origin (left bar) and destination (right bar), stacked by range class as in
Fig. 3e, top ten states and the rest as "Other".

Input: results/optimization_2/Optimization_2_Results_R1/ (fuel saving in
Money_Cost_Saving) and the nine flight_map_* images in
results/optimization_2/route_maps/.
Assemble with assemble.py --figure figure8.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import agg
import style as st
from fig7 import (DATA, FS, N_TOP, RATES, ROOT, W, grid_headers,
                  open_results, read)

ROUTE_MAPS = os.path.join(ROOT, "results", "optimization_2", "route_maps")
OUT = os.path.join(ROOT, "figures", "main", "fig08_flight_selection", "panels")


def state_classes(flights, side):
    return (flights.pivot_table(index="STATE_%s" % side, columns="cls",
                                values="total_usd", aggfunc="sum", observed=True)
                   .reindex(columns=agg.CLASS_NAMES).fillna(0.0))


def panel_b(data):
    tabs = {}
    for fp in RATES:
        for ip in RATES:
            f = read(data, "flight", fp, ip,
                     usecols=["STATE_Origin", "STATE_Destination", "Distance_km",
                              "Money_Cost_Saving", "CO2_Emissions_Reduction"])
            f["cls"] = pd.cut(f.Distance_km, agg.CLASS_EDGES,
                              labels=agg.CLASS_NAMES, right=False)
            # Money_Cost_Saving is the fuel saving only; the total adds the
            # CO2 cost as in agg.load_flights
            f["total_usd"] = (f.Money_Cost_Saving
                              + f.CO2_Emissions_Reduction * agg.CARBON_PRICE)
            o, d = state_classes(f, "Origin"), state_classes(f, "Destination")
            idx = o.index.union(d.index)
            o, d = o.reindex(idx).fillna(0.0), d.reindex(idx).fillna(0.0)
            order = (o.sum(axis=1) + d.sum(axis=1)).sort_values(ascending=False).index
            top, rest = list(order[:N_TOP]), list(order[N_TOP:])
            o2 = pd.concat([o.loc[top], o.loc[rest].sum().to_frame("Other").T])
            d2 = pd.concat([d.loc[top], d.loc[rest].sum().to_frame("Other").T])
            tabs[(fp, ip)] = (o2, d2, f.total_usd.sum())
    ymax = max(max(o.sum(axis=1).max(), d.sum(axis=1).max())
               for o, d, _ in tabs.values())

    fig, axes = plt.subplots(3, 3, figsize=(W, 10), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.88, bottom=0.12,
                        wspace=0.06, hspace=0.42)
    width = 0.38
    labels = ["Short range", "Medium range", "Long range"]
    for i, fp in enumerate(RATES):
        for j, ip in enumerate(RATES):
            o, d, _ = tabs[(fp, ip)]
            ax = axes[i, j]
            x = np.arange(len(o))
            for side, tab in enumerate((o, d)):
                bottom = np.zeros(len(tab))
                for k, cls in enumerate(agg.CLASS_NAMES):
                    ax.bar(x + (side - 0.5) * width, tab[cls].values / 1e6,
                           width, bottom=bottom, color=st.CLASS_COLORS[k],
                           edgecolor="black", linewidth=0.6)
                    bottom = bottom + tab[cls].values / 1e6
            ax.set_xticks(x)
            ax.set_xticklabels(list(o.index), rotation=90, fontsize=FS["tick"])
            ax.tick_params(axis="y", labelsize=FS["tick"])
            ax.set_xlim(-0.6, len(o) - 0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if j == 0:
                ax.set_ylabel("Total cost\nreduction ($ million)",
                              fontsize=FS["label"])
    axes[0, 0].set_ylim(0, ymax / 1e6 * 1.05)
    grid_headers(fig, axes, row_x=-0.30, title_x=0.12)
    fig.legend(handles=[mpatches.Patch(facecolor=c, edgecolor="black", label=l)
                        for c, l in zip(st.CLASS_COLORS, labels)],
               loc="lower center", ncol=3, frameon=False, fontsize=FS["tick"],
               bbox_to_anchor=(0.57, 0.0))
    return fig, tabs


def panel_a(folder):
    fig, axes = plt.subplots(3, 3, figsize=(W, 10.8))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.89, bottom=0.01,
                        wspace=0.03, hspace=0.04)
    for i, fp in enumerate(RATES):
        for j, ip in enumerate(RATES):
            name = "flight_map_%sfarm_%sflight.png" % (fp, ip)
            axes[i, j].imshow(Image.open(os.path.join(folder, name)).convert("RGB"))
            axes[i, j].axis("off")
    grid_headers(fig, axes)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA)
    p.add_argument("--route-maps", default=ROUTE_MAPS)
    a = p.parse_args()

    st.apply()
    for f in st.save(panel_a(a.route_maps),
                     os.path.join(OUT, "figure8a")):
        print("wrote", f)
    fig, tabs = panel_b(open_results(a.data))
    for f in st.save(fig, os.path.join(OUT, "figure8b")):
        print("wrote", f)

    print("\npanel b: farm p, flight p, total $, top three states")
    for (fp, ip), (o, d, tot) in tabs.items():
        top = (o.sum(axis=1) + d.sum(axis=1)).drop("Other").index[:3]
        print("  %.1f %.1f %12s  %s" % (fp, ip, format(tot, ",.0f"), ", ".join(top)))


if __name__ == "__main__":
    main()
