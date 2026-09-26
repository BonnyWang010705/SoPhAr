"""Fig. 7 of the arXiv version: solar farms selected by the
farm-and-flight choice optimization, redrawn in the style of the main Fig. 2.

a, 3x3 maps of the selected farms at solar farm penetration (rows) and flight
penetration (columns) of 20, 50 and 80%, states filled by the summed safety
capacity of their selected farms as in Fig. 2a; b, how many of the 100
scenarios select each farm, by state, as in Fig. 2e.

Every panel spans the full printed width, and text is sized with style.pt so
that it prints at 5-7 pt. fig8.py reuses the shared helpers defined here.

Input, read inside the zip without extracting it:
results/optimization_2/Optimization_2_Results_R1.zip (solar_results_* and
flight_results_*, one file per scenario).
Assemble with assemble.py --figure figure7.
"""
import argparse
import os
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd

import agg
import maps
import style as st
from fig2be import PANEL_E_ORDER

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
DATA = os.path.join(ROOT, "results", "optimization_2",
                    "Optimization_2_Results_R1.zip")
DATA_DIR = "Optimization_2_Results_R1"
OUT = os.path.join(ROOT, "figures", "main", "fig07_solar_farm_selection", "panels")

RATES = [0.2, 0.5, 0.8]
ALL_RATES = [round(0.1 * k, 1) for k in range(1, 11)]
N_SCENARIOS = len(ALL_RATES) ** 2
CAP_MAX = 8000          # MW, the colour-bar ceiling of Fig. 2a
N_TOP = 10
FARM_DOT = "#1f3b57"

W = 16.0                # every panel spans the printed width
FS = {"head": st.pt(7.0, W), "label": st.pt(7.0, W), "tick": st.pt(6.0, W)}


def open_results(path=DATA):
    data = zipfile.ZipFile(path)
    n = sum(1 for m in data.namelist()
            if os.path.basename(m).startswith("solar_results_"))
    if n != N_SCENARIOS:
        raise ValueError("expected %d solar_results files, found %d"
                         % (N_SCENARIOS, n))
    return data


def read(data, kind, farm_p, flight_p, **kw):
    """One scenario's file from the open results zip."""
    name = "%s/%s_results_%sfarm_%sflight.csv" % (DATA_DIR, kind, farm_p, flight_p)
    with data.open(name) as f:
        return pd.read_csv(f, low_memory=False, **kw)


def grid_headers(fig, axes, row_x=-0.04, title_x=0.045):
    for j, r in enumerate(RATES):
        axes[0, j].set_title("%d%%" % (100 * r), fontsize=FS["head"], pad=8)
    for i, r in enumerate(RATES):
        axes[i, 0].annotate("%d%%" % (100 * r), xy=(row_x, 0.5),
                            xycoords="axes fraction", ha="right", va="center",
                            rotation=90, fontsize=FS["head"])
    top = max(ax.get_position().y1 for ax in axes[0])
    left = min(ax.get_position().x0 for ax in axes[:, 0])
    mid_x = np.mean([axes[0, 0].get_position().x0, axes[0, -1].get_position().x1])
    mid_y = np.mean([axes[0, 0].get_position().y1, axes[-1, 0].get_position().y0])
    fig.text(mid_x, top + 0.06, "Flight penetration", ha="center",
             fontsize=FS["head"])
    fig.text(left - title_x, mid_y, "Solar farm penetration", ha="center",
             va="center", rotation=90, fontsize=FS["head"])


def panel_a(data, states):
    fig, axes = plt.subplots(3, 3, figsize=(W, 9.4))
    fig.subplots_adjust(left=0.07, right=0.86, top=0.87, bottom=0.02,
                        wspace=0.03, hspace=0.06)
    rows = []
    for i, fp in enumerate(RATES):
        for j, ip in enumerate(RATES):
            sel = read(data, "solar", fp, ip)
            by = sel.groupby("p_state")["p_cap_safe"].sum()
            g = states.copy()
            g["value"] = g.STUSPS.map(by).fillna(0.0)
            ax = axes[i, j]
            g.plot(ax=ax, column="value", cmap=st.CMAP_WARM, vmin=0,
                   vmax=CAP_MAX, edgecolor=maps.EDGE, linewidth=maps.STATE_LW)
            pts = gpd.GeoSeries(gpd.points_from_xy(sel.xlong, sel.ylat),
                                crs="EPSG:4326").to_crs(maps.ALBERS)
            ax.scatter(pts.x, pts.y, s=4, color=FARM_DOT, linewidths=0)
            ax.margins(0)
            ax.axis("off")
            rows.append((fp, ip, len(sel), sel.p_cap_safe.sum(), by.max()))
    grid_headers(fig, axes)
    cax = fig.add_axes([0.885, 0.10, 0.016, 0.62])
    sm = plt.cm.ScalarMappable(cmap=st.CMAP_WARM,
                               norm=plt.Normalize(vmin=0, vmax=CAP_MAX))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Selected capacity (MW)", labelpad=10, fontsize=FS["label"])
    cb.ax.tick_params(labelsize=FS["tick"])
    dot = plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                     color=FARM_DOT, label="Selected\nsolar farm")
    fig.legend(handles=[dot], loc="lower left", bbox_to_anchor=(0.875, 0.76),
               frameon=False, fontsize=FS["tick"], handletextpad=0.2)
    return fig, rows


def panel_b(data, height=6.2):
    counts = pd.Series(dtype=float)
    info = []
    for fp in ALL_RATES:
        for ip in ALL_RATES:
            sel = read(data, "solar", fp, ip,
                       usecols=["FID", "p_state", "p_cap_safe"])
            counts = counts.add(pd.Series(1.0, index=sel.FID), fill_value=0.0)
            info.append(sel)
    farms = pd.concat(info).drop_duplicates("FID").set_index("FID")
    farms["count"] = counts.reindex(farms.index)
    names = farms.p_state.map(st.full_name)
    pos = {n: k for k, n in enumerate(PANEL_E_ORDER)}
    unknown = sorted(set(names) - set(pos))
    if unknown:
        raise ValueError("states missing from PANEL_E_ORDER: %s" % unknown)
    d = farms.assign(state_name=names, order=names.map(pos)).sort_values("order")

    fig = plt.figure(figsize=(W, height))
    gs = fig.add_gridspec(1, 2, width_ratios=[40, 1.2], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    for gx in range(len(PANEL_E_ORDER)):
        ax.axvline(gx, color="0.9", linewidth=0.6, zorder=0)
    sc = ax.scatter(d["order"], d["count"], c=d.p_cap_safe, cmap="viridis",
                    s=25, zorder=3)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Capacity (MW)", labelpad=15, fontsize=FS["label"])
    cb.ax.tick_params(labelsize=FS["tick"])
    ax.set_ylabel("Scenarios selecting the farm", labelpad=15,
                  fontsize=FS["label"])
    ax.set_xlabel("State", fontsize=FS["label"])
    ax.set_ylim(0, N_SCENARIOS + 4)
    ax.set_xlim(-0.7, len(PANEL_E_ORDER) - 0.3)
    ax.set_xticks(np.arange(len(PANEL_E_ORDER)))
    ax.set_xticklabels([st.STATE_CODE[n] for n in PANEL_E_ORDER],
                       fontsize=FS["tick"])
    ax.tick_params(axis="y", labelsize=FS["tick"])
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.14, top=0.97)
    return fig, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA)
    a = p.parse_args()
    data = open_results(a.data)

    st.apply()
    states, _ = maps.load_boundaries()
    fig, rows = panel_a(data, states)
    for f in st.save(fig, os.path.join(OUT, "figure7a")):
        print("wrote", f)
    fig, farms = panel_b(data)
    for f in st.save(fig, os.path.join(OUT, "figure7b")):
        print("wrote", f)

    print("\npanel a: farm p, flight p, farms, selected MW, largest state MW")
    for r in rows:
        print("  %.1f %.1f %4d %9.0f %7.0f" % r)
    print("panel b: %d farms ever selected, %d in all %d scenarios, %d states"
          % (len(farms), (farms["count"] == N_SCENARIOS).sum(), N_SCENARIOS,
             farms.p_state.nunique()))


if __name__ == "__main__":
    main()
