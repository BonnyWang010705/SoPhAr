import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import agg
import maps
import style as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
OUT = os.path.join(ROOT, "figures", "main", "fig02_solar_farm_analysis", "panels")

CAP_MAX = 8000

PANEL_E_ORDER = [
    "Texas", "California", "Nevada", "Florida", "Arizona", "Georgia", "Utah",
    "Virginia", "Ohio", "North Carolina", "Colorado", "Wisconsin", "Indiana",
    "New Mexico", "South Carolina", "Arkansas", "Illinois", "Alabama",
    "Michigan", "Mississippi", "Idaho", "Tennessee", "Iowa", "Wyoming",
    "South Dakota", "Washington", "Louisiana", "Oregon", "Montana",
    "Minnesota", "Pennsylvania", "Oklahoma"]

# Panel b sits beside a map, so it occupies half the printed width; panel e
# spans the whole width. Sizes are set from the printed target so that all
# the text lands inside the 5-7 pt range the journal allows.
B_W, E_W = 14.245, 15.0      # B_W matches the maps' width
HALF = (3600 - 60) / 2 / 3600
FS = {"title": st.pt(7.0, B_W, HALF), "tick": st.pt(6.0, B_W, HALF),
      "ann": st.pt(6.0, B_W, HALF), "cbar": st.pt(7.0, B_W, HALF),
      "donut": st.pt(5.0, B_W, HALF), "centre": st.pt(5.5, B_W, HALF),
      "axis": st.pt(7.0, B_W, HALF)}
E_FS = {"label": st.pt(7.0, E_W), "tick": st.pt(6.0, E_W),
        "count": st.pt(6.0, E_W), "cbar": st.pt(7.0, E_W),
        "bar_value": st.pt(5.5, E_W)}

def panel_b(tab):
    states = list(tab.index)[::-1]
    total_vals = list(tab.total_pct)[::-1]
    depth_cap = list(tab.capacity_mw)[::-1]

    y = np.arange(len(states))
    height = 0.7

    fig, ax = plt.subplots(figsize=(B_W, 8.04))
    fig.subplots_adjust(left=0.195, right=0.80, bottom=0.17, top=0.88)

    # capacity uses the same colormap as the maps in panels a, c and d
    cmap_cap = plt.get_cmap(st.CMAP_WARM)
    norm_cap = mcolors.Normalize(vmin=0, vmax=CAP_MAX)

    rects = ax.barh(y, total_vals, height,
                    color=cmap_cap(norm_cap(depth_cap)), edgecolor="black")

    ax.set_title("Top-ten states by total cost saving", pad=25,
                 fontsize=FS["title"], fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(states, fontsize=FS["tick"])
    ax.tick_params(axis="x", labelsize=FS["tick"])
    ax.set_xlabel("Share of total cost saving (%)", fontsize=FS["axis"],
                  labelpad=10)

    for rect in rects:
        width = rect.get_width()
        ax.annotate("%g%%" % round(width, 2),
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=FS["ann"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(total_vals) + 6)

    cax = fig.add_axes(maps.BAR_BOX)
    sm = plt.cm.ScalarMappable(cmap=cmap_cap, norm=norm_cap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Total capacity (MW)", rotation=90, labelpad=15,
                 fontsize=FS["cbar"])
    cb.ax.tick_params(labelsize=FS["tick"])

    return fig

def panel_e(farms):
    pos = {name: i for i, name in enumerate(PANEL_E_ORDER)}
    names = farms.p_state.map(st.full_name)
    unknown = sorted(set(names) - set(pos))
    if unknown:
        raise ValueError("states missing from PANEL_E_ORDER: %s" % unknown)
    d = farms.assign(state_name=names, order=names.map(pos)).sort_values("order")
    counts = agg.farm_counts_by_state(farms)
    code_by_name = {name: code for code, name in st.STATE_NAME.items()}
    count_values = np.array([counts.get(code_by_name[name], 0)
                             for name in PANEL_E_ORDER])

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[40, 1.2],
                          height_ratios=[5.0, 1.0], hspace=0.05, wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    count_ax = fig.add_subplot(gs[1, 0], sharex=ax)
    cax = fig.add_subplot(gs[:, 1])
    for gx in range(len(PANEL_E_ORDER)):
        for a in (ax, count_ax):
            a.axvline(gx, color="0.9", linewidth=0.6, zorder=0)
    cost_max = 20000 * np.ceil(d.total_usd.max() / 20000)
    scatter = ax.scatter(d["order"], d.p_cap_safe, c=d.total_usd,
                         cmap="viridis", s=25, zorder=3,
                         vmin=0, vmax=cost_max)

    cb = fig.colorbar(scatter, cax=cax)
    cb.set_ticks(np.arange(0, cost_max + 1, 20000))
    cb.set_label("Total Cost Reduction ($)", labelpad=20,
                 fontsize=E_FS["cbar"])
    cb.ax.tick_params(labelsize=E_FS["tick"])
    ax.set_ylabel("Capacity (MW)", labelpad=20, fontsize=E_FS["label"])
    ax.tick_params(axis="y", labelsize=E_FS["tick"])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.spines["bottom"].set_visible(False)

    x = np.arange(len(PANEL_E_ORDER))
    bars = count_ax.bar(x, count_values, width=0.68, color="#9e9e9e",
                        edgecolor="black", linewidth=0.6)
    count_ax.set_ylabel("Number of\nsolar farms", labelpad=12,
                        fontsize=E_FS["count"])
    count_ax.set_xlabel("State", fontsize=E_FS["label"])
    count_ax.set_xticks(x)
    count_ax.set_xticklabels([st.STATE_CODE[n] for n in PANEL_E_ORDER],
                             fontsize=E_FS["tick"])
    count_ax.tick_params(axis="y", labelsize=E_FS["count"])
    count_ax.spines["top"].set_visible(False)
    count_ax.spines["right"].set_visible(False)
    count_ax.set_ylim(0, count_values.max() * 1.18)
    for bar, value in zip(bars, count_values):
        count_ax.annotate(str(value),
                          (bar.get_x() + bar.get_width() / 2,
                           bar.get_height()),
                          xytext=(0, 2), textcoords="offset points",
                          ha="center", va="bottom",
                          fontsize=E_FS["bar_value"])

    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.14, top=0.96)
    return fig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--altitude", type=int, default=12100)
    a = p.parse_args()

    st.apply()
    farms = agg.load_farms(a.altitude)
    tab = agg.farm_state_ranking(farms, other="Other States")

    for f in st.save(panel_b(tab), os.path.join(OUT, "figure2b")):
        print("wrote", f)
    for f in st.save(panel_e(farms), os.path.join(OUT, "figure2e")):
        print("wrote", f)

    print("\nfarms %d in %d states, %.0f MW safety capacity, $%s delivered"
          % (len(farms), farms.p_state.nunique(), farms.p_cap_safe.sum(),
             format(farms.total_usd.sum(), ",.0f")))
    print("max |fuel share - CO2 share| across states: %.4f pp"
          % (tab.fuel_pct - tab.co2_pct).abs().max())
    print("capacity colour-bar ceiling: %d MW (largest state %.0f)"
          % (CAP_MAX, tab.capacity_mw.max()))

if __name__ == "__main__":
    main()
