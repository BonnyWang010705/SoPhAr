"""Supplementary Figs. 4-23: the market-penetration figures, in the main-figure style.

Twenty figures, as in the submitted document:

  farm selection   (Supplementary Figs. 4-13) one figure per solar farm penetration (10-100%), showing
                   the selected farms at every flight penetration: ten maps
                   (a-j, as in Fig. 2a) and ten scatter plots of farm capacity
                   against total cost reduction (k-t, as in Fig. 2e).
  flight selection (Supplementary Figs. 14-23) one figure per flight penetration, showing the selected
                   flights at every solar farm penetration: ten route maps
                   (a-j, as in Fig. 3b-d) and ten state bar
                   panels (k-t, as in Fig. 3e, every state).

Panels are drawn at the printed width so their text lands in the 5-7 pt range.
Each figure follows the submitted document's layout: every row pairs a map on
the left with its corresponding quantitative panel on the right, and the ten
rows are split into a six-row image (a-f and k-p) and a four-row image (g-j
and q-t).

Inputs: results/optimization_2/Optimization_2_Results_R1.zip (farm selections),
results/optimization_2/penetration_state_totals.csv (written by
supp_fig14_23_state_totals.py) and
results/optimization_2/route_maps/Figure_8_All_Penetrations_R1.zip
(route maps drawn by the optimization visualization).
"""
import argparse
import os
import shutil
import sys
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "code", "4_figures"))

import agg          # noqa: E402
import assemble     # noqa: E402
import maps         # noqa: E402
import style as st  # noqa: E402

OPT2 = os.path.join(ROOT, "results", "optimization_2")
ARCHIVE = os.path.join(OPT2, "Optimization_2_Results_R1.zip")
INSIDE = "Optimization_2_Results_R1"
TOTALS = os.path.join(OPT2, "penetration_state_totals.csv")
ROUTE_ZIP = os.path.join(ROOT, "results", "optimization_2", "route_maps",
                          "Figure_8_All_Penetrations_R1.zip")
ROUTE_INSIDE = "Figure_8_All_Penetrations"
OUT = os.path.join(ROOT, "figures", "supplementary", "fig04_23_penetration")

# saved so that the pictures print at about 300 ppi: the canvas is 16 in wide
# and goes into the document at 6.5 in, so 125 dpi gives 2,000 px, or 308 ppi.
# The default 200 dpi would put nearly 500 ppi in the file and make the
# document twice as large for no visible gain.
SAVE_DPI = 125

W_FULL = 16.0           # the canvas width; every page prints at 6.5 in
RATES = [round(0.1 * k, 1) for k in range(1, 11)]
MAP_LETTERS = "abcdefghij"
PLOT_LETTERS = "klmnopqrst"
# the scales the submitted supplementary document uses: capacity 0-6,000 MW
# in steps of 1,000, farm cost 0-200,000 $ in steps of 25,000, farm capacity
# 0-400 MW, and dollars on the bar axis. The two ceilings are the only places
# where R1 cannot keep the submitted number: the largest state now holds
# 7,650 MW, which 6,000 would clip, and the largest farm saves $96,818, which
# 200,000 would squeeze into the bottom half of the colour bar. Both are
# extended on the submitted step size, and main() checks the data against them.
CAP_MAX = 8000          # MW, the state capacity colour bar
CAP_STEP = 1000
COST_MAX = 100000       # $, the farm cost colour bar
COST_STEP = 25000
FARM_CAP_MAX = 430      # MW, the scatter axis; the submitted document stops
FARM_CAP_TICKS = 100    # at 400, where the largest farm (392 MW) touches the
                        # frame, so the axis is lifted clear of it and keeps
                        # the submitted ticks
BAR_MAX = 1.4e6         # $, the bar axis, shared by all ten flight figures
FARM_DOT = "#1f3b57"
PAGE_GROUPS = (range(0, 6), range(6, 10))
# row heights that reproduce the submitted images once they are placed at the
# 6.5 in text width: 6.5 x 7.99 in and 6.5 x 5.33 in for the farm figures,
# 6.09 x 9.00 in and 6.5 x 6.41 in for the flight figures
# The six-row image has to fit the 8.9 in of page height at the 6.5 in text
# width, or Word scales that page down and the two images of one figure come
# out at different sizes. These row heights put the six-row image just inside
# the page, and the four-row image then keeps exactly the same row height.
# One row of map plus panel, in canvas inches. The submitted document gets its
# large maps by spending nothing on anything else: the rows touch, there is no
# title line above them and no legend band below, so six maps fill the 9 in of
# page height and each one is 2.3 in wide. The layout below does the same. The
# map fills its row; the panel beside it is inset within the row, which leaves
# the space its letter, its state labels and its axis name need (that is where
# the submitted document puts them too).
TOP_IN = 0.05
BOTTOM_IN = 0.10
MAX_PRINT_IN = 8.98        # what one page holds at the 6.5 in text width
ROW = (MAX_PRINT_IN * W_FULL / 6.5 - TOP_IN - BOTTOM_IN) / 6
LEFT_IN = 0.52           # the row letter and its rate, left of the map
GUTTER_IN = 0.85           # the panel's tick labels and its axis name
PANEL_TOP_IN = 0.34        # the panel letter, above the axes
PANEL_BOTTOM_IN = 0.68     # the rotated state labels and the axis name
COLOURBAR_IN = 1.95        # the two colour bars of the farm pages
# the map keeps its own aspect ratio, so the map column is sized to it; any
# extra width would leave the map floating in white space, and the panel beside
# it would look oversized
ROUTE_ASPECT = 2325 / 1501.0     # the route map images


def page_size(rows):
    return (W_FULL, rows * ROW + TOP_IN + BOTTOM_IN)


def row_axes(fig, rows, row, aspect, right_in):
    """The map axes and the panel axes of one row, in canvas inches.

    The map takes the whole row height and whatever width its own aspect
    ratio then asks for; the panel takes the rest of the width and is inset
    within the row so that its letter and its labels stay inside it."""
    height = rows * ROW + TOP_IN + BOTTOM_IN
    top = height - TOP_IN - row * ROW
    map_w = aspect * ROW

    def box(x, y, w, h):
        return fig.add_axes([x / W_FULL, y / height, w / W_FULL, h / height])

    map_ax = box(LEFT_IN, top - ROW, map_w, ROW)
    panel_x = LEFT_IN + map_w + GUTTER_IN
    panel_ax = box(panel_x, top - ROW + PANEL_BOTTOM_IN,
                   W_FULL - right_in - panel_x,
                   ROW - PANEL_TOP_IN - PANEL_BOTTOM_IN)
    return map_ax, panel_ax


W_HALF = 8.0            # a map, two to a row
FS_FULL = {"label": st.pt(7.0, W_FULL), "tick": st.pt(6.0, W_FULL)}
FS_HALF = {"label": st.pt(7.0, W_HALF, 0.5), "tick": st.pt(6.0, W_HALF, 0.5)}


def load_states():
    """Load the state boundaries without relying on the PRJ sidecar.

    The source data are NAD83 (EPSG:4269). The other shapefile parts are copied
    to a temporary directory and the known CRS is assigned explicitly.
    """
    stem = "cb_2023_us_state_20m"
    with tempfile.TemporaryDirectory() as tmp:
        for ext in ("shp", "shx", "dbf", "cpg"):
            shutil.copyfile(os.path.join(maps.BOUNDARIES, stem + "." + ext),
                            os.path.join(tmp, stem + "." + ext))
        states = gpd.read_file(os.path.join(tmp, stem + ".shp"))
    return (states[~states.STUSPS.isin(maps.NON_CONUS)]
            .set_crs("EPSG:4269", allow_override=True).to_crs(maps.ALBERS))


def farms_of(archive, farm_p, flight_p):
    name = "%s/solar_results_%sfarm_%sflight.csv" % (INSIDE, farm_p, flight_p)
    with archive.open(name) as f:
        return pd.read_csv(f, low_memory=False)


def row_label(fig, ax, letter, description, fontsize):
    """The row letter and its rate, in the margin left of the map.

    The submitted document puts its letters there. Nothing is drawn over the
    map itself: the map is as wide as the row is high, so the margin costs the
    panel beside it a little width and the map nothing."""
    box = ax.get_position()
    x = box.x0 / 2
    fig.text(x, box.y1, letter + ".", ha="center", va="top",
             fontsize=fontsize, fontweight="bold", family=st.LETTER_FAMILY)
    fig.text(x, (box.y0 + box.y1) / 2, description, rotation=90,
             ha="center", va="center", fontsize=fontsize)


def panel_label(ax, letter, fontsize):
    """The panel letter above the axes, at the left, as the main figures set
    them and as the submitted document places them."""
    ax.text(-0.012, 1.02, letter + ".", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=fontsize, fontweight="bold",
            family=st.LETTER_FAMILY)


def check_ceilings(*checks):
    """Stop if the data outgrow a fixed axis, rather than clipping silently.

    The ceilings come from the submitted document and are not derived from
    the data, so a later data revision has to be told about them."""
    for seen, ceiling, what in checks:
        if seen > ceiling:
            raise SystemExit("%s reaches %s but the axis stops at %s"
                             % (what, format(seen, ",.0f"),
                                format(ceiling, ",.0f")))


def farm_page(archive, states, farm_p, order, indices):
    """One submitted-layout page of paired farm maps and scatter panels."""
    indices = list(indices)
    tables = [farms_of(archive, farm_p, flight_p) for flight_p in RATES]
    check_ceilings((max(t.groupby("p_state").p_cap_safe.sum().max()
                        for t in tables), CAP_MAX, "state capacity (MW)"),
                   (max((t.Money_Cost_Saving + t.CO2_Emissions_Reduction
                         * agg.CARBON_PRICE).max() for t in tables),
                    COST_MAX, "farm cost reduction ($)"),
                   (max(t.p_cap_safe.max() for t in tables),
                    FARM_CAP_MAX, "farm capacity (MW)"))
    pos = {state: i for i, state in enumerate(order)}
    size = page_size(len(indices))
    fig = plt.figure(figsize=size)
    map_aspect = float(states.total_bounds[2] - states.total_bounds[0]) / (
        states.total_bounds[3] - states.total_bounds[1])
    boxes = [row_axes(fig, len(indices), row, map_aspect, COLOURBAR_IN)
             for row in range(len(indices))]
    axes = np.array(boxes)
    scatters = []
    for row, k in enumerate(indices):
        flight_p, sel = RATES[k], tables[k]
        ax = axes[row, 0]
        g = states.copy()
        g["value"] = g.STUSPS.map(
            sel.groupby("p_state")["p_cap_safe"].sum()).fillna(0.0)
        g.plot(ax=ax, column="value", cmap=st.CMAP_WARM, vmin=0, vmax=CAP_MAX,
               edgecolor=maps.EDGE, linewidth=maps.STATE_LW)
        pts = gpd.GeoSeries(gpd.points_from_xy(sel.xlong, sel.ylat),
                            crs="EPSG:4326").to_crs(maps.ALBERS)
        ax.scatter(pts.x, pts.y, s=3, color=FARM_DOT, linewidths=0)
        ax.margins(0)
        ax.axis("off")
        row_label(fig, ax, MAP_LETTERS[k], "Flight %d%%" % (100 * flight_p),
                  FS_HALF["label"])

        ax = axes[row, 1]
        sel = sel.assign(total=sel.Money_Cost_Saving
                         + sel.CO2_Emissions_Reduction * agg.CARBON_PRICE,
                         x=sel.p_state.map(pos))
        for gx in range(len(order)):
            ax.axvline(gx, color="0.9", linewidth=0.6, zorder=0)
        scatters.append(ax.scatter(sel.x, sel.p_cap_safe, c=sel.total,
                                   cmap="viridis", s=18, zorder=3,
                                   vmin=0, vmax=COST_MAX))
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=FS_FULL["tick"])
        ax.tick_params(axis="y", labelsize=FS_FULL["tick"])
        ax.set_ylim(0, FARM_CAP_MAX)
        ax.set_yticks(np.arange(0, 401, FARM_CAP_TICKS))
        ax.set_ylabel("Capacity (MW)", fontsize=FS_FULL["label"])
        if row == len(indices) - 1:
            ax.set_xlabel("State", fontsize=FS_FULL["label"])
        panel_label(ax, PLOT_LETTERS[k], FS_FULL["label"])

    half = 0.40 * (size[1] - TOP_IN - BOTTOM_IN) / size[1]
    bar_x = (W_FULL - COLOURBAR_IN + 0.30) / W_FULL
    map_cax = fig.add_axes([bar_x, 0.52, 0.010, half])
    sm = plt.cm.ScalarMappable(cmap=st.CMAP_WARM,
                               norm=plt.Normalize(vmin=0, vmax=CAP_MAX))
    map_cb = fig.colorbar(sm, cax=map_cax)
    map_cb.set_ticks(np.arange(0, CAP_MAX + 1, CAP_STEP))
    map_cb.set_label("Capacity (MW)", labelpad=7, fontsize=FS_FULL["label"])
    map_cb.ax.tick_params(labelsize=FS_FULL["tick"])
    cost_cax = fig.add_axes([bar_x, 0.52 - half - 0.04, 0.010, half])
    cost_cb = fig.colorbar(scatters[0], cax=cost_cax)
    cost_cb.set_ticks(np.arange(0, COST_MAX + 1, COST_STEP))
    cost_cb.set_label("Total Cost Reduction ($)", labelpad=8,
                      fontsize=FS_FULL["label"])
    cost_cb.ax.tick_params(labelsize=FS_FULL["tick"])
    dot = plt.Line2D([], [], marker="o", linestyle="", markersize=4,
                     color=FARM_DOT, label="Selected\nSolar Farms")
    fig.legend(handles=[dot], loc="upper center", frameon=False,
               fontsize=FS_FULL["tick"], handletextpad=0.3,
               bbox_to_anchor=((W_FULL - COLOURBAR_IN / 2) / W_FULL,
                               0.52 - half - 0.055))
    return fig


def flight_page(archive, totals, flight_p, indices):
    """One submitted-layout page of paired route maps and state bar panels."""
    indices = list(indices)
    sub = totals[totals.flight_p == flight_p]
    order = (sub.groupby("state").total_usd.sum()
                .sort_values(ascending=False).index.tolist())
    ymax = 0
    tables = {}
    for farm_p in RATES:
        scene = sub[sub.farm_p == farm_p]
        by_side = {}
        for side in ("Origin", "Destination"):
            t = (scene[scene.side == side]
                 .pivot_table(index="state", columns="cls", values="total_usd",
                              aggfunc="sum")
                 .reindex(index=order, columns=agg.CLASS_NAMES).fillna(0.0))
            by_side[side] = t
            ymax = max(ymax, t.sum(axis=1).max())
        tables[farm_p] = by_side
    check_ceilings((ymax, BAR_MAX, "state cost reduction ($)"))

    size = page_size(len(indices))
    fig = plt.figure(figsize=size)
    axes = np.array([row_axes(fig, len(indices), row, ROUTE_ASPECT, LEFT_IN)
                     for row in range(len(indices))])
    width = 0.38
    for row, k in enumerate(indices):
        farm_p = RATES[k]
        ax = axes[row, 0]
        name = "%s/flight_map_%sfarm_%sflight.png" % (ROUTE_INSIDE, farm_p,
                                                      flight_p)
        with archive.open(name) as f:
            ax.imshow(Image.open(f).convert("RGB"))
        ax.axis("off")
        row_label(fig, ax, MAP_LETTERS[k], "Solar farm %d%%" % (100 * farm_p),
                  FS_HALF["label"])

        ax = axes[row, 1]
        x = np.arange(len(order))
        for side_index, side in enumerate(("Origin", "Destination")):
            table = tables[farm_p][side]
            bottom = np.zeros(len(order))
            for j, cls in enumerate(agg.CLASS_NAMES):
                values = table[cls].values
                ax.bar(x + (side_index - 0.5) * width, values, width,
                       bottom=bottom, color=st.CLASS_COLORS[j],
                       edgecolor="black", linewidth=0.4)
                bottom = bottom + values
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=90, fontsize=FS_FULL["tick"])
        ax.tick_params(axis="y", labelsize=FS_FULL["tick"])
        ax.set_xlim(-0.6, len(order) - 0.4)
        # the legend goes inside the panel, where the submitted document keeps
        # it; a band for it under the page would cost every map its height
        handles = [mpatches.Patch(facecolor=c, edgecolor="black", label=l)
                   for c, l in zip(st.CLASS_COLORS,
                                   ["Short range", "Medium range",
                                    "Long range"])]
        ax.legend(handles=handles, loc="upper right", ncol=3, frameon=False,
                  fontsize=FS_FULL["tick"], handlelength=1.1,
                  handletextpad=0.35, columnspacing=1.0, borderaxespad=0.2)
        # only the last panel names the axis; the state codes are under every
        # one of them
        if row == len(indices) - 1:
            ax.set_xlabel("State", fontsize=FS_FULL["label"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_label(ax, PLOT_LETTERS[k], FS_FULL["label"])
    for ax in axes[:, 1]:
        ax.set_ylim(0, BAR_MAX)
    map_edge = max(ax.get_position().x1 for ax in axes[:, 0])
    fig.text(map_edge + 0.006, 0.5, "Cost Reduction ($)",
             rotation=90, va="center", ha="left", fontsize=FS_FULL["label"])
    return fig


def scatter_order(archive):
    """States on the x axis, ordered by capacity as in Fig. 2e."""
    sel = farms_of(archive, 1.0, 1.0)
    return (sel.groupby("p_state").p_cap_safe.sum()
               .sort_values(ascending=False).index.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=("farms", "flights"))
    p.add_argument("--rates", type=float, nargs="*", default=RATES)
    a = p.parse_args()

    st.apply()
    os.makedirs(OUT, exist_ok=True)
    archive = zipfile.ZipFile(ARCHIVE)

    if a.only != "flights":
        states = load_states()
        order = scatter_order(archive)
        for farm_p in a.rates:
            farm_p = round(farm_p, 1)
            name = "farm_selection_%02d" % (100 * farm_p)
            st.save(farm_page(archive, states, farm_p, order, PAGE_GROUPS[0]),
                    os.path.join(OUT, name + "_maps"), dpi=SAVE_DPI)
            st.save(farm_page(archive, states, farm_p, order, PAGE_GROUPS[1]),
                    os.path.join(OUT, name + "_scatter"), dpi=SAVE_DPI)
            print("wrote %s (farm penetration %d%%)" % (name, 100 * farm_p),
                  flush=True)

    if a.only != "farms":
        totals = pd.read_csv(TOTALS)
        routes = zipfile.ZipFile(ROUTE_ZIP)
        for flight_p in a.rates:
            flight_p = round(flight_p, 1)
            name = "flight_selection_%02d" % (100 * flight_p)
            st.save(flight_page(routes, totals, flight_p, PAGE_GROUPS[0]),
                    os.path.join(OUT, name + "_maps"), dpi=SAVE_DPI)
            st.save(flight_page(routes, totals, flight_p, PAGE_GROUPS[1]),
                    os.path.join(OUT, name + "_bars"), dpi=SAVE_DPI)
            print("wrote %s (flight penetration %d%%)" % (name, 100 * flight_p),
                  flush=True)


if __name__ == "__main__":
    main()
