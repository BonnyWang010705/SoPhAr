"""Supplementary Fig. 2: state-level total cost reduction by origin state and
by destination state, for short-, medium- and long-range flights (a-f).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import agg
import style as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
OUT = os.path.join(ROOT, "figures", "supplementary", "fig02_state_rankings")
N_TOP = 10
OTHER = "Other states"

# light to dark around each range class colour, so the hue names the class
# and the shade carries the flight distance
CLASS_ENDS = {"short": ("#e4ecfb", "#2b4a86"),
              "medium": ("#f7e7e4", "#8c4238"),
              "long": ("#f1eaf5", "#6f4f7c")}


def class_cmap(cls):
    light, dark = CLASS_ENDS[cls]
    middle = st.CLASS_COLORS[agg.CLASS_NAMES.index(cls)]
    return mcolors.LinearSegmentedColormap.from_list(
        "%s_cmap" % cls, [light, middle, dark])


CEILINGS = {"short": (16000000, 16000),
            "medium": (25000000, 10000),
            "long": (2500000, 500)}

# the colour scale reads in millions of kilometres: the tables give the
# distance in km, and a colour bar labelled in km would carry an unreadable
# "1e7" above it
MILLION = 1e6
DISTANCE_LABEL = "Total flight distance (million km)"

# the assembled figure: three range classes by origin and destination state,
# drawn on a 16 in canvas like the other R1 figures and printed 6.5 in wide
GRID_W, GRID_H = 16.0, 17.2
ROWS = [("short", "Short-range"), ("medium", "Medium-range"),
        ("long", "Long-range")]
SIDES = [("Origin", "origin"), ("Destination", "destination")]

def top_table(flights, cls, side):
    g = agg.state_by_class(flights, side)
    s = g[g.cls == cls].copy()
    s["carbon_usd"] = s.co2_kg * agg.CARBON_PRICE
    s = s.sort_values("total_usd", ascending=False)
    top, rest = s.head(N_TOP), s.iloc[N_TOP:]
    if len(rest):
        pooled = {"state": OTHER}
        for c in ("fuel_usd", "carbon_usd", "total_usd", "distance_km", "flights"):
            pooled[c] = rest[c].sum()
        top = pooled_append(top, pooled)
    top["fuel_pct"] = 100 * top.fuel_usd / s.fuel_usd.sum()
    top["carbon_pct"] = 100 * top.carbon_usd / s.carbon_usd.sum()
    top["state"] = [v if v != OTHER else "Other" for v in top.state]
    return top, s.total_usd.sum(), s.fuel_usd.sum(), s.carbon_usd.sum()

def pooled_append(df, row):
    import pandas as pd
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

def bars(ax, tab, total, cmap, norm, xmax, fig_w, annotate_pt):
    """Top-ten states by share, bars shaded by total flight distance.

    The hue is the range class, so the panels are told apart the way Figs. 3,
    5 and 8 separate the classes."""
    states = list(tab.state)[::-1]
    shares = [100 * v / total for v in tab.total_usd][::-1]
    distance = [v / MILLION for v in tab.distance_km][::-1]

    y = np.arange(len(states))
    rects = ax.barh(y, shares, 0.7, color=cmap(norm(distance)),
                    edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(states)
    for rect, value in zip(rects, shares):
        ax.annotate("%g%%" % round(value, 2),
                    xy=(value, rect.get_y() + rect.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=annotate_pt)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, xmax)
    return shares


def draw(tab, title, total, cls, dist_max):
    """One panel on its own, at the width a single panel is drawn at."""
    st.scale(14)
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(left=0.10, right=0.80, bottom=0.17, top=0.88)

    cmap = class_cmap(cls)
    norm = mcolors.Normalize(vmin=0, vmax=dist_max / MILLION)
    shares = bars(ax, tab, total, cmap, norm, 0, 14, st.pt(6.0, 14))
    ax.set_xlim(0, max(shares) + 6)
    ax.set_title(title, pad=25, fontsize=st.pt(7.0, 14), fontweight="bold")
    ax.set_xlabel("Share of %s-range total cost saving (%%)" % cls)

    cax = fig.add_axes([0.83, 0.25, 0.02, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(DISTANCE_LABEL, rotation=90, labelpad=15)
    return fig


def panel_label(ax, letter, fontsize):
    """The panel letter above the axes, at the left, as the submitted
    document and the other R1 figures place them."""
    ax.text(-0.012, 1.02, letter + ".", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=fontsize, fontweight="bold",
            family=st.LETTER_FAMILY)


def assembled(flights):
    """The six panels as one figure, the way the document prints them.

    Rows are the range classes and columns the origin and destination state,
    so a row compares the two ends of the same flights; the row shares one
    colour scale, since both panels cover the same flights."""
    st.scale(GRID_W)
    letter_pt, label_pt = st.pt(8.0, GRID_W), st.pt(7.0, GRID_W)
    fig = plt.figure(figsize=(GRID_W, GRID_H))
    gs = fig.add_gridspec(3, 3, width_ratios=[40, 40, 1.2],
                          left=0.085, right=0.925, bottom=0.055, top=0.955,
                          wspace=0.22, hspace=0.20)

    letters = "abcdef"
    index = 0
    for row, (cls, row_name) in enumerate(ROWS):
        tabs = {side: top_table(flights, cls, side) for side, _ in SIDES}
        cmap = class_cmap(cls)
        norm = mcolors.Normalize(vmin=0, vmax=CEILINGS[cls][0] / MILLION)
        xmax = max(100 * t[0].total_usd.max() / t[1] for t in tabs.values()) + 6
        axes = []
        for column, (side, word) in enumerate(SIDES):
            tab, total = tabs[side][0], tabs[side][1]
            ax = fig.add_subplot(gs[row, column])
            bars(ax, tab, total, cmap, norm, xmax, GRID_W,
                 st.pt(6.0, GRID_W))
            panel_label(ax, letters[index], letter_pt)
            index += 1
            if row == 0:
                ax.set_title("%s state" % word.capitalize(), pad=10,
                             fontsize=label_pt, fontweight="bold")
            if row == len(ROWS) - 1:
                ax.set_xlabel("Share of the total cost saving of the range "
                              "class (%)", fontsize=label_pt)
            axes.append(ax)

        cax = fig.add_subplot(gs[row, 2])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(DISTANCE_LABEL, rotation=90, labelpad=12,
                     fontsize=label_pt)
        # the class name in the margin, left of the row, as the penetration
        # figures name their rows
        box = axes[0].get_position()
        fig.text(box.x0 / 3, (box.y0 + box.y1) / 2, row_name, rotation=90,
                 ha="center", va="center", fontsize=label_pt)
    return fig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--altitude", type=int, default=12100)
    a = p.parse_args()

    st.apply()
    flights = agg.load_flights(a.altitude)
    national = flights.total_usd.sum()

    panel = "abcdef"
    i = 0
    for cls in agg.CLASS_NAMES:
        tabs = {s: top_table(flights, cls, s) for s in ("Origin", "Destination")}
        dist_max, flight_max = CEILINGS[cls]
        seen_dist = max(t[0].distance_km.max() for t in tabs.values())
        seen_flights = max(t[0].flights.max() for t in tabs.values())
        for side, word in (("Origin", "origin"), ("Destination", "destination")):
            tab, total, fuel, carbon = tabs[side]
            title = ("Top-ten states as %s by total cost saving (%s-range)"
                     % (word, cls))
            fig = draw(tab, title, total, cls, dist_max)
            name = "supp_fig02%s_%s_%s" % (panel[i], cls, word)
            for f in st.save(fig, os.path.join(OUT, "panels", name)):
                print("wrote", f)
            i += 1
        print("  %-6s ceilings %s km / %s flights, largest %s / %s%s | "
              "class total $%s, fuel/carbon %.1f/%.1f, "
              "max |fuel-carbon| %.4f pp"
              % (cls, format(dist_max, ","), format(flight_max, ","),
                 format(int(seen_dist), ","), format(int(seen_flights), ","),
                 "  ** past ceiling **"
                 if seen_dist > dist_max or seen_flights > flight_max else "",
                 format(tabs["Origin"][1], ",.0f"),
                 100 * tabs["Origin"][2] / tabs["Origin"][1],
                 100 * tabs["Origin"][3] / tabs["Origin"][1],
                 (tabs["Origin"][0].fuel_pct
                  - tabs["Origin"][0].carbon_pct).abs().max()))

    fig = assembled(flights)
    for f in st.save(fig, os.path.join(OUT, "supp_fig02")):
        print("wrote", f)

if __name__ == "__main__":
    main()
