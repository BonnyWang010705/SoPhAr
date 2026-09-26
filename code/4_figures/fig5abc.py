import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import agg
import style as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(ROOT, "figures", "main", "fig05_schedule_optimization", "panels")
SUPP = os.path.join(ROOT, "figures", "drafts", "fig5_carbon190")   # not used in the paper
BASE = os.path.join(ROOT, "results")

HALF = (3600 - 60) / 2 / 3600
RANGES = ["Short", "Medium", "Long"]
EDGES = [0, 1500, 4000, np.inf]
PANEL = {9100: "a", 12100: "b", 15100: "c"}

BASELINE = {
    9100: ("baseline", "flight_analysis_results_9100m_merged_R1.csv"),
    12100: ("baseline", "flight_analysis_results_merged_R1.csv"),
    15100: ("baseline", "flight_analysis_results_15100m_merged_R1.csv")}
OPTIMIZED = {
    9100: ("optimization_1",
           "Optimization_1_flight_analysis_results_9100m_merged_R1.csv"),
    12100: ("optimization_1",
            "Optimization_1_flight_analysis_results_merged_R1.csv"),
    15100: ("optimization_1",
            "Optimization_1_flight_analysis_results_15100m_merged_R1.csv")}

FUEL = {"base": "Fuel_Cost_Savings_by_Energy", "opt": "Money_Cost_Saving"}
CO2 = {"base": "CO2_Emissions_Reduction_by_Energy", "opt": "CO2_Emissions_Reduction"}

def by_class(path, col):
    d = pd.read_csv(path, low_memory=False)
    cls = pd.cut(d["Distance_km"], EDGES, labels=RANGES, right=False)
    return d.groupby(cls, observed=True)[col].sum().reindex(RANGES).fillna(0.0)

def series(altitude, carbon=agg.CARBON_PRICE):
    b = os.path.join(BASE, *BASELINE[altitude])
    o = os.path.join(BASE, *OPTIMIZED[altitude])
    before_money = by_class(b, FUEL["base"])
    before_co2 = by_class(b, CO2["base"]) * carbon
    opt_money = by_class(o, FUEL["opt"]) - before_money
    opt_co2 = by_class(o, CO2["opt"]) * carbon - before_co2
    return (before_money.to_dict(), before_co2.to_dict(),
            opt_money.to_dict(), opt_co2.to_dict())

X_POS = [0.0, 1.35, 2.7]
WIDTH = 0.3
# white backing behind every value, so guide lines pass behind the numbers
LABEL_BOX = dict(facecolor="white", edgecolor="none", pad=0.6)
COLUMN_NAMES = ["CO$_2$ cost\nsaving", "Fuel cost\nsaving", "Total cost\nreduction"]

def draw(before_money, before_co2, opt_money, opt_co2, legend=False):
    ranges = RANGES
    colors = st.CLASS_COLORS

    sum_before_co2 = sum(before_co2.values())
    total_before = sum_before_co2 + sum(before_money.values())
    sum_opt_co2 = sum(opt_co2.values())
    total_opt = sum_opt_co2 + sum(opt_money.values())
    final_top = total_before + total_opt

    st.scale(8, HALF)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    label_pt = st.pt(5.5, 8, HALF)

    # waterfall: CO2 first, fuel stacked on top of it, total alongside; the
    # optimization gain sits above the baseline in every column
    segments = [[], [], []]            # per column: (bottom, value)
    for i, r in enumerate(ranges):
        segments[0].append((sum(list(before_co2.values())[:i]), before_co2[r], i))
        segments[1].append((sum_before_co2 + sum(list(before_money.values())[:i]),
                            before_money[r], i))
        segments[2].append((sum(before_co2[q] + before_money[q] for q in ranges[:i]),
                            before_co2[r] + before_money[r], i))
    for i, r in enumerate(ranges):
        segments[0].append((total_before + sum(list(opt_co2.values())[:i]),
                            opt_co2[r], i))
        segments[1].append((total_before + sum_opt_co2
                            + sum(list(opt_money.values())[:i]), opt_money[r], i))
        segments[2].append((total_before + sum(opt_co2[q] + opt_money[q]
                                               for q in ranges[:i]),
                            opt_co2[r] + opt_money[r], i))

    # every value is labelled to the right of its bar; labels of thin
    # segments are pushed apart vertically so none of them overlap
    gap = 0.052 * final_top
    label_top = final_top
    for col, x in enumerate(X_POS):
        for bottom, val, i in segments[col]:
            ax.bar(x, val, bottom=bottom, color=colors[i], width=WIDTH,
                   edgecolor="black", linewidth=0.8)
        # the topmost segment has free space above it, so its value sits
        # there; it is lifted when a side label reaches the same height
        top_b, top_v, _ = segments[col][-1]
        top_y = top_b + top_v
        wanted = sorted((b + v / 2, v) for b, v, _ in segments[col][:-1])
        placed = []
        for mid, v in wanted:
            y = mid if not placed else max(mid, placed[-1][1] + gap)
            placed.append((mid, y, v))
        above_y = max(top_y + 0.016 * final_top, placed[-1][1] + 0.9 * gap)
        ax.plot([x, x], [top_y, above_y - 0.004 * final_top], color="black",
                linewidth=0.6)
        ax.text(x, above_y, format(top_v, ",.0f"), ha="center", va="bottom",
                fontsize=label_pt, bbox=LABEL_BOX)
        label_top = max(label_top, above_y + gap)
        edge = x + WIDTH / 2
        for mid, y, v in placed:
            ax.plot([edge, edge + 0.06, edge + 0.1], [mid, y, y],
                    color="black", linewidth=0.6, clip_on=False)
            ax.text(edge + 0.12, y, format(v, ",.0f"), va="center", ha="left",
                    fontsize=label_pt, bbox=LABEL_BOX)
            label_top = max(label_top, y)

    ax.set_xticks(X_POS)
    ax.set_xticklabels(COLUMN_NAMES)
    ax.set_ylabel("Cost saving (\\$ million)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: ("%.1f" % (v / 1e6)).rstrip("0").rstrip(".")))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    brace_x = X_POS[-1] + 0.85
    def draw_brace(y_min, y_max, text):
        w = 0.2
        ax.plot([brace_x, brace_x + w, brace_x + w, brace_x],
                [y_min, y_min, y_max, y_max], color="black", lw=0.9,
                clip_on=False)
        ax.text(brace_x + w + 0.08, (y_min + y_max) / 2, text, va="center",
                ha="left")

    draw_brace(0, total_before * 0.99, "Before\noptimization")
    draw_brace(total_before * 1.01, final_top, "Optimized")

    for line in (total_before, final_top, sum_before_co2,
                 sum_opt_co2 + total_before):
        ax.hlines(y=line, xmin=X_POS[0] - WIDTH, xmax=X_POS[-1] + WIDTH / 2,
                  colors="#9a9a9a", linestyles="--", linewidth=0.5, zorder=0)

    ax.set_xlim(X_POS[0] - 0.45, brace_x - 0.05)
    ax.set_ylim(0, label_top + gap)
    # one legend for the whole of Fig. 5, above the first panel; d-f name
    # their range classes on the axis
    if legend:
        ax.legend(handles=[Line2D([0], [0], color=c, lw=4, label=l)
                           for c, l in zip(colors, ["Short range", "Medium range",
                                                    "Long range"])],
                  loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
                  frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(right=0.73)
    return fig, total_before, final_top

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--altitude", type=int, default=0,
                   choices=(0, 9100, 12100, 15100))
    # --carbon and --prefix drive the carbon-price sensitivity: rerun with a
    # different price and a different name, e.g.
    #   python fig5abc.py --carbon 0.19 --prefix suppCarbon190
    p.add_argument("--carbon", type=float, default=agg.CARBON_PRICE,
                   help="carbon price in $/kg CO2 (default: the EU ETS value)")
    p.add_argument("--prefix", default="figure5",
                   help="output name prefix; a name starting with 'supp' is "
                        "written to the supplemental folder")
    a = p.parse_args()
    st.apply()

    for alt in ([a.altitude] if a.altitude else sorted(PANEL)):
        bm, bc, om, oc = series(alt, a.carbon)
        # the panel carries its legend, as the panels of Fig. 6 do; assemble.py
        # draws panels a-c again without it for Fig. 5, under one shared legend
        fig, before, top = draw(bm, bc, om, oc, legend=True)
        name = "%s%s_%dm" % (a.prefix, PANEL[alt], alt)
        out = SUPP if a.prefix.startswith("supp") else MAIN
        for f in st.save(fig, os.path.join(out, name)):
            print("wrote", f)
        print("  %-6s baseline $%s -> optimized $%s  (+%.1f%%)"
              % ("%dm" % alt, format(before, ",.0f"), format(top, ",.0f"),
                 100 * (top / before - 1)))

def legend_strip(path):
    """One legend across the top of the assembled Fig. 5, as in Fig. 6.
    Drawn at the full printed width so its text prints at 6 pt."""
    import matplotlib.patches as mpatches
    width = st.PRINT_MM / 25.4
    st.scale(width, 1.0)
    fig = plt.figure(figsize=(width, 0.22))
    fig.legend(handles=[mpatches.Patch(color=c, label=l)
                        for c, l in zip(st.CLASS_COLORS,
                                        ["Short range", "Medium range",
                                         "Long range"])],
               loc="center", ncol=3, frameon=False, handlelength=1.6,
               columnspacing=2.4)
    return st.save(fig, path)

if __name__ == "__main__":
    main()
