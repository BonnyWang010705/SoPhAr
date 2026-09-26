"""Supplementary Fig. 1: the week of 7-13 April 2025 against the 51 complete
weeks of 2025 (Supplementary Note 1).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

import style as st

# Drawn to the main-figure standard (code/4_figures/style.py, copied here as
# style.py): matplotlib defaults, text 5-7 pt at the 180 mm printed width,
# panel letters "a." in Times bold, no titles or notes inside the figure.
# Run after 1 and 6: reads the tables written by 1_weekly_flight_hours.py and
# 6_weekly_energy_estimate.py.

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
DATA = os.path.join(ROOT, "results", "representativeness")
OUT = os.path.join(ROOT, "figures", "supplementary", "fig01_representativeness")

REFERENCE_WEEK = pd.Timestamp("2025-04-07")

# Fig. 4: daytime in the warm family, nighttime in the blue family
DAY, NIGHT = "#F28C28", "#2980B9"
OD, CLASSES = "tab:blue", "tab:orange"
ENERGY = "tab:green"

W, H = 7.2, 6.0

thousands = FuncFormatter(lambda v, _: format(v, ",.0f"))


def load():
    hours = pd.read_csv(os.path.join(DATA, "weekly_flight_hours.csv"),
                        parse_dates=["week"])
    hours = hours[~hours["partial"]].set_index("week")
    weekly = pd.read_csv(os.path.join(DATA, "weekly_energy_estimate.csv"),
                         parse_dates=["week"]).set_index("week")
    assert hours.index.equals(weekly.index), "weekly tables disagree on weeks"
    return hours, weekly


def main():
    st.apply()
    st.scale(W, 1.0, label=7.0, tick=6.0)
    hours, weekly = load()
    weeks = hours.index
    x = np.arange(len(weeks))
    i_ref = list(weeks).index(REFERENCE_WEEK)

    # Bar i stands for the week starting weeks[i], Monday 00:00 to Sunday
    # 24:00, and sits at that week's midpoint. Dates on the same scale put the
    # month ticks on the 1st of each month, and the axis spans exactly 2025.
    def at(date):
        return ((pd.Timestamp(date) - weeks[0]).days - 3.5) / 7.0

    fig, axes = plt.subplots(3, 1, figsize=(W, H), sharex=True)
    ref_line = dict(color="#555555", linewidth=0.8, linestyle="--", zorder=4)
    median_line = dict(color="black", linewidth=0.8, linestyle=":", zorder=4)
    legend_kw = dict(loc="center left", bbox_to_anchor=(1.01, 0.5),
                     frameon=False, handlelength=1.8)
    mark = dict(linewidth=1.2, marker="o", markersize=2.2, zorder=3)

    # a, weekly flight hours, daytime and nighttime
    ax = axes[0]
    ax.bar(x, hours["daytime_hours"], width=0.75, color=DAY, linewidth=0,
           zorder=3)
    ax.bar(x, hours["nighttime_hours"], width=0.75,
           bottom=hours["daytime_hours"], color=NIGHT, linewidth=0, zorder=3)
    ax.axhline(hours["elapsed_hours"].median(), **median_line)
    ax.set_ylabel("Elapsed time (h)")
    ax.yaxis.set_major_formatter(thousands)
    ax.set_ylim(0, 400_000)
    ax.legend(handles=[Patch(color=DAY, label="Daytime"),
                       Patch(color=NIGHT, label="Nighttime"),
                       Line2D([], [], label="Median", **median_line),
                       Line2D([], [], label="7–13 April 2025", **ref_line)],
              **legend_kw)

    # b, distance of each week's flight mix from the annual mix
    ax = axes[1]
    ax.plot(x, weekly["od_tvd_vs_annual_pct"], color=OD,
            label="Routes", **mark)
    ax.plot(x, weekly["range_class_tvd_vs_annual_pct"], color=CLASSES,
            label="Range classes", **mark)
    ax.set_ylabel("Difference from\nannual mix (%)")
    ax.set_ylim(0, 10)
    ax.legend(**legend_kw)

    # c, estimated delivered energy per week
    ax = axes[2]
    ax.plot(x, weekly["energy_estimate_mwh"], color=ENERGY,
            label="Estimated energy", **mark)
    ax.axhline(weekly["energy_estimate_mwh"].median(), **median_line)
    ax.set_ylabel("Energy supplied (MWh)")
    ax.yaxis.set_major_formatter(thousands)
    ax.set_ylim(28_000, 38_000)
    ax.legend(handles=[Line2D([], [], color=ENERGY, label="Estimated energy",
                              **mark),
                       Line2D([], [], label="Median", **median_line)],
              **legend_kw)

    for ax in axes:
        ax.axvline(i_ref, **ref_line)
        ax.grid(True, linewidth=0.4, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

    ax = axes[-1]
    starts = pd.date_range("2025-01-01", periods=12, freq="MS")
    ax.set_xticks([at(d) for d in starts])
    ax.set_xticklabels([d.strftime("%b") for d in starts])
    ax.set_xlim(at("2025-01-01"), at("2026-01-01"))
    ax.set_xlabel("2025")

    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0.02, 0, 1, 0.98), h_pad=1.6)
    for ax, letter in zip(axes, "abc"):
        box = ax.get_position()
        fig.text(0.005, box.y1 + 0.01, letter + ".", family=st.LETTER_FAMILY,
                 fontsize=st.pt(9.0, W), fontweight="bold",
                 va="bottom", ha="left")

    # the canvas is 7.2 in wide and the figure prints 6.5 in wide in the
    # Supplementary Materials, so it is saved well above the 300 ppi that
    # Nature asks of a raster figure at its printed size
    for f in st.save(fig, os.path.join(OUT, "supp_fig01"),
                     dpi=400):
        print("wrote", f)


if __name__ == "__main__":
    main()
