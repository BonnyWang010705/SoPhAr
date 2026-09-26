import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import style as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures", "main", "fig05_schedule_optimization", "panels")
SHIFTS = os.path.join(ROOT, "results", "optimization_1",
                      "optimized_shift_flight_merged_R1.csv")

HALF = (3600 - 60) / 2 / 3600
RANGES = ["Short", "Medium", "Long"]
EDGES = [0, 1500, 4000, np.inf]
PANEL = {9100: "d", 12100: "e", 15100: "f"}
POSITIONS = [1.25, 2.5, 3.5]
LIMITS = [-1800, 1800]

def load(altitude):
    d = pd.read_csv(SHIFTS, low_memory=False)
    col = "time_shift_seconds_%dm" % altitude
    d = d[d[col].notna()].copy()
    d["cls"] = pd.cut(d["Distance_km"], EDGES, labels=RANGES, right=False)
    return [d.loc[d.cls == r, col].tolist() for r in RANGES]

def draw(data_sets):
    colors = st.CLASS_COLORS
    counts = [len(x) for x in data_sets]
    calc_widths = [np.sqrt(c / max(counts)) for c in counts]

    st.scale(10, HALF)
    # same aspect ratio as panels a-c (8 x 5.4) so each row lines up
    fig, ax = plt.subplots(figsize=(10, 6.75))
    parts = ax.violinplot(data_sets, positions=POSITIONS, showmedians=True,
                          widths=calc_widths, quantiles=[[0.25, 0.75]] * 3)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(1.0)          # same solid colours as panels a-c
    for name in ("cmins", "cmaxes", "cmedians", "cquantiles"):
        if name in parts:
            parts[name].set_edgecolor("#333333")
            parts[name].set_linewidth(1.5)
    parts["cbars"].set_visible(False)

    def label(x, y, val):
        ax.text(x + 0.05, y, "%s" % format(val, ",.0f"), va="center", ha="left",
                fontsize=st.pt(6.0, 10, HALF),
                color="#333333",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1))

    def trim(name, per_body):
        segments = parts[name].get_segments()
        for i, pc in enumerate(parts["bodies"]):
            v = pc.get_paths()[0].vertices
            right = v[v[:, 0] >= POSITIONS[i]]
            right = right[right[:, 1].argsort()]
            for j in range(per_body):
                idx = i * per_body + j
                y = segments[idx][0, 1]
                w = np.interp(y, right[:, 1], right[:, 0]) - POSITIONS[i]
                segments[idx][:, 0] = [POSITIONS[i] - w, POSITIONS[i] + w]
                label(POSITIONS[i] + w, y, y)
        parts[name].set_segments(segments)

    if "cmedians" in parts:
        trim("cmedians", 1)
    if "cquantiles" in parts:
        trim("cquantiles", 2)

    for t in LIMITS:
        ax.axhline(y=t, color="#555555", linestyle="--", linewidth=0.8,
                   zorder=1)

    fig.canvas.draw()
    ymin, ymax = ax.get_ylim()
    natural = [n for n in range(-20000, 20001, 1000) if n not in (-2000, 2000)]
    ax.set_yticks([t for t in sorted(set(natural + LIMITS)) if ymin <= t <= ymax])

    ax.set_xlim(0.5, 4.2)

    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xticks(POSITIONS)
    # flight counts sit under each violin instead of in a legend, which
    # used to cover the long-range distribution
    ax.set_xticklabels(["%s range\n(n = %s)" % (r, format(c, ","))
                        for r, c in zip(RANGES, counts)])
    ax.set_ylabel("Time shift (s)")
    # open frame, as in panels a-c
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig, counts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--altitude", type=int, default=0,
                   choices=(0, 9100, 12100, 15100))
    a = p.parse_args()
    st.apply()

    for alt in ([a.altitude] if a.altitude else sorted(PANEL)):
        data = load(alt)
        fig, counts = draw(data)
        name = "figure5%s_%dm" % (PANEL[alt], alt)
        for f in st.save(fig, os.path.join(OUT, name)):
            print("wrote", f)
        q = [np.percentile(x, [25, 50, 75]) for x in data]
        print("  %-6s flights %s   quartiles %s"
              % ("%dm" % alt, [format(c, ",") for c in counts],
                 [[int(round(v)) for v in row] for row in q]))

if __name__ == "__main__":
    main()
