import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

import agg
import style as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
OUT = os.path.join(ROOT, "figures", "main", "fig06_market_penetration", "panels")
ASSEMBLED = os.path.join(ROOT, "figures", "main", "fig06_market_penetration")
SUMMARY = os.path.join(ROOT, "results", "optimization_2",
                       "Optimization_2_summary_R1.csv")

# panel letters (line, contour), axis label, column, file stem
METRICS_ALL = [
    ("a", "f", "Energy supplied (MWh)", "Total_Energy_Supplied_MWh",
     "energy"),
    ("b", "g", "Duration supported (h)",
     "Total_Flight_Duration_Supported_Hours", "duration"),
    ("c", "h", "Total cost saving (\\$)", "total_usd", "total"),
    ("d", "i", "Fuel cost saving (\\$)", "Money_Cost_Saving", "fuel"),
    ("e", "j", "CO$_2$ cost saving (\\$)", "carbon_usd", "co2"),
]

# Fuel and CO2 cost savings are fixed shares of the total cost saving (63.5% and
# 36.5%) in every scenario, so the main figure keeps energy, duration and total
# cost saving only; panels read a-c (lines) and d-f (contours).
METRICS = [("a", "d") + METRICS_ALL[0][2:],
           ("b", "e") + METRICS_ALL[1][2:],
           ("c", "f") + METRICS_ALL[2][2:]]

thousands = FuncFormatter(lambda v, _: format(v, ",.0f"))

def compact(v):
    if v >= 1e6:
        return ("%.2f" % (v / 1e6)).rstrip("0").rstrip(".") + "M"
    if v >= 1e3:
        return "%.0fk" % (v / 1e3)
    return "%.0f" % v

def load():
    d = pd.read_csv(SUMMARY)
    d["carbon_usd"] = d["CO2_Emissions_Reduction"] * agg.CARBON_PRICE
    d["total_usd"] = d["Money_Cost_Saving"] + d["carbon_usd"]
    return d

def draw_lines(ax, grid, solar, flight, label):
    for p in flight:
        ax.plot(solar, grid.loc[p].values, marker="o", markersize=4,
                label="%d%%" % round(100 * p))
    ax.set_xlabel("Solar farm penetration")
    ax.set_ylabel(label)
    ax.set_xticks(solar)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(thousands)
    ax.grid(True, linewidth=0.4, alpha=0.6)

def interior_points(iso, solar, flight, min_margin=0.08):
    """One label position per contour level: the point on the line farthest
    from the axes frame. Lines that hug a corner have no room for a label and
    are left unlabelled rather than clipped."""
    x0, x1 = min(solar), max(solar)
    y0, y1 = min(flight), max(flight)
    points = []
    for path in iso.get_paths():
        best, best_margin = None, -1.0
        for seg in path.to_polygons(closed_only=False):
            if len(seg) < 2:
                continue
            t = np.linspace(0, 1, 25)[:, None]
            dense = np.vstack([a + t * (b - a) for a, b in zip(seg[:-1], seg[1:])])
            u = (dense[:, 0] - x0) / (x1 - x0)
            w = (dense[:, 1] - y0) / (y1 - y0)
            margin = np.minimum.reduce([u, 1 - u, w, 1 - w])
            i = int(np.argmax(margin))
            if margin[i] > best_margin:
                best, best_margin = tuple(dense[i]), margin[i]
        if best is not None and best_margin >= min_margin:
            points.append(best)
    return points

def draw_contour(fig, ax, grid, solar, flight, label):
    X, Y = np.meshgrid(solar, flight)
    Z = grid.values
    cf = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    levels = MaxNLocator(6).tick_values(Z.min(), Z.max())
    iso = ax.contour(X, Y, Z, levels=levels, colors="white",
                     linewidths=0.5, alpha=0.8)
    # automatic placement can put a label on the axis edge, where it is
    # clipped; place each label at the point of its line farthest from the
    # frame instead
    ax.clabel(iso, inline=True, fontsize=plt.rcParams["xtick.labelsize"],
              fmt=compact, colors="white",
              manual=interior_points(iso, solar, flight))
    ax.set_xlabel("Solar farm penetration")
    ax.set_ylabel("Flight penetration")
    ax.set_xticks(solar)
    ax.set_yticks(flight)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(PercentFormatter(xmax=1))
    bar = fig.colorbar(cf, ax=ax, format=thousands, pad=0.02)
    bar.set_label(label)

def lines(grid, solar, flight, label):
    st.scale(7.5)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    draw_lines(ax, grid, solar, flight, label)
    ax.legend(title="Flight penetration", loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False,
              fontsize=st.pt(6.0, 7.5), title_fontsize=st.pt(7.0, 7.5))
    fig.tight_layout()
    return fig

def contour(grid, solar, flight, label):
    st.scale(6)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    draw_contour(fig, ax, grid, solar, flight, label)
    fig.tight_layout()
    return fig

def composite(grids, solar, flight):
    """Draft of the assembled figure: five rows, lines left and contours
    right, one legend for the line panels across the top."""
    st.scale(11)
    fig, axes = plt.subplots(len(METRICS), 2,
                             figsize=(11, 1.0 + 2.4 * len(METRICS)),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    for row, (la, lc, label, _, _) in enumerate(METRICS):
        ax, cx = axes[row]
        draw_lines(ax, grids[row], solar, flight, label)
        draw_contour(fig, cx, grids[row], solar, flight, label)
        for a, letter in ((ax, la), (cx, lc)):
            a.text(-0.21, 1.05, letter + ".", transform=a.transAxes,
                   family=st.LETTER_FAMILY, fontsize=st.pt(9.0, 11),
                   fontweight="bold",
                   va="bottom", ha="left")
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Flight penetration", ncol=10,
               loc="upper center", bbox_to_anchor=(0.5, 0.995),
               frameon=False, fontsize=st.pt(6.0, 11),
               title_fontsize=st.pt(7.0, 11),
               columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(rect=(0.02, 0, 0.99, 0.955), h_pad=1.4, w_pad=2.6)
    return fig

def main():
    st.apply()
    d = load()
    solar = np.sort(d.p_solar_farm.unique())
    flight = np.sort(d.p_flight.unique())

    grids = []
    for letter_line, letter_cont, label, col, stem in METRICS:
        grid = (d.pivot(index="p_flight", columns="p_solar_farm", values=col)
                 .reindex(index=flight, columns=solar))
        grids.append(grid)
        for letter, fn, kind in ((letter_line, lines, "lines"),
                                 (letter_cont, contour, "contour")):
            fig = fn(grid, solar, flight, label)
            name = "figure6%s_%s_%s" % (letter, stem, kind)
            for f in st.save(fig, os.path.join(OUT, name)):
                print("wrote", f)
            plt.close(fig)

    fig = composite(grids, solar, flight)
    for f in st.save(fig, os.path.join(ASSEMBLED, "figure6")):
        print("wrote", f)
    plt.close(fig)

    print("\n%d scenarios, solar %d-%d%%, flight %d-%d%%"
          % (len(d), 100 * solar.min(), 100 * solar.max(),
             100 * flight.min(), 100 * flight.max()))
    for _, _, label, col, _ in METRICS:
        print("  %-40s %14s ~ %s"
              % (label, format(d[col].min(), ",.0f"),
                 format(d[col].max(), ",.0f")))

if __name__ == "__main__":
    main()
