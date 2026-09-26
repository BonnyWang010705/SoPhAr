import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

import agg
import style as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
OUT = os.path.join(ROOT, "figures", "main", "fig04_temporal_analysis", "panels")

DAY_COLORS = {"short": "#FAD02E", "medium": "#F28C28", "long": "#D35400"}
NIGHT_COLORS = {"short": "#A9CCE3", "medium": "#2980B9", "long": "#1B4F72"}

ROWS_6 = [("Arrival total", "Destination", "total_usd"),
          ("Departure total", "Origin", "total_usd"),
          ("Arrival $", "Destination", agg.FUEL),
          ("Departure $", "Origin", agg.FUEL),
          ("Arrival CO$_2$", "Destination", "carbon_usd"),
          ("Departure CO$_2$", "Origin", "carbon_usd")]

def state_matrix(flights, rows):
    f = agg.label_day_night(flights)
    out = {}
    for i, (_, side, col) in enumerate(rows):
        g = f.groupby(["STATE_%s" % side, "window", "cls"], observed=True)[col].sum()
        for (state, window, cls), v in g.items():
            out.setdefault(state, np.zeros((len(rows), 2, 3)))
            out[state][i, 0 if window == "day" else 1,
                       agg.CLASS_NAMES.index(cls)] = v
    return out

ROWS_2 = [("Arrival", "Destination", "total_usd"),
          ("Departure", "Origin", "total_usd")]

def nice_limit(v):
    """Round a column maximum up to a readable number (800k, 50k, 5k)."""
    exp = 10 ** np.floor(np.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if v <= m * exp:
            return m * exp
    return 10 * exp

def compact(v):
    v = abs(v)
    if v >= 1e6:
        return ("%.1f" % (v / 1e6)).rstrip("0").rstrip(".") + "M"
    if v >= 1e3:
        return ("%.1f" % (v / 1e3)).rstrip("0").rstrip(".") + "k"
    return "%.0f" % v

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=2, choices=(2, 6),
                   help="2: departure and arrival totals (main text); "
                        "6: also the fuel and CO2 rows of the submitted version")
    p.add_argument("--altitude", type=int, default=12100)
    a = p.parse_args()

    rows = ROWS_2 if a.rows == 2 else ROWS_6
    categories = [r[0] for r in rows]
    st.apply()
    flights = agg.load_flights(a.altitude)
    col_lists = agg.power_groups(flights)
    mat = state_matrix(flights, rows)

    # drawn at the printed width, so every size below is the printed size
    W, H = 7.2, 6.6
    st.scale(W, 1.0, label=6.0, tick=5.0)
    row_pt, state_pt = st.pt(5.5, W), st.pt(6.0, W)
    title_pt, legend_pt = st.pt(7.0, W), st.pt(6.0, W)

    n_cols = 3
    n_rows = max(len(c) for c in col_lists)
    limits = [nice_limit(max(mat[s].sum(axis=2).max() for s in col if s in mat))
              for col in col_lists]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(W, H), squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.905, bottom=0.07,
                        wspace=0.42, hspace=0.32)
    y_pos = np.arange(len(categories))

    for c in range(n_cols):
        limit = limits[c]
        for r in range(n_rows):
            ax = axes[r][c]
            if r >= len(col_lists[c]) or col_lists[c][r] not in mat:
                ax.axis("off")
                continue
            state = col_lists[c][r]
            v = mat[state]
            curr_n = np.zeros(len(categories))
            curr_d = np.zeros(len(categories))
            for k, cls in enumerate(agg.CLASS_NAMES):
                nw, dw = v[:, 1, k], v[:, 0, k]
                ax.barh(y_pos, nw, left=curr_n - nw, height=0.78,
                        color=NIGHT_COLORS[cls], linewidth=0)
                ax.barh(y_pos, dw, left=curr_d, height=0.78,
                        color=DAY_COLORS[cls], linewidth=0)
                curr_n = curr_n - nw
                curr_d = curr_d + dw

            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-0.55, len(categories) - 0.45)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories, fontsize=row_pt)
            ax.tick_params(axis="y", length=0, pad=1.5)
            # state code inside the panel, top left; night bars never reach
            # that far left on these scales
            ax.text(0.0, 1.0, state, transform=ax.transAxes, ha="left",
                    va="top", fontsize=state_pt, fontweight="bold",
                    color="#222222")
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.5)
            ticks = [-limit, -limit / 2, 0, limit / 2, limit]
            ax.set_xticks(ticks)
            last = r == len(col_lists[c]) - 1
            if last:
                # one shared scale per column, labelled once at its foot
                ax.set_xticklabels([compact(t) for t in ticks])
                ax.set_xlabel("Night  |  Day\nTotal cost reduction (\$)",
                              fontsize=st.pt(5.5, W), labelpad=2)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="x", length=1.5, width=0.5, pad=1)

    titles = ["High-power states", "Medium-power states", "Low-power states"]
    for c in range(n_cols):
        pos = axes[0][c].get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.925, titles[c], ha="center",
                 va="bottom", fontsize=title_pt, fontweight="bold")

    # one legend on top, as in Figs 5 and 6: columns are range classes,
    # rows are day and night
    handles = []
    for cls in agg.CLASS_NAMES:
        handles.append(mpatches.Patch(color=DAY_COLORS[cls],
                                      label="%s range, daytime" % cls.capitalize()))
        handles.append(mpatches.Patch(color=NIGHT_COLORS[cls],
                                      label="%s range, nighttime" % cls.capitalize()))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False, fontsize=legend_pt, handlelength=1.4,
               columnspacing=2.0, handletextpad=0.5)

    name = "figure4" + ("" if a.rows == 2 else "_six_rows")
    for f in st.save(fig, os.path.join(OUT, name)):
        print("wrote", f)

    f = agg.label_day_night(flights)
    d = f.groupby("window").total_usd.sum()
    print("\ndaytime flights %.1f%%, daytime cost reduction %.1f%%"
          % (100 * (f.window == "day").mean(), 100 * d["day"] / d.sum()))
    print("columns: %s" % [len(c) for c in col_lists])
    print("column scales: %s" % ["$%s" % format(int(m), ",") for m in limits])

if __name__ == "__main__":
    main()
