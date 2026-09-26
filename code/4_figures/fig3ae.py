import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

import agg
import style as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
OUT = os.path.join(ROOT, "figures", "main", "fig03_flight_analysis", "panels")

def panel_a(totals, share):
    energy = share.energy_mwh.values
    duration = share.duration_min.values
    cls_share = share.total_usd.values / 100     # class shares of every $ figure
    fuel = totals.fuel_usd.sum()
    carbon = totals.carbon_usd.sum()
    total = fuel + carbon
    fuel_pct = 100 * fuel / total
    colors = st.CLASS_COLORS
    height = 0.5

    st.scale(12)
    fig, ax = plt.subplots(figsize=(12, 5.6))

    def money(v):
        return "\\$" + format(v, ",.0f")

    def callout(y, x_end, text, x_text=104):
        ax.plot([x_end - 0.75, x_text - 1], [y, y], color="black",
                linewidth=1.5, zorder=5, clip_on=False)
        ax.text(x_text, y, text, va="center", ha="left", fontweight="bold",
                fontsize=st.pt(7.0, 12), clip_on=False)

    # bars 1 and 2: energy and duration by range class
    for y, seg in ((4, energy), (3, duration)):
        left = 0
        for j, w in enumerate(seg):
            ax.barh(y, w, left=left, color=colors[j], edgecolor="black",
                    linewidth=1, height=height, clip_on=False)
            if j == 2:
                callout(y, left + w, "%.1f%%" % w)
            else:
                ax.text(left + w / 2, y, "%.1f%%" % w, va="center",
                        ha="center", fontweight="bold")
            left += w

    # bar 3: the total, split into fuel and CO2
    ax.barh(2, fuel_pct, color=st.FUEL, edgecolor="black", linewidth=1,
            height=height, clip_on=False)
    ax.barh(2, 100 - fuel_pct, left=fuel_pct, color=st.CARBON_FILL,
            edgecolor="black", linewidth=1, height=height, clip_on=False)
    ax.text(fuel_pct / 2, 2, "Fuel cost saving %s (%.1f%%)"
            % (money(fuel), fuel_pct), va="center", ha="center",
            fontweight="bold")
    ax.text(fuel_pct + (100 - fuel_pct) / 2, 2,
            "CO$_2$ cost saving %s (%.1f%%)" % (money(carbon), 100 - fuel_pct),
            va="center", ha="center", fontweight="bold")

    # bars 4 and 5: fuel and CO2 by range class, each drawn under its own
    # part of the total
    for y, amount, start in ((1, fuel, 0.0), (0, carbon, fuel_pct)):
        width = 100 * amount / total
        left = start
        for j, f in enumerate(cls_share):
            w = width * f
            ax.barh(y, w, left=left, color=colors[j], edgecolor="black",
                    linewidth=1, height=height, clip_on=False)
            text = money(amount * f)
            if j == 2:
                callout(y, left + w, text,
                        x_text=left + w + 1.5 if start == 0 else 104)
            else:
                ax.text(left + w / 2, y, text, va="center", ha="center",
                        fontweight="bold")
            left += w

    ax.plot([fuel_pct, fuel_pct], [-height / 2 - 0.15, 2 + height / 2],
            color="black", linestyle=(0, (3, 1)), linewidth=1.5, zorder=6)

    ax.set_ylim(-0.75, 4.5)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([4, 3, 2, 1, 0])
    # every row carries its own total, so the panel reads without the text
    ax.set_yticklabels(
        ["Energy\n(%s MWh)" % format(totals.energy_mwh.sum(), ",.0f"),
         "Duration\n(%s min)" % format(totals.duration_min.sum(), ",.0f"),
         "Total cost reduction\n(%s)" % money(total),
         "Fuel cost saving\n(%s)" % money(fuel),
         "CO$_2$ cost saving\n(%s)" % money(carbon)])

    handles = [mpatches.Patch(color=c, label=l) for c, l in
               zip(colors, ["Short range", "Medium range", "Long range"])]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=3, frameon=False)
    fig.tight_layout()
    return fig

def panel_e(flights):
    o = agg.state_totals(flights, "Origin")
    d = agg.state_totals(flights, "Destination")
    states = o.index.union(d.index)
    o, d = o.reindex(states).fillna(0.0), d.reindex(states).fillna(0.0)
    order = (o.sum(axis=1) + d.sum(axis=1)).sort_values(ascending=False).index
    o, d = o.loc[order], d.loc[order]

    n_states = len(order)
    bar_width = 0.35
    index = np.arange(n_states)
    colors = st.CLASS_COLORS
    labels = ["Short range", "Medium range", "Long range"]

    st.scale(15)
    fig, ax = plt.subplots(figsize=(15, 5))
    for i, tab in enumerate((o, d)):
        offset = (i - 0.5) * bar_width
        bottoms = np.zeros(n_states)
        for k, cls in enumerate(agg.CLASS_NAMES):
            values = tab[cls].values
            ax.bar(index + offset, values, bar_width, bottom=bottoms,
                   color=colors[k], label=labels[k] if i == 0 else "",
                   edgecolor="black")
            bottoms = bottoms + values

    ax.set_xlabel("State")
    ax.set_ylabel("Total cost reduction ($)")
    ax.set_xticks(index)
    ax.set_xticklabels(list(order), rotation=90, fontsize=st.pt(6.0, 15))
    ax.legend(loc="upper right", ncol=1, frameon=False)
    fig.tight_layout()
    return fig, o, d

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--altitude", type=int, default=12100)
    a = p.parse_args()

    st.apply()
    flights = agg.load_flights(a.altitude)
    totals, share = agg.range_class_decomposition(flights)

    for f in st.save(panel_a(totals, share), os.path.join(OUT, "figure3a")):
        print("wrote", f)
    fig, o, d = panel_e(flights)
    for f in st.save(fig, os.path.join(OUT, "figure3e")):
        print("wrote", f)

    fuel, carbon = totals.fuel_usd.sum(), totals.carbon_usd.sum()
    print("\npanel a")
    print("  class shares of energy   %s" % np.round(share.energy_mwh.values, 1))
    print("  class shares of duration %s" % np.round(share.duration_min.values, 1))
    print("  fuel / carbon of total   %.1f / %.1f"
          % (100 * fuel / (fuel + carbon), 100 * carbon / (fuel + carbon)))
    print("panel e: %d states, origin $%s, destination $%s"
          % (len(o), format(o.values.sum(), ",.0f"), format(d.values.sum(), ",.0f")))

if __name__ == "__main__":
    main()
