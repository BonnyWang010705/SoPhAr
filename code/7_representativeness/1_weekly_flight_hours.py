import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import airport_match
import natstyle as ns
from daynight import REQUIRED_COLUMNS, split_daytime_hours, week_of

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "flights", "processed", "flights_2025.parquet")
OUT = os.path.join(ROOT, "figures", "drafts", "representativeness")
DATA_OUT = os.path.join(ROOT, "results", "representativeness")

REFERENCE_WEEK = pd.Timestamp("2025-04-07")
FIRST_COMPLETE = pd.Timestamp("2025-01-06")
LAST_COMPLETE = pd.Timestamp("2025-12-22")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DATA_OUT, exist_ok=True)
    ns.apply()

    df = pd.read_parquet(DATA, columns=REQUIRED_COLUMNS + ["Air time (Minutes)",
                                                           "reference_carrier_set"]
                         + airport_match.COLUMNS)
    df = airport_match.apply(df).reset_index(drop=True)
    daytime, nighttime, n_fallback = split_daytime_hours(df)

    week = week_of(df["Wheels-off time (UTC)"])

    wk = pd.DataFrame({
        "week": week,
        "daytime_h": daytime,
        "nighttime_h": nighttime,
        "elapsed_h": df["Actual elapsed time (Minutes)"] / 60.0,
        "air_h": df["Air time (Minutes)"] / 60.0,
        "ref_elapsed_h": df["Actual elapsed time (Minutes)"].where(
            df["reference_carrier_set"], 0.0) / 60.0,
    }).groupby("week").agg(
        flights=("elapsed_h", "size"),
        elapsed_hours=("elapsed_h", "sum"),
        daytime_hours=("daytime_h", "sum"),
        nighttime_hours=("nighttime_h", "sum"),
        air_hours=("air_h", "sum"),
        elapsed_hours_17_carriers=("ref_elapsed_h", "sum"),
    ).reset_index()

    wk["daytime_share_pct"] = wk["daytime_hours"] / wk["elapsed_hours"] * 100
    wk["partial"] = (wk["week"] < FIRST_COMPLETE) | (wk["week"] > LAST_COMPLETE)

    full = wk[~wk["partial"]]
    ref = wk.loc[wk["week"] == REFERENCE_WEEK].iloc[0]
    median = full["elapsed_hours"].median()
    rank = int((full["elapsed_hours"] > ref["elapsed_hours"]).sum()) + 1
    below = (full["elapsed_hours"] <= ref["elapsed_hours"]).mean() * 100
    share = full["daytime_share_pct"]

    lines = [
        f"weeks: {len(wk)} ({len(full)} complete, {int(wk['partial'].sum())} partial)",
        f"reference week {REFERENCE_WEEK.date()}: {ref['elapsed_hours']:,.0f} hours, "
        f"{int(ref['flights']):,} flights",
        f"  daytime {ref['daytime_hours']:,.0f} h ({ref['daytime_share_pct']:.1f}%), "
        f"nighttime {ref['nighttime_hours']:,.0f} h",
        f"rank among complete weeks: {rank} of {len(full)} "
        f"({ref['elapsed_hours']/median-1:+.2%} vs median, {below:.0f}th percentile)",
        f"complete weeks: min {full['elapsed_hours'].min():,.0f}, "
        f"median {median:,.0f}, max {full['elapsed_hours'].max():,.0f} hours",
        f"spread: max/min = {full['elapsed_hours'].max()/full['elapsed_hours'].min():.2f}, "
        f"CV = {full['elapsed_hours'].std()/full['elapsed_hours'].mean()*100:.1f}%",
        f"daytime share: min {share.min():.1f}%, median {share.median():.1f}%, "
        f"max {share.max():.1f}% (reference {ref['daytime_share_pct']:.1f}%)",
        f"flights airborne across a daylight-saving transition: {n_fallback:,}",
    ]
    text = "\n".join(lines)
    print(text)

    out = wk.copy()
    out["week"] = out["week"].dt.date
    for c in ["elapsed_hours", "daytime_hours", "nighttime_hours", "air_hours",
              "elapsed_hours_17_carriers", "daytime_share_pct"]:
        out[c] = out[c].round(1)
    out.to_csv(os.path.join(DATA_OUT, "weekly_flight_hours.csv"), index=False)
    with open(os.path.join(DATA_OUT, "weekly_flight_hours.txt"), "w") as fh:
        fh.write(text + "\n")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(ns.MANUSCRIPT_WIDTH, 70 * ns.MM),
                           layout="constrained")
    x = range(len(wk))
    alpha = [0.45 if p else 1.0 for p in wk["partial"]]

    day_k = wk["daytime_hours"] / 1000.0
    night_k = wk["nighttime_hours"] / 1000.0
    ax.bar(x, day_k, width=0.78, color=ns.DAY, alpha=1.0, zorder=3,
           linewidth=0)
    ax.bar(x, night_k, width=0.78, bottom=day_k, color=ns.NIGHT, zorder=3,
           linewidth=0)
    for i, a in enumerate(alpha):                      # fade the partial weeks
        if a < 1:
            ax.bar([i], [day_k.iloc[i]], width=0.78, color=ns.SURFACE,
                   alpha=1 - a, zorder=4, linewidth=0)
            ax.bar([i], [night_k.iloc[i]], width=0.78, bottom=day_k.iloc[i],
                   color=ns.SURFACE, alpha=1 - a, zorder=4, linewidth=0)

    total_k = wk["elapsed_hours"] / 1000.0
    ax.axhline(median / 1000.0, color=ns.INK_2, lw=0.6, ls=(0, (3, 2)), zorder=2)

    i_ref = int(wk.index[wk["week"] == REFERENCE_WEEK][0])
    top = total_k.max() * 1.10
    ax.annotate("7–13 Apr", xy=(i_ref, total_k.iloc[i_ref]),
                xytext=(i_ref, top), ha="center", va="bottom",
                fontsize=ns.SMALL_PT, color=ns.INK,
                arrowprops=dict(arrowstyle="-", color=ns.INK, lw=0.5,
                                shrinkA=1, shrinkB=2))

    ns.strip(ax)
    ax.set_ylim(0, top * 1.16)
    ax.margins(x=0.008)
    ax.set_ylabel("Flight hours per week (thousands)", color=ns.INK_2)

    months = wk["week"].dt.to_period("M")
    ticks = [i for i in range(len(wk)) if i == 0 or months.iloc[i] != months.iloc[i - 1]]
    ax.set_xticks(ticks[1:13])
    ax.set_xticklabels([wk["week"].iloc[i].strftime("%b") for i in ticks[1:13]],
                       color=ns.INK_2)

    ax.legend(handles=[Patch(facecolor=ns.DAY, label="Daytime (06:00–18:00 CT)"),
                       Patch(facecolor=ns.NIGHT, label="Nighttime (18:00–06:00 CT)"),
                       Line2D([], [], color=ns.INK_2, lw=0.6, ls=(0, (3, 2)),
                              label="Median of complete weeks")],
              loc="upper left", bbox_to_anchor=(0, 1.06), ncol=3,
              frameon=False, handlelength=1.1, handleheight=0.9,
              columnspacing=1.2, borderpad=0)

    for p in ns.save(fig, OUT, "weekly_flight_hours"):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
