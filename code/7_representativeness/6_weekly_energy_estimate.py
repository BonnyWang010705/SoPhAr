"""Where the reference week sits among the 51 complete weeks of 2025, in the
flight inputs the model reads and in an estimate of the energy it delivers.

In the model, farm capacity, aircraft and prices are the same in every week, so
a different week changes the results only through its flights. Two steps:

1. Inputs. Total elapsed hours, and the distance of each week's flight mix from
   the annual mix, over range classes and over origin-destination pairs, by
   total variation distance (TVD): half the sum of the absolute differences in
   shares, i.e. the share of flights that would have to move to make the two
   mixes equal.
2. Energy. The delivered energy per flight on each origin-destination pair in
   the reference week (the baseline run at 12,100 m, which sums to the 36,637
   MWh of the manuscript) is applied to every week's flights on that pair.
   Pairs the reference week does not fly take the reference week's mean energy
   per flight of their range class. The estimate holds each pair's energy per
   flight fixed, so it leaves out farms shared between simultaneous flights.
"""
import os

import numpy as np
import pandas as pd

import airport_match
from daynight import week_of

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
DATA = os.path.join(ROOT, "data", "flights", "processed", "flights_2025.parquet")
RESULTS = os.path.join(ROOT, "results", "baseline",
                       "flight_analysis_results_merged_R1.csv")
DATA_OUT = os.path.join(ROOT, "results", "representativeness")

REFERENCE_WEEK = pd.Timestamp("2025-04-07")
FIRST_COMPLETE, LAST_COMPLETE = pd.Timestamp("2025-01-06"), pd.Timestamp("2025-12-22")
CLASS_EDGES = [0, 1500, 4000, float("inf")]
CLASS_NAMES = ["short", "medium", "long"]
MANUSCRIPT_MWH = 36637.0


def tvd(p, q):
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum()


def main():
    df = pd.read_parquet(DATA, columns=[
        "Wheels-off time (UTC)", "Actual elapsed time (Minutes)",
        "Distance (km)", "Origin Airport", "Destination Airport"])
    df = airport_match.apply(df)
    df["week"] = week_of(df["Wheels-off time (UTC)"])
    df = df[(df["week"] >= FIRST_COMPLETE) & (df["week"] <= LAST_COMPLETE)].copy()
    df["od"] = df["Origin Airport"] + "-" + df["Destination Airport"]
    df["cls"] = pd.cut(df["Distance (km)"], CLASS_EDGES, labels=CLASS_NAMES,
                       right=False)
    ref = df[df["week"] == REFERENCE_WEEK]

    # ---- inputs ----
    hours = df.groupby("week")["Actual elapsed time (Minutes)"].sum() / 60.0
    counts = df.groupby(["week", "od"]).size().unstack(fill_value=0)
    annual_od = counts.sum() / counts.values.sum()
    od_tvd = counts.apply(lambda r: tvd(r / r.sum(), annual_od), axis=1)
    cls = (df.groupby(["week", "cls"], observed=True).size()
             .unstack("cls")[CLASS_NAMES])
    annual_cls = cls.sum() / cls.values.sum()
    cls_tvd = cls.apply(lambda r: tvd(r / r.sum(), annual_cls), axis=1)

    # ---- energy ----
    res = pd.read_csv(RESULTS, usecols=["Origin Airport", "Destination Airport",
                                        "Total_Power_Received_MWh"])
    assert abs(res["Total_Power_Received_MWh"].sum() - MANUSCRIPT_MWH) < 1, \
        "baseline results do not sum to the manuscript's delivered energy"
    res["od"] = res["Origin Airport"] + "-" + res["Destination Airport"]
    energy_od = res.groupby("od")["Total_Power_Received_MWh"].sum()
    flights_od = ref.groupby("od").size()
    assert set(energy_od.index) <= set(flights_od.index)
    per_flight = energy_od.reindex(flights_od.index).fillna(0.0) / flights_od
    per_class = (ref.assign(e=ref["od"].map(per_flight))
                    .groupby("cls", observed=True)["e"].mean())
    od_class = df.groupby("od")["cls"].first()
    known = counts.columns.isin(per_flight.index)
    e = (per_flight.reindex(counts.columns)
         .fillna(od_class.reindex(counts.columns).map(per_class).astype(float)))
    energy = pd.Series(counts.to_numpy() @ e.to_numpy(), index=counts.index)
    off_ref = counts.loc[:, ~known].sum(axis=1) / counts.sum(axis=1) * 100

    table = pd.DataFrame({
        "flights": counts.sum(axis=1),
        "elapsed_hours": hours.round(1),
        "range_class_tvd_vs_annual_pct": (cls_tvd * 100).round(3),
        "od_tvd_vs_annual_pct": (od_tvd * 100).round(3),
        "energy_estimate_mwh": energy.round(1),
        "flights_on_pairs_absent_from_reference_pct": off_ref.round(3),
    })
    table.index = table.index.date
    os.makedirs(DATA_OUT, exist_ok=True)
    table.to_csv(os.path.join(DATA_OUT, "weekly_energy_estimate.csv"),
                 index_label="week")

    # ---- summary ----
    others = energy.index != REFERENCE_WEEK
    ref_e = energy[REFERENCE_WEEK]
    dev = energy[others] / ref_e - 1

    def top(s):
        return int((s > s[REFERENCE_WEEK]).sum()) + 1

    print(f"reference week: {len(ref):,} flights, {hours[REFERENCE_WEEK]:,.0f} h "
          f"({hours[REFERENCE_WEEK]/hours.median()-1:+.2%} vs median, "
          f"{top(hours)} of 51 from the top)")
    print(f"range-class TVD vs annual: reference {cls_tvd[REFERENCE_WEEK]:.2%}, "
          f"median {cls_tvd.median():.2%}, max {cls_tvd.max():.2%}")
    print(f"OD TVD vs annual: reference {od_tvd[REFERENCE_WEEK]:.2%}, "
          f"median {od_tvd.median():.2%}, range {od_tvd.min():.2%}-{od_tvd.max():.2%}; "
          f"{int((od_tvd[others] > od_tvd[REFERENCE_WEEK]).sum())} of 50 other weeks are farther")
    print(f"energy estimate: reference {ref_e:,.0f} MWh (manuscript {MANUSCRIPT_MWH:,.0f}), "
          f"{top(energy)} of 51 from the top; range {energy.min():,.0f}-{energy.max():,.0f}")
    print(f"  median week {energy.median():,.0f} ({energy.median()/ref_e-1:+.2%}), "
          f"mean {energy.mean():,.0f} ({energy.mean()/ref_e-1:+.2%}), "
          f"lowest {energy.idxmin().date()} ({energy.min()/ref_e-1:+.2%})")
    print(f"  other weeks within 5%: {int((dev.abs() <= 0.05).sum())} of 50, "
          f"within 10%: {int((dev.abs() <= 0.10).sum())} of 50")
    print(f"  flights on pairs absent from the reference week: max {off_ref.max():.2f}%, "
          f"median {off_ref.median():.2f}%")
    per_hour = energy / hours
    print(f"energy per flight hour: reference {per_hour[REFERENCE_WEEK]*1000:.1f} kWh/h, "
          f"{top(per_hour)} of 51 from the top")


if __name__ == "__main__":
    main()
