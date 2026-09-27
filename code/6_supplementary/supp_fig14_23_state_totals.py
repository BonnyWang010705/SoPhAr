"""Aggregate the 100 farm-and-flight choice scenarios once.

Reading every scenario's flight file costs a couple of gigabytes, so the
state-level totals the bar panels of Supplementary Figs. 14-23 need are written to
`results/optimization_2/penetration_state_totals.csv`:

    farm_p, flight_p, side, state, cls, total_usd

Totals are the manuscript's basis: fuel saving plus CO2 reduction at the
$84/t carbon price (Money_Cost_Saving is the fuel saving only).
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
OPT2 = os.path.join(ROOT, "results", "optimization_2")
RESULTS = os.path.join(OPT2, "Optimization_2_Results_R1")
OUT = os.path.join(OPT2, "penetration_state_totals.csv")

CARBON_PRICE = 0.084
RATES = [round(0.1 * k, 1) for k in range(1, 11)]
CLASS_EDGES = [0, 1500, 4000, float("inf")]
CLASS_NAMES = ["short", "medium", "long"]
COLUMNS = ["STATE_Origin", "STATE_Destination", "Distance_km",
           "Money_Cost_Saving", "CO2_Emissions_Reduction"]


def scenario(farm_p, flight_p):
    name = "flight_results_%sfarm_%sflight.csv" % (farm_p, flight_p)
    d = pd.read_csv(os.path.join(RESULTS, name), usecols=COLUMNS, low_memory=False)
    d["cls"] = pd.cut(d.Distance_km, CLASS_EDGES, labels=CLASS_NAMES,
                      right=False)
    d["total_usd"] = d.Money_Cost_Saving + d.CO2_Emissions_Reduction * CARBON_PRICE
    frames = []
    for side in ("Origin", "Destination"):
        g = (d.groupby(["STATE_%s" % side, "cls"], observed=True)["total_usd"]
              .sum().reset_index())
        g.columns = ["state", "cls", "total_usd"]
        g["side"] = side
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["farm_p"], out["flight_p"] = farm_p, flight_p
    return out[["farm_p", "flight_p", "side", "state", "cls", "total_usd"]]


def main():
    frames = []
    for farm_p in RATES:
        for flight_p in RATES:
            frames.append(scenario(farm_p, flight_p))
            print("  %.1f farm / %.1f flight: $%s"
                  % (farm_p, flight_p,
                     format(frames[-1].total_usd.sum() / 2, ",.0f")), flush=True)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(OUT, index=False)
    print("wrote %s (%d rows)" % (OUT, len(table)))


if __name__ == "__main__":
    main()
