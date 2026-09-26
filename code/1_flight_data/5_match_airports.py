"""Keep the reference-week flights whose airports have coordinates.

Flight records name airports by IATA code; the airport dataset
(data/airports/airport_data.geojson, ArcGIS USA Airports) keys them by FAA
identifier. The R1 coverage and optimization runs matched the two directly,
without translating codes, so a flight is kept only when both its origin and
destination codes appear as an FAA_ID. This step reproduces that rule: it turns
the 149,739 reference-week flights into the 148,814 flights analysed in R1.

Two kinds of airport are lost this way:
  - IATA and FAA codes differ although the airport is in the dataset
    (YUM = NYL, MQT = SAW, USA = JQF);
  - the airport is not in the dataset under either code (AZA, FCA, SCE, HHH,
    CLD, SPN) or only under a closed predecessor (XWA, listed as ISN).
Flights pairing a kept airport with a lost one are dropped too.
"""
import json
import os
from collections import Counter

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
DATA = os.path.join(ROOT, "data", "flights", "processed")
AIRPORTS = os.path.join(ROOT, "data", "airports", "airport_data.geojson")

# IATA codes whose FAA identifier differs while the airport is in the dataset.
# Not applied: the R1 results were computed without this translation. Kept here
# so the difference stays documented and can be switched on for a future run.
IATA_TO_FAA = {"YUM": "NYL", "MQT": "SAW", "USA": "JQF"}
TRANSLATE = False


def main():
    week = pd.read_parquet(os.path.join(DATA, "flights_reference_week.parquet"))
    with open(AIRPORTS) as f:
        faa = {feat["properties"]["FAA_ID"] for feat in json.load(f)["features"]}

    origin, destination = week["Origin Airport"], week["Destination Airport"]
    if TRANSLATE:
        origin, destination = origin.replace(IATA_TO_FAA), destination.replace(IATA_TO_FAA)
    keep = origin.isin(faa) & destination.isin(faa)
    matched = week[keep].reset_index(drop=True)

    matched.to_parquet(os.path.join(DATA, "flights_reference_week_matched.parquet"),
                       index=False)

    lost = Counter(code for o, d in zip(origin[~keep], destination[~keep])
                   for code in (o, d) if code not in faa)
    print(f"reference week       : {len(week):,} flights")
    print(f"airports matched     : {len(matched):,} flights")
    print(f"dropped              : {int((~keep).sum()):,} flights")
    print("unmatched codes      : "
          + ", ".join(f"{c} {n}" for c, n in lost.most_common()))
    print(f"code translation     : {'on' if TRANSLATE else 'off'}")


if __name__ == "__main__":
    main()
