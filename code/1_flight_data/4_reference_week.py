import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
DATA = os.path.join(ROOT, "data", "flights", "processed")

REFERENCE_TZ = "America/Chicago"
WEEK_START = pd.Timestamp("2025-04-07", tz=REFERENCE_TZ)
WEEK_END = pd.Timestamp("2025-04-14", tz=REFERENCE_TZ)


def main():
    annual = pd.read_parquet(os.path.join(DATA, "flights_2025.parquet"))
    utc = pd.to_datetime(annual["Wheels-off time (UTC)"]).dt.tz_localize("UTC")
    week = annual[(utc >= WEEK_START) & (utc < WEEK_END)].reset_index(drop=True)

    week.to_parquet(os.path.join(DATA, "flights_reference_week.parquet"),
                    index=False)
    week.to_csv(os.path.join(DATA, "flights_reference_week.csv"), index=False)

    print(f"annual         : {len(annual):,} flights")
    print(f"reference week : {len(week):,} flights")
    print(f"window         : {WEEK_START} to {WEEK_END}")
    print(f"               : {WEEK_START.tz_convert('UTC')} to "
          f"{WEEK_END.tz_convert('UTC')} UTC")
    print(f"carriers       : {week['Carrier Code'].nunique()}")
    print(f"airports       : "
          f"{len(set(week['Origin Airport']) | set(week['Destination Airport']))}")


if __name__ == "__main__":
    main()
