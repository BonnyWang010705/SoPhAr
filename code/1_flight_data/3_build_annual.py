import argparse
import glob
import os
import time

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
HERE = os.path.dirname(os.path.abspath(__file__))
BY_MONTH = os.path.join(ROOT, "data", "flights", "processed", "by_month")
DATA = os.path.join(ROOT, "data", "flights", "processed")

TARGET_YEAR = 2025
LOCAL_DATE = "Date (MM/DD/YYYY)"
DEPARTURE_UTC = "Wheels-off time (UTC)"
ARRIVAL_UTC = "Landing time (UTC)"
ARRIVAL_LOCAL = "Landing time (local)"


def log(report, line):
    print(line)
    report.append(line)


def load_month_table(year):
    paths = sorted(glob.glob(os.path.join(BY_MONTH, f"flights_{year}_*.parquet")))
    if not paths:
        raise SystemExit(
            f"no monthly tables for {year} in {BY_MONTH}\n"
            f"run: python 2_preprocess.py"
        )
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["_source_year"] = year
    return df


def annotate_clocks(df):
    df["_dep_local_year"] = pd.to_datetime(df[LOCAL_DATE], format="%m/%d/%Y").dt.year
    df["_dep_utc_year"] = pd.to_datetime(df[DEPARTURE_UTC]).dt.year
    df["_arr_utc_year"] = pd.to_datetime(df[ARRIVAL_UTC]).dt.year
    df["_arr_local_year"] = pd.to_datetime(df[ARRIVAL_LOCAL]).dt.year
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-csv", action="store_true",
                    help="write only the parquet file")
    args = ap.parse_args()

    report, started = [], time.time()
    log(report, f"Assembling the {TARGET_YEAR} flight table")
    log(report, f"Source: {BY_MONTH}\n")

    frames = [annotate_clocks(load_month_table(y)) for y in (2024, TARGET_YEAR, 2026)]
    pool = pd.concat(frames, ignore_index=True)
    log(report, f"Pool of candidate flights: {len(pool):,}")

    in_2025 = (
        (pool["_dep_utc_year"] == TARGET_YEAR)
        | (pool["_arr_utc_year"] == TARGET_YEAR)
    )
    annual = pool[in_2025].copy()

    # Filed under a 2025 local date, yet outside 2025 in UTC at both ends.
    dropped_local = pool[(pool["_dep_local_year"] == TARGET_YEAR) & ~in_2025]
    log(report, f"Excluded, local date 2025 but outside 2025 in UTC: "
                f"{len(dropped_local):,} flights")

    log(report, "")
    log(report, "Composition")
    for year in (2024, TARGET_YEAR, 2026):
        block = annual[annual["_source_year"] == year]
        if year == TARGET_YEAR:
            log(report, f"  {year} tables            : {len(block):,} flights")
        else:
            local_dates = sorted({str(d) for d in
                                  pd.to_datetime(block[LOCAL_DATE],
                                                 format="%m/%d/%Y").dt.date})
            utc_dates = sorted({str(d) for d in
                                pd.to_datetime(block[DEPARTURE_UTC]).dt.date})
            log(report, f"  {year} adjacent month    : {len(block):,} flights "
                        f"(local {', '.join(local_dates)} -> UTC {', '.join(utc_dates)})")

    # Flights whose UTC departure lies outside 2025, kept because they land in it.
    for other in (2024, 2026):
        by_arrival = annual[(annual["_dep_utc_year"] == other)]
        log(report, f"  UTC departure {other}, kept by arrival: {len(by_arrival):,} flights")

    log(report, "")
    log(report, f"Annual table: {len(annual):,} flights")
    log(report, f"  operating carriers : {annual['Carrier Code'].nunique()}")
    log(report, f"  airports           : "
                f"{len(set(annual['Origin Airport']) | set(annual['Destination Airport']))}")
    local = pd.to_datetime(annual[LOCAL_DATE], format="%m/%d/%Y")
    utc = pd.to_datetime(annual[DEPARTURE_UTC])
    log(report, f"  local date range   : {local.min().date()} to {local.max().date()}")
    log(report, f"  UTC date range     : {utc.min().date()} to {utc.max().date()}")

    first = utc[utc.dt.date == pd.Timestamp(f"{TARGET_YEAR}-01-01").date()].min()
    log(report, f"  first flight on 1 January UTC: {first.strftime('%H:%M')}")

    annual = annual.drop(columns=[c for c in annual.columns if c.startswith("_")])

    log(report, "")
    pq = os.path.join(DATA, "flights_2025.parquet")
    annual.to_parquet(pq, index=False)
    log(report, f"Wrote {pq}  ({os.path.getsize(pq) / 1e6:.0f} MB)")

    if not args.no_csv:
        for name, kwargs in (("csv", {}), ("csv.gz", {"compression": "gzip"})):
            path = os.path.join(DATA, f"flights_2025.{name}")
            annual.to_csv(path, index=False, **kwargs)
            log(report, f"Wrote {path}  ({os.path.getsize(path) / 1e6:.0f} MB)")

    log(report, f"Elapsed {time.time() - started:.0f}s")

    with open(os.path.join(DATA, "build_annual_report.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
