import argparse
import io
import os
import sys
import time
import zipfile

import airportsdata
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "flights", "raw")
OUT_DIR = os.path.join(ROOT, "data", "flights", "processed", "by_month")

REFERENCE_CARRIERS = {
    "9E", "AA", "AS", "B6", "DL", "F9", "G4", "HA", "MQ",
    "NK", "OH", "OO", "QX", "UA", "WN", "YV", "YX",
}

# Columns pulled from the 120-column source table.
SOURCE_COLS = [
    "FlightDate",
    "Operating_Airline ",  # trailing space is present in the BTS header
    "Flight_Number_Operating_Airline",
    "Marketing_Airline_Network",
    "Tail_Number",
    "Origin",
    "Dest",
    "WheelsOff",
    "ActualElapsedTime",
    "AirTime",
    "Distance",
    "Cancelled",
    "Diverted",
]

MANUAL_TZ = {
    "USA": "America/New_York", "UTM": "America/Chicago", "UST": "America/New_York",
    "TKI": "America/Anchorage", "BQN": "America/Puerto_Rico", "PPG": "Pacific/Pago_Pago",
    "GUM": "Pacific/Guam", "SPN": "Pacific/Saipan", "ROR": "Pacific/Palau",
    "FCA": "America/Denver", "RIW": "America/Denver", "VPS": "America/Chicago",
}

MILES_TO_KM = 1.609344


def log(report, message):
    print(message)
    report.append(message)


def month_path(year, month):
    return os.path.join(
        RAW_DIR,
        "On_Time_Marketing_Carrier_On_Time_Performance_"
        f"Beginning_January_2018_{year}_{month}.zip")


def read_month(path):
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(name) as fh:
            data = io.BytesIO(fh.read())
    df = pd.read_csv(data, low_memory=False, usecols=SOURCE_COLS)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_tz_map(codes, report):
    table = airportsdata.load("IATA")
    mapping, unknown = {}, []
    for code in codes:
        entry = table.get(code)
        tz = entry["tz"] if entry else MANUAL_TZ.get(code)
        if tz:
            mapping[code] = tz
        else:
            unknown.append(code)
    if unknown:
        log(report, f"  WARNING: no timezone for {len(unknown)} airports: {sorted(unknown)}")
        log(report, "           rows touching these airports will be dropped")
    return mapping


def combine_local(dates, hhmm):
    v = pd.to_numeric(hhmm, errors="coerce")
    hours = (v // 100).astype("Int64")
    minutes = (v % 100).astype("Int64")
    rollover = hours == 24
    hours = hours.where(~rollover, 0)
    out = (
        dates
        + pd.to_timedelta(hours.astype("float"), unit="h")
        + pd.to_timedelta(minutes.astype("float"), unit="m")
        + pd.to_timedelta(rollover.astype("float"), unit="D")
    )
    return out, int(rollover.sum())


def localize_by_tz(naive, tz_codes, report, label):
    out = pd.Series(pd.NaT, index=naive.index, dtype="datetime64[ns, UTC]")
    flagged = 0
    for tz, idx in tz_codes.groupby(tz_codes).groups.items():
        block = naive.loc[idx]
        valid = block.notna()
        if not valid.any():
            continue
        localized = block[valid].dt.tz_localize(
            tz, ambiguous=False, nonexistent="shift_forward"
        )
        out.loc[block[valid].index] = localized.dt.tz_convert("UTC")
        flagged += int(
            block[valid].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").isna().sum()
        )
    if flagged:
        log(report, f"  {label}: {flagged:,} timestamps sat on a DST transition (resolved)")
    return out


def process_month(year, month, report):
    path = month_path(year, month)
    df = read_month(path)
    raw_n = len(df)

    # --- record filter ---
    keep = (
        (df["Cancelled"] == 0)
        & df["WheelsOff"].notna()
        & df["ActualElapsedTime"].notna()
        & (df["ActualElapsedTime"] != 0)
    )
    dropped = {
        "cancelled": int((df["Cancelled"] != 0).sum()),
        "no wheels-off": int(df["WheelsOff"].isna().sum()),
        "no/zero elapsed": int(
            (df["ActualElapsedTime"].isna() | (df["ActualElapsedTime"] == 0)).sum()
        ),
    }
    df = df[keep].copy()

    # --- timezone lookup ---
    codes = sorted(set(df["Origin"]) | set(df["Dest"]))
    tz_map = build_tz_map(codes, report)
    df["origin_tz"] = df["Origin"].map(tz_map)
    df["dest_tz"] = df["Dest"].map(tz_map)
    no_tz = int(df["origin_tz"].isna().sum() + df["dest_tz"].isna().sum())
    df = df[df["origin_tz"].notna() & df["dest_tz"].notna()].copy()

    # --- time unification ---
    flight_date = pd.to_datetime(df["FlightDate"])
    dep_local, rollovers = combine_local(flight_date, df["WheelsOff"])
    dep_utc = localize_by_tz(dep_local, df["origin_tz"], report, "departure")
    arr_utc = dep_utc + pd.to_timedelta(df["ActualElapsedTime"].astype(float), unit="m")

    arr_local = pd.Series(pd.NaT, index=df.index, dtype="object")
    for tz, idx in df["dest_tz"].groupby(df["dest_tz"]).groups.items():
        block = arr_utc.loc[idx]
        arr_local.loc[idx] = block.dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M")

    out = pd.DataFrame(
        {
            # --- core columns ---
            "Date (MM/DD/YYYY)": flight_date.dt.strftime("%m/%d/%Y"),
            "Wheels-off time (Local)": dep_local.dt.strftime("%Y-%m-%d %H:%M"),
            "Landing time (local)": arr_local,
            "IANA Timezone ID": df["origin_tz"],
            "Wheels-off time (UTC)": dep_utc.dt.strftime("%Y-%m-%d %H:%M"),
            "Landing time (UTC)": arr_utc.dt.strftime("%Y-%m-%d %H:%M"),
            "Actual elapsed time (Minutes)": df["ActualElapsedTime"].astype(int),
            "Origin Airport": df["Origin"],
            "Destination Airport": df["Dest"],
            "Carrier Code": df["Operating_Airline"],
            "Flight Number": df["Flight_Number_Operating_Airline"].astype(int),
            "Tail Number": df["Tail_Number"],
            # --- additional fields available in the monthly archives ---
            "Marketing Carrier": df["Marketing_Airline_Network"],
            "Air time (Minutes)": df["AirTime"],
            "Distance (km)": (df["Distance"].astype(float) * MILES_TO_KM).round(2),
            "Diverted": df["Diverted"].astype(int),
            "Destination Timezone ID": df["dest_tz"],
            "reference_carrier_set": df["Operating_Airline"].isin(REFERENCE_CARRIERS),
        }
    )
    # Sort by date, then origin, then carrier.
    out = out.sort_values(
        ["Date (MM/DD/YYYY)", "Origin Airport", "Carrier Code"], kind="stable"
    )

    log(
        report,
        f"  {year}-{month:02d}: {raw_n:,} raw -> {len(out):,} kept "
        f"(cancelled {dropped['cancelled']:,}, no wheels-off {dropped['no wheels-off']:,}, "
        f"no/zero elapsed {dropped['no/zero elapsed']:,}, no timezone {no_tz:,}, "
        f"2400 rollovers {rollovers:,})",
    )
    return out


def parse_month(text):
    year, month = text.split("-")
    return int(year), int(month)


def default_months():
    return [(2024, 12)] + [(2025, m) for m in range(1, 13)] + [(2026, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=parse_month, nargs="+",
                    default=default_months(), metavar="YYYY-MM")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    report, started = [], time.time()

    missing = [f"{y}-{m:02d}" for y, m in args.months
               if not os.path.exists(month_path(y, m))]
    if missing:
        print(f"ERROR: missing archives for {missing}. Run 1_download.py first.")
        return 1

    log(report, f"Source: {RAW_DIR}\n")

    for year, month in args.months:
        table = process_month(year, month, report)
        stem = os.path.join(OUT_DIR, f"flights_{year}_{month:02d}")
        table.to_parquet(f"{stem}.parquet", index=False)
        table.to_csv(f"{stem}.csv.gz", index=False, compression="gzip")
        log(report, f"  wrote flights_{year}_{month:02d}.parquet  "
                    f"({os.path.getsize(stem + '.parquet') / 1e6:.0f} MB)")

    log(report, f"Elapsed {time.time() - started:.0f}s")
    with open(os.path.join(OUT_DIR, "preprocess_report.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
