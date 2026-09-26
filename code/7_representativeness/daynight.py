import numpy as np
import pandas as pd

REFERENCE_TZ = "America/Chicago"

DAY_START_MIN = 6 * 60
DAY_END_MIN = 18 * 60
DAY_LENGTH_MIN = DAY_END_MIN - DAY_START_MIN
MINUTES_PER_DAY = 24 * 60

UTC_DEPARTURE = "Wheels-off time (UTC)"
UTC_ARRIVAL = "Landing time (UTC)"
ELAPSED = "Actual elapsed time (Minutes)"

REQUIRED_COLUMNS = [UTC_DEPARTURE, UTC_ARRIVAL, ELAPSED]


def _cumulative_daytime_minutes(minutes):
    whole_days = np.floor(minutes / MINUTES_PER_DAY)
    into_day = minutes - whole_days * MINUTES_PER_DAY
    return (whole_days * DAY_LENGTH_MIN
            + np.clip(into_day - DAY_START_MIN, 0.0, DAY_LENGTH_MIN))


def to_reference_clock(utc_series):
    return (pd.to_datetime(utc_series)
              .dt.tz_localize("UTC")
              .dt.tz_convert(REFERENCE_TZ)
              .dt.tz_localize(None))


def week_of(utc_series):
    local = to_reference_clock(utc_series)
    return (local - pd.to_timedelta(local.dt.weekday, unit="D")).dt.normalize()


def _to_reference_minutes(utc_series):
    return to_reference_clock(utc_series).to_numpy("datetime64[m]").astype(float)


def _offset_minutes(utc_series):
    utc = pd.to_datetime(utc_series)
    local = utc.dt.tz_localize("UTC").dt.tz_convert(REFERENCE_TZ).dt.tz_localize(None)
    return (local - utc).dt.total_seconds().to_numpy() / 60.0


def split_daytime_hours(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    elapsed_min = df[ELAPSED].to_numpy(dtype=float)
    start = _to_reference_minutes(df[UTC_DEPARTURE])
    end = start + elapsed_min

    n_across_dst = int((_offset_minutes(df[UTC_DEPARTURE])
                        != _offset_minutes(df[UTC_ARRIVAL])).sum())

    span = np.where(elapsed_min > 0, elapsed_min, 1.0)
    daytime_share = np.where(
        elapsed_min > 0,
        (_cumulative_daytime_minutes(end) - _cumulative_daytime_minutes(start))
        / span,
        0.0,
    )
    daytime_share = np.clip(daytime_share, 0.0, 1.0)
    elapsed_h = elapsed_min / 60.0

    daytime = pd.Series(elapsed_h * daytime_share, index=df.index)
    nighttime = pd.Series(elapsed_h * (1.0 - daytime_share), index=df.index)
    return daytime, nighttime, n_across_dst
