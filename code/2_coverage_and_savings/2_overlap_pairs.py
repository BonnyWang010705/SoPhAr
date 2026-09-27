"""Find when each flight passes within beaming range of each solar farm.

Every flight is a straight line between its airports, flown at constant speed
from wheels-off to landing (Methods, Section 3.2.1). Each solar farm polygon is
buffered by the cruise altitude, the farm's energy boundary at that altitude.
A spatial join keeps the flight-farm pairs whose line crosses the buffer; for
each crossing the entry and exit points and the times the aircraft passes them
are recorded.

    python 2_overlap_pairs.py                    # 12,100 m (reference)
    python 2_overlap_pairs.py --altitude 9100
    python 2_overlap_pairs.py --altitude 15100

Writes results/baseline/flight_solar_overlap_pairs[_9100m|_15100m]_R1.csv, one
row per crossing, for all 5,712 farms (0.8-1.3 GB; 3_system_level.py keeps the
qualified ones):

    Trip_ID, Solar_Farm_ID, Solar_Farm_Power (DC capacity, MW),
    Entry_Time, Entry_Longitude, Entry_Latitude,
    Exit_Time, Exit_Longitude, Exit_Latitude, Duration_Overlap_Sec

This is the vectorized form of the per-row loop of the R1 notebooks; it gives
the same events.
"""
import argparse
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

import model

LINESTRING, MULTILINESTRING = 1, 5      # shapely geometry type ids
CHUNK = 250_000                         # flight-farm pairs per batch


def load_inputs(altitude):
    gdf_solar, _, _, _ = model.load_solar_farms(qualified=False)
    flights = model.load_flights()
    flights = flights.rename(columns={"Trip_Duration_Sec": "duration_sec"})
    lines = [shapely.LineString([(lon_o, lat_o), (lon_d, lat_d)]) for lon_o, lat_o, lon_d, lat_d
             in zip(flights["Longitude_Origin"], flights["Latitude_Origin"],
                    flights["Longitude_Destination"], flights["Latitude_Destination"])]
    gdf_flights = gpd.GeoDataFrame(flights, geometry=lines, crs=model.WGS84_CRS)

    flights_proj = gdf_flights.to_crs(model.METRIC_CRS)
    solar_proj = gdf_solar.to_crs(model.METRIC_CRS)
    # the energy boundary: the farm polygon buffered by the cruise altitude
    solar_proj["buffered_geom"] = solar_proj.geometry.buffer(altitude)
    solar_proj = solar_proj.set_geometry("buffered_geom")
    return flights_proj, solar_proj


def crossings(flight_geom, buffer_geom, wheels_off_ns, duration_sec):
    """Entry and exit of each flight line through each farm buffer.

    Inputs are aligned arrays, one element per candidate pair. Returns the
    position of the pair for each crossing segment and the segment's entry and
    exit points and times.
    """
    total_dist = shapely.length(flight_geom)
    ok = (duration_sec > 0) & (total_dist > 0)
    inter = shapely.intersection(flight_geom, buffer_geom)
    kind = shapely.get_type_id(inter)
    ok &= ~shapely.is_empty(inter) & np.isin(kind, (LINESTRING, MULTILINESTRING))

    pair = np.flatnonzero(ok)
    segments, part_of = shapely.get_parts(inter[pair], return_index=True)
    pair = pair[part_of]
    keep = shapely.length(segments) > 0
    segments, pair = segments[keep], pair[keep]

    entry = shapely.get_coordinates(shapely.get_point(segments, 0))
    exit_ = shapely.get_coordinates(shapely.get_point(segments, -1))
    origin = shapely.get_coordinates(shapely.get_point(flight_geom[pair], 0))
    speed = total_dist[pair] / duration_sec[pair]

    def passing_time(xy):
        dist = np.sqrt((xy[:, 0] - origin[:, 0]) ** 2 + (xy[:, 1] - origin[:, 1]) ** 2)
        # nanoseconds truncated toward zero, as pd.Timedelta(seconds=...) does
        return wheels_off_ns[pair] + (dist / speed * 1e9).astype(np.int64)

    return pair, entry, exit_, passing_time(entry), passing_time(exit_)


def total_seconds_us(delta):
    """Durations in seconds at microsecond resolution, as the scalar
    pd.Timedelta.total_seconds() of the R1 run returns them."""
    ns = delta.values.astype("timedelta64[ns]").astype(np.int64)
    days, rest = np.divmod(ns, 86_400 * 10**9)
    seconds, rest = np.divmod(rest, 10**9)
    micro = rest // 1000
    return (days * 86_400 + seconds).astype(np.float64) + micro / 1_000_000


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--altitude", type=int, default=model.REFERENCE_ALTITUDE,
                    choices=model.ALTITUDES, help="cruise altitude in m (default 12100)")
    a = ap.parse_args()
    started = time.time()

    flights_proj, solar_proj = load_inputs(a.altitude)
    # only the columns the crossings need, so that the join stays small
    left = flights_proj[["Trip_ID", "Wheels-off time (UTC)", "duration_sec", "geometry"]]
    right = solar_proj[["FID", "p_cap_dc", "buffered_geom"]]
    joined = gpd.sjoin(left, right, how="inner", predicate="intersects").reset_index(drop=True)
    print("candidate flight-farm pairs: %s" % format(len(joined), ","), flush=True)

    flight_geom = joined.geometry.values.to_numpy()
    buffer_geom = right.geometry.values.to_numpy()[
        right.index.get_indexer(joined["index_right"])]
    wheels_off = joined["Wheels-off time (UTC)"].values.astype("datetime64[ns]").astype(np.int64)
    duration = joined["duration_sec"].to_numpy(dtype=np.float64)

    frames = []
    for start in range(0, len(joined), CHUNK):
        sl = slice(start, start + CHUNK)
        pair, entry, exit_, t_in, t_out = crossings(
            flight_geom[sl], buffer_geom[sl], wheels_off[sl], duration[sl])
        pair = pair + start
        frames.append(pd.DataFrame({
            "Trip_ID": joined["Trip_ID"].values[pair],
            "Solar_Farm_ID": joined["FID"].values[pair],
            "Solar_Farm_Power": joined["p_cap_dc"].values[pair],
            "Entry_Time": t_in.astype("datetime64[ns]"),
            "Entry_X": entry[:, 0], "Entry_Y": entry[:, 1],
            "Exit_Time": t_out.astype("datetime64[ns]"),
            "Exit_X": exit_[:, 0], "Exit_Y": exit_[:, 1],
        }))
        print("  %s / %s pairs" % (format(min(start + CHUNK, len(joined)), ","),
                                   format(len(joined), ",")), flush=True)
    df = pd.concat(frames, ignore_index=True)
    df["Duration_Overlap_Sec"] = total_seconds_us(df["Exit_Time"] - df["Entry_Time"])

    to_wgs = Transformer.from_crs(model.METRIC_CRS, model.WGS84_CRS, always_xy=True)
    df["Entry_Longitude"], df["Entry_Latitude"] = to_wgs.transform(df["Entry_X"].values,
                                                                  df["Entry_Y"].values)
    df["Exit_Longitude"], df["Exit_Latitude"] = to_wgs.transform(df["Exit_X"].values,
                                                                df["Exit_Y"].values)
    df = df[["Trip_ID", "Solar_Farm_ID", "Solar_Farm_Power",
             "Entry_Time", "Entry_Longitude", "Entry_Latitude",
             "Exit_Time", "Exit_Longitude", "Exit_Latitude", "Duration_Overlap_Sec"]]

    out = model.overlap_file(a.altitude)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print("overlap events: %s (%d farms, %s flights)"
          % (format(len(df), ","), df["Solar_Farm_ID"].nunique(),
             format(df["Trip_ID"].nunique(), ",")))
    print("wrote %s  (%.0f s)" % (out, time.time() - started))


if __name__ == "__main__":
    main()
