"""Attach airport coordinates to the analysed week and number the flights.

Reads the 2025 flight table (code/1_flight_data) and the airport file, and
writes data/flights/processed/flight_with_od_coor.csv, the flight table every
later step reads:

  - the flights of 7-13 April 2025 (Central Time), selected as in
    code/1_flight_data/4_reference_week.py; the column `index` keeps each
    flight's row in flights_2025.parquet;
  - origin and destination state and coordinates, matched on the FAA
    identifier (as in 5_match_airports.py: flights whose origin or
    destination has no match are dropped, 149,739 -> 148,814 flights);
  - Trip_ID, the flight's row in the reference week before the airport match,
    which is the key of every result table;
  - Distance_km, the length of the straight origin-destination line in the
    CONUS Albers projection (EPSG:5070), used for the range classes.
"""
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

import model

REFERENCE_TZ = "America/Chicago"
WEEK_START = pd.Timestamp("2025-04-07 00:00:00", tz=REFERENCE_TZ)
WEEK_END = pd.Timestamp("2025-04-13 23:59:59", tz=REFERENCE_TZ)


def reference_week():
    annual = pd.read_parquet(model.FLIGHTS_2025)
    chicago = (pd.to_datetime(annual["Wheels-off time (UTC)"], utc=True)
                 .dt.tz_convert(REFERENCE_TZ))
    annual["Wheels-off time (America/Chicago)"] = chicago
    week = annual[(chicago >= WEEK_START) & (chicago <= WEEK_END)]
    # `index` = row in flights_2025.parquet; the new RangeIndex = row in the week
    return week.reset_index()


def airports():
    airport_data = gpd.read_file(model.AIRPORTS)
    airport_data["Longitude"] = airport_data.geometry.x
    airport_data["Latitude"] = airport_data.geometry.y
    return airport_data[["FAA_ID", "STATE", "Longitude", "Latitude"]]


def main():
    if not os.path.exists(model.FLIGHTS_2025):
        raise SystemExit("missing %s\nrun the scripts in code/1_flight_data, or "
                         "python download_data.py" % model.FLIGHTS_2025)
    flights = reference_week()
    ports = airports()

    merged = pd.merge(flights, ports, left_on="Origin Airport", right_on="FAA_ID",
                      how="left").drop(columns=["FAA_ID"])
    merged = pd.merge(merged, ports, left_on="Destination Airport", right_on="FAA_ID",
                      how="left", suffixes=("_Origin", "_Destination")).drop(columns=["FAA_ID"])
    missing_o = set(merged.loc[merged["Latitude_Origin"].isna(), "Origin Airport"])
    missing_d = set(merged.loc[merged["Latitude_Destination"].isna(), "Destination Airport"])

    flight_od = merged.dropna(subset=["Latitude_Origin", "Latitude_Destination"]).copy()
    flight_od["Trip_ID"] = flight_od.index.astype(int)

    lines = gpd.GeoSeries(
        [LineString([(lon_o, lat_o), (lon_d, lat_d)]) for lon_o, lat_o, lon_d, lat_d in
         zip(flight_od["Longitude_Origin"], flight_od["Latitude_Origin"],
             flight_od["Longitude_Destination"], flight_od["Latitude_Destination"])],
        crs=model.WGS84_CRS)
    flight_od["Distance_km"] = (lines.to_crs(model.METRIC_CRS).length / 1000).values

    os.makedirs(os.path.dirname(model.FLIGHT_OD), exist_ok=True)
    flight_od.to_csv(model.FLIGHT_OD, index=False)

    print("reference week      : %s flights" % format(len(flights), ","))
    print("unmatched origins   : %s" % ", ".join(sorted(missing_o)))
    print("unmatched dest.     : %s" % ", ".join(sorted(missing_d)))
    print("analysed flights    : %s" % format(len(flight_od), ","))
    print("wrote %s" % model.FLIGHT_OD)


if __name__ == "__main__":
    main()
