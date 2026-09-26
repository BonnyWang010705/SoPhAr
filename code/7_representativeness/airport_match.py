"""The airport match, shared by the scripts in this folder.

Flight records name airports by IATA code; the airport file keys them by FAA
identifier, and the coverage and optimization runs compare the two directly. A
flight enters the analysis only when both of its codes appear as an FAA_ID.
With MATCHED = True every script here reads the same flights the model reads,
148,814 in the reference week; with MATCHED = False it reads every collected
record, 149,739.
"""
import json
import os

MATCHED = True

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
AIRPORTS = os.path.join(ROOT, "data", "airports", "airport_data.geojson")

ORIGIN = "Origin Airport"
DESTINATION = "Destination Airport"
COLUMNS = [ORIGIN, DESTINATION]


def faa_ids():
    with open(AIRPORTS) as f:
        return {ft["properties"]["FAA_ID"] for ft in json.load(f)["features"]}


def apply(df):
    """Keep the flights whose two airports are in the airport file."""
    if not MATCHED:
        return df
    faa = faa_ids()
    keep = df[ORIGIN].isin(faa) & df[DESTINATION].isin(faa)
    return df[keep].copy()


def label():
    return "matched" if MATCHED else "collected"
