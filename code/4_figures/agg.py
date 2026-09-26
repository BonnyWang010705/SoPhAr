import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
HERE = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.join(ROOT, "results", "baseline")

# Main analysis prices carbon at the EU ETS level; the social cost of
# carbon is reported separately in the Discussion, not added to the total.
CARBON_PRICE = 0.084          # $/kg CO2  = $84/t, EU ETS 2026
SOCIAL_CARBON = 0.19          # $/kg CO2  = $190/t, U.S. EPA social cost

CLASS_EDGES = [0, 1500, 4000, np.inf]
CLASS_NAMES = ["short", "medium", "long"]

FLIGHT_FILE = {9100: "flight_analysis_results_9100m_merged_R1.csv",
               12100: "flight_analysis_results_merged_R1.csv",
               15100: "flight_analysis_results_15100m_merged_R1.csv"}
FARM_FILE = {9100: "solar_farm_analysis_results_9100m_merged_R1.csv",
             12100: "solar_farm_analysis_results_merged_R1.csv",
             15100: "solar_farm_analysis_results_15100m_merged_R1.csv"}

FUEL = "Fuel_Cost_Savings_by_Energy"
CO2 = "CO2_Emissions_Reduction_by_Energy"

def load_flights(altitude=12100):
    d = pd.read_csv(os.path.join(R1, FLIGHT_FILE[altitude]), low_memory=False)
    d["cls"] = pd.cut(d["Distance_km"], CLASS_EDGES, labels=CLASS_NAMES, right=False)
    if d["cls"].isna().any():
        raise ValueError("%d flights fall outside the distance classes"
                         % int(d["cls"].isna().sum()))
    d["carbon_usd"] = d[CO2] * CARBON_PRICE
    d["total_usd"] = d[FUEL] + d["carbon_usd"]
    return d

def load_farms(altitude=12100):
    d = pd.read_csv(os.path.join(R1, FARM_FILE[altitude]), low_memory=False)
    d["carbon_usd"] = d[CO2] * CARBON_PRICE
    d["total_usd"] = d[FUEL] + d["carbon_usd"]
    return d

def range_class_decomposition(flights):
    g = flights.groupby("cls", observed=True).agg(
        energy_mwh=("Total_Power_Received_MWh", "sum"),
        duration_min=("Duration_Supported_Min", "sum"),
        fuel_usd=(FUEL, "sum"),
        carbon_usd=("carbon_usd", "sum"),
        total_usd=("total_usd", "sum"),
        flights=("Trip_ID", "size")).reindex(CLASS_NAMES)
    share = g.div(g.sum(axis=0), axis=1) * 100
    return g, share

def state_by_class(flights, side="Origin"):
    key = "STATE_%s" % side
    g = flights.groupby([key, "cls"], observed=True).agg(
        co2_kg=(CO2, "sum"),
        fuel_usd=(FUEL, "sum"),
        total_usd=("total_usd", "sum"),
        distance_km=("Distance_km", "sum"),
        flights=("Trip_ID", "size")).reset_index().rename(columns={key: "state"})
    return g

def state_totals(flights, side="Origin"):
    key = "STATE_%s" % side
    t = (flights.pivot_table(index=key, columns="cls", values="total_usd",
                             aggfunc="sum", observed=True)
                .reindex(columns=CLASS_NAMES).fillna(0.0))
    t.index.name = "state"
    return t.loc[t.sum(axis=1).sort_values(ascending=False).index]

DAY_START_MIN = 6 * 60        # 06:00 on the reference clock
DAY_END_MIN = 18 * 60         # 18:00 on the reference clock
REFERENCE_CLOCK = "Wheels-off time (America/Chicago)"

def _cumulative_daytime(minutes):
    full_days = np.floor(minutes / 1440)
    within = np.clip((minutes % 1440) - DAY_START_MIN, 0,
                     DAY_END_MIN - DAY_START_MIN)
    return full_days * (DAY_END_MIN - DAY_START_MIN) + within

def daytime_share(flights):
    t = pd.to_datetime(flights[REFERENCE_CLOCK], utc=True).dt.tz_convert(
        "America/Chicago").dt.tz_localize(None)
    start = (t - t.dt.normalize()).dt.total_seconds() / 60
    dur = flights["Actual elapsed time (Minutes)"].astype(float)
    return (_cumulative_daytime(start + dur) - _cumulative_daytime(start)) / dur

def label_day_night(flights):
    f = flights.copy()
    f["window"] = np.where(daytime_share(f) >= 0.5, "day", "night")
    return f

def power_groups(flights, n_groups=3):
    total = (state_totals(flights, "Origin").sum(axis=1)
             .add(state_totals(flights, "Destination").sum(axis=1),
                  fill_value=0.0))
    order = list(total.sort_values(ascending=False).index)
    per = -(-len(order) // n_groups)
    return [order[i * per:(i + 1) * per] for i in range(n_groups)]

def farm_state_ranking(farms, n_top=10, other="Other states"):
    g = farms.groupby("p_state").agg(
        fuel_usd=(FUEL, "sum"),
        co2_kg=(CO2, "sum"),
        carbon_usd=("carbon_usd", "sum"),
        total_usd=("total_usd", "sum"),
        capacity_mw=("p_cap_safe", "sum"),
        farms=("Solar_Farm_ID", "size")).sort_values("total_usd", ascending=False)
    top, rest = g.head(n_top), g.iloc[n_top:]
    if len(rest):
        top = pd.concat([top, rest.sum().to_frame(other).T])
    top["fuel_pct"] = 100 * top.fuel_usd / g.fuel_usd.sum()
    top["co2_pct"] = 100 * top.co2_kg / g.co2_kg.sum()
    top["total_pct"] = 100 * top.total_usd / g.total_usd.sum()
    return top


def farm_counts_by_state(farms):
    """Number of solar farms represented in the farm-level results, by state."""
    return farms.groupby("p_state")["Solar_Farm_ID"].size()
