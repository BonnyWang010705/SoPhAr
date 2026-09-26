import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd

import agg
import style as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
HERE = os.path.dirname(os.path.abspath(__file__))
BOUNDARIES = os.path.join(ROOT, "data", "US_boundaries")
OUT = os.path.join(ROOT, "figures", "main", "fig02_solar_farm_analysis", "panels")

ALBERS = "EPSG:5070"
NON_CONUS = {"AK", "HI", "PR", "VI", "GU", "MP", "AS"}

FIGSIZE = (14.245, 8.04)
EDGE_COLOR = "grey"
STATE_LW = 0.5
COUNTY_LW = 0.10
MAP_BOX = [0.020, 0.066, 0.790, 0.872]      # left, bottom, width, height
# shared with panel b so the two right-hand colour bars match
BAR_BOX = [0.832, 0.150, 0.0263, 0.700]
# maps sit two to a row, so each occupies half the printed figure width
HALF = (3600 - 60) / 2 / 3600
CBAR_LABEL_PT = st.pt(7.0, FIGSIZE[0], HALF)
CBAR_TICK_PT = st.pt(6.0, FIGSIZE[0], HALF)
EDGE = EDGE_COLOR

def load_boundaries():
    states = gpd.read_file(os.path.join(BOUNDARIES, "cb_2023_us_state_20m.shp"))
    counties = gpd.read_file(os.path.join(BOUNDARIES, "cb_2023_us_county_500k.shp"))
    states = states[~states.STUSPS.isin(NON_CONUS)].to_crs(ALBERS)
    counties = counties[counties.STATEFP.isin(set(states.STATEFP))]
    return states, counties.to_crs(ALBERS)

def draw(gdf, column, label, vmax, states=None, cmap=st.CMAP_WARM):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(MAP_BOX)
    gdf.plot(ax=ax, column=column, cmap=cmap, vmin=0, vmax=vmax,
             edgecolor=EDGE, linewidth=COUNTY_LW if states is not None
             else STATE_LW)
    if states is not None:
        states.boundary.plot(ax=ax, edgecolor=EDGE, linewidth=STATE_LW)
    ax.margins(0)
    ax.axis("off")

    cax = fig.add_axes(BAR_BOX)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, rotation=90, labelpad=10, fontsize=CBAR_LABEL_PT)
    cb.ax.tick_params(labelsize=CBAR_TICK_PT)
    return fig

# column, colour-bar label, colormap, state cap, county cap, file name
# (state capacity, county energy and county duration are Fig. 2a, c, d)
QUANTITIES = (
    ("p_cap_safe", "Power Capacity (MW)", st.CMAP_WARM, 2000, 200, "capacity"),
    ("Total_Energy_Supplied_MWh", "Energy (MWh)", st.CMAP_WARM, 2000, 200,
     "energy"),
    ("Total_Flight_Duration_Supported_Hours", "Duration (Hours)", st.CMAP_WARM,
     800, 80, "duration"),
    ("CO2_Emissions_Reduction_by_Energy", "CO2 Reduction (kg)", st.CMAP_COOL,
     3e6, 1e5, "co2"),
    ("Fuel_Cost_Savings_by_Energy", "Cost Savings ($)", st.CMAP_COOL,
     3e5, 2.5e4, "fuel"),
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--altitude", type=int, default=12100)
    a = p.parse_args()

    st.apply()
    farms = agg.load_farms(a.altitude)
    states, counties = load_boundaries()

    key = farms.p_state.str.upper() + "|" + farms.p_county.str.strip().str.lower()
    counties = counties.copy()
    counties["key"] = (counties.STUSPS.str.upper() + "|"
                       + counties.NAME.str.strip().str.lower())
    missing = sorted(set(key[~key.isin(set(counties.key))]))
    if missing:
        raise ValueError("counties with no boundary match: %s" % missing)
    duplicated = set(counties.key[counties.key.duplicated(keep=False)])
    ambiguous = sorted(set(key) & duplicated)
    if ambiguous:
        raise ValueError("county name matches more than one boundary, join on "
                         "GEOID instead: %s" % ambiguous)
    farms = farms.assign(key=key)

    for col, label, cmap, cap_state, cap_county, short in QUANTITIES:
        for level, cap in (("state", cap_state), ("county", cap_county)):
            if level == "state":
                by = farms.groupby("p_state")[col].sum()
                g = states.copy()
                g["value"] = g.STUSPS.map(by).fillna(0.0)
                fig = draw(g, "value", label, cap, cmap=cmap)
            else:
                by = farms.groupby("key")[col].sum()
                g = counties.copy()
                g["value"] = g.key.map(by).fillna(0.0)
                fig = draw(g, "value", label, cap, states=states, cmap=cmap)
            path = os.path.join(OUT, "map_%s_%s" % (short, level))
            for f in st.save(fig, path):
                print("wrote", f)
            print("  %-22s %-6s %d of %d carry value, max %.4g, capped at %g"
                  % (label, level, (g.value > 0).sum(), len(g), by.max(), cap))

if __name__ == "__main__":
    main()
