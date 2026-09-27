# SoPhAr aviation: code

Code for *Solar phased-array-based wireless power transfer for commercial aviation decarbonization*
(T. Wang, Y. Xu, J. Byeon, J. Jiao, J. Mohammadi, K. Kockelman, C. Claudel and A. Bayen; under review at *Nature Sustainability*).

The release has two parts:

| Part | Where | Contents |
|---|---|---|
| Code | this repository | the scripts in `code/` |
| Data | Hugging Face dataset [`Solarphasedarray/SoPhAr`](https://huggingface.co/datasets/Solarphasedarray/SoPhAr) | `data/` (raw and processed inputs), `results/` (model outputs), `figures/` (paper figures) |

## Quick start

```bash
git clone https://github.com/BonnyWang010705/SoPhAr.git
cd SoPhAr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_data.py            # adds data/, results/ and figures/ to the repository root
python code/4_figures/fig4.py      # for example, redraw Fig. 4
```

Every script locates its files relative to the repository root. `download_data.py` places the dataset
there, so no paths need to be edited:

```
SoPhAr/
├── code/        this repository
├── data/        from the dataset
├── results/     from the dataset
└── figures/     from the dataset
```

Use `python download_data.py --include "results/*" "figures/*"` to skip the raw flight archives (about 0.5 GB).
The flight-farm overlap tables in `results/baseline` are large (0.8–1.3 GB each); they are needed only to
rerun the model scripts, not to redraw the figures.

## What each folder does

| Folder | Purpose | Reads | Writes |
|---|---|---|---|
| `code/1_flight_data` | Download the BTS Marketing Carrier On-Time Performance archives, build the 2025 flight table and the analysed week (7–13 April 2025), and match airports | `data/flights/raw`, `data/airports` | `data/flights/processed` |
| `code/2_coverage_and_savings` | Model: flight coordinates, flight-farm overlaps, and the power, energy, cost and CO₂ of beaming at each cruise altitude | `data/flights/processed`, `data/airports`, `data/solar_farms` | `data/flights/processed/flight_with_od_coor.csv`, `results/baseline` |
| `code/3_optimization` | Flight schedule optimization (Fig. 5) and farm-and-flight choice optimization (Fig. 6); Adam or greedy solver, or Gurobi | `results/baseline`, `data/solar_farms` | `results/optimization_1`, `results/optimization_2` |
| `code/4_figures` | Main Figs. 2–6, Figs. 7–8 of the arXiv version, Supplementary Fig. 2, and the route maps of Fig. 3b–d and Fig. 8a | `results/`, `data/US_boundaries` | `figures/main`, `figures/supplementary/fig02_state_rankings`, `results/*/route_maps` |
| `code/5_sensitivity_analysis` | Parameter sensitivity: Supplementary Figs. 32–37 and Table 19 | `results/baseline`, `data/solar_farms` | `results/sensitivity`, `figures/supplementary/fig32_37_sensitivity` |
| `code/6_supplementary` | Supplementary Figs. 3–31 and Supplementary Tables 1–15 | `results/`, `data/case_study` | `figures/supplementary/fig03_selection_frequency`, `fig04_23_penetration`, `fig24_29_case_studies`, `fig30_31_flight_case_study` |
| `code/7_representativeness` | Supplementary Note 1 and Supplementary Fig. 1 | `data/flights/processed`, `results/baseline` | `results/representativeness`, `figures/supplementary/fig01_representativeness` |

Some scripts also write files that the paper does not use: CSV copies of the flight tables
(`2_preprocess.py`, `3_build_annual.py`), seven further maps (`maps.py`) and a working figure
(`7_representativeness/1_weekly_flight_hours.py`). These are not in the dataset.

Main-figure scripts are named after the panels they draw (`fig2be.py` draws Fig. 2b and 2e), and
Supplementary scripts after the Supplementary figures or tables they produce (`supp_fig04_23_penetration.py`
produces Supplementary Figs. 4–23).

Each of the folders `2_coverage_and_savings`, `3_optimization`, `5_sensitivity_analysis` and
`7_representativeness` has its own README with the details of its scripts.

## Reproducing the model results

The dataset holds every result table, so the figures can be redrawn without this step. To
recompute the results from the inputs, run the model scripts in this order (times on a laptop):

```bash
cd code/2_coverage_and_savings
python 1_flight_od.py                                   # analysed flights with coordinates
for h in 12100 9100 15100; do
    python 2_overlap_pairs.py --altitude $h             # flight-farm overlaps, 7-10 min each
    python 3_system_level.py --altitude $h              # energy, cost, CO2; 1 min each
done

cd ../3_optimization
for h in 12100 9100 15100; do
    python 1_schedule_optimization.py --altitude $h     # Fig. 5, Supplementary Tables 8-14
done
python 2_merge_shifts.py                                # Fig. 5d-f
python 3_farm_flight_selection.py                       # Fig. 6, Supplementary Figs. 3-23 and Table 15

cd ../5_sensitivity_analysis
python 1_parameter_sweep.py                             # Supplementary Notes 6-10
```

The 12,100 m results feed Figs. 2–4 and Supplementary Tables 1–7; the three altitudes feed Fig. 5.

`1_schedule_optimization.py` runs Adam in PyTorch, on a GPU when one is available; on a CPU it
takes about 1.5 h per altitude. Both optimization scripts take `--solver gurobi` to solve the
problems as mixed-integer programs with Gurobi instead (needs `gurobipy` and a Gurobi licence);
see `code/3_optimization/README.md`.

`model.py` in `code/2_coverage_and_savings` holds the system parameters (Supplementary
Tables 16–17). At 12,100 m the reference case gives 36,637 MWh of delivered energy, a fuel cost
saving of $3,352,818 and a CO₂ reduction of 22,970 t (Results 1.1–1.2).

## Reproducing the figures and tables

### Main figures

```bash
cd code/4_figures
python maps.py                          # maps for Fig. 2a, c and d
python fig2be.py                        # Fig. 2b and 2e
python fig3ae.py                        # Fig. 3a and 3e
python fig4.py                          # Fig. 4
python fig5abc.py && python fig5def.py  # Fig. 5a-c and 5d-f
python fig6.py                          # Fig. 6, assembled by the script itself
python fig7.py && python fig8.py        # Figs. 7-8 (arXiv version), panels a and b
python assemble.py                      # Figs. 2-5, 7 and 8 from their panels
```

Fig. 1 is a schematic and is not produced by code. Fig. 3b–d are route maps read from
`results/baseline/route_maps`; `python route_maps_fig3bd.py` draws them again (the basemap tiles
are downloaded, so it needs internet access).

Each main figure has its own folder in `figures/main` (`fig02_solar_farm_analysis/` to
`fig08_flight_selection/`), holding the assembled figure and its `panels/`. Each panel file stands on
its own with its legend. The assembled Fig. 5 shows one legend for panels a–c: `assemble.py` draws
panels a–c again without their legend, with the code of `fig5abc.py`, and places the legend above them.

Figs. 7–8 are the market-penetration figures of the arXiv version, redrawn with the R1 results for
the scenarios of 20, 50 and 80% penetration. The article does not include them; Supplementary
Figs. 3–23 give the same content for all 100 scenarios. Fig. 8a places the route maps read from
`results/optimization_2/route_maps`, which `python route_maps_fig8.py` draws again from the
farm-and-flight choice results (internet access needed for the basemap).

### Supplementary figures

| Supplementary figure | Produced by | Output folder in `figures/supplementary` |
|---|---|---|
| Fig. 1 | `code/7_representativeness/3_supp_fig01_representativeness.py` | `fig01_representativeness` |
| Fig. 2 | `code/4_figures/supp_fig02_state_rankings.py` | `fig02_state_rankings` |
| Fig. 3 | `code/6_supplementary/supp_fig03_selection_frequency.py` | `fig03_selection_frequency` |
| Figs. 4–23 | `code/6_supplementary/supp_fig14_23_state_totals.py`, then `supp_fig04_23_penetration.py` | `fig04_23_penetration` |
| Figs. 24–29 | `code/6_supplementary/supp_fig24_29_farm_case_studies.py` | `fig24_29_case_studies` |
| Figs. 30–31 | `code/6_supplementary/supp_fig30_31_flight_case_study.py` | `fig30_31_flight_case_study` |
| Figs. 32–37 | `code/5_sensitivity_analysis/3_carbon_price.py` to `8_cost_scenario_2.py` | `fig32_37_sensitivity` |

Figs. 4–13 show the selected farms at each solar farm penetration and Figs. 14–23 the selected
flights at each flight penetration. Each of these figures has two images: `_maps` holds rows a–f
with their panels k–p, and `_scatter` or `_bars` holds rows g–j with q–t.
`supp_fig14_23_state_totals.py` reads the 100 scenarios once and writes the state totals that the
bar panels of Figs. 14–23 use; the route maps of those figures are read from
`results/optimization_2/route_maps`.

`supp_fig03_selection_frequency.py` draws Supplementary Fig. 3 with the function of
`code/4_figures/fig7.py` that draws Fig. 7b, on a taller canvas.

Figs. 24–29 are the case studies of two farms, FID 230 (Desert Sunlight 300, CA) and FID 4746
(TX): hourly energy delivered, location and beaming range, and the routes that cross the range.
Figs. 30–31 follow flight AA 1459 from Austin to Los Angeles on 9 April 2025. This case study
was drawn from the flight data of the first submission, in which the flight is Trip_ID 22663;
its record and its farm crossings are in `data/case_study`. The case-study maps download
OpenStreetMap tiles (internet access needed; `--no-maps` skips them).

Figs. 32–37 show the carbon price, maximum received power per flight, receiving aperture,
transfer efficiency and the two electricity cost scenarios. They are drawn from the sweep of
`code/5_sensitivity_analysis/1_parameter_sweep.py` (`results/sensitivity`).

### Supplementary tables

`code/6_supplementary/supp_table01_15.py` computes the numbers of Supplementary Tables 1–15 and writes
them into the Supplementary Word document, which is not part of this release. Its functions give the
same numbers directly (run from the repository root):

```python
import sys; sys.path.insert(0, "code/6_supplementary")
import supp_table01_15 as t
t.farm_table("results/baseline/" + t.FARMS)        # Table 1 (Table 8: results/optimization_1/ + t.OPT_FARMS)
t.flight_tables("results/baseline/" + t.FLIGHTS)   # Tables 2-7, keyed by range class and origin/destination
                                                   # (Tables 9-14: results/optimization_1/ + t.OPT_FLIGHTS)
```

Columns `co2`, `fuel`, `third` and `total` are the table columns CO₂ reduction, fuel cost saving,
total capacity (farms) or total flight distance (flights), and total cost saving. Table 15 is
`results/optimization_2/Optimization_2_summary_R1.csv`, with the total cost saving
`Money_Cost_Saving + 0.084 × CO2_Emissions_Reduction`. The tables print ten significant digits
(`t.fmt`). Tables 16–18 list parameters and notation (the values are set in
`code/2_coverage_and_savings/model.py`). Table 19 (carbon price) follows from two totals of
`results/baseline/flight_analysis_results_merged_R1.csv`: the net saving ($3.35M, the sum of
`Fuel_Cost_Savings_by_Energy`) and the net CO₂ reduction (22.97 kt, the sum of
`CO2_Emissions_Reduction_by_Energy`); the carbon value is the CO₂ reduction times the carbon price.
`code/5_sensitivity_analysis/3_carbon_price.py` writes it to `results/sensitivity/carbon_price_table.csv`.

### Representativeness of the analysed week (Supplementary Note 1)

```bash
cd code/7_representativeness
python 1_weekly_flight_hours.py        # weekly flight volume and daytime share, 2025
python 2_weekly_energy_estimate.py     # flight mix against the annual mix, and weekly energy
python 3_supp_fig01_representativeness.py
```

The flight basis is set in `airport_match.py`.

### Rebuilding the flight data (optional)

The processed tables are in the dataset. To rebuild them from the BTS archives:

```bash
cd code/1_flight_data
python 1_download.py          # monthly archives, December 2024 – January 2026
python 2_preprocess.py
python 3_build_annual.py      # 7,599,787 records in 2025
python 4_reference_week.py    # 149,739 records, 7–13 April 2025 (Central Time)
python 5_match_airports.py    # 148,814 flights analysed
```

`code/2_coverage_and_savings/1_flight_od.py` then attaches the airport coordinates to the analysed
flights and numbers them (`Trip_ID`), as the model scripts need.

## Data

Inputs the model scripts read from the dataset:

| Path | Contents |
|---|---|
| `data/flights/processed/flights_2025.parquet` | 2025 flight table (`code/1_flight_data`) |
| `data/flights/processed/flight_with_od_coor.csv` | analysed flights with airport coordinates (`1_flight_od.py`) |
| `data/airports/airport_data.geojson` | USA Airports (Esri, ArcGIS) |
| `data/solar_farms/uspvdb_v3_0_20250430.geojson` | U.S. Large-Scale Solar Photovoltaic Database v3.0 (LBNL/USGS) |
| `data/case_study/flight_22663_*.csv` | record and farm crossings of the flight of Supplementary Figs. 30–31 |
| `data/US_boundaries/` | Census cartographic boundary files for the maps |

## Requirements

Python 3.9 or newer. The figures were checked with Python 3.12, pandas 2.3.3, numpy 2.0.2,
matplotlib 3.9.4 and geopandas 1.0.1. Text anti-aliasing can differ by a pixel between font and
FreeType versions; the plotted data do not.

The model scripts also need numba, pyproj and PyTorch, and the route maps and case studies need
contextily and seaborn (all in `requirements.txt`). They were checked with Python 3.11, pandas
2.3.3 and numpy 2.0.2. Gurobi (`gurobipy`) is optional and only used with `--solver gurobi`.

## License

The code is released under the MIT License (see `LICENSE`).

## Contact

Christian Claudel (christian.claudel@utexas.edu)
