# 2. Coverage and savings (Results 1.1–1.3, Methods 3.2–3.3)

Which flights pass within beaming range of which solar farms, how much power each farm delivers
to them, and what that saves in jet fuel and CO₂. Run once per cruise altitude; 12,100 m is the
reference, 9,100 and 15,100 m are the altitude cases of Fig. 5.

| Script | Output |
|---|---|
| `1_flight_od.py` | `data/flights/processed/flight_with_od_coor.csv`: the 148,814 analysed flights with airport coordinates, `Trip_ID` and `Distance_km` |
| `2_overlap_pairs.py [--altitude H]` | `results/baseline/flight_solar_overlap_pairs[_9100m\|_15100m]_R1.csv`: every crossing of a flight through a farm's beaming range, with entry and exit points and times |
| `3_system_level.py [--altitude H]` | `results/baseline/solar_farm_analysis_results[tag][_merged]_R1.csv` and `flight_analysis_results[tag][_merged]_R1.csv`: energy, supported duration, cost and CO₂ by farm and by flight |
| `model.py` | shared parameters, loaders and power model, imported by the scripts here and in `3_optimization`, `5_sensitivity_analysis` and `6_supplementary` |

```bash
cd code/2_coverage_and_savings
python 1_flight_od.py
for h in 12100 9100 15100; do
    python 2_overlap_pairs.py --altitude $h     # 7-10 min and 0.8-1.3 GB each
    python 3_system_level.py --altitude $h      # about 1 min each
done
```

Of these outputs, the dataset holds only the `_merged` tables of step 3; the scripts create the others.

## Model

- **Flights**: straight lines between the airports, flown at constant speed from wheels-off to
  landing, in the CONUS Albers projection (EPSG:5070).
- **Energy boundary**: each farm polygon buffered by the cruise altitude. A flight line that
  crosses the buffer gives one overlap event per crossing.
- **Qualified farms**: effective capacity `p_cap_safe = min(p_cap_dc, 20 W/m² × p_area)` of at
  least 27.7 MW; 445 farms qualify and 438 of them serve at least one flight at 12,100 m.
- **Power**: every crossing is sampled every 0.2 s on a time grid shared by all flights of the
  farm. At each sample the power at full farm capacity is
  `P = p_cap_safe × η_sys × A_r / (π λ z)` with `z = sqrt(d² + h²)` (Eq. 11), where
  η_sys = 0.4478, A_r = 261.6 m², λ = 0.05 m. The farm serves the flights in range nearest first,
  each up to its 10.3 MW cruise demand, until the capacity is used up.
- **Savings**: delivered energy costs $58/MWh and emits 48 kg CO₂/MWh; it displaces jet fuel
  worth $1,540/10.3 per MWh and 6,952/10.3 kg CO₂ per MWh (2,200 kg/h at $0.7/kg and
  3.16 kg CO₂/kg, for 10.3 MW). The merged tables keep two bases: `*_by_Energy` (fuel priced
  per delivered MWh), which the R1 results use, and `*_by_Duration` (fuel priced per hour of
  supported flight, first submission).

At 12,100 m the reference case delivers 36,637 MWh over 667,833 flight-minutes, a fuel cost
saving of $3,352,818 and a CO₂ reduction of 22,970 t ($5,282,266 in total at $84/t).

## Reproducibility

The crossing points come from GEOS geometry operations. Other GEOS or PROJ versions can move a
crossing time by a nanosecond, which occasionally moves one 0.2 s sample in or out of a
crossing. Against the R1 files this changes 7 of 86,084 flights by one sample at 12,100 m
(totals agree to 6 × 10⁻⁹); the 9,100 m results agree to 10⁻¹³.
