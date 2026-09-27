# 5. Sensitivity analysis (Supplementary Notes 6–10, Supplementary Figs. 32–37, Table 19)

How the reference case (12,100 m) responds to the carbon price, the power a flight can receive,
the receiving aperture, the transfer efficiency and the electricity cost. The altitude cases of
Fig. 5 are runs of `code/2_coverage_and_savings` and `code/3_optimization` with `--altitude`.

| Script | Output |
|---|---|
| `1_parameter_sweep.py [--force]` | `results/sensitivity/`: `physical_sensitivity_full.csv` (every evaluated value of the maximum received power, aperture and efficiency), `<analysis>_key_points.csv` for the five analyses, `baseline_by_farm.csv`, `manifest.json`, `validation.json` |
| `2_validate_sweep.py` | `results/sensitivity/engine_validation.json`: checks of the sweep method (below) |
| `3_carbon_price.py` | Table 19 (`results/sensitivity/carbon_price_table.csv`, `.tex`) and `carbon_price_sensitivity.png` |
| `4_electric_propulsion.py` | `electric_propulsion_sensitivity.png`: maximum power received per flight, 0–10.3 MW |
| `5_receiving_aperture.py` | `receiving_aperture_sensitivity.png`: receiving aperture, 0–261.6 m² |
| `6_transfer_efficiency.py` | `transfer_efficiency_sensitivity.png`: transfer efficiency, 0–1 |
| `7_cost_scenario_1.py` | `LCOE_sensitivity_s1.png`: daytime without storage ($38–78/MWh), nighttime with storage ($50–131/MWh) |
| `8_cost_scenario_2.py` | `LCOE_sensitivity_s2.png`: daytime and nighttime with storage ($50–131/MWh) |

Figures go to `figures/supplementary/fig32_37_sensitivity/` (PNG, PDF and SVG). Each figure
script has a settings block at the top for the figure's appearance.

```bash
cd code/5_sensitivity_analysis
python 1_parameter_sweep.py
python 2_validate_sweep.py
for s in 3_carbon_price 4_electric_propulsion 5_receiving_aperture 6_transfer_efficiency \
         7_cost_scenario_1 8_cost_scenario_2; do python $s.py; done
```

## Method

One parameter changes at a time; everything else stays at the reference values. Daytime is
06:00–18:00 America/Chicago for every farm (daylight-saving time applied), nighttime the rest.
The electricity cost scenarios change the price of the daytime and nighttime energy only.

The sweep generates the sample positions and distances once per farm, at the original 0.2 s
step. For a given instant, the nearest-first allocation is a piecewise-linear function of the
ratio r of per-flight power cap to the aperture × efficiency multiplier. With nearest-first
full-capacity powers b_j, H_j = Σ_(i≤j) 1/b_i and b_(n+1) = 0,
F(r) = n·r + Σ_j H_j (b_(j+1) − b_j) max(r − 1/H_j, 0), and the received power is
multiplier × F(cap / multiplier). All parameter values therefore come from one pass
(about 1 min).

Checks: the farm energies at the reference parameters match
`results/baseline/solar_farm_analysis_results_R1.csv` (`validation.json`), and
`2_validate_sweep.py` compares the method with the direct sample-by-sample allocation on two
farms at every key point, on synthetic contention cases, and across the daylight-saving
transitions.
