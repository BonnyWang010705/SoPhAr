# 3. Optimization (Results 1.4–1.5, Methods 3.4)

Two optimization problems on the overlap events of `code/2_coverage_and_savings`. Each can be
solved with the solver used for the paper or with Gurobi (optional).

| Script | Problem | Solvers (`--solver`) | Output |
|---|---|---|---|
| `1_schedule_optimization.py [--altitude H]` | flight schedule: shift each flight by up to ±1,800 s so that fewer flights compete for the same farm; all farms and flights take part | `adam` (default), `gurobi` | `results/optimization_1/optimized_flight_shifts[tag]_R1.csv`, `Optimization_1_solar_farm_analysis_results[tag]_merged_R1.csv`, `Optimization_1_flight_analysis_results[tag]_merged_R1.csv` |
| `2_merge_shifts.py` | collects the shifts of the three altitudes (Fig. 5d–f) | – | `results/optimization_1/optimized_shift_flight_merged_R1.csv` |
| `3_farm_flight_selection.py` | farm-and-flight choice: which farms and flights to equip at solar farm and flight penetration rates of 10–100% (100 scenarios), fixed schedules, 12,100 m | `greedy` (default), `gurobi` | `results/optimization_2/Optimization_2_Results_R1/`, `Optimization_2_summary_R1.csv` |
| `solvers.py` | the solvers, imported by the scripts | | |

```bash
cd code/3_optimization
for h in 12100 9100 15100; do python 1_schedule_optimization.py --altitude $h; done
python 2_merge_shifts.py
python 3_farm_flight_selection.py
```

## Solvers

**Flight schedule.** Crossings of the same farm that are no more than 3,600 s apart can be made to
overlap, so only those pairs enter the problem. The objective is the total time during which two
flights are inside the same farm's range together.

- `adam`: the shifts are continuous variables, updated by gradient descent (PyTorch Adam) on
  the total pairwise overlap and clipped to ±1,800 s after each step. The epochs and learning
  rate of each altitude are set in the script (9,100 m: 10,000 and 70; 12,100 m: 9,000 and
  160; 15,100 m: 10,000 and 180) and can be changed with `--epochs` and `--lr`. The paper's
  shifts were computed on a GPU; a CPU run (about 0.5 s per epoch at 12,100 m) or another GPU
  can give slightly different shifts.
- `gurobi`: the same objective as a mixed-integer linear program. The overlap of each pair is
  linearised with two binaries and a big-M, and the pairs are solved in consecutive time windows
  (`--window-hours`, default 2); shifts fixed in a window are kept in the later ones.
  Other options: `--time-limit` (per window), `--threads`.

**Farm-and-flight choice.** The objective is the capacity-weighted overlap time,
Σ `p_cap_safe` × overlap seconds, of the pairs whose farm and flight are both selected, with at
most ρ_F|F| farms and ρ_I|I| flights.

- `greedy`: multi-start alternating local search. Start from the farms and flights with the
  largest potential, then alternately pick the best flights for the chosen farms and the best
  farms for the chosen flights until the objective stops improving; keep the best of five
  starts. All 100 scenarios take about 1 h on a laptop.
- `gurobi`: the integer program with binary farm, flight and pair variables. Options:
  `--time-limit` (per scenario), `--mip-gap`, `--threads`, `--verbose`.
  `python 3_farm_flight_selection.py --compare` solves the 10%/10% scenario both ways and
  prints the gap between the two objectives (below 2%).

`--farm-rates` and `--flight-rates` run a subset of the scenarios; the results folder and the summary
keep the other scenarios of an earlier run. Both solvers write the same file names.

With `--solver gurobi` the scripts need `gurobipy` and a Gurobi licence
(`pip install gurobipy`; see gurobi.com for academic licences). The power model is rerun on the
selected farms and flights, or on the shifted schedules, in the same way whichever solver is
used.
