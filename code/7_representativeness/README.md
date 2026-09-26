# 7. Representativeness of the analysed week (Supplementary Note 1)

How representative the week of 7–13 April 2025 is. In the model, farm capacity, aircraft and prices
are the same in every week, so a week differs only in its flights. The analysed week is compared with
the 51 complete weeks of 2025 in its flight inputs and in an estimate of the energy it delivers.

| Script | Output |
|---|---|
| `1_weekly_flight_hours.py` | weekly elapsed time and daytime share, 2025 |
| `2_weekly_energy_estimate.py` | route mix against the annual mix (total variation distance), and estimated weekly energy |
| `3_supp_fig01_representativeness.py` | Supplementary Fig. 1, from the tables of scripts 1 and 2 |

Tables go to `results/representativeness/` and Supplementary Fig. 1 to
`figures/supplementary/fig01_representativeness/`. Script 1 also draws a working figure in
`figures/drafts/representativeness/`, which is not in the paper.

The scripts read the flights the model reads: records whose origin and destination both appear as an
`FAA_ID` in the airport file (148,814 in the analysed week). Setting `MATCHED = False` in
`airport_match.py` uses all collected records instead (149,739).

Helpers: `airport_match.py` (airport match and basis switch), `daynight.py` (Central-time weeks and
the daytime/nighttime split), `natstyle.py` (style of the working figure) and `style.py` (style of
Supplementary Fig. 1).
