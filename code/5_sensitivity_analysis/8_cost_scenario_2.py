"""Electricity cost, scenario 2: daytime and nighttime with storage
($50-131/MWh each).

Draws one of Supplementary Figs. 32-37 from the saved sweep of
1_parameter_sweep.py (results/sensitivity/cost_scenario_2_key_points.csv) and writes
figures/supplementary/fig32_37_sensitivity/LCOE_sensitivity_s2.png (also .pdf, .svg).
The settings block below controls the appearance.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------ EDIT THESE SETTINGS ------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "results" / "sensitivity" / "cost_scenario_2_key_points.csv"
OUTPUT_DIR = ROOT / "figures" / "supplementary" / "fig32_37_sensitivity"
OUTPUT_NAME = "LCOE_sensitivity_s2"
SAVE_FORMATS = ["png", "pdf", "svg"]
DPI = 300

FIGSIZE = (12, 6)            # Width, height in inches
FONT_FAMILY = "DejaVu Sans"     # For example: "Arial", "Times New Roman"
FONT_SIZE = 10
TITLE_SIZE = 14
PANEL_TITLE_SIZE = 11

TITLE = "Scenario 2: daytime and nighttime with storage"
SUBTITLE = (
    "{n_cases} day/night combinations · daytime 06:00–18:00 America/Chicago · "
    "daylight saving time applied"
)
PANEL_TITLES = ["(a)", "(b)"]
DAY_COST_LABEL = "Daytime cost ($/MWh)"
NIGHT_COST_LABEL = "Nighttime cost ($/MWh)"
SAVING_LABEL = "Net cost saving ($M)"
SAVING_DIVISOR = 1e6            # Convert dollars to millions

# The heatmap spans these cost ranges. Each daytime value becomes one line
# in the right panel; nighttime values determine the markers along each line.
DAY_COSTS = [50, 70, 90, 110, 131]
NIGHT_COSTS = [50, 70, 90, 110, 131]

HEATMAP_COLORMAP = "YlGnBu"
HEATMAP_RESOLUTION = 180        # Grid samples along each cost axis
CONTOUR_LEVELS = 24
HEATMAP_VALUE_LIMITS = None     # For a common scale, e.g. (0.5, 4.0), in $M
SHOW_CONTOUR_LABELS = False
SHOW_KEY_POINTS = False
HEATMAP_MARKER_SIZE = 30        # Scatter marker area in points squared
HEATMAP_MARKER_FACE = "white"
HEATMAP_MARKER_EDGE = "#273A4A"

LINE_COLORMAP = "viridis"
LINE_COLORS = None              # Or supply one color per daytime value
LINE_WIDTH = 1.8
LINE_STYLE = "-"
LINE_MARKER = "o"               # Use "" to hide markers
LINE_MARKER_SIZE = 4.5          # Marker diameter in points
LINE_Y_LIMITS = None            # For example: (0, 4.2), in $M
GRID_ALPHA = 0.15
SHOW_LEGEND = True
LEGEND_POSITION = (1.02, 1.02)  # Relative to the right panel
LEGEND_FONT_SIZE = 8.6

# Reserve space on the right for the legend and below for the notes.
LAYOUT = dict(left=0.065, right=0.84, bottom=0.25, top=0.77, wspace=0.38)
PANEL_WIDTH_RATIOS = [1.17, 1]
SHOW_FOOTNOTES = False
MODEL_NOTE = (
    "Storage changes the specified LCOE only; no additional storage losses, "
    "capacity limits, or emissions are assumed.\n"
    "Results cover the observed flight sample and are not annualized."
)

# --------------------------------------------------------------------


def net_savings(day_cost, night_cost, reference):
    """Return savings in dollars, for scalar or NumPy-array cost inputs.

    Costs are per delivered MWh, following the original R1 model.
    Daytime/nighttime energy and displaced fuel cost stay fixed.
    """
    return (
        reference["displaced_fuel_cost_usd"]
        - reference["daytime_energy_mwh"] * day_cost
        - reference["nighttime_energy_mwh"] * night_cost
    )


def make_figure():
    if not DATA_FILE.exists():
        sys.exit("missing %s\nrun: python code/5_sensitivity_analysis/1_parameter_sweep.py" % DATA_FILE)
    """Return (figure, axes) for additional edits before saving.

    axes[0] is the heatmap; axes[1] is the line comparison.
    """
    reference = pd.read_csv(DATA_FILE).iloc[0]
    day_values = np.unique(np.asarray(DAY_COSTS, dtype=float))
    night_values = np.unique(np.asarray(NIGHT_COSTS, dtype=float))
    if len(day_values) < 2 or len(night_values) < 2:
        raise ValueError("The heatmap needs at least two distinct costs on each axis.")

    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#253447",
        "text.color": "#253447",
        "axes.edgecolor": "#98A3AE",
        "xtick.color": "#536170",
        "ytick.color": "#536170",
        "pdf.fonttype": 42,       # Embed TrueType text in PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # Preserve text in SVG for editing
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI,
                             gridspec_kw={"width_ratios": PANEL_WIDTH_RATIOS})
    fig.subplots_adjust(**LAYOUT)
    # fig.suptitle(TITLE, x=0.065, y=0.975, ha="left", fontsize=TITLE_SIZE, fontweight="bold")
    # fig.text(0.065, 0.89, SUBTITLE.format(n_cases=len(day_values)*len(night_values)),
    #          fontsize=FONT_SIZE, color="#617180")

    # Left panel: calculate the exact linear cost response over a dense grid.
    d, n = np.meshgrid(
        np.linspace(day_values.min(), day_values.max(), HEATMAP_RESOLUTION),
        np.linspace(night_values.min(), night_values.max(), HEATMAP_RESOLUTION),
    )
    saving = net_savings(d, n, reference) / SAVING_DIVISOR
    levels = (CONTOUR_LEVELS if HEATMAP_VALUE_LIMITS is None else
              np.linspace(*HEATMAP_VALUE_LIMITS, CONTOUR_LEVELS + 1))
    heat = axes[0].contourf(d, n, saving, levels=levels, cmap=HEATMAP_COLORMAP,
                            extend="neither" if HEATMAP_VALUE_LIMITS is None else "both")
    fig.colorbar(heat, ax=axes[0], label=SAVING_LABEL, fraction=0.045, pad=0.035)

    if SHOW_CONTOUR_LABELS:
        contours = axes[0].contour(d, n, saving, levels=6, colors="#536170", linewidths=0.6)
        axes[0].clabel(contours, inline=True, fontsize=8, fmt="%.2f")

    if SHOW_KEY_POINTS:
        key_d, key_n = np.meshgrid(day_values, night_values)
        axes[0].scatter(key_d.ravel(), key_n.ravel(), s=HEATMAP_MARKER_SIZE,
                        facecolor=HEATMAP_MARKER_FACE, edgecolor=HEATMAP_MARKER_EDGE,
                        clip_on=False, zorder=4)

    axes[0].set(xlabel=DAY_COST_LABEL, ylabel=NIGHT_COST_LABEL,
                xlim=(day_values.min(), day_values.max()),
                ylim=(night_values.min(), night_values.max()))
    axes[0].set_xticks(day_values, [f"{v:g}" for v in day_values])
    axes[0].set_yticks(night_values, [f"{v:g}" for v in night_values])

    # Right panel: one curve for each daytime cost, with nighttime-cost markers.
    colors = (plt.get_cmap(LINE_COLORMAP)(np.linspace(0.12, 0.82, len(day_values)))
              if LINE_COLORS is None else LINE_COLORS)
    if len(colors) != len(day_values):
        raise ValueError("LINE_COLORS must contain one color per distinct daytime cost.")
    for day_cost, color in zip(day_values, colors):
        y = net_savings(day_cost, night_values, reference) / SAVING_DIVISOR
        axes[1].plot(night_values, y, color=color, linewidth=LINE_WIDTH,
                      linestyle=LINE_STYLE, marker=LINE_MARKER, markersize=LINE_MARKER_SIZE,
                      label=f"Daytime ${day_cost:g}/MWh")

    padding = (night_values.max() - night_values.min()) * 0.025
    axes[1].set(xlabel=NIGHT_COST_LABEL, ylabel=SAVING_LABEL,
                xlim=(night_values.min()-padding, night_values.max()+padding))
    axes[1].set_xticks(night_values, [f"{v:g}" for v in night_values])
    if LINE_Y_LIMITS is not None:
        axes[1].set_ylim(*LINE_Y_LIMITS)
    axes[1].grid(axis="y", alpha=GRID_ALPHA)
    if SHOW_LEGEND:
        axes[1].legend(frameon=False, fontsize=LEGEND_FONT_SIZE,
                        loc="upper left", bbox_to_anchor=LEGEND_POSITION)

    for ax, title in zip(axes, PANEL_TITLES):
        ax.set_title(title, fontsize=PANEL_TITLE_SIZE, fontweight="bold", loc="left", pad=10)

    if SHOW_FOOTNOTES:
        energy_note = (
            f"Energy fixed: {reference['daytime_energy_mwh']/1000:.3f} GWh daytime + "
            f"{reference['nighttime_energy_mwh']/1000:.3f} GWh nighttime. "
            f"Net CO₂ reduction fixed: {reference['net_co2_reduction_t']/1000:.3f} kt."
        )
        fig.text(0.065, 0.115, energy_note, fontsize=FONT_SIZE-1)
        fig.text(0.065, 0.025, MODEL_NOTE, fontsize=FONT_SIZE-1)
    return fig, axes


def save_figure(fig):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in SAVE_FORMATS:
        path = OUTPUT_DIR / f"{OUTPUT_NAME}.{extension}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {path}")


if __name__ == "__main__":
    fig, axes = make_figure()

    # Optional edits after plotting. Examples:
    # axes[0].set_title("Cost sensitivity")
    # axes[1].set_ylim(0, 4.2)
    # axes[1].set_ylabel("Net savings ($ million)")

    save_figure(fig)
