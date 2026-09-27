"""End-to-end wireless power transfer efficiency, 0-1; the range above the
0.76 theoretical maximum is drawn separately.

Draws one of Supplementary Figs. 32-37 from the saved sweep of
1_parameter_sweep.py (results/sensitivity/physical_sensitivity_full.csv) and writes
figures/supplementary/fig32_37_sensitivity/transfer_efficiency_sensitivity.png (also .pdf, .svg).
The settings block below controls the appearance.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ------------------------ EDIT THESE SETTINGS ------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "results" / "sensitivity" / "physical_sensitivity_full.csv"
OUTPUT_DIR = ROOT / "figures" / "supplementary" / "fig32_37_sensitivity"
OUTPUT_NAME = "transfer_efficiency_sensitivity"
SAVE_FORMATS = ["png", "pdf", "svg"]
DPI = 300

FIGSIZE = (10, 5)             # Width, height in inches
FONT_FAMILY = "DejaVu Sans"      # For example: "Arial", "Times New Roman"
FONT_SIZE = 10
TITLE_SIZE = 15
LINE_WIDTH = 2.0
KEY_POINT_SIZE = 26              # Scatter marker area, in points squared
BASELINE_SIZE = 95
GRID_ALPHA = 0.15

TITLE = "Wireless power transfer efficiency sensitivity"
SUBTITLE = "Observed flight sample · 438 farms · other model inputs held fixed"
FOOTNOTE = (
    "Reference labels (user-specified): 0.45 ours (rounded); "
    "0.54 experimental maximum; 0.76 theoretical maximum.\n"
    "Coded baseline = 0.4478. Electricity cost = $58/MWh; "
    "beaming emissions = 48 kg CO₂/MWh. Totals are not annualized."
)
X_LABEL = "Wireless power transfer efficiency"
X_LIMITS = (0, 1)
X_TICKS = [0, 0.15, 0.30, 0.45, 0.54, 0.60, 0.76, 1.00]
X_TICK_ROTATION = 45             # Prevent nearby tick labels from overlapping
DOTTED_FROM = 0.76               # This boundary must exist in the CSV grid
SHOW_KEY_POINTS = True
SHOW_BASELINE = False
SHOW_LEGEND = False

# Each panel: CSV column, unit divisor, y-axis label, line color, y-axis limits.
# Set limits to None for automatic limits, or use a tuple such as (0, 80).
# Remove or reorder panels here to change the figure composition.
PANELS = [
    ("energy_mwh", 1000, "Energy delivered (GWh)", "#236D9D", None),
    ("net_cost_saving_usd", 1e6, "Net cost saving ($M)", "#CC7042", None),
    ("net_co2_reduction_t", 1000, "Net CO₂ reduction (kt)", "#25836A", None),
]

# Subplot positions and spacing, expressed as fractions of figure size.
LAYOUT = dict(left=0.06, right=0.985, bottom=0.30, top=0.78, wspace=0.32)
LEGEND_POSITION = (0.055, 0.075)
FOOTNOTE_POSITION = (0.06, 0.015)

# --------------------------------------------------------------------


def make_figure():
    if not DATA_FILE.exists():
        sys.exit("missing %s\nrun: python code/5_sensitivity_analysis/1_parameter_sweep.py" % DATA_FILE)
    """Return (figure, axes) so you can make additional edits before saving."""
    full = pd.read_csv(DATA_FILE)
    data = full.loc[full["analysis"] == "transfer_efficiency"].sort_values("value")
    keys = data.loc[data["is_key_point"]]
    # Use the exact coded baseline (0.4478), distinct from the 0.45 key point.
    baseline = data.loc[data["is_baseline"]].iloc[0]
    x = data["value"].to_numpy()
    if DOTTED_FROM not in x:
        raise ValueError("Choose DOTTED_FROM from the saved 'value' grid.")

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

    fig, axes_grid = plt.subplots(1, len(PANELS), figsize=FIGSIZE, dpi=DPI, squeeze=False)
    axes = axes_grid[0]
    fig.subplots_adjust(**LAYOUT)
    # fig.suptitle(TITLE, x=0.06, y=0.98, ha="left", fontsize=TITLE_SIZE, fontweight="bold")
    # fig.text(0.06, 0.90, SUBTITLE, fontsize=FONT_SIZE, color="#617180")

    for ax, (column, divisor, ylabel, color, ylim) in zip(axes, PANELS):
        y = data[column].to_numpy() / divisor

        # Include the boundary in both segments so they join at 0.76.
        solid = x <= DOTTED_FROM
        dotted = x >= DOTTED_FROM
        ax.plot(x[solid], y[solid], color=color, linewidth=LINE_WIDTH, linestyle="-")
        ax.plot(x[dotted], y[dotted], color=color, linewidth=LINE_WIDTH, linestyle=":")

        if SHOW_KEY_POINTS:
            ax.scatter(keys["value"], keys[column] / divisor,
                       color=color, s=KEY_POINT_SIZE, zorder=4)

        if SHOW_BASELINE:
            baseline_y = baseline[column] / divisor
            ax.axhline(baseline_y, color="#ABB5BC", linewidth=0.9,
                       linestyle="--", zorder=0)
            ax.scatter(baseline["value"], baseline_y, marker="*", color="#1D2939",
                       s=BASELINE_SIZE, zorder=5, clip_on=False)

        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(ylabel)
        ax.set_xlim(*X_LIMITS)
        ax.set_ylim(*(ylim if ylim is not None else (0, y.max() * 1.12)))
        ax.set_xticks(X_TICKS, [f"{v:g}" for v in X_TICKS])
        ax.tick_params(axis="both", labelsize=FONT_SIZE - 1)
        ax.tick_params(axis="x", labelrotation=X_TICK_ROTATION)
        ax.grid(axis="y", alpha=GRID_ALPHA)

    if SHOW_LEGEND:
        handles = []
        if SHOW_KEY_POINTS:
            handles.append(Line2D([], [], marker="o", color="#536170", linewidth=0,
                                  label="Requested key points"))
        if SHOW_BASELINE:
            handles.append(Line2D([], [], marker="*", color="#1D2939", linewidth=0,
                                  markersize=9, label=f"Coded baseline ({baseline['value']:g})"))
        handles.append(Line2D([], [], color="#536170", linestyle=":", linewidth=LINE_WIDTH,
                              label=f"Dotted range: {DOTTED_FROM:.2f}–{x.max():.2f}"))
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=LEGEND_POSITION,
                   ncol=len(handles), frameon=False, fontsize=FONT_SIZE - 1)

    # fig.text(*FOOTNOTE_POSITION, FOOTNOTE, fontsize=FONT_SIZE - 1.4)
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
    # axes[0].set_ylim(0, 80)
    # axes[1].set_title("Economic benefit")
    # axes[2].set_ylabel("Net CO₂ reduction (kt)")

    save_figure(fig)
