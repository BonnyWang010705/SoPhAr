"""Carbon price sensitivity: Supplementary Table 19 and one of Supplementary
Figs. 32-37.

The net fuel cost saving (private cash flow) and the net CO2 reduction of the
reference case are fixed; the carbon value is the CO2 reduction times the
carbon price, $0-200/t. Both totals are read from
results/baseline/flight_analysis_results_merged_R1.csv (3_system_level.py):
$3.352818M and 22,970 t, rounded as in the manuscript.

Writes results/sensitivity/carbon_price_table.csv and .tex (Table 19) and
figures/supplementary/fig32_37_sensitivity/carbon_price_sensitivity.png
(also .pdf, .svg).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "results" / "baseline" / "flight_analysis_results_merged_R1.csv"
TABLE_DIR = ROOT / "results" / "sensitivity"
TABLE_PRICES = [0, 25, 84, 100, 150, 190, 200]   # $/t CO2, rows of Table 19


def baseline_totals():
    """Net fuel cost saving ($M, to the dollar) and net CO2 reduction (t)."""
    if not BASELINE.exists():
        sys.exit("missing %s\nrun: python code/2_coverage_and_savings/3_system_level.py"
                 % BASELINE)
    d = pd.read_csv(BASELINE, usecols=["Fuel_Cost_Savings_by_Energy",
                                       "CO2_Emissions_Reduction_by_Energy"])
    return (round(d["Fuel_Cost_Savings_by_Energy"].sum() / 1e6, 6),
            float(round(d["CO2_Emissions_Reduction_by_Energy"].sum() / 1000)))

PRIVATE_M, CO2_NET_T = baseline_totals()
KEY_PRICES = [0, 25, 84, 100, 150, 190, 200]  # USD per tonne CO2

OUTPUT_DIR = ROOT / "figures" / "supplementary" / "fig32_37_sensitivity"
OUTPUT_NAME = "carbon_price_sensitivity"
SAVE_FORMATS = ["png", "pdf", "svg"]
SAVE_FIGURE = True
DPI = 300

FIGSIZE = (10.6, 4.8)     # Width, height in inches
FONT_FAMILY = "DejaVu Sans"  # For example: "Arial", "Times New Roman"
FONT_SIZE = 10
LINE_WIDTH = 2.0
KEY_POINT_SIZE = 28      # Scatter marker area, in points squared
GRID_ALPHA = 0.18

PRIVATE_COLOR = "#236D9D"
CARBON_COLOR = "#CC7042"
TOTAL_COLOR = "#25836A"
PRIVATE_STYLE = "--"
CARBON_STYLE = "-."
TOTAL_STYLE = "-"

X_LABEL = r"Carbon price (\$/t CO$_2$)"
X_LIMITS = (0, 200)
X_TICKS = KEY_PRICES
X_TICK_ROTATION = 60
VALUE_Y_LABEL = "Value ($M)"
SHARE_Y_LABEL = "Share of total value (%)"
VALUE_Y_LIMITS = (0, 9)   # Set to None for automatic limits
SHARE_Y_LIMITS = (0, 105)

PANEL_TITLES = ["(a) Monetary value", "(b) Share of total value"]
SHOW_KEY_POINTS = True
SHOW_LEGEND = True
SHOW_TITLE = False
TITLE = "Carbon price sensitivity"

# Margins and spacing are fractions of the figure size.
LAYOUT = dict(left=0.075, right=0.98, bottom=0.21, top=0.81, wspace=0.30)
LEGEND_POSITION = (0.5, 0.98)

# --------------------------------------------------------------------


def calculate_results(prices):
    """Return unrounded results with the same column names as the notebook."""
    prices = np.asarray(prices, dtype=float)
    carbon_m = CO2_NET_T * prices / 1e6
    total_m = PRIVATE_M + carbon_m
    return pd.DataFrame({
        "Carbon price ($/t)": prices,
        "Private cash flow ($M)": np.full_like(prices, PRIVATE_M),
        "Private (%)": 100 * PRIVATE_M / total_m,
        "Carbon value ($M)": carbon_m,
        "Carbon (%)": 100 * carbon_m / total_m,
        "Total ($M)": total_m,
    })


def make_figure():
    """Return (figure, axes) for further editing before saving or displaying."""
    # Compute shares directly on a dense grid: percentage curves are nonlinear.
    curve = calculate_results(np.linspace(*X_LIMITS, 501))
    keys = calculate_results(KEY_PRICES)

    style = {
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#98A3AE",
        "axes.labelcolor": "#253447",
        "text.color": "#253447",
        "xtick.color": "#536170",
        "ytick.color": "#536170",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # Keep SVG text editable
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
        fig.subplots_adjust(**LAYOUT)

        def draw(ax, column, color, linestyle, label):
            ax.plot(curve["Carbon price ($/t)"], curve[column],
                    color=color, linestyle=linestyle, linewidth=LINE_WIDTH,
                    label=label)
            if SHOW_KEY_POINTS:
                ax.scatter(keys["Carbon price ($/t)"], keys[column],
                           color=color, s=KEY_POINT_SIZE, zorder=4,
                           clip_on=False)

        draw(axes[0], "Private cash flow ($M)", PRIVATE_COLOR,
             PRIVATE_STYLE, "Private cash flow")
        draw(axes[0], "Carbon value ($M)", CARBON_COLOR,
             CARBON_STYLE, "Carbon value")
        draw(axes[0], "Total ($M)", TOTAL_COLOR,
             TOTAL_STYLE, "Total value")
        draw(axes[1], "Private (%)", PRIVATE_COLOR,
             PRIVATE_STYLE, "Private cash flow")
        draw(axes[1], "Carbon (%)", CARBON_COLOR,
             CARBON_STYLE, "Carbon value")

        for ax, title, ylabel, ylim in zip(
            axes, PANEL_TITLES,
            [VALUE_Y_LABEL, SHARE_Y_LABEL],
            [VALUE_Y_LIMITS, SHARE_Y_LIMITS],
        ):
            ax.set_title(title, loc="left", fontsize=FONT_SIZE + 1, pad=12)
            ax.set_xlabel(X_LABEL)
            ax.set_ylabel(ylabel)
            ax.set_xlim(*X_LIMITS)
            ax.set_xticks(X_TICKS, [f"{p:g}" for p in X_TICKS])
            ax.tick_params(axis="x", labelrotation=X_TICK_ROTATION,
                           labelsize=FONT_SIZE - 1)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=GRID_ALPHA)

        axes[1].set_yticks([0, 20, 40, 60, 80, 100])
        if SHOW_LEGEND:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=LEGEND_POSITION, ncol=3, frameon=False)
        if SHOW_TITLE:
            fig.suptitle(TITLE, y=1.04, fontsize=FONT_SIZE + 4)

    return fig, axes


def save_figure(fig):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"}):
        for extension in SAVE_FORMATS:
            path = OUTPUT_DIR / f"{OUTPUT_NAME}.{extension}"
            fig.savefig(path, dpi=DPI, bbox_inches="tight",
                        facecolor="white")
            print(f"Saved: {path}")


def table():
    """Supplementary Table 19, rounded to two decimals as printed."""
    df = calculate_results(TABLE_PRICES).round(2)
    df["Carbon price ($/t)"] = df["Carbon price ($/t)"].astype(int)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "carbon_price_table.csv", index=False)
    df.to_latex(TABLE_DIR / "carbon_price_table.tex", index=False, float_format="%.2f")
    print(df.to_string(index=False))
    print("Saved: %s" % (TABLE_DIR / "carbon_price_table.csv"))


if __name__ == "__main__":
    table()
    fig, axes = make_figure()
    if SAVE_FIGURE:
        save_figure(fig)
