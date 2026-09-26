import os

import matplotlib.pyplot as plt

DPI = 200

# Nature branded research journals, guide to preparing final artwork:
# figure text must sit between 5 pt and 7 pt at the printed width, and a
# two-column figure in original research prints 180 mm wide.
PRINT_MM = 180.0
MAX_PT, MIN_PT = 7.0, 5.0

def pt(target_pt, fig_w_in, frac=1.0, print_mm=PRINT_MM):
    """Point size to draw with so the text prints at target_pt.

    fig_w_in is the panel's own width in inches; frac is the share of the
    printed figure width that the panel occupies once it is assembled.
    """
    return target_pt * fig_w_in * 25.4 / (print_mm * frac)

SHORT, MEDIUM, LONG = "#6495ed", "#ca7064", "#a77fb5"
CLASS_COLORS = [SHORT, MEDIUM, LONG]

FUEL, CARBON, CARBON_FILL = "#d2b48c", "#bc5a4b", "#f4f4f4"
# CO2 partner for FUEL where a wedge or bar needs a visible fill
CARBON_TINT = "#d9d9d9"

CMAP_WARM, CMAP_COOL = "OrRd", "Greens"

# The submitted figures label panels "a.", "b.", ... in Times New Roman
# Bold. The fallbacks are metric-compatible Times clones, then faces that
# ship with matplotlib, so the letters render without Times installed.
LETTER_FAMILY = ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                 "Tinos", "STIXGeneral", "DejaVu Serif"]

STATE_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida",
    "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "PR": "Puerto Rico", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "VI": "Virgin Islands",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin",
    "WY": "Wyoming",
}

def apply():
    # matplotlib defaults: DejaVu Sans, which ships inside matplotlib and so
    # renders identically on every platform; fonts are embedded in the PDFs
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

def scale(fig_w_in, frac=1.0, label=7.0, tick=6.0):
    """Size a panel's text from the printed target.

    Call after apply(), before creating the figure. fig_w_in is the panel's
    width in inches and frac the share of the printed figure width it takes.
    """
    lab, tk = pt(label, fig_w_in, frac), pt(tick, fig_w_in, frac)
    plt.rcParams.update({
        "font.size": tk, "axes.titlesize": lab, "axes.labelsize": lab,
        "xtick.labelsize": tk, "ytick.labelsize": tk,
        "legend.fontsize": tk, "legend.title_fontsize": lab,
    })

def full_name(code):
    return STATE_NAME.get(code, code)

def save(fig, path, dpi=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        f = "%s.%s" % (path, ext)
        fig.savefig(f, dpi=dpi or DPI, facecolor="white")
        out.append(f)
    plt.close(fig)
    return out
