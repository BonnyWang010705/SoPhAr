"""Assemble the main-text figures from their panel images.

Fig. 2, 4 and 5 follow the submitted layout; Fig. 3 places panel a on top,
the three route maps (b-d) in the middle and the state
bars (e) at the bottom. Fig. 6 is assembled by fig6.py itself
because its line panels share one legend. Figs. 7-8 are the
market-penetration figures of the arXiv version (fig7.py, fig8.py).

The panel files of Fig. 5 carry their own legend, so panels a-c are drawn again
here without it, by fig5abc.py, and placed under one shared legend.
"""
import argparse
import os
import tempfile

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

import style as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
MAIN = os.path.join(ROOT, "figures", "main")
# one folder per figure, holding the figure and a panels/ subfolder
FOLDERS = {"figure2": "fig02_solar_farm_analysis", "figure3": "fig03_flight_analysis",
           "figure4": "fig04_temporal_analysis", "figure5": "fig05_schedule_optimization",
           "figure7": "fig07_solar_farm_selection", "figure8": "fig08_flight_selection"}
P2, P3, P4, P5, P7, P8 = (os.path.join(MAIN, FOLDERS[f], "panels") for f in
                          ("figure2", "figure3", "figure4", "figure5", "figure7", "figure8"))
YIMING = os.path.join(ROOT, "results", "baseline", "route_maps")
SUPP = os.path.join(ROOT, "figures", "supplementary")

WIDTH = 3600          # assembled width in pixels
GAP = 60              # between columns and rows
BAND = 120            # space above each panel for its letter
# ~9 pt at the printed width, matching the submitted figures
LETTER_PT = round(9 * (WIDTH / 180) * 25.4 / 72)

# each figure: rows of (letter, image path); None letter = no label
FIGURES = {
    "figure2": [
        [("a", os.path.join(P2, "map_capacity_state.png")),
         ("b", os.path.join(P2, "figure2b.png"))],
        [("c", os.path.join(P2, "map_energy_county.png")),
         ("d", os.path.join(P2, "map_duration_county.png"))],
        [("e", os.path.join(P2, "figure2e.png"))],
    ],
    "figure3": [
        [("a", os.path.join(P3, "figure3a.png"))],
        [("b", os.path.join(YIMING, "R1_Flight_Short_Range_3486_Unique_Routes.png")),
         ("c", os.path.join(YIMING, "R1_Flight_Medium_Range_1944_Unique_Routes.png")),
         ("d", os.path.join(YIMING, "R1_Flight_Long_Range_40_Unique_Routes.png"))],
        [("e", os.path.join(P3, "figure3e.png"))],
    ],
    "figure4": [
        [(None, os.path.join(P4, "figure4.png"))],
    ],
    "figure5": None,     # rows built by figure5_rows()
    # Figs. 7-8 of the arXiv version, drawn by fig7.py and fig8.py
    "figure7": [
        [("a", os.path.join(P7, "figure7a.png"))],
        [("b", os.path.join(P7, "figure7b.png"))],
    ],
    "figure8": [
        [("a", os.path.join(P8, "figure8a.png"))],
        [("b", os.path.join(P8, "figure8b.png"))],
    ],
}

def letter_font():
    path = font_manager.findfont(font_manager.FontProperties(
        family=st.LETTER_FAMILY, weight="bold"))
    return ImageFont.truetype(path, LETTER_PT)

def fit(im, width):
    return im.resize((width, round(im.height * width / im.width)),
                     Image.LANCZOS)

def assemble(rows, font):
    built = []
    for row in rows:
        images = [Image.open(p).convert("RGB") for _, p in row]
        # share the width in proportion to aspect ratio so that every
        # panel in the row comes out the same height
        # equal widths, so a panel in the left column always starts and ends
        # at the same x as the panel above it and their colour bars align
        free = WIDTH - GAP * (len(row) - 1)
        widths = [free // len(row)] * len(row)
        widths[-1] = free - sum(widths[:-1])
        cells = [(letter, fit(im, w))
                 for (letter, _), im, w in zip(row, images, widths)]
        band = BAND if any(l for l, _ in cells) else 0
        row_h = band + max(im.height for _, im in cells)
        canvas = Image.new("RGB", (WIDTH, row_h), "white")
        draw = ImageDraw.Draw(canvas)
        x = 0
        for letter, im in cells:
            canvas.paste(im, (x, band))
            if letter:
                draw.text((x, 0), letter + ".", fill="black", font=font)
            x += im.width + GAP
        built.append(canvas)
    total_h = sum(b.height for b in built) + GAP * (len(built) - 1)
    fig = Image.new("RGB", (WIDTH, total_h), "white")
    y = 0
    for b in built:
        fig.paste(b, (0, y))
        y += b.height + GAP
    return fig

def figure5_rows(tmp):
    """Fig. 5: the shared legend on top, panels a-c redrawn without their
    legend by fig5abc.py, and panels d-f."""
    import fig5abc
    st.apply()
    for alt in sorted(fig5abc.PANEL):
        fig, _, _ = fig5abc.draw(*fig5abc.series(alt))
        st.save(fig, os.path.join(tmp, "figure5%s_%dm" % (fig5abc.PANEL[alt], alt)))
    fig5abc.legend_strip(os.path.join(tmp, "figure5_legend"))
    return [
        [(None, os.path.join(tmp, "figure5_legend.png"))],
        [("a", os.path.join(tmp, "figure5a_9100m.png")),
         ("d", os.path.join(P5, "figure5d_9100m.png"))],
        [("b", os.path.join(tmp, "figure5b_12100m.png")),
         ("e", os.path.join(P5, "figure5e_12100m.png"))],
        [("c", os.path.join(tmp, "figure5c_15100m.png")),
         ("f", os.path.join(P5, "figure5f_15100m.png"))],
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=sorted(FIGURES),
                        help="assemble one figure; the default assembles all")
    args = parser.parse_args()

    font = letter_font()
    selected = ({args.figure: FIGURES[args.figure]}
                if args.figure else FIGURES)
    for name, rows in selected.items():
        if name == "figure5":
            with tempfile.TemporaryDirectory() as tmp:
                fig = assemble(figure5_rows(tmp), font)
        else:
            fig = assemble(rows, font)
        out_dir = SUPP if name.startswith("supp") else os.path.join(MAIN, FOLDERS[name])
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            path = os.path.join(out_dir, "%s.%s" % (name, ext))
            fig.save(path, resolution=300.0)
            print("wrote", path)
        print("  %s: %d x %d px" % (name, *fig.size))

if __name__ == "__main__":
    main()
