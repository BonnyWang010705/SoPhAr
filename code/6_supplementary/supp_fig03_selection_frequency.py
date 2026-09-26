"""Draw the farm selection frequency for the Supplementary Materials.

The document gives this figure a page of its own, so it is drawn taller than
the panel of the same name in Fig. 7 of the arXiv version: the same canvas width,
so the text still prints between 5 and 7 pt, with the 100 scenarios spread
over the height of half a page instead of a strip.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "4_figures"))

import fig7                                            # noqa: E402
import style as st                                     # noqa: E402

OUT = os.path.join(fig7.ROOT, "figures", "supplementary",
                   "fig03_selection_frequency")
HEIGHT = 12.8           # prints 5.2 in tall at the 6.5 in text width


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=fig7.DATA)
    p.add_argument("--height", type=float, default=HEIGHT)
    a = p.parse_args()

    st.apply()
    data = fig7.open_results(a.data)
    fig, farms = fig7.panel_b(data, height=a.height)
    for f in st.save(fig, os.path.join(OUT, "supp_fig03")):
        print("wrote", f)
    print("%d farms ever selected, %d in all %d scenarios"
          % (len(farms), (farms["count"] == fig7.N_SCENARIOS).sum(),
             fig7.N_SCENARIOS))


if __name__ == "__main__":
    main()
