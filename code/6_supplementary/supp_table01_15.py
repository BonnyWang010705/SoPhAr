"""Write the R1 numbers into the submitted Supplementary Materials.

The submitted document is kept as it is: its styles, fonts, table layout,
table of contents, headings and figures are untouched. Only the numbers
inside Tables 1-17 are replaced, cell by cell, in the copy
`supplementary_materials/Supplementary Materials_R1.docx`. Numbers are
written with ten significant digits, the way the submitted tables print them.

R1 changes every value in two ways: the flight data are now the Marketing
Carrier records (148,814 flights, 438 qualified farms) and the savings are
taken on the delivered-energy basis with the $84/t carbon price, the basis the
manuscript reports. The submitted tables used the supported-duration basis and
the $190/t social cost of carbon.
"""
import argparse
import copy
import io
import os
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
# Word rejects the file if the namespace prefixes change, because mc:Ignorable
# names them, so every prefix of the original part is registered before the
# tree is written back out.
XML_DECLARATION = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
SUPP = os.path.join(ROOT, "supplementary_materials")
ORIGINAL = os.path.join(SUPP, "Supplementary Materials_original.docx")
TARGET = os.path.join(SUPP, "Supplementary Materials_R1.docx")
DOC = "word/document.xml"

BASELINE = os.path.join(ROOT, "results", "baseline")
OPT1 = os.path.join(ROOT, "results", "optimization_1")
OPT2 = os.path.join(ROOT, "results", "optimization_2")

CARBON_PRICE = 0.084          # $/kg CO2 = $84/t, as in code/4_figures/agg.py
# the baseline files carry both bases, the optimization files only the
# delivered-energy one, under shorter names
CO2_NAMES = ("CO2_Emissions_Reduction_by_Energy", "CO2_Emissions_Reduction")
FUEL_NAMES = ("Fuel_Cost_Savings_by_Energy", "Money_Cost_Saving")
CLASS_EDGES = [0, 1500, 4000, float("inf")]
CLASS_NAMES = ["short", "medium", "long"]
FLIGHTS_ANALYSED = 148814     # after the IATA-FAA airport match

# Tables 2-7 and 9-14, in the order the document uses them
FLIGHT_TABLES = [(c, s) for s in ("Origin", "Destination") for c in CLASS_NAMES]

FLIGHTS = "flight_analysis_results_merged_R1.csv"
FARMS = "solar_farm_analysis_results_merged_R1.csv"
OPT_FLIGHTS = "Optimization_1_flight_analysis_results_merged_R1.csv"
OPT_FARMS = "Optimization_1_solar_farm_analysis_results_merged_R1.csv"


def q(tag):
    return "{%s}%s" % (W, tag)


def register_prefixes(payload):
    for _, (prefix, uri) in ET.iterparse(io.BytesIO(payload),
                                         events=("start-ns",)):
        ET.register_namespace(prefix, uri)


def document_tag(payload):
    match = re.match(rb"<w:document\b[^>]*>",
                     payload[payload.index(b"<w:document"):])
    if not match:
        raise ValueError("could not find the w:document tag")
    return match.group(0)


def serialize(root, original):
    """Write the tree back, keeping every namespace the original declared.

    ElementTree only declares the namespaces it happens to use, but
    mc:Ignorable names prefixes that appear nowhere else, and Word reports a
    missing declaration as damaged content.
    """
    body = ET.tostring(root, encoding="utf-8")
    new_tag = document_tag(body)
    declared = dict(re.findall(rb'xmlns:([A-Za-z0-9]+)="([^"]*)"', new_tag))
    missing = [b'xmlns:%s="%s"' % (prefix, uri) for prefix, uri
               in re.findall(rb'xmlns:([A-Za-z0-9]+)="([^"]*)"',
                             document_tag(original))
               if prefix not in declared]
    merged = new_tag[:-1] + b" " + b" ".join(missing) + b">" if missing else new_tag
    return XML_DECLARATION + body.replace(new_tag, merged, 1)


def cell_text(tc):
    return " ".join(tc.itertext()).strip()


def set_cell(tc, value):
    """Replace a cell's text, keeping the formatting of its first run."""
    paragraphs = tc.findall(q("p"))
    runs = [r for p in paragraphs for r in p.findall(q("r"))]
    if not runs:
        raise ValueError("cell has no run to copy the formatting from")
    first, rest = runs[0], runs[1:]
    texts = first.findall(q("t"))
    if not texts:
        texts = [ET.SubElement(first, q("t"))]
    texts[0].text = value
    texts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for extra in texts[1:]:
        first.remove(extra)
    for p in paragraphs:
        for r in p.findall(q("r")):
            if r in rest:
                p.remove(r)
    for p in paragraphs[1:]:
        if not p.findall(q("r")):
            tc.remove(p)


def rows_of(tbl):
    return [tr.findall(q("tc")) for tr in tbl.findall(q("tr"))]


def fmt(value):
    """Ten significant digits, as the submitted tables print their numbers."""
    return "%.10g" % float(value)


def load(path, usecols=None):
    return pd.read_csv(path, low_memory=False, usecols=usecols)


def savings_columns(columns):
    co2 = next(c for c in CO2_NAMES if c in columns)
    fuel = next(c for c in FUEL_NAMES if c in columns)
    return co2, fuel


def farm_table(path):
    d = load(path)
    co2, fuel = savings_columns(d.columns)
    g = d.groupby("p_state").agg(co2=(co2, "sum"), fuel=(fuel, "sum"),
                                 third=("p_cap_safe", "sum"))
    g["total"] = g.fuel + g.co2 * CARBON_PRICE
    return g


def flight_tables(path):
    """One table per range class and per origin/destination state."""
    co2, fuel = savings_columns(pd.read_csv(path, nrows=0).columns)
    d = load(path, ["STATE_Origin", "STATE_Destination", "Distance_km",
                    co2, fuel])
    d["cls"] = pd.cut(d.Distance_km, CLASS_EDGES, labels=CLASS_NAMES,
                      right=False)
    if d["cls"].isna().any():
        raise ValueError("flights outside the distance classes")
    out = {}
    for cls, side in FLIGHT_TABLES:
        sub = d[d.cls == cls]
        g = sub.groupby("STATE_%s" % side).agg(
            co2=(co2, "sum"), fuel=(fuel, "sum"), third=("Distance_km", "sum"))
        g["total"] = g.fuel + g.co2 * CARBON_PRICE
        out[(cls, side)] = g
    return out


def write_state_table(tbl, table):
    """Fill a State | CO2 | Fuel | <third> | Total table, Total row included."""
    rows = rows_of(tbl)
    header = [cell_text(c) for c in rows[0]]
    if header[0] != "State" or len(header) != 5:
        raise ValueError("unexpected header: %s" % header)
    for cells in rows[1:]:
        state = cell_text(cells[0])
        if state == "Total":
            values = table.sum()
        elif state in table.index:
            values = table.loc[state]
        else:
            raise ValueError("state %r is not in the R1 results" % state)
        for tc, column in zip(cells[1:], ["co2", "fuel", "third", "total"]):
            set_cell(tc, fmt(values[column]))
    return len(rows) - 1, header


def write_penetration_table(tbl, summary):
    """Table 15: one row per (farm, flight) penetration pair."""
    rows = rows_of(tbl)
    header = [cell_text(c) for c in rows[0]]
    if header[0] != "Solar Farm":
        raise ValueError("unexpected header: %s" % header)
    keyed = {(round(r.p_solar_farm, 1), round(r.p_flight, 1)): r
             for r in summary.itertuples()}
    for cells in rows[1:]:
        key = (round(float(cell_text(cells[0])), 1),
               round(float(cell_text(cells[1])), 1))
        row = keyed[key]
        total = row.Money_Cost_Saving + row.CO2_Emissions_Reduction * CARBON_PRICE
        values = [row.Total_Energy_Supplied_MWh,
                  row.Total_Flight_Duration_Supported_Hours,
                  row.CO2_Emissions_Reduction, row.Money_Cost_Saving, total]
        for tc, value in zip(cells[2:], values):
            set_cell(tc, fmt(value))
    return len(rows) - 1, header


W14 = "http://schemas.microsoft.com/office/word/2010/wordml"


def add_social_cost_row(tbl):
    """Table 16: the social cost of carbon, beside the carbon price.

    The revision values CO2 at the EU ETS carbon price and reports the U.S.
    EPA social cost separately, so the parameter table has to carry both. The
    row is a copy of the carbon price row, which keeps the table's fonts,
    borders and the italic subscript of the symbol; only the subscript text,
    the description and the value change."""
    for tr in tbl.findall(q("tr")):
        cells = tr.findall(q("tc"))
        if len(cells) < 3 or cell_text(cells[1]) != "Carbon price":
            continue
        row = copy.deepcopy(tr)
        for p in row.iter(q("p")):
            for name in ("paraId", "textId"):
                p.attrib.pop("{%s}%s" % (W14, name), None)
        cells = row.findall(q("tc"))
        runs = [r for p in cells[0].findall(q("p")) for r in p.findall(q("r"))]
        subscripts = [r for r in runs
                      if r.find(q("rPr") + "/" + q("vertAlign")) is not None]
        if len(subscripts) != 1:
            raise ValueError("the carbon price symbol is not C with one "
                             "subscript")
        subscripts[0].find(q("t")).text = "social"
        set_cell(cells[1], "Carbon social cost")
        set_cell(cells[2], "0.19")
        tbl.insert(list(tbl).index(tr) + 1, row)
        return "added the social cost of carbon, 0.19 $/kg CO2"
    raise ValueError("the carbon price row was not found")


def write_parameters(tbl):
    """Table 16: the carbon price is the manuscript's $84/t, and the $190/t
    social cost of carbon follows it on its own row."""
    changed = []
    for cells in rows_of(tbl)[1:]:
        if cell_text(cells[1]) == "Social cost of carbon":
            set_cell(cells[1], "Carbon price")
            set_cell(cells[2], "0.084")
            changed.append("carbon price 0.19 -> 0.084 $/kg CO2")
    if not changed:
        raise ValueError("the carbon price row was not found")
    changed.append(add_social_cost_row(tbl))
    return changed


def set_number(tc, value):
    """Replace the number in a cell, leaving its runs alone.

    The set sizes are written as three runs, "Set of flights, |", an italic
    "I" and "| = 143,152", so the whole cell must not be rewritten: that
    would drop the italic and leave spaces inside the bars. Only the run
    that carries the number is touched."""
    for t in tc.iter(q("t")):
        if re.search(r"=\s*[\d,]+", t.text or ""):
            t.text = re.sub(r"=\s*[\d,]+", "= %s" % format(value, ","), t.text)
            return t.text
    raise ValueError("no run carries a number: %r" % cell_text(tc))


def write_notation(tbl, flights, farms):
    """Table 17: the set sizes |I| and |F|."""
    changed = []
    for cells in rows_of(tbl)[1:]:
        text = cell_text(cells[-1])
        if text.startswith("Set of flights"):
            changed.append(set_number(cells[-1], flights))
        elif text.startswith("Set of qualified solar farms"):
            changed.append(set_number(cells[-1], farms))
    if len(changed) != 2:
        raise ValueError("expected to update |I| and |F|, changed %s" % changed)
    return changed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true",
                   help="start again from the submitted document")
    p.add_argument("--path", default=TARGET,
                   help="the document to write (the R1 copy by default)")
    a = p.parse_args()
    target = a.path
    if a.reset or not os.path.exists(target):
        shutil.copyfile(ORIGINAL, target)
        print("copied the submitted document to", os.path.basename(target))

    with zipfile.ZipFile(target) as z:
        names = z.namelist()
        payload = {n: z.read(n) for n in names}
    register_prefixes(payload[DOC])
    root = ET.fromstring(payload[DOC])
    tables = [el for el in root[0] if el.tag == q("tbl")]
    if len(tables) != 17:
        raise ValueError("expected 17 tables, found %d" % len(tables))

    # Tables 1-7: the baseline run at 12,100 m
    baseline_farms = farm_table(os.path.join(BASELINE, FARMS))
    n, _ = write_state_table(tables[0], baseline_farms)
    print("Table 1  farms                 %2d rows" % n)
    baseline_flights = flight_tables(os.path.join(BASELINE, FLIGHTS))
    for i, key in enumerate(FLIGHT_TABLES):
        n, _ = write_state_table(tables[1 + i], baseline_flights[key])
        print("Table %-2d %-6s %-11s      %2d rows" % (2 + i, key[0], key[1], n))

    # Tables 8-14: the same after schedule optimization at 12,100 m
    opt_farms = farm_table(os.path.join(OPT1, OPT_FARMS))
    n, _ = write_state_table(tables[7], opt_farms)
    print("Table 8  farms, optimized      %2d rows" % n)
    opt_flights = flight_tables(os.path.join(OPT1, OPT_FLIGHTS))
    for i, key in enumerate(FLIGHT_TABLES):
        n, _ = write_state_table(tables[8 + i], opt_flights[key])
        print("Table %-2d %-6s %-11s opt  %2d rows" % (9 + i, key[0], key[1], n))

    # Table 15: the 100 market penetration scenarios
    summary = load(os.path.join(OPT2, "Optimization_2_summary_R1.csv"))
    n, _ = write_penetration_table(tables[14], summary)
    print("Table 15 penetration           %2d rows" % n)

    # Tables 16-17: parameters and notation
    farms_count = len(load(os.path.join(BASELINE, FARMS), ["FID"]))
    print("Table 16", write_parameters(tables[15]))
    print("Table 17", write_notation(tables[16], FLIGHTS_ANALYSED, farms_count))

    payload[DOC] = serialize(root, payload[DOC])
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.writestr(name, payload[name])
    print("wrote", target)

    print("\nnational totals on the R1 basis")
    for label, table in (("baseline ", baseline_farms), ("optimized", opt_farms)):
        print("  %s: CO2 %s kg, fuel $%s, total $%s"
              % (label, format(table.co2.sum(), ",.0f"),
                 format(table.fuel.sum(), ",.0f"),
                 format(table.total.sum(), ",.0f")))


if __name__ == "__main__":
    main()
