import argparse
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip"
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "data", "flights", "raw")

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
CHUNK = 1 << 20  # 1 MiB
MAX_RETRIES = 3


def target_path(year, month):
    return os.path.join(
        OUT_DIR,
        "On_Time_Marketing_Carrier_On_Time_Performance_"
        f"Beginning_January_2018_{year}_{month}.zip")


def verify(path):
    if not os.path.exists(path):
        return False, "missing"
    size = os.path.getsize(path)
    if size == 0:
        return False, "empty file"
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad is not None:
                return False, f"corrupt member: {bad}"
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if len(csvs) != 1:
                return False, f"expected 1 csv, found {len(csvs)}"
            rows = csvs[0]
    except zipfile.BadZipFile as exc:
        return False, f"bad zip ({exc})"
    return True, f"{size / 1e6:.1f} MB, {rows}"


def download(url, path):
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        declared = resp.headers.get("Content-Length")
        declared = int(declared) if declared else None
        written = 0
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
                if declared:
                    pct = 100 * written / declared
                    print(
                        f"\r      {written / 1e6:6.1f} / {declared / 1e6:.1f} MB "
                        f"({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
        print()
    if declared is not None and written != declared:
        os.remove(tmp)
        raise IOError(f"size mismatch: got {written}, expected {declared}")
    os.replace(tmp, path)


def fetch_month(year, month, force=False):
    path = target_path(year, month)
    label = f"{year}-{month:02d}"

    if not force:
        ok, msg = verify(path)
        if ok:
            print(f"[{label}] already present  ({msg})")
            return True, "cached"

    url = BASE_URL.format(year=year, month=month)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[{label}] downloading (attempt {attempt}/{MAX_RETRIES})")
            download(url, path)
        except (urllib.error.URLError, IOError, TimeoutError) as exc:
            print(f"[{label}] transfer failed: {exc}")
        else:
            ok, msg = verify(path)
            if ok:
                print(f"[{label}] OK  ({msg})")
                return True, "downloaded"
            print(f"[{label}] verification failed: {msg}")
            if os.path.exists(path):
                os.remove(path)
        if attempt < MAX_RETRIES:
            time.sleep(5 * attempt)

    print(f"[{label}] GIVING UP after {MAX_RETRIES} attempts")
    return False, "failed"


def default_months():
    return [(2024, 12)] + [(2025, m) for m in range(1, 13)] + [(2026, 1)]


def parse_month(text):
    year, month = text.split("-")
    return int(year), int(month)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=parse_month, nargs="+", default=default_months(),
                    metavar="YYYY-MM")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")
    print(f"Months: {', '.join(f'{y}-{m:02d}' for y, m in args.months)}\n")

    results = {}
    started = time.time()
    for year, month in args.months:
        results[(year, month)] = fetch_month(year, month, force=args.force)

    print("\n" + "=" * 58)
    print(f"{'month':<10}{'status':<14}{'file':<34}")
    print("-" * 58)
    total_bytes = 0
    for year, month in args.months:
        ok, how = results[(year, month)]
        path = target_path(year, month)
        if ok:
            total_bytes += os.path.getsize(path)
        print(
            f"{year}-{month:02d}   "
            f"{('OK (' + how + ')') if ok else 'FAILED':<14}"
            f"{os.path.basename(path) if ok else '-':<34}"
        )
    print("-" * 58)
    n_ok = sum(1 for ok, _ in results.values() if ok)
    print(
        f"{n_ok}/{len(args.months)} files   "
        f"{total_bytes / 1e6:.0f} MB   "
        f"{time.time() - started:.0f}s elapsed"
    )

    return 0 if n_ok == len(args.months) else 1


if __name__ == "__main__":
    sys.exit(main())
