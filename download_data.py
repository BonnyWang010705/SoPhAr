"""Download the Hugging Face dataset into the root of this repository.

Afterwards data/, results/ and figures/ sit next to code/, which is where every
script looks for them.

    python download_data.py
    python download_data.py --include "results/*" "figures/*"   # skip data/
"""
import argparse
import os
import sys

# Hugging Face dataset id
REPO_ID = "Solarphasedarray/SoPhAr"
ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo-id", default=os.environ.get("SOPHAR_HF_DATASET", REPO_ID),
                   help="Hugging Face dataset id (default: %(default)s)")
    p.add_argument("--include", nargs="*", default=None,
                   help='download only paths matching these patterns, e.g. "results/*"')
    a = p.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is missing: pip install huggingface_hub")

    # README.md and .gitattributes belong to the dataset repository and would
    # overwrite this repository's files of the same name.
    snapshot_download(repo_id=a.repo_id, repo_type="dataset", local_dir=ROOT,
                      allow_patterns=a.include,
                      ignore_patterns=["README.md", ".gitattributes"])
    print("data/, results/ and figures/ are in", ROOT)


if __name__ == "__main__":
    main()
