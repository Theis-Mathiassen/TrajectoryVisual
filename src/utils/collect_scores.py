#!/usr/bin/env python3
import argparse
import csv
import pickle
import pathlib
from typing import Any, Dict, List

#Aggregate all score-pickle files into a single CSV.
# example use: uv run collect_scores.py results  scores_summary.csv

def _flatten(rec: Dict[str, Any]) -> Dict[str, float]:
    f1_1, f1_2, f1_3, f1_4, f1_5 = rec["f1Scores"]
    se_1, se_2 = rec["simplificationError"]
    return {
        "cr": float(rec["cr"]),
        "f1_1": float(f1_1),
        "f1_2": float(f1_2),
        "f1_3": float(f1_3),
        "f1_4": float(f1_4),
        "f1_5": float(f1_5),
        "simpl_err_1": float(se_1),
        "simpl_err_2": float(se_2),
    }

def _collect(root: pathlib.Path) -> List[dict]:
    rows: List[dict] = []
    for pkl_path in root.rglob("*.pkl"):
        # skip non-score pickles like the logger outputs
        if pkl_path.name.startswith("script_"):
            continue

        try:
            data = pickle.load(pkl_path.open("rb"))
        except Exception as exc:
            print(f"   Skipping {pkl_path}: {exc}")
            continue

        for rec in data:
            row = {
                "folder": pkl_path.parent.name,
                "file": pkl_path.name,
                **_flatten(rec),
            }
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect score pickles into CSV")
    ap.add_argument("root", help="Top-level directory containing result folders")
    ap.add_argument("out_csv", nargs="?", default="scores_summary.csv",
                    help="Output CSV name (by default: scores_summary.csv)")
    args = ap.parse_args()

    rows = _collect(pathlib.Path(args.root))
    if not rows:
        print("No score pickles found – nothing written.")
        return

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()