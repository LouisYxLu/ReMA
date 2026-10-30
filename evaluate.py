"""Optional evaluation of already saved matches on the paper's [30,80) window."""

import argparse
import csv
from pathlib import Path
from data import SCENES
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--results", type=Path, default=root / "results")
    args = parser.parse_args()
    total = dict(N=0, N0=0, C=0, A=0, F0=0)
    for scene in SCENES:
        path = args.results / f"{scene}.csv"
        if not path.exists():
            continue
        with path.open(newline="") as f:
            predictions = {(r["Time"], r["VideoID"]): r["MMSI"] for r in csv.DictReader(f)}
        with (args.data / scene / "ground_truth.csv").open(newline="") as f:
            truth = list(csv.DictReader(f))
        t0 = min((np.datetime64(r["Time"], "ns") for r in truth))
        counts = dict(N=0, N0=0, C=0, A=0, F0=0)
        for row in truth:
            minutes = (np.datetime64(row["Time"], "ns") - t0) / np.timedelta64(1, "m")
            if not 30 <= minutes < 80 or row["Ambiguous"] == "1":
                continue
            key = (row["Time"], row["VideoID"])
            if key not in predictions:
                raise ValueError(f"Missing prediction: {scene} {key}")
            predicted, expected = (predictions[key], row["MMSI"])
            counts["N" if expected else "N0"] += 1
            counts["A"] += bool(predicted)
            counts["C"] += bool(expected and predicted == expected)
            counts["F0"] += bool(not expected and predicted)
        for key, value in counts.items():
            total[key] += value
        show(scene, counts)
    if not total["N"]:
        raise ValueError("No matching result CSVs found")
    show("Pooled", total)


def show(name, x):
    precision = 100 * x["C"] / max(x["A"], 1)
    recall = 100 * x["C"] / max(x["N"], 1)
    f1 = 200 * x["C"] / max(x["A"] + x["N"], 1)
    far = 100 * x["F0"] / max(x["N0"], 1)
    print(f"{name:8s} P={precision:.3f} R={recall:.3f} F1={f1:.3f} FAR={far:.3f}")


if __name__ == "__main__":
    main()
