"""Read CSV observations, run ReMA, and write Time/VideoID/MMSI matches."""

import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[name] = "1"
import argparse
from pathlib import Path
import time
import torch
from data import SCENES, load_scene, save_matches
from rema import match_scene


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES + ["all"], default="all")
    parser.add_argument("--method", choices=["rema", "rema-c"], default="rema")
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--models", type=Path, default=root / "models")
    parser.add_argument("--output", type=Path, default=root / "results")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for scene in SCENES if args.scene == "all" else [args.scene]:
        start = time.perf_counter()
        print(f"{scene}: {args.method}", flush=True)
        inputs, visual_ids, ais_ids = load_scene(args.data / scene)
        predictions = match_scene(
            inputs,
            args.models / f"{scene}.pt",
            args.method == "rema-c",
            progress=lambda message: print(message, flush=True),
        )
        path = args.output / f"{scene}.csv"
        save_matches(path, inputs, visual_ids, ais_ids, predictions)
        print(f"  saved {path} ({time.perf_counter() - start:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
