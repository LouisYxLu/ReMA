"""Read independent sensor CSVs. Matching never opens ground_truth.csv."""

import csv
import hashlib
from pathlib import Path
import numpy as np

SCENES = ["Area-I", "Area-II", "Area-III", "Area-IV"]


def _order(ids, scene, modality):
    return sorted(
        set(ids), key=lambda x: hashlib.sha256(f"{scene}:{modality}:{x}".encode()).digest()
    )


def load_scene(directory):
    directory = Path(directory)
    with (directory / "video.csv").open(newline="") as f:
        video = list(csv.DictReader(f))
    with (directory / "ais.csv").open(newline="") as f:
        ais = list(csv.DictReader(f))
    times = np.array(
        sorted({r["Time"] for r in video} | {r["Time"] for r in ais}), dtype="datetime64[ns]"
    )
    if len(times) <= 120 or not np.all(np.diff(times) == np.timedelta64(5, "s")):
        raise ValueError("Expected more than ten minutes of consecutive 5-s samples")
    visual_ids = _order([r["VideoID"] for r in video], directory.name, "video")
    ais_ids = _order([r["MMSI"] for r in ais], directory.name, "ais")
    ti = {str(t): i for i, t in enumerate(times)}
    vi = {name: i for i, name in enumerate(visual_ids)}
    ai = {name: i for i, name in enumerate(ais_ids)}
    T, V, A = (len(times), len(vi), len(ai))
    inputs = dict(
        times=times,
        video_xy=np.full((T, V, 2), np.nan),
        video_size=np.full((T, V, 2), np.nan),
        video_mask=np.zeros((T, V), bool),
        ais_xy=np.full((T, A, 2), np.nan),
        ais_speed=np.full((T, A), np.nan),
        ais_cog=np.full((T, A), np.nan),
        ais_mask=np.zeros((T, A), bool),
    )
    for row in video:
        t, i = (ti[str(np.datetime64(row["Time"], "ns"))], vi[row["VideoID"]])
        if inputs["video_mask"][t, i]:
            raise ValueError("Duplicate Time/VideoID")
        values = np.array([float(row[k]) for k in ["CenterX", "CenterY", "Width", "Height"]])
        if not np.isfinite(values).all() or (values[2:] <= 0).any():
            raise ValueError("Invalid visual observation")
        inputs["video_xy"][t, i], inputs["video_size"][t, i] = (values[:2], values[2:])
        inputs["video_mask"][t, i] = True
    for row in ais:
        t, i = (ti[str(np.datetime64(row["Time"], "ns"))], ai[row["MMSI"]])
        if inputs["ais_mask"][t, i]:
            raise ValueError("Duplicate Time/MMSI")
        xy = np.array([float(row["East_m"]), float(row["North_m"])])
        if not np.isfinite(xy).all():
            raise ValueError("Invalid AIS position")
        inputs["ais_xy"][t, i] = xy
        inputs["ais_speed"][t, i] = float(row["Speed_mps"]) if row["Speed_mps"] else np.nan
        inputs["ais_cog"][t, i] = float(row["Course_deg"]) if row["Course_deg"] else np.nan
        inputs["ais_mask"][t, i] = True
    return (inputs, visual_ids, ais_ids)


def save_matches(path, inputs, visual_ids, ais_ids, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "VideoID", "MMSI"])
        for t in range(120, len(inputs["times"])):
            for i in np.flatnonzero(inputs["video_mask"][t]):
                j = int(predictions[t, i])
                writer.writerow(
                    [str(inputs["times"][t]), visual_ids[i], ais_ids[j] if j >= 0 else ""]
                )
