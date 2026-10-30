# ReMA

Inference code and four-scene CSV data for **Continual Evidence Learning for
Self-Calibrating Cross-Modal Vessel Association in Maritime IoT**.

ReMA associates visual vessel tracks with AIS identities. It initializes spatial
alignment from unlabeled motion, then updates geometry, historical memory and
identity evidence while keeping source-trained neural weights fixed.

## Quick start

Requires Python 3.10 or later. CPU inference is sufficient.

```bash
python -m pip install -r requirements.txt
python match.py
```

Run these commands from the repository root. This runs all four regions and saves `results/Area-I.csv` through
`results/Area-IV.csv`. To run one region or the continuity extension:

```bash
python match.py --scene Area-I
python match.py --method rema-c --output results_continuity
```

Each output row is one visible target at one time:

```csv
Time,VideoID,MMSI
```

A nonempty MMSI is the assigned AIS identity. An empty MMSI means unmatched.
Output starts after the ten-minute initialization period. Initialization uses
unlabeled motion, not known visual–AIS correspondences. Detection images are
not required because the CSV files already contain visual track observations.

## Files

```text
match.py        Read data, run the method, save matches
rema.py         Historical memory, delayed evidence, identity competition
network.py      GRU encoders, memory attention, two graph layers, scoring heads
features.py     Temporal and relational features
geometry.py     Unlabeled initialization, guarded mapping updates, assignment
evidence.py     Physical scores and counterfactual audits
data.py         CSV input and matching output
evaluate.py     Optional evaluation of saved outputs
models/         Four source-trained checkpoints and fixed settings
data/Area-*/    video.csv, ais.csv, ground_truth.csv
```

No experiment scheduler, figure generator, dataset cache, or saved target
mapping is needed. Geometry, memory and identity evidence are recomputed from
the sensor CSVs. Neural weights and calibration remain fixed during inference.
ReMA-C additionally applies the source-selected acceptance support.

## Data

The release contains the frozen paper benchmark `audit80_20261027_v1`.
Each region spans 80 minutes at 5-s intervals. Times are local recording
timestamps. Sensor streams are deliberately separated from reference matches.

| File | Columns |
| --- | --- |
| `video.csv` | `Time, VideoID, CenterX, CenterY, Width, Height` |
| `ais.csv` | `Time, MMSI, East_m, North_m, Speed_mps, Course_deg` |
| `ground_truth.csv` | `Time, VideoID, MMSI, Ambiguous` and original audit flags |

Visual coordinates and sizes are in pixels. AIS positions use the fixed local
east/north metric frame of the paper. Speeds are m/s and courses are degrees.
`data/scenes.csv` records each time range and geographic origin. Numeric
observations are exported without rounding. MMSIs and VideoIDs are opaque
strings and are not network features. Rows are chronological within each file.

Visual observations with no usable AIS correspondence are retained.
In `ground_truth.csv`, an empty MMSI means no verified usable correspondence
at that timestamp, not confirmed absence of AIS equipment. `Ambiguous=1` rows
are excluded by the evaluator. The additional flags preserve source absence,
quality-control unavailability, unresolved labels, occlusion and mooring status.
Blank AIS speed/course fields mean unavailable measurements, not zero.

`match.py` reads only `video.csv` and `ais.csv`. It does not read
`ground_truth.csv`. Do not join the reference identity column into model inputs.

The four checkpoints use source-training seed 20261014. The model for each
region was trained on the other three regions. The corresponding JSON records
the source regions, fixed calibration and source-selected evidence strengths.
This small release provides pretrained inference, not source-training scripts.

## Evaluate (optional)

```bash
python evaluate.py
python evaluate.py --results results_continuity
```

The primary evaluation uses minutes `[30,80)`, while adaptation and output
start at minute 10. Evaluation is separate from inference and uses reference
matches only after predictions have been saved. The expected pooled metrics are:

| Method | Precision (%) | Recall (%) | F1 (%) | FAR (%) |
| --- | ---: | ---: | ---: | ---: |
| ReMA | 96.058 | 91.906 | 93.937 | 6.821 |
| ReMA-C | 96.094 | 92.068 | 94.038 | 6.843 |

Inference is chronological replay of the supplied synchronized streams, not a
native AIS receiver. On the validation environment (Python 3.12, NumPy 2.3.5,
PyTorch 2.14.1, CPU), both methods match their corresponding archived experiments
for all 73,954 visible observations after initialization. Other numerical libraries may cause
small floating-point differences near decision boundaries.

## Citation

Please cite the associated manuscript when using this code or dataset:

> Yuxu Lu, Jingming Zhang, Xiaoyu Li, Haifeng Ding, and Dong Yang.
> Continual Evidence Learning for Self-Calibrating Cross-Modal Vessel
> Association in Maritime IoT.

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).
