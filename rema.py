"""ReMA replay: sensor input -> neural scores -> partial one-to-one matches.

Geometry and neural history are computed chronologically first. Delayed events
are then replayed by timestamp, always after that timestamp's output. No target
truth, saved predictions, saved mappings, or saved audit events enter inference.
"""

from pathlib import Path
import json
import numpy as np
import torch
from geometry import Config, bootstrap, run_teacher, select_teacher, assignment_with_rejection
from features import SceneFeatures
from network import GraphMemoryNetwork
from evidence import window_evidence, witnesses


def assign(frame, scores, visual_count):
    cost = -scores.copy()
    cost[~frame["candidate"]] = np.inf
    rows, columns = assignment_with_rejection(cost, 0.0)
    out = np.full(visual_count, -1, np.int32)
    out[frame["vi"][rows]] = frame["ai"][columns]
    return out


def score_frames(model, data, calibration):
    frames = [None] * data.T
    bank = {}
    last_seen = np.full(data.V, -999)
    for t in range(120, data.T):
        last_seen[data.inputs["video_mask"][t]] = t
        bank = {i: r for i, r in bank.items() if t - last_seen[i] <= 24}
        data.memory[t] = sorted(bank.values(), key=lambda r: r[3], reverse=True)[:32]
        f = data.frame(t)
        ev = window_evidence(data, t, vi=f["vi"], ai=f["ai"])
        with torch.no_grad():
            logits = model(f["tensors"]).numpy()
        margin = logits[:, :-1] - logits[:, -1, None]
        pair = f["tensors"]["pair"].numpy()
        row_best = np.max(margin, axis=1, initial=-12)
        column_best = np.max(margin, axis=0, initial=-12)
        phi = np.concatenate(
            [
                np.ones((*margin.shape, 1)),
                margin[..., None] / 4,
                pair,
                ev["features"],
                np.broadcast_to(row_best[:, None, None] / 4, (*margin.shape, 1)),
                np.broadcast_to(column_best[None, :, None] / 4, (*margin.shape, 1)),
            ],
            -1,
        ).astype(np.float32)
        phi = np.clip(np.nan_to_num(phi), -6, 6)
        frames[t] = dict(
            vi=f["vi"],
            ai=f["ai"],
            candidate=f["tensors"]["candidate"].numpy(),
            base=margin + np.clip(phi @ calibration, -4, 4),
        )
        if t >= 126 and t % 6 == 0:
            obs = t - 6
            old = frames[obs]
            audit = witnesses(window_evidence(data, t, start=obs - 17, vi=old["vi"], ai=old["ai"]))
            for row, i in enumerate(old["vi"]):
                label = int(audit["label"][row])
                j = int(old["ai"][label]) if label >= 0 else label
                if j >= 0 and data.inputs["video_mask"][obs, i] and data.inputs["ais_mask"][obs, j]:
                    bank[int(i)] = (obs, int(i), j, t, float(max(0.5, audit["quality"][row])))
                elif j == -1:
                    bank.pop(int(i), None)
    return frames


def audit_events(frames, data, offset, weak_weight):
    events, metadata = ([], {})
    for t in range(126, data.T, 6):
        obs = t - 6
        for row, i in enumerate(frames[obs]["vi"]):
            metadata[obs, int(i)] = dict(row=row, label=-2)
        for obs in sorted({k[0] for k in metadata}):
            if t - obs > 24:
                continue
            old = frames[obs]
            audit = witnesses(
                window_evidence(
                    data, t, start=obs - 17, end=min(t + 1, obs + 25), vi=old["vi"], ai=old["ai"]
                ),
                offset=offset,
            )
            for key, m in metadata.items():
                if key[0] != obs:
                    continue
                row = m["row"]
                local = int(audit["label"][row])
                label = int(old["ai"][local]) if local >= 0 else local
                if label == -1 and audit["reason"][row] != 3:
                    label = -2
                previous = m["label"]
                if previous == label:
                    continue
                if previous != -2:
                    events.append(
                        dict(
                            step=t,
                            event="retract",
                            observation=obs,
                            visual=key[1],
                            previous=previous,
                            next=label,
                        )
                    )
                m["label"] = label
                if label != -2:
                    events.append(
                        dict(
                            step=t,
                            event="positive" if label >= 0 else "null",
                            observation=obs,
                            visual=key[1],
                            label=label,
                            weight=1.0,
                        )
                    )
        metadata = {k: v for k, v in metadata.items() if t - k[0] <= 24}
    if weak_weight <= 0:
        return events
    basepred = np.full((data.T, data.V), -1, np.int32)
    by_time = {}
    extra = []
    for e in events:
        by_time.setdefault(e["step"], []).append(e)
    for t in range(120, data.T):
        f = frames[t]
        basepred[t] = assign(f, f["base"], data.V)
        if t < 126 or t % 6:
            continue
        obs = t - 6
        old = frames[obs]
        notified = {
            e["visual"]
            for e in by_time.get(t, [])
            if e["observation"] == obs and e["event"] in ["positive", "null"]
        }
        for row, i in enumerate(old["vi"]):
            if int(i) in notified:
                continue
            j = int(basepred[obs, i])
            if (
                j < 0
                or np.sum(basepred[obs : t + 1, i] == j) < 6
                or np.sum(data.consensus[obs : t + 1, i] == j) < 4
            ):
                continue
            columns = np.flatnonzero(old["ai"] == j)
            if not len(columns):
                continue
            c = int(columns[0])
            rivals = old["base"][row].copy()
            chosen = float(rivals[c])
            rivals[c] = -np.inf
            if chosen - max(0.0, float(np.max(rivals, initial=-np.inf))) < 0.5:
                continue
            extra.append(
                dict(
                    step=t,
                    event="positive",
                    observation=obs,
                    visual=int(i),
                    label=j,
                    weight=weak_weight,
                )
            )
    return sorted(events + extra, key=lambda e: (e["step"], e["event"] == "positive"))


def replay_identity(frames, data, events, strength, continuity=0.0):
    by_time = {}
    for e in events:
        by_time.setdefault(e["step"], []).append(e)
    roots = {}
    last = np.full(data.V, -999)
    pixel = np.zeros((data.V, 2))
    reset = np.zeros(data.V, int)
    cache = {}
    basepred = {}
    out = np.full((data.T, data.V), -1, np.int32)
    for t in range(120, data.T):
        f = frames[t]
        visible = np.flatnonzero(data.inputs["video_mask"][t])
        changed = [
            i
            for i in visible
            if last[i] >= 0
            and (
                t - last[i] > 6
                or np.linalg.norm((data.center[t, i] - pixel[i]) / data.scale) > 0.25
            )
        ]
        for i in changed:
            reset[i] = t
            cache.pop(int(i), None)
        roots = {k: v for k, v in roots.items() if t - k[0] <= 36 and k[0] >= reset[k[1]]}
        last[visible] = t
        pixel[visible] = data.center[t, visible]
        votes = {}
        for (obs, i), entry in roots.items():
            j, weight = entry[:2]
            if t - last[i] <= 6:
                group = votes.setdefault(j, {})
                group[i] = group.get(i, 0) + weight * np.exp(-(t - obs) / 24.0)
        scores = f["base"].copy()
        mask = f["candidate"]
        vi = {int(i): r for r, i in enumerate(f["vi"])}
        ai = {int(j): c for c, j in enumerate(f["ai"])}
        owners = {}
        if strength:
            for j, weights in votes.items():
                i, mass = max(weights.items(), key=lambda x: (x[1], -x[0]))
                total = sum(weights.values())
                count = sum((k[1] == i and v[0] == j for k, v in roots.items()))
                if mass < 1 or mass / total < 0.8 or count < 2 or (j not in ai):
                    continue
                row, column = (vi.get(i), ai[j])
                if row is not None and (not mask[row, column]):
                    continue
                support = mass / total * (mass / (mass + 1))
                owners[j] = i
                scores[:, column] -= strength * support
                if row is not None:
                    scores[row, column] += 2 * strength * support
        if continuity:
            cache = {
                i: v
                for i, v in cache.items()
                if t - v[0] <= 36 and t - last[i] <= 6 and (v[0] >= reset[i])
            }
            offset = np.zeros(len(f["vi"]))
            if len(f["ai"]):
                ranks = np.argmax(np.where(mask, scores, -np.inf), 1)
                for row, ii in enumerate(f["vi"]):
                    i = int(ii)
                    if not mask[row].any() or i not in cache:
                        continue
                    obs, j = cache[i]
                    if int(f["ai"][ranks[row]]) != j or owners.get(j, i) != i:
                        continue
                    mass = np.exp(-(t - obs) / 24.0)
                    offset[row] = continuity * mass / (mass + 1)
            scores = scores.astype(np.float64) + offset[:, None]
        out[t] = assign(f, scores, data.V)
        for e in by_time.get(t, []):
            key = (e["observation"], e["visual"])
            if key[0] < reset[key[1]]:
                continue
            old = roots.get(key)
            if e["event"] == "positive":
                roots[key] = (int(e["label"]), float(e.get("weight", 1)))
            elif old is not None:
                if e["event"] == "retract" and e.get("next", -1) >= 0:
                    roots.pop(key)
                elif len(old) == 2:
                    roots[key] = (old[0], old[1] / 2, old[1])
        if continuity:
            basepred[t] = assign(f, f["base"], data.V)
            if t >= 126 and t % 6 == 0:
                obs = t - 6
                for i in frames[obs]["vi"]:
                    if basepred[obs][i] >= 0 and obs >= reset[i]:
                        cache[int(i)] = (obs, int(basepred[obs][i]))
    return out


def match_scene(inputs, model_file, use_continuity=False, progress=print):
    """Infer from sensor fields alone, using a source-trained checkpoint."""
    model_file = Path(model_file)
    with model_file.with_suffix(".json").open() as f:
        settings = json.load(f)
    teachers = []
    for seed in settings["geometry_seeds"]:
        progress(f"  initialize and refine geometry, seed {seed}")
        cfg = Config(seed=seed)
        H, info, anchors = bootstrap(inputs, cfg)
        if H is None:
            raise RuntimeError(f"Unlabeled initialization failed: {info['status']}")
        teachers.append(run_teacher(inputs, H, cfg, anchors))
    selection = select_teacher(inputs, teachers, eta=0.5)
    data = SceneFeatures(inputs, teachers, selection)
    model = GraphMemoryNetwork().eval()
    model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
    model.requires_grad_(False)
    progress("  score observations with historical memory")
    frames = score_frames(model, data, np.asarray(settings["calibration"]))
    progress("  audit evidence and emit identity matches")
    events = audit_events(frames, data, settings["beta"], settings["weak"])
    lam = settings["continuity"] if use_continuity else 0.0
    return replay_identity(frames, data, events, settings["eta"], lam)
