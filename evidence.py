"""Physical counterfactual audits independent of ownership support."""

import numpy as np
from geometry import apply_homography, assignment_with_rejection, _hungarian_rectangular


def window_evidence(data, t, start=None, end=None, vi=None, ai=None):
    inp = data.inputs
    start = max(0, t - 35) if start is None else max(0, start)
    end = t + 1 if end is None else min(t + 1, end)
    vi = np.flatnonzero(inp["video_mask"][t]) if vi is None else np.asarray(vi)
    ai = np.flatnonzero(inp["ais_mask"][t]) if ai is None else np.asarray(ai)
    vm = inp["video_mask"][start:end][:, vi]
    am = inp["ais_mask"][start:end][:, ai]
    p = apply_homography(data.pixel[start:end][:, vi], data.H[t])
    a = data.ais[start:end][:, ai]
    good = (
        vm[:, :, None]
        & am[:, None, :]
        & np.isfinite(p).all(-1)[:, :, None]
        & np.isfinite(a).all(-1)[:, None, :]
    )
    count = good.sum(0)
    delta = p[:, :, None] - a[:, None]
    delta = np.where(good[..., None], delta, np.nan)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmedian(delta, axis=0)
        dist = np.linalg.norm(center, axis=-1)
        spread = np.nanmedian(np.linalg.norm(delta - center[None], axis=-1), axis=0)
        sec = np.asarray(
            (inp["times"][start:end] - inp["times"][t]) / np.timedelta64(1, "s"), float
        )
        x = sec[:, None, None] - np.sum(good * sec[:, None, None], axis=0) / np.maximum(count, 1)
        den = np.sum(good * x * x, axis=0)
        dv = np.sum(
            np.where(good[..., None], p[:, :, None] * x[..., None], 0), axis=0
        ) / np.maximum(den[..., None], 1)
        da = np.sum(np.where(good[..., None], a[:, None] * x[..., None], 0), axis=0) / np.maximum(
            den[..., None], 1
        )
        sv = np.linalg.norm(dv, axis=-1)
        sa = np.linalg.norm(da, axis=-1)
        motion = np.linalg.norm(dv - da, axis=-1)
        cos = np.sum(dv * da, axis=-1) / np.maximum(sv * sa, 1e-06)
    features = np.stack(
        [
            dist / 80,
            spread / 20,
            motion / 2,
            sv / 5,
            sa / 5,
            cos,
            count / 36,
            np.minimum(sv, sa) > 0.5,
        ],
        axis=-1,
    )
    return dict(
        features=np.clip(np.nan_to_num(features, nan=6, posinf=6, neginf=-6), -6, 6).astype(
            np.float32
        ),
        count=count,
        dist=np.nan_to_num(dist, nan=1000000.0),
        spread=np.nan_to_num(spread, nan=1000000.0),
        motion=motion,
        sv=sv,
        sa=sa,
        cos=cos,
        vi=vi,
        ai=ai,
    )


def physical_score(e):
    return 3.0 - e["dist"] / 40.0 - e["spread"] / 8.0 - e["motion"] / 1.5


def global_assign(score, allowed):
    cost = -np.asarray(score, float).copy()
    cost[~allowed] = np.inf
    r, c = assignment_with_rejection(cost, 0.0)
    result = np.full(len(score), -1, int)
    result[r] = c
    return result


def assignment_value(score, allowed, forced_row=None):
    if forced_row is None:
        out = global_assign(score, allowed)
        rows = np.flatnonzero(out >= 0)
        return float(score[rows, out[rows]].sum())
    n, a = score.shape
    if not a or not np.any(allowed[forced_row]):
        return -np.inf
    cost = np.zeros((n, a + n))
    cost[:, :a] = np.where(allowed, -score, 1000000.0)
    cost[forced_row, a:] = 1000000.0
    rows, cols = _hungarian_rectangular(cost)
    where = np.flatnonzero(rows == forced_row)
    if not len(where) or cols[where[0]] >= a or (not allowed[forced_row, cols[where[0]]]):
        return -np.inf
    keep = cols < a
    rows = rows[keep]
    cols = cols[keep]
    return float(score[rows, cols].sum())


def counterfactual_margins(score, allowed, result):
    """Gap to the best assignment that reverses a row's match/null decision."""
    rows = np.flatnonzero(result >= 0)
    opt = float(score[rows, result[rows]].sum())
    gap = np.zeros(len(result))
    for i, j in enumerate(result):
        if j >= 0:
            banned = allowed.copy()
            banned[i, j] = False
            alternative = assignment_value(score, banned)
        else:
            alternative = assignment_value(score, allowed, forced_row=i)
        gap[i] = max(0.0, opt - alternative)
    return gap


def witnesses(e, offset=0.0, margin=0.5):
    score = physical_score(e) + offset
    allowed = (e["count"] >= 8) & (e["dist"] < 160)
    result = global_assign(score, allowed)
    label = np.full(len(result), -2, int)
    quality = np.zeros(len(result))
    reason = np.zeros(len(result), int)
    gap = counterfactual_margins(score, allowed, result)
    for i, j in enumerate(result):
        if j < 0:
            continue
        if (
            allowed[i, j]
            and gap[i] >= margin
            and (e["spread"][i, j] < 10)
            and (e["motion"][i, j] < 1.5)
        ):
            label[i] = j
            quality[i] = min(1.0, gap[i] / 2)
            reason[i] = 1
    owners = {int(j): i for i, j in enumerate(label) if j >= 0}
    for i in range(len(result)):
        if result[i] >= 0 or gap[i] < margin:
            continue
        seen = e["count"][i] >= 8
        if not len(seen) or not np.any(seen):
            continue
        if np.any((e["count"][i] > 0) & ~seen & (e["dist"][i] < 160)):
            continue
        plausible = seen & (score[i] > -margin) & (e["dist"][i] < 160)
        other = np.flatnonzero(plausible)
        if len(other) == 0:
            label[i] = -1
            quality[i] = 0.5
            reason[i] = 2
        elif all(
            (
                int(j) in owners
                and owners[int(j)] != i
                and (score[owners[int(j)], j] - score[i, j] > margin)
                for j in other
            )
        ):
            label[i] = -1
            quality[i] = 0.5
            reason[i] = 3
    return dict(
        label=label,
        quality=quality,
        reason=reason,
        score=score,
        result=result,
        allowed=allowed,
        counterfactual_gap=gap,
    )
