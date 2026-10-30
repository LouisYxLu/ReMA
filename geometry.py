"""Unlabeled initialization, guarded mapping refinement and teacher selection.

Every mapping at step t uses only observations available by that step.
No saved target homographies or annotation correspondences are loaded.
"""

from collections import deque
from dataclasses import dataclass
import numpy as np

INPUT_KEYS = (
    "times",
    "video_xy",
    "video_size",
    "video_mask",
    "ais_xy",
    "ais_speed",
    "ais_cog",
    "ais_mask",
)


def apply_homography(points, H):
    """Apply a 3 x 3 transform to (..., 2) points; poles become NaN."""
    points = np.asarray(points, dtype=float)
    H = np.asarray(H, dtype=float)
    if points.ndim < 1 or points.shape[-1] != 2 or H.shape != (3, 3):
        raise ValueError("Expected points (..., 2) and H (3, 3)")
    homogeneous = np.concatenate((points, np.ones(points.shape[:-1] + (1,))), axis=-1)
    projected = homogeneous @ H.T
    denominator_scale = np.sum(np.abs(homogeneous * H[2]), axis=-1)
    valid = np.isfinite(projected).all(axis=-1) & (
        np.abs(projected[..., 2]) > 1e-12 * np.maximum(denominator_scale, np.finfo(float).tiny)
    )
    result = np.full(points.shape, np.nan)
    np.divide(projected[..., :2], projected[..., 2, None], out=result, where=valid[..., None])
    return result


def _points_and_weights(src, dst, weights, minimum):
    src, dst = (np.asarray(src, dtype=float), np.asarray(dst, dtype=float))
    if src.ndim != 2 or src.shape[1:] != (2,) or dst.shape != src.shape:
        raise ValueError("src and dst must have equal (N, 2) shapes")
    weights = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(src),):
        raise ValueError("weights must have shape (N,)")
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    valid &= np.isfinite(weights) & (weights > 0)
    src, dst, weights = (src[valid], dst[valid], weights[valid])
    if len(src) < minimum:
        return None
    return (src, dst, weights / weights.max())


def _normalization(points, weights):
    centroid = np.average(points, axis=0, weights=weights)
    centered = points - centroid
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[-1] <= np.finfo(float).tiny or eigenvalues[0] <= 1e-08 * eigenvalues[-1]:
        return None
    scale = np.sqrt(2.0 / np.trace(covariance))
    transform = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1.0]]
    )
    return (centered * scale, transform)


def _canonical_homography(H):
    if not np.isfinite(H).all():
        return None
    scale = H[2, 2] if abs(H[2, 2]) > 1e-12 * np.linalg.norm(H) else np.linalg.norm(H)
    if scale == 0:
        return None
    return H / scale


def fit_homography(src, dst, weights=None):
    """Weighted, Hartley-normalized DLT; return H or None if underdetermined.

    Nonfinite points and nonpositive/nonfinite weights are excluded. At least
    four points with noncollinear support in both coordinate systems are needed.
    """
    prepared = _points_and_weights(src, dst, weights, 4)
    if prepared is None:
        return None
    src, dst, weights = prepared
    norm_src, norm_dst = (_normalization(src, weights), _normalization(dst, weights))
    if norm_src is None or norm_dst is None:
        return None
    src_n, T_src = norm_src
    dst_n, T_dst = norm_dst
    x, y = src_n.T
    u, v = dst_n.T
    A = np.zeros((2 * len(src), 9))
    A[0::2] = np.column_stack((-x, -y, -np.ones(len(x)), np.zeros((len(x), 3)), u * x, u * y, u))
    A[1::2] = np.column_stack((np.zeros((len(x), 3)), -x, -y, -np.ones(len(x)), v * x, v * y, v))
    A *= np.repeat(np.sqrt(weights), 2)[:, None]
    try:
        _, singular, vh = np.linalg.svd(A, full_matrices=A.shape[0] < 9)
        rank_value = singular[7]
        if rank_value <= singular[0] * 1e-10:
            return None
        H_n = vh[-1].reshape(3, 3)
        h_singular = np.linalg.svd(H_n, compute_uv=False)
        if h_singular[-1] <= h_singular[0] * 1e-12:
            return None
        return _canonical_homography(np.linalg.solve(T_dst, H_n @ T_src))
    except np.linalg.LinAlgError:
        return None


def refine_homography(src, dst, H=None, weights=None, huber_delta=5.0, iterations=5):
    """Approximate IRLS refinement of normalized DLT with destination errors.

    This is an algebraic DLT refit with reprojection-based weights, not nonlinear
    geometric optimization. Use only after robust inlier selection.
    """
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    prepared = _points_and_weights(src, dst, weights, 4)
    if prepared is None:
        return None
    src, dst, weights = prepared
    current = fit_homography(src, dst, weights) if H is None else np.asarray(H, dtype=float)
    if current is None:
        return None
    for _ in range(iterations):
        residuals = np.linalg.norm(apply_homography(src, current) - dst, axis=1)
        robust = np.minimum(1.0, huber_delta / np.maximum(residuals, 1e-12))
        robust[~np.isfinite(robust)] = 0
        candidate = fit_homography(src, dst, weights * robust)
        if candidate is None:
            break
        current = candidate
    return current


def _coverage_ratio(points):
    centered = points - np.mean(points, axis=0)
    values = np.linalg.eigvalsh(centered.T @ centered)
    return float(max(0.0, values[0]) / max(values[-1], np.finfo(float).tiny))


def ransac_homography(src, dst, threshold, rng=None, iterations=200, groups=None):
    """Return ``(H_or_None, inlier_mask, diagnostics)`` using destination errors.

    ``groups`` optionally contains a track/group ID per correspondence. Samples
    and the returned support must contain at least two distinct groups. This
    requirement alone does not make repeated points from a track independent.
    """
    src, dst = (np.asarray(src, dtype=float), np.asarray(dst, dtype=float))
    if src.ndim != 2 or src.shape[1:] != (2,) or dst.shape != src.shape:
        raise ValueError("src and dst must have equal (N, 2) shapes")
    if not np.isfinite(threshold) or threshold <= 0 or iterations < 1:
        raise ValueError("threshold must be positive and iterations >= 1")
    rng = np.random.default_rng(rng)
    mask = np.zeros(len(src), dtype=bool)
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    indices = np.flatnonzero(valid)
    s, d = (src[valid], dst[valid])
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(src),):
            raise ValueError("groups must have shape (N,)")
        groups = groups[valid]
    diagnostic = {
        "status": "insufficient_support",
        "n_input": len(src),
        "n_valid": len(s),
        "n_inliers": 0,
        "iterations": 0,
    }
    if len(s) < 4 or (groups is not None and len(np.unique(groups)) < 2):
        return (None, mask, diagnostic)
    best_H, best_mask, best_score = (None, None, (-1, -np.inf))
    for iteration in range(iterations):
        diagnostic["iterations"] = iteration + 1
        sample = rng.choice(len(s), 4, replace=False)
        if groups is not None and len(np.unique(groups[sample])) < 2:
            continue
        H = fit_homography(s[sample], d[sample])
        if H is None:
            continue
        error = np.linalg.norm(apply_homography(s, H) - d, axis=1)
        inliers = np.isfinite(error) & (error <= threshold)
        if inliers.sum() < 4 or (groups is not None and len(np.unique(groups[inliers])) < 2):
            continue
        if min(_coverage_ratio(s[inliers]), _coverage_ratio(d[inliers])) <= 1e-08:
            continue
        score = (int(inliers.sum()), -float(np.median(error[inliers])))
        if score > best_score:
            best_H, best_mask, best_score = (H, inliers, score)
    if best_H is None:
        diagnostic["status"] = "no_nondegenerate_consensus"
        return (None, mask, diagnostic)
    for _ in range(3):
        candidate = refine_homography(
            s[best_mask], d[best_mask], huber_delta=threshold / 2, iterations=4
        )
        if candidate is None:
            break
        error = np.linalg.norm(apply_homography(s, candidate) - d, axis=1)
        inliers = np.isfinite(error) & (error <= threshold)
        if inliers.sum() < 4 or (groups is not None and len(np.unique(groups[inliers])) < 2):
            break
        if min(_coverage_ratio(s[inliers]), _coverage_ratio(d[inliers])) <= 1e-08:
            break
        if inliers.sum() < best_mask.sum():
            break
        unchanged = np.array_equal(inliers, best_mask)
        best_H, best_mask = (candidate, inliers)
        if unchanged:
            break
    error = np.linalg.norm(apply_homography(s, best_H) - d, axis=1)
    mask[indices[best_mask]] = True
    diagnostic.update(
        status="ok",
        n_inliers=int(mask.sum()),
        inlier_fraction=float(best_mask.mean()),
        rmse=float(np.sqrt(np.mean(error[best_mask] ** 2))),
        median_error=float(np.median(error[best_mask])),
        source_coverage_ratio=_coverage_ratio(s[best_mask]),
        destination_coverage_ratio=_coverage_ratio(d[best_mask]),
        n_groups=None if groups is None else len(np.unique(groups[best_mask])),
    )
    return (best_H, mask, diagnostic)


def _hungarian_rectangular(cost):
    """Dense Hungarian shortest augmenting path; requires finite n <= m."""
    n, m = cost.shape
    if n > m or not np.isfinite(cost).all():
        raise ValueError("Hungarian input must be finite and n <= m")
    u, v = (np.zeros(n + 1), np.zeros(m + 1))
    p, way = (np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int))
    for row in range(1, n + 1):
        p[0] = row
        col = 0
        minimum, used = (np.full(m + 1, np.inf), np.zeros(m + 1, dtype=bool))
        while True:
            used[col] = True
            active_row = p[col]
            candidates = np.flatnonzero(~used[1:]) + 1
            reduced = cost[active_row - 1, candidates - 1] - u[active_row] - v[candidates]
            improves = reduced < minimum[candidates]
            improved_cols = candidates[improves]
            minimum[improved_cols] = reduced[improves]
            way[improved_cols] = col
            new_col = candidates[np.argmin(minimum[candidates])]
            delta = minimum[new_col]
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            col = new_col
            if p[col] == 0:
                break
        while True:
            previous = way[col]
            p[col] = p[previous]
            col = previous
            if col == 0:
                break
    rows = np.flatnonzero(p[1:] > 0)
    assigned_rows, assigned_cols = (p[1:][rows] - 1, rows)
    order = np.argsort(assigned_rows)
    return (assigned_rows[order], assigned_cols[order])


def assignment_with_rejection(cost, rejection_cost=1.0):
    """Return matched ``(row_indices, column_indices)`` with free rejection.

    The objective is sum(matched costs) + rejection_cost/2 for EACH unmatched
    node on EITHER side. Nonfinite edges and edges >= rejection_cost are banned.
    A rectangular dummy formulation differs from this objective by a constant:
    row-dummy cost rejection_cost and implicit unused columns give the same
    optimum. Thus the returned matches need not have maximum cardinality.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D matrix")
    if not np.isfinite(rejection_cost) or rejection_cost < 0:
        raise ValueError("rejection_cost must be finite and nonnegative")
    n, m = cost.shape
    if not n or not m:
        return (np.array([], dtype=int), np.array([], dtype=int))
    transposed = n > m
    oriented = cost.T if transposed else cost
    n_rows, n_cols = oriented.shape
    allowed = np.isfinite(oriented) & (oriented < rejection_cost)
    penalty = rejection_cost + max(1.0, abs(rejection_cost))
    augmented = np.full((n_rows, n_cols + n_rows), rejection_cost)
    augmented[:, :n_cols] = np.where(allowed, oriented, penalty)
    solver = _hungarian_rectangular
    rows, cols = solver(augmented)
    keep = cols < n_cols
    rows, cols = (rows[keep], cols[keep])
    keep = allowed[rows, cols]
    rows, cols = (rows[keep], cols[keep])
    if transposed:
        rows, cols = (cols, rows)
    order = np.argsort(rows)
    return (rows[order], cols[order])


def geometric_refine(src, dst, H=None, weights=None, huber=15.0, iterations=30):
    src, dst = (np.asarray(src, float), np.asarray(dst, float))
    if src.ndim != 2 or src.shape[1:] != (2,) or dst.shape != src.shape:
        raise ValueError("Source and target must have equal (N,2) shapes")
    if huber <= 0:
        raise ValueError("Huber scale must be positive")
    weights = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    if weights.shape != (len(src),):
        raise ValueError("Weights must have shape (N,)")
    valid = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    valid &= np.isfinite(weights) & (weights > 0)
    src, dst = (src[valid], dst[valid])
    weights = weights[valid]
    if len(src) < 4:
        return None
    for points in (src, dst):
        centered = points - np.average(points, axis=0, weights=weights)
        values = np.linalg.eigvalsh((centered * weights[:, None]).T @ centered / weights.sum())
        if values[-1] <= np.finfo(float).tiny or values[0] <= 1e-08 * values[-1]:
            return None
    H = fit_homography(src, dst, weights) if H is None else H.copy()
    if H is None or np.asarray(H).shape != (3, 3) or (not np.isfinite(H).all()):
        return None
    sc, dc = (np.mean(src, 0), np.mean(dst, 0))
    ss = max(np.sqrt(np.mean((src - sc) ** 2)), 1.0)
    ds = max(np.sqrt(np.mean((dst - dc) ** 2)), 1.0)
    S = np.array([[1 / ss, 0, -sc[0] / ss], [0, 1 / ss, -sc[1] / ss], [0, 0, 1.0]])
    D = np.array([[1 / ds, 0, -dc[0] / ds], [0, 1 / ds, -dc[1] / ds], [0, 0, 1.0]])
    x, y = (src - sc).T / ss
    target = (dst - dc) / ds
    current = D @ H @ np.linalg.inv(S)
    if abs(current[2, 2]) <= 1e-12 * np.linalg.norm(current):
        return None
    current /= current[2, 2]
    delta = huber / ds

    def loss(h):
        q = apply_homography(np.c_[x, y], h)
        r = np.linalg.norm(q - target, axis=1)
        return np.sum(weights * np.where(r < delta, 0.5 * r * r, delta * (r - 0.5 * delta)))

    damping = 0.001
    for _ in range(iterations):
        q = apply_homography(np.c_[x, y], current)
        residual = q - target
        norm = np.linalg.norm(residual, axis=1)
        if not np.isfinite(norm).all():
            break
        w = weights * np.minimum(1.0, delta / np.maximum(norm, 1e-12))
        denominator = current[2, 0] * x + current[2, 1] * y + 1
        z = np.zeros_like(x)
        one = np.ones_like(x)
        J = np.empty((2 * len(src), 8))
        J[0::2] = np.c_[x, y, one, z, z, z, -q[:, 0] * x, -q[:, 0] * y] / denominator[:, None]
        J[1::2] = np.c_[z, z, z, x, y, one, -q[:, 1] * x, -q[:, 1] * y] / denominator[:, None]
        jw = J * np.sqrt(np.repeat(w, 2))[:, None]
        rw = residual.ravel() * np.sqrt(np.repeat(w, 2))
        gram = jw.T @ jw
        try:
            step = np.linalg.solve(
                gram + damping * np.diag(np.maximum(np.diag(gram), 1e-06)), -jw.T @ rw
            )
        except np.linalg.LinAlgError:
            break
        candidate = np.r_[current.ravel()[:8] + step, 1.0].reshape(3, 3)
        if loss(candidate) < loss(current):
            current = candidate
            damping = max(damping / 3, 1e-09)
            if np.linalg.norm(step) < 1e-09:
                break
        else:
            damping *= 10
            if damping > 10000000000.0:
                break
    return np.linalg.solve(D, current @ S)


class SpatialTrustRegion:
    """Bound cumulative displacement on every finite historical visual point.

    The reference is the initial, provisional H: comparing only adjacent H
    versions would allow many small updates to accumulate into large drift.
    Candidate activation must call allow() again, since new image locations may
    have appeared during prospective validation.
    """

    def __init__(self, initial_H, maximum_shift_m=40.0):
        self.reference = np.asarray(initial_H, float).copy()
        self.maximum_shift_m = float(maximum_shift_m)
        self._controls = {}
        self._source = None
        self._reference_projection = None
        self.last_maximum_shift_m = 0.0

    def observe(self, points):
        points = np.asarray(points, float).reshape(-1, 2)
        changed = False
        for point in points[np.isfinite(points).all(axis=1)]:
            key = tuple(point)
            if key not in self._controls:
                self._controls[key] = point.copy()
                changed = True
        if changed:
            self._source = None
            self._reference_projection = None

    def allow(self, candidate_H):
        if not self._controls:
            return False
        if self._source is None:
            self._source = np.asarray(list(self._controls.values()))
            self._reference_projection = apply_homography(self._source, self.reference)
        target = apply_homography(self._source, candidate_H)
        usable = np.isfinite(self._reference_projection).all(axis=1)
        if not usable.any():
            return False
        shift = np.linalg.norm(target[usable] - self._reference_projection[usable], axis=1)
        self.last_maximum_shift_m = float(np.max(shift))
        return bool(np.isfinite(shift).all() and self.last_maximum_shift_m <= self.maximum_shift_m)

    @property
    def control_points(self):
        return len(self._controls)


@dataclass
class Config:
    seed: int = 20261013
    warmup_seconds: int = 600
    window: int = 6
    motion_window: int = 18
    stationary_speed_mps: float = 0.2
    moving_speed_mps: float = 1.5
    direction_speed_mps: float = 0.7
    reference_height_fraction: float = 0.5
    bootstrap_hypotheses: int = 10000
    bootstrap_gate_m: float = 80.0
    gate_m: float = 80.0
    min_overlap: int = 3
    confirmation_steps: int = 3
    pseudo_max_distance_m: float = 30.0
    pseudo_margin: float = 0.12
    geometry_every: int = 24
    geometry_total_shift_m: float = 40.0
    validate_steps: int = 12
    memory_points: int = 800


def prepare_observations(inputs, height_fraction=0.0):
    """Mask stale values and derive the geometric point without changing inputs.

    A fraction of 0.5 moves the annotated box centre to its bottom midpoint.
    This is only an approximate water-surface reference, not an AIS antenna.
    """
    if set(inputs) != set(INPUT_KEYS):
        raise ValueError("Expected sensor inputs only")
    result = dict(inputs)
    result["video_xy"] = np.array(inputs["video_xy"], dtype=float, copy=True)
    result["video_xy"][..., 1] += height_fraction * inputs["video_size"][..., 1]
    result["video_xy"][~inputs["video_mask"]] = np.nan
    result["ais_xy"] = np.array(inputs["ais_xy"], dtype=float, copy=True)
    result["ais_xy"][~inputs["ais_mask"]] = np.nan
    return result


def bootstrap(inputs, cfg):
    """Coarse centre-track association, then robust water-reference fitting.

    Both stages use only the warmup. Associations remain provisional. No labels
    select anchors, geometry or the image reference point.
    """
    centres = prepare_observations(inputs)
    H, info, memory = _bootstrap_centres(centres, cfg)
    if H is None:
        return (H, info, memory)
    water = prepare_observations(inputs, cfg.reference_height_fraction)
    memory = [(t, i, j, water["video_xy"][t, i].copy(), dst) for t, i, j, src, dst in memory]
    groups = np.asarray([m[1] for m in memory])
    if len(np.unique(groups)) < 3:
        info.update(status="insufficient_water_reference_support")
        return (None, info, [])
    source = np.asarray([m[3] for m in memory])
    target = np.asarray([m[4] for m in memory])
    weights = np.asarray([1 / np.sum(groups == g) for g in groups])
    refined = geometric_refine(
        source, target, H if cfg.reference_height_fraction == 0 else None, weights, huber=15.0
    )
    if refined is None or not np.isfinite(apply_homography(source, refined)).all():
        info.update(status="water_reference_fit_failed")
        return (None, info, [])
    info.update(
        reference_height_fraction=cfg.reference_height_fraction,
        anchor_tracks=len(np.unique(groups)),
        water_reference_anchor_median_m=float(
            np.median(np.linalg.norm(apply_homography(source, refined) - target, axis=1))
        ),
        geometry_method="Centre-track coarse association; robust metric bottom-reference refinement.",
    )
    return (refined, info, memory)


def warmup_count(inputs, cfg):
    elapsed = (inputs["times"] - inputs["times"][0]) / np.timedelta64(1, "s")
    return int(np.sum(elapsed < cfg.warmup_seconds))


def _span(points):
    good = points[np.isfinite(points).all(axis=1)]
    return 0.0 if len(good) < 2 else float(np.linalg.norm(np.ptp(good, axis=0)))


def _progress(points):
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(distances)]
    return cumulative / max(cumulative[-1], 1e-06)


def _greedy(cost, cutoff):
    matches, used_r, used_c = ([], set(), set())
    for flat in np.argsort(cost, axis=None):
        r, c = np.unravel_index(flat, cost.shape)
        if not np.isfinite(cost[r, c]) or cost[r, c] >= cutoff:
            break
        if r not in used_r and c not in used_c:
            matches.append((int(r), int(c)))
            used_r.add(r)
            used_c.add(c)
    return matches


def _bootstrap_centres(inputs, cfg):
    """Multiple two-trajectory geometric hypotheses, using warmup only.

    Temporal progress ranks hypotheses; trajectory birth/death equality and ID
    strings are not features. All candidates are scored jointly across ships.
    A returned model is a provisional hypothesis, not a certified calibration.
    """
    rng = np.random.default_rng(cfg.seed)
    n = warmup_count(inputs, cfg)
    v, a = (inputs["video_xy"][:n], inputs["ais_xy"][:n])
    vm, am = (inputs["video_mask"][:n], inputs["ais_mask"][:n])
    moving_v = np.array(
        [i for i in range(v.shape[1]) if vm[:, i].sum() >= 12 and _span(v[:, i]) > 25], int
    )
    moving_a = np.array(
        [i for i in range(a.shape[1]) if am[:, i].sum() >= 12 and _span(a[:, i]) > 50], int
    )
    info = dict(
        status="insufficient_evidence",
        warmup_steps=n,
        moving_video_tracks=len(moving_v),
        moving_ais_tracks=len(moving_a),
        hypotheses_attempted=0,
        hypotheses_valid=0,
    )
    if len(moving_v) < 3 or len(moving_a) < 3:
        return (None, info, [])
    pair_options = []
    for i in moving_v:
        options = []
        for j in moving_a:
            ts = np.flatnonzero(vm[:, i] & am[:, j])
            if len(ts) < 12 or _span(v[ts, i]) < 25 or _span(a[ts, j]) < 50:
                continue
            samples = ts[np.linspace(0, len(ts) - 1, min(14, len(ts)), dtype=int)]
            error = float(np.mean(np.abs(_progress(v[samples, i]) - _progress(a[samples, j]))))
            options.append((error, int(i), int(j), samples))
        pair_options.extend(sorted(options, key=lambda x: x[0]))
    if len(pair_options) < 3:
        return (None, info, [])
    snapshots = np.linspace(0, n - 1, min(12, n), dtype=int)
    vv, aa = (v[snapshots][:, moving_v], a[snapshots][:, moving_a])
    common = vm[snapshots][:, moving_v, None] & am[snapshots][:, None, moving_a]
    count = common.sum(axis=0)

    def score(H, exact=False):
        projected = apply_homography(vv, H)
        residual = np.linalg.norm(projected[:, :, None] - aa[:, None], axis=-1)
        valid = common & np.isfinite(residual)
        mean = np.sum(np.where(valid, np.minimum(residual, 4 * cfg.bootstrap_gate_m), 0), axis=0)
        mean /= np.maximum(valid.sum(axis=0), 1)
        mean[(count < 3) | (valid.sum(axis=0) != count)] = np.inf
        if exact:
            rr, cc = assignment_with_rejection(mean, cfg.bootstrap_gate_m)
            pairs = list(zip(rr, cc))
        else:
            pairs = _greedy(mean, cfg.bootstrap_gate_m)
        benefit = sum((1 - mean[i, j] / cfg.bootstrap_gate_m for i, j in pairs))
        return (
            float(benefit),
            [(int(moving_v[i]), int(moving_a[j]), float(mean[i, j])) for i, j in pairs],
        )

    candidates = []
    for trial in range(cfg.bootstrap_hypotheses):
        info["hypotheses_attempted"] += 1
        x, y = rng.choice(len(pair_options), 2, replace=False)
        _, i1, j1, ts1 = pair_options[x]
        _, i2, j2, ts2 = pair_options[y]
        if i1 == i2 or j1 == j2:
            continue
        source = np.concatenate([v[ts1, i1], v[ts2, i2]])
        dest = np.concatenate([a[ts1, j1], a[ts2, j2]])
        H = fit_homography(source, dest)
        if H is None:
            continue
        info["hypotheses_valid"] += 1
        benefit, pairs = score(H)
        if len(pairs) >= 3:
            candidates.append((benefit, H))
    if not candidates:
        return (None, info, [])
    best = None
    for _, H in sorted(candidates, key=lambda x: x[0], reverse=True)[:12]:
        benefit, pairs = score(H, exact=True)
        if best is None or benefit > best[0]:
            best = (benefit, H, pairs)
    benefit, H, pairs = best
    for _ in range(2):
        src, dst, groups = ([], [], [])
        for i, j, error in pairs:
            if error > cfg.bootstrap_gate_m * 0.65:
                continue
            ts = np.flatnonzero(vm[:, i] & am[:, j])
            ts = ts[np.linspace(0, len(ts) - 1, min(10, len(ts)), dtype=int)]
            src.extend(v[ts, i])
            dst.extend(a[ts, j])
            groups.extend([i] * len(ts))
        if len(set(groups)) < 3:
            break
        candidate, mask, diag = ransac_homography(
            np.asarray(src),
            np.asarray(dst),
            threshold=25.0,
            rng=rng,
            iterations=250,
            groups=np.asarray(groups),
        )
        if candidate is None:
            break
        new_benefit, new_pairs = score(candidate, exact=True)
        if new_benefit < benefit:
            break
        benefit, H, pairs = (new_benefit, candidate, new_pairs)
    errors = [p[2] for p in pairs]
    info.update(
        status="provisional" if len(pairs) >= 3 else "insufficient_evidence",
        supported_tracks=len(pairs),
        unsupervised_benefit=benefit,
        median_association_residual_m=float(np.median(errors)),
        note="Confidence is internal consistency, not measured ground-truth accuracy.",
    )
    if len(pairs) < 3:
        return (None, info, [])
    memory = []
    for i, j, error in pairs:
        if error > cfg.pseudo_max_distance_m:
            continue
        ts = np.flatnonzero(vm[:, i] & am[:, j])
        for t in ts[np.linspace(0, len(ts) - 1, min(8, len(ts)), dtype=int)]:
            memory.append((int(t), int(i), int(j), v[t, i].copy(), a[t, j].copy()))
    return (H, info, memory)


def _teacher_features(inputs, t, H, cfg, previous):
    """Short-window location plus longer-window velocity evidence.

    Velocity is a least-squares slope on shared past timestamps. This suppresses
    five-second center jitter without forcing a long history onto location
    matching. Opposing motion and clear stationary/moving disagreement are
    explicit rejection evidence; neither proves that AIS is absent.
    """
    vi = np.flatnonzero(inputs["video_mask"][t])
    ai = np.flatnonzero(inputs["ais_mask"][t])
    if not len(vi) or not len(ai):
        return (vi, ai, None, None, None)
    start = max(0, t - cfg.window + 1)
    v = inputs["video_xy"][start : t + 1][:, vi]
    a = inputs["ais_xy"][start : t + 1][:, ai]
    pv = apply_homography(v, H)
    mask = (
        np.isfinite(pv).all(-1)[:, :, None]
        & np.isfinite(a).all(-1)[:, None, :]
        & inputs["video_mask"][start : t + 1][:, vi, None]
        & inputs["ais_mask"][start : t + 1][:, None, ai]
    )
    distances = np.linalg.norm(pv[:, :, None] - a[:, None], axis=-1)
    counts = mask.sum(0)
    mean = np.sum(np.where(mask, distances, 0), axis=0) / np.maximum(counts, 1)
    current = distances[-1]
    mm = mask[1:] & mask[:-1]
    var = np.sum(np.where(mask, (distances - mean) ** 2, 0), axis=0) / np.maximum(counts, 1)
    motion_start = max(0, t - cfg.motion_window + 1)
    mpv = apply_homography(inputs["video_xy"][motion_start : t + 1][:, vi], H)
    ma = inputs["ais_xy"][motion_start : t + 1][:, ai]
    common = (
        np.isfinite(mpv).all(-1)[:, :, None]
        & np.isfinite(ma).all(-1)[:, None, :]
        & inputs["video_mask"][motion_start : t + 1][:, vi, None]
        & inputs["ais_mask"][motion_start : t + 1][:, None, ai]
    )
    seconds = np.asarray(
        (inputs["times"][motion_start : t + 1] - inputs["times"][t]) / np.timedelta64(1, "s"), float
    )
    num = common.sum(0)
    centered = seconds[:, None, None] - np.sum(
        common * seconds[:, None, None], axis=0
    ) / np.maximum(num, 1)
    denominator = np.sum(common * centered**2, axis=0)
    video_velocity = np.sum(
        np.where(common[:, :, :, None], centered[:, :, :, None] * mpv[:, :, None, :], 0), axis=0
    ) / np.maximum(denominator[:, :, None], 1)
    ais_velocity = np.sum(
        np.where(common[:, :, :, None], centered[:, :, :, None] * ma[:, None, :, :], 0), axis=0
    ) / np.maximum(denominator[:, :, None], 1)
    video_speed = np.linalg.norm(video_velocity, axis=-1)
    ais_speed = np.linalg.norm(ais_velocity, axis=-1)
    motion = np.linalg.norm(video_velocity - ais_velocity, axis=-1)
    direction_cosine = np.sum(video_velocity * ais_velocity, axis=-1) / np.maximum(
        video_speed * ais_speed, 1e-06
    )
    continuity = (previous[vi, None] == ai[None]).astype(float)
    known_previous = (previous[vi, None] >= 0).astype(float)
    feat = np.stack(
        [
            np.ones_like(mean),
            np.minimum(mean / cfg.gate_m, 3),
            np.minimum(current / cfg.gate_m, 3),
            np.minimum(motion / 3, 3),
            np.minimum(np.sqrt(var) / cfg.gate_m, 3),
            continuity,
            np.broadcast_to(known_previous, mean.shape),
            np.minimum(counts / cfg.window, 1),
        ],
        axis=-1,
    )
    feat = np.nan_to_num(feat, nan=3.0, posinf=3.0)
    cost = 0.6 * mean / cfg.gate_m + 0.25 * np.minimum(motion / 4, 2) + 0.1 * current / cfg.gate_m
    cost += 0.05 * known_previous * (1 - continuity)
    invalid = (
        (counts < cfg.min_overlap) | (mm.sum(0) < 1) | ~np.isfinite(mean) | ~np.isfinite(current)
    )
    invalid |= (mean > cfg.gate_m) | (current > cfg.gate_m * 1.5)
    invalid |= (
        (video_speed > cfg.direction_speed_mps)
        & (ais_speed > cfg.direction_speed_mps)
        & (direction_cosine < 0)
    )
    invalid |= (video_speed < cfg.stationary_speed_mps) & (ais_speed > cfg.moving_speed_mps)
    invalid |= (ais_speed < cfg.stationary_speed_mps) & (video_speed > cfg.moving_speed_mps)
    cost[invalid] = np.inf
    return (vi, ai, feat, cost, mean)


def run_teacher(inputs, H_init, cfg, initial_memory=()):
    if set(inputs) != set(INPUT_KEYS):
        raise ValueError("Expected sensor inputs only")
    inputs = prepare_observations(inputs, cfg.reference_height_fraction)
    rng = np.random.default_rng(cfg.seed + 17)
    T, V = inputs["video_mask"].shape
    start = warmup_count(inputs, cfg)
    predictions = np.full((T, V), -1, dtype=np.int32)
    locations = np.full((T, V, 2), np.nan)
    H_trace = np.full((T, 3, 3), np.nan)
    H = None if H_init is None else H_init.copy()
    trust = None if H is None else SpatialTrustRegion(H, cfg.geometry_total_shift_m)
    if trust is not None:
        trust.observe(inputs["video_xy"][:start].reshape(-1, 2))
    previous = np.full(V, -1, int)
    streak = np.zeros(V, int)
    recent = deque(maxlen=cfg.memory_points // 2)
    anchors = list(initial_memory)[: cfg.memory_points // 2]
    pending_H = None
    rollback = None
    events = []
    counters = dict(
        pseudo_positive=0,
        pseudo_negative=0,
        geometry_proposed=0,
        geometry_accepted=0,
        geometry_rejected=0,
        geometry_rolled_back=0,
        geometry_guard_rejected=0,
    )
    if H is None:
        return dict(
            predictions=predictions,
            locations=locations,
            H_trace=H_trace,
            events=events,
            counters=counters,
            status="bootstrap_failed",
        )
    for t in range(start, T):
        trust.observe(inputs["video_xy"][t])
        H_trace[t] = H
        locations[t] = apply_homography(inputs["video_xy"][t], H)
        vi, ai, feat, base, distance = _teacher_features(inputs, t, H, cfg, previous)
        if base is None:
            previous[:] = -1
            streak[:] = 0
            continue
        cost = base.copy()
        rr, cc = assignment_with_rejection(cost, rejection_cost=0.85)
        current = np.full(V, -1, int)
        for r, c in zip(rr, cc):
            current[vi[r]] = ai[c]
        predictions[t] = current
        streak = np.where(
            (current == previous) & (current >= 0), streak + 1, np.where(current >= 0, 1, 0)
        )
        confident = []
        for r, c in zip(rr, cc):
            row = np.delete(base[r], c)
            col = np.delete(base[:, c], r)
            alternate = min(np.min(row, initial=np.inf), np.min(col, initial=np.inf), 0.85)
            if (
                streak[vi[r]] >= cfg.confirmation_steps
                and distance[r, c] < cfg.pseudo_max_distance_m
                and (alternate - base[r, c] > cfg.pseudo_margin)
            ):
                confident.append((int(r), int(c)))
        previous = current
        fresh = []
        for r, c in confident:
            i, j = (int(vi[r]), int(ai[c]))
            src = inputs["video_xy"][t, i].copy()
            dst = inputs["ais_xy"][t, j].copy()
            fresh.append((i, src, dst))
            if t % 3 == 0:
                recent.append((t, i, j, src, dst))
        if pending_H is not None and t > pending_H["time"]:
            for i, src, dst in fresh:
                pending_H["old"].append(float(np.linalg.norm(apply_homography(src, H) - dst)))
                pending_H["new"].append(
                    float(np.linalg.norm(apply_homography(src, pending_H["H"]) - dst))
                )
                pending_H["groups"].add(i)
            if t - pending_H["time"] >= cfg.validate_steps:
                old = np.array(pending_H["old"])
                new = np.array(pending_H["new"])
                accept = len(old) >= 12 and len(pending_H["groups"]) >= 3 and np.isfinite(new).all()
                if accept:
                    accept = np.median(new) < 0.98 * np.median(old) and np.quantile(
                        new, 0.9
                    ) <= 1.05 * np.quantile(old, 0.9)
                if accept and (not trust.allow(pending_H["H"])):
                    accept = False
                    counters["geometry_guard_rejected"] += 1
                if accept:
                    rollback = dict(H=H.copy(), time=t, old=[], new=[], groups=set())
                    H = pending_H["H"]
                    counters["geometry_accepted"] += 1
                    events.append(
                        dict(
                            step=t,
                            event="geometry_accept",
                            validation_pairs=len(old),
                            old_median=float(np.median(old)),
                            new_median=float(np.median(new)),
                            maximum_historical_shift_m=trust.last_maximum_shift_m,
                            historical_control_points=trust.control_points,
                        )
                    )
                else:
                    counters["geometry_rejected"] += 1
                pending_H = None
        if rollback is not None and t > rollback["time"]:
            for i, src, dst in fresh:
                rollback["old"].append(
                    float(np.linalg.norm(apply_homography(src, rollback["H"]) - dst))
                )
                rollback["new"].append(float(np.linalg.norm(apply_homography(src, H) - dst)))
                rollback["groups"].add(i)
            if t - rollback["time"] >= cfg.validate_steps:
                old = np.asarray(rollback["old"])
                new = np.asarray(rollback["new"])
                if (
                    len(old) >= 12
                    and len(rollback["groups"]) >= 3
                    and (np.median(new) > 1.15 * np.median(old))
                ):
                    H = rollback["H"]
                    counters["geometry_rolled_back"] += 1
                    events.append(dict(step=t, event="geometry_rollback"))
                rollback = None
        if (
            t % cfg.geometry_every == 0
            and pending_H is None
            and (rollback is None)
            and (len(recent) >= 30)
        ):
            pool = anchors + list(recent)
            chosen = []
            for group in sorted(set((x[1] for x in pool))):
                rows = [x for x in pool if x[1] == group]
                chosen.extend(
                    (rows[k] for k in np.linspace(0, len(rows) - 1, min(12, len(rows)), dtype=int))
                )
            if len(set((x[1] for x in chosen))) >= 3:
                src = np.array([x[3] for x in chosen])
                dst = np.array([x[4] for x in chosen])
                groups = np.array([x[1] for x in chosen])
                candidate, inlier, diag = ransac_homography(src, dst, 20.0, rng, 160, groups)
                if candidate is not None:
                    inlier_groups = groups[inlier]
                    weights = np.asarray([1 / np.sum(inlier_groups == g) for g in inlier_groups])
                    refined = geometric_refine(
                        src[inlier], dst[inlier], candidate, weights, huber=15.0
                    )
                    if refined is not None:
                        candidate = refined
                if candidate is not None and (not trust.allow(candidate)):
                    counters["geometry_guard_rejected"] += 1
                    candidate = None
                if candidate is not None:
                    shifts = np.linalg.norm(
                        apply_homography(src, candidate) - apply_homography(src, H), axis=1
                    )
                    if np.isfinite(shifts).all() and np.quantile(shifts, 0.9) < 40:
                        pending_H = dict(H=candidate, time=t, old=[], new=[], groups=set())
                        counters["geometry_proposed"] += 1
    return dict(predictions=predictions, locations=locations, H_trace=H_trace)


def majority(predictions):
    stack = np.asarray(predictions)
    out = np.full(stack.shape[1:], -1, dtype=np.int32)
    for k in range(len(stack)):
        accepted = (stack[k] >= 0) & (np.sum(stack == stack[k], axis=0) > len(stack) // 2)
        out[accepted] = stack[k][accepted]
    return out


def select_teacher(inputs, results, eta=0.5):
    T, V = inputs["video_mask"].shape
    votes = majority([r["predictions"] for r in results])
    log_weight = np.zeros(len(results))
    pred = np.full((T, V), -1, int)
    loc = np.full((T, V, 2), np.nan)
    weight = np.full((T, len(results)), np.nan)
    which = np.full(T, -1, int)
    events = []
    for t in range(120, T):
        weights = np.exp(log_weight - np.max(log_weight))
        weights /= weights.sum()
        weight[t] = weights
        k = int(np.argmax(weights))
        which[t] = k
        pred[t] = results[k]["predictions"][t]
        loc[t] = results[k]["locations"][t]
        rows = []
        targets = []
        for i in np.flatnonzero(inputs["video_mask"][t]):
            j = int(votes[t, i])
            if j < 0 or not np.all(votes[t - 2 : t + 1, i] == j):
                continue
            b = max(0, t - 17)
            a = inputs["ais_xy"][b : t + 1, j]
            valid = inputs["ais_mask"][b : t + 1, j] & np.isfinite(a).all(1)
            if valid.sum() < 6:
                continue
            sec = np.asarray(
                (inputs["times"][b : t + 1] - inputs["times"][t]) / np.timedelta64(1, "s"), float
            )[valid]
            x = sec - sec.mean()
            velocity = np.sum(a[valid] * x[:, None], axis=0) / max(float(x @ x), 1.0)
            if np.linalg.norm(velocity) > 0.5:
                rows.append(i)
                targets.append(inputs["ais_xy"][t, j])
        if len(rows) < 2:
            continue
        estimates = np.stack([r["locations"][t, rows] for r in results])
        error = np.linalg.norm(estimates - np.asarray(targets)[None], axis=-1)
        loss = np.mean(np.minimum(np.nan_to_num(error, nan=80.0, posinf=80.0) / 80.0, 1), axis=1)
        log_weight -= eta * loss
        events.append(dict(step=t, loss=loss.tolist(), probe_tracks=len(rows)))
    return dict(
        predictions=pred, locations=loc, weight_trace=weight, selected_trace=which, events=events
    )
