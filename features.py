"""Trajectory, graph and current-map historical-memory features."""

import numpy as np
import torch
from geometry import apply_homography


def compress(sequence, mask):
    n = sequence.shape[1]
    out = np.zeros((n, 18, 8), np.float32)
    length = np.ones(n, np.int64)
    for k in range(n):
        values = sequence[:, k][mask[:, k]]
        length[k] = max(len(values), 1)
        if len(values):
            out[k, : len(values)] = values
    return (out, length)


class SceneFeatures:

    def __init__(self, inputs, teachers, selection):
        allowed = {
            "times",
            "video_xy",
            "video_size",
            "video_mask",
            "ais_xy",
            "ais_speed",
            "ais_cog",
            "ais_mask",
        }
        if set(inputs) != allowed:
            raise ValueError("Only sensor whitelist is accepted")
        self.inputs = inputs
        self.T, self.V = inputs["video_mask"].shape
        self.A = inputs["ais_mask"].shape[1]
        self.predictions = np.stack([r["predictions"] for r in teachers], 0)
        self.hypotheses = np.stack([r["H_trace"] for r in teachers], 1)
        self.teacher = selection["predictions"]
        self.chosen = selection["selected_trace"]
        self.H = np.asarray([self.hypotheses[t, max(int(k), 0)] for t, k in enumerate(self.chosen)])
        self.center = np.asarray(inputs["video_xy"], float).copy()
        self.center[~inputs["video_mask"]] = np.nan
        self.pixel = self.center.copy()
        self.pixel[..., 1] += 0.5 * inputs["video_size"][..., 1]
        a = np.asarray(inputs["ais_xy"], float).copy()
        a[~inputs["ais_mask"]] = np.nan
        self.ais = a
        p = self.center[:120][inputs["video_mask"][:120]]
        self.origin = np.nanmedian(p, axis=0)
        self.scale = np.maximum(
            np.nanpercentile(p, 95, axis=0) - np.nanpercentile(p, 5, axis=0), 20
        )
        self.vx = np.zeros((self.T, self.V, 8), np.float32)
        self.ax = np.zeros((self.T, self.A, 8), np.float32)
        self.vx[..., :2] = (self.center - self.origin) / self.scale
        self.vx[1:, :, 2:4] = (self.center[1:] - self.center[:-1]) / self.scale * 10
        self.vx[..., 4:6] = np.log(np.maximum(inputs["video_size"] / self.scale, 0.0001))
        self.vx[..., 7] = inputs["video_mask"]
        self.ax[1:, :, :2] = (a[1:] - a[:-1]) / 50
        self.ax[..., 2] = np.linalg.norm(self.ax[..., :2], axis=-1)
        self.ax[..., 3] = inputs["ais_speed"] / 10
        cog = np.deg2rad(inputs["ais_cog"])
        self.ax[..., 4] = np.where(np.isfinite(cog), np.sin(cog), 0)
        self.ax[..., 5] = np.where(np.isfinite(cog), np.cos(cog), 0)
        self.ax[..., 7] = inputs["ais_mask"]
        for features, mask in [(self.vx, inputs["video_mask"]), (self.ax, inputs["ais_mask"])]:
            last = np.full(mask.shape[1], -1)
            for t in range(self.T):
                dt = np.where(last >= 0, t - last, 1)
                features[t, :, 6] = dt
                last[mask[t]] = t
        for x, mask in [(self.vx, inputs["video_mask"]), (self.ax, inputs["ais_mask"])]:
            x[:] = np.clip(np.nan_to_num(x, nan=0, posinf=5, neginf=-5), -6, 6)
            x[~mask] = 0
        votes = np.full((self.T, self.V), -1, np.int32)
        for k in range(3):
            ok = (self.predictions[k] >= 0) & (
                (self.predictions == self.predictions[k]).sum(0) >= 2
            )
            votes[ok] = self.predictions[k][ok]
        self.consensus = votes
        self.memory = [[] for _ in range(self.T)]

    def frame(self, t):
        inp = self.inputs
        vi = np.flatnonzero(inp["video_mask"][t])
        ai = np.flatnonzero(inp["ais_mask"][t])
        nv, na = (len(vi), len(ai))
        b = max(0, t - 17)
        vm = inp["video_mask"][b : t + 1][:, vi]
        am = inp["ais_mask"][b : t + 1][:, ai]
        vs, vl = compress(self.vx[b : t + 1][:, vi], vm)
        ass, al = compress(self.ax[b : t + 1][:, ai], am)
        h = self.H[t]
        vp = apply_homography(self.pixel[b : t + 1][:, vi], h)
        ap = self.ais[b : t + 1][:, ai]
        vpos = vp[-1]
        apos = ap[-1]
        common = (
            vm[:, :, None]
            & am[:, None, :]
            & np.isfinite(vp).all(-1)[:, :, None]
            & np.isfinite(ap).all(-1)[:, None, :]
        )
        sec = np.asarray(
            (inp["times"][b : t + 1] - inp["times"][t]) / np.timedelta64(1, "s"), float
        )
        count = common.sum(0)
        x = sec[:, None, None] - np.sum(common * sec[:, None, None], 0) / np.maximum(count, 1)
        den = np.sum(common * x * x, 0)
        vv = np.sum(
            np.where(common[..., None], vp[:, :, None, :] * x[..., None], 0), 0
        ) / np.maximum(den[..., None], 1)
        av = np.sum(
            np.where(common[..., None], ap[:, None, :, :] * x[..., None], 0), 0
        ) / np.maximum(den[..., None], 1)
        speedv = np.linalg.norm(vv, axis=-1)
        speeda = np.linalg.norm(av, axis=-1)
        motion = np.linalg.norm(vv - av, axis=-1)
        cos = np.sum(vv * av, -1) / np.maximum(speedv * speeda, 1e-06)
        delta = vpos[:, None] - apos[None]
        dist = np.linalg.norm(delta, axis=-1)
        distance = np.linalg.norm(vp[:, :, None] - ap[:, None], axis=-1)
        short = common[-6:]
        num = short.sum(0)
        mean = np.sum(np.where(short, distance[-6:], 0), 0) / np.maximum(num, 1)
        std = np.sqrt(
            np.sum(np.where(short, (distance[-6:] - mean) ** 2, 0), 0) / np.maximum(num, 1)
        )
        teacher = (self.teacher[t, vi, None] == ai[None]).astype(float)
        votes = np.mean(self.predictions[:, t, vi, None] == ai[None, None], axis=0)
        projections = np.stack(
            [apply_homography(self.pixel[t, vi], hh) for hh in self.hypotheses[t]]
        )
        disagreement = np.linalg.norm(np.std(projections, axis=0), axis=-1)
        pair = np.stack(
            [
                delta[..., 0] / 80,
                delta[..., 1] / 80,
                dist / 80,
                mean / 80,
                motion / 5,
                speedv / 5,
                speeda / 5,
                cos,
                count / 18,
                std / 80,
                teacher,
                votes,
                np.broadcast_to(disagreement[:, None] / 80, (nv, na)),
                num / 6,
                (mean < 30).astype(float),
                np.ones((nv, na)),
            ],
            -1,
        )
        candidate = np.isfinite(dist) & (dist < 200) & (num >= 2)

        def within(position):
            d = position[:, None] - position[None]
            norm = np.linalg.norm(d, axis=-1)
            return (np.concatenate([d / 100, norm[..., None] / 100], -1), norm < 150)

        vv_edge, vv_mask = within(vpos)
        aa_edge, aa_mask = within(apos)
        mem = self.memory[t]
        mv = []
        ma = []
        mvl = []
        mal = []
        mp = []
        mapos = []
        mi = []
        mj = []
        age = []
        quality = []
        for obs, i, j, confirmed, q in mem:
            bb = max(0, obs - 17)
            z, l = compress(
                self.vx[bb : obs + 1, i : i + 1], inp["video_mask"][bb : obs + 1, i : i + 1]
            )
            mv.append(z[0])
            mvl.append(l[0])
            z, l = compress(
                self.ax[bb : obs + 1, j : j + 1], inp["ais_mask"][bb : obs + 1, j : j + 1]
            )
            ma.append(z[0])
            mal.append(l[0])
            mp.append(apply_homography(self.pixel[obs, i], h))
            mapos.append(self.ais[obs, j])
            mi.append(i)
            mj.append(j)
            age.append((t - obs) / 60)
            quality.append(q)
        nm = len(mem)
        mp = np.asarray(mp).reshape(nm, 2)
        mapos = np.asarray(mapos).reshape(nm, 2)

        def memory_edge(position, old, indices, ids):
            d = np.linalg.norm(position[:, None] - old[None], axis=-1)
            same = indices[:, None] == np.asarray(ids)[None]
            edge = np.stack(
                [
                    d / 80,
                    np.broadcast_to(np.asarray(age)[None], d.shape),
                    np.broadcast_to(np.asarray(quality)[None], d.shape),
                    same.astype(float),
                ],
                -1,
            )
            return (edge, (d < 150) | same)

        ve, vmask = memory_edge(vpos, mp, vi, mi)
        ae, amask = memory_edge(apos, mapos, ai, mj)
        null = np.stack(
            [
                (self.teacher[t, vi] < 0).astype(float),
                np.min(dist, axis=1, initial=400) / 80,
                disagreement / 80,
            ],
            -1,
        )
        arrays = dict(
            vseq=vs,
            vlength=vl,
            aseq=ass,
            alength=al,
            pair=pair,
            candidate=candidate,
            vv_edge=vv_edge,
            vv_mask=vv_mask,
            aa_edge=aa_edge,
            aa_mask=aa_mask,
            vpos=vpos,
            apos=apos,
            null_features=null,
            mvseq=np.asarray(mv).reshape(nm, 18, 8),
            mvlength=np.asarray(mvl, dtype=np.int64),
            maseq=np.asarray(ma).reshape(nm, 18, 8),
            malength=np.asarray(mal, dtype=np.int64),
            vm_edge=ve,
            vm_mask=vmask,
            am_edge=ae,
            am_mask=amask,
        )
        tensor = {}
        for k, value in arrays.items():
            a = np.asarray(value)
            if a.dtype == bool:
                tensor[k] = torch.as_tensor(a.copy(), dtype=torch.bool)
            elif k.endswith("length"):
                tensor[k] = torch.as_tensor(a.copy(), dtype=torch.long)
            else:
                tensor[k] = (
                    torch.as_tensor(
                        np.clip(np.nan_to_num(a, nan=0, posinf=6, neginf=-6), -6, 6).astype(
                            np.float32
                        )
                    )
                    if k not in ("vpos", "apos")
                    else torch.as_tensor(np.nan_to_num(a).astype(np.float32))
                )
        return dict(tensors=tensor, vi=vi, ai=ai, time=t, memory_records=mem)
