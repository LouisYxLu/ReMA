"""Frozen source-trained GRUs, memory attention, graph layers and scoring heads."""

import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def mlp(a, b, c):
    return nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, c))


def masked_softmax(score, mask):
    out = torch.softmax(score.masked_fill(~mask, -10000.0), dim=-1) * mask.to(score.dtype)
    return out / out.sum(-1, keepdim=True).clamp_min(1e-08)


class RelationAttention(nn.Module):

    def __init__(self, d, edge):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.bias = mlp(edge, 16, 1)
        self.norm = nn.LayerNorm(d)

    def forward(self, query, source, edge, mask):
        if not len(query) or not len(source):
            return query
        score = self.q(query) @ self.k(source).T / math.sqrt(query.shape[-1]) + self.bias(
            edge
        ).squeeze(-1)
        attention = masked_softmax(score, mask)
        return self.norm(query + attention @ self.v(source))


class GraphLayer(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.vv = RelationAttention(d, 3)
        self.aa = RelationAttention(d, 3)
        self.va = RelationAttention(d, 16)
        self.av = RelationAttention(d, 16)

    def forward(self, v, a, f):
        v = self.vv(v, v, f["vv_edge"], f["vv_mask"])
        a = self.aa(a, a, f["aa_edge"], f["aa_mask"])
        v1 = self.va(v, a, f["pair"], f["candidate"])
        a1 = self.av(a, v, f["pair"].transpose(0, 1), f["candidate"].T)
        return (v1, a1)


class GraphMemoryNetwork(nn.Module):

    def __init__(self, d=32, use_graph=True, use_memory=True, use_gru=True):
        super().__init__()
        self.d = d
        self.use_graph = use_graph
        self.use_memory = use_memory
        self.use_gru = use_gru
        self.visual = nn.GRU(8, d, batch_first=True)
        self.ais = nn.GRU(8, d, batch_first=True)
        self.memory_content = mlp(2 * d, 2 * d, d)
        self.vm = RelationAttention(d, 4)
        self.am = RelationAttention(d, 4)
        self.layers = nn.ModuleList([GraphLayer(d), GraphLayer(d)])
        self.pair_head = mlp(4 * d + 16, 64, 1)
        self.null_head = mlp(d + 3, 32, 1)
        nn.init.zeros_(self.pair_head[-1].weight)
        nn.init.zeros_(self.pair_head[-1].bias)
        nn.init.zeros_(self.null_head[-1].weight)
        nn.init.zeros_(self.null_head[-1].bias)

    def encode(self, module, x, length):
        if not len(x):
            return x.new_zeros((0, self.d))
        if not self.use_gru:
            last = x[torch.arange(len(x), device=x.device), length - 1][:, None]
            return module(last)[1][0]
        packed = pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        return module(packed)[1][0]

    def forward(self, f, return_features=False):
        nv, na = (len(f["vseq"]), len(f["aseq"]))
        nm = len(f["mvseq"]) if self.use_memory else 0
        vs = torch.cat([f["vseq"], f["mvseq"]], 0) if nm else f["vseq"]
        vl = torch.cat([f["vlength"], f["mvlength"]], 0) if nm else f["vlength"]
        ass = torch.cat([f["aseq"], f["maseq"]], 0) if nm else f["aseq"]
        al = torch.cat([f["alength"], f["malength"]], 0) if nm else f["alength"]
        ve = self.encode(self.visual, vs, vl)
        ae = self.encode(self.ais, ass, al)
        v, a = (ve[:nv], ae[:na])
        if nm:
            memory = self.memory_content(torch.cat([ve[nv:], ae[na:]], -1))
            v = self.vm(v, memory, f["vm_edge"], f["vm_mask"])
            a = self.am(a, memory, f["am_edge"], f["am_mask"])
        if self.use_graph:
            for layer in self.layers:
                v, a = layer(v, a, f)
        vv = v[:, None, :].expand(nv, na, self.d)
        aa = a[None, :, :].expand(nv, na, self.d)
        pair_features = torch.cat([vv, aa, torch.abs(vv - aa), vv * aa, f["pair"]], -1)
        pair_prior = 4 * f["pair"][..., 10] - 2 * f["pair"][..., 2].clamp(0, 5)
        score = self.pair_head(pair_features).squeeze(-1) + pair_prior
        score = score.masked_fill(~f["candidate"], -10000.0)
        null_features = torch.cat([v, f["null_features"]], -1)
        null = self.null_head(null_features).squeeze(-1) + 4 * f["null_features"][:, 0]
        logits = torch.cat([score, null[:, None]], -1)
        if return_features:
            return (
                logits,
                dict(
                    pair_features=pair_features,
                    null_features=null_features,
                    pair_prior=pair_prior,
                    null_prior=4 * f["null_features"][:, 0],
                    candidate=f["candidate"],
                ),
            )
        return logits
