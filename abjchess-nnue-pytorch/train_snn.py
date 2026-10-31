#!/usr/bin/env python3
"""Train an SNN small net (engine/src/snnue.h: the AB-JChess big-net architecture at a small FT width) by
distillation from the ABJ big net, on TrainRecords (112-byte self-play records, src/selfplay.cpp).

Architecture (per perspective; same 15264 inputs as the big net, see snn_ref.features):
  FT 15264 -> W int16 (+16 PSQT buckets) ; pairwise product of the halves -> W/2 per side, [stm, opp] -> W uint8
  -> stack (16 = 8 piece-count x {face-down pieces or not}): W -> 16 ; [sqr-clip(0..14), clip(0..14)] -> 32 -> 1
  + 600 x L1 output 15 (skip) + PSQT ; then AB-JChess runtime post-processing (material scale, 80-ply damping).

Float semantics used in training (exactly the integer engine up to rounding):
  acc = ft(x) (unit 1/127), a, b = clamp(halves, 0, 1), x = a * b * 127/128
  f = L1(x) ; s = [min(1, 0.99219 f^2), clamp(f, 0, 1)] (f[:15]) ; h = clamp(L2(s), 0, 1)
  nn = out(h) + 600 f[15] + (P_stm - P_opp) / 2 ; value = nn * matscale / 1024 * (218 - min(nocap, 120)) / 176

Target (win-probability space, K fitted): wt * sigmoid(teacher / K) + ws * sigmoid(score / K) + wr * (result + 1) / 2,
where teacher = the big net's static value of the position (computed once on the GPU), score = the record's search
score and result = the game result (side to move).

  python3 train_snn.py DATA.bin --val-file VAL.bin --teacher TEACHER.nnue --width 512 --out runs/snn512 \
      [--init-abj] [--factor] --epochs 12
"""
import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402  (record loading, fields, fit_k, rec_to_fen)
import snn_ref as R  # noqa: E402  (numpy reference, ABJ constants)

NFEAT, NPSQT, NST = R.NFEAT, 16, 16
NVIRT = 14 * 90  # factoriser: board features shared by all king buckets and both variants
SQR_C = 8128.0 ** 2 / 2 ** 19 / 127.0  # 0.99219: sqr-clip in float units
X_C = 127.0 / 128.0


# ---------------------------------------------------------------- features (GPU, batched; mirrors snn_ref.features)
class Featurizer:
    def __init__(self, dev):
        self.dev = dev
        sq = torch.arange(90, device=dev)
        self.sq = sq
        self.mir = (sq // 9) * 9 + (8 - sq % 9)
        self.KB = torch.tensor(R.KB, device=dev)
        self.PLANE = torch.tensor(R.PLANE, device=dev)
        self.REST = torch.tensor(R.REST_TYPE, device=dev)
        self.PV = torch.tensor(R.PIECE_VAL, device=dev)
        self.IC = torch.tensor(R.INIT_CNT, device=dev)

    def __call__(self, raw):
        """raw (B,112) uint8 on device -> feats (B,2,48) long (-1 pad, perspective 0 = side to move), virtual
        (B,2,36) long (-1 pad), bucket (B,), post (B,) float: value = nn * post."""
        B = len(raw)
        board = raw[:, :90].long()
        typ, col = board & 15, board >> 4
        stm = raw[:, 90].long()
        urn = raw[:, 91:103].to(torch.int8).long().view(B, 2, 6)
        nocap = raw[:, 111].long()
        occ = typ > 0
        dark_total = ((typ == 8) & occ).sum(1)
        pieces = occ.sum(1)
        dv = dark_total > 0
        ks = torch.stack([((typ == 1) & (col == c)).long().argmax(1) for c in range(2)], 1)
        # compact the occupied squares (<= 32 pieces) first
        order = torch.argsort((~occ).to(torch.int8), dim=1, stable=True)[:, :36]
        t_o, c_o = typ.gather(1, order), col.gather(1, order)
        occ_o = t_o > 0
        feats, virt = [], []
        for k in range(2):
            p = stm if k == 0 else 1 - stm  # absolute colour of this perspective
            own = ks.gather(1, p[:, None])[:, 0]
            opp = ks.gather(1, (1 - p)[:, None])[:, 0]
            kb = self.KB[own]
            mirror = (own % 9 > 4) | ((own % 9 == 4) & (opp % 9 > 4))
            s = torch.where(mirror[:, None], self.mir[order], order)
            s = torch.where((p == 0)[:, None], (9 - s // 9) * 9 + s % 9, s)
            plane = self.PLANE[t_o] + ((t_o != 1) & (t_o != 8) & (c_o != p[:, None])).long()
            local = plane * 90 + s
            f = torch.where(dv, 0, R.VARIANT_DIMS)[:, None] + kb[:, None] * 1260 + local
            f = torch.where(occ_o, f, -1)
            owner_is_p = torch.arange(2, device=self.dev)[None, :] == p[:, None]  # (B,2) over owner colour
            rest = R.BOARD_DIMS + torch.where(owner_is_p, 0, 36)[:, :, None] + self.REST[None, None, :] * 6 + urn
            rest = torch.where(dv[:, None, None], rest, -1).view(B, 12)
            feats.append(torch.cat([f, rest], 1))
            virt.append(torch.where(occ_o, NFEAT + local, -1))
        up = torch.stack([torch.stack([((typ == sl + 2) & (col == c)).sum(1) for sl in range(6)], 1) for c in range(2)], 1)
        revealed = (self.IC[None, None, :] - urn).clamp(min=0)
        mat = (urn * self.PV).sum((1, 2)) + (torch.minimum(up, revealed)[:, :, :5] * self.PV[:5]).sum((1, 2))
        ms = 935 + torch.div(mat * 93, 5116, rounding_mode="floor")
        post = ms.float() / 1024.0 * (218 - nocap.clamp(max=120)).float() / 176.0
        bucket = torch.clamp(torch.div(pieces - 1, 4, rounding_mode="floor"), max=7) + torch.where(dv, 8, 0)
        return torch.stack(feats, 1), torch.stack(virt, 1), bucket, post


def bag(table, idx):
    """sum of table rows over idx (B,K) with -1 padding."""
    valid = idx >= 0
    return F.embedding_bag(idx.clamp(min=0), table, per_sample_weights=valid.to(table.dtype), mode="sum")


# ---------------------------------------------------------------- teacher (ABJ big net on the GPU, integer semantics)
class Teacher:
    def __init__(self, path, dev):
        n = R.Net(path)
        self.n = n
        self.ftW = torch.from_numpy(n.ftW.astype(np.float32)).to(dev)
        self.ftB = torch.from_numpy(n.ftB.astype(np.float32)).to(dev)
        self.psqt = torch.from_numpy(n.psqt.astype(np.float64)).to(dev)
        self.l1w = torch.from_numpy(n.l1w.astype(np.float32)).to(dev)   # (16,16,2048)
        self.l1b = torch.from_numpy(n.l1b.astype(np.float64)).to(dev)
        self.l2w = torch.from_numpy(n.l2w.astype(np.float64)).to(dev)   # (16,32,32)
        self.l2b = torch.from_numpy(n.l2b.astype(np.float64)).to(dev)
        self.fb = torch.from_numpy(n.fb.astype(np.float64)).to(dev)
        self.fw = torch.from_numpy(n.fw.astype(np.float64)).to(dev)
        self.full = n.full_psqt

    @torch.no_grad()
    def x_first(self, feats, bucket):
        B = len(feats)
        acc = bag(self.ftW, feats.view(B * 2, -1)).view(B, 2, -1) + self.ftB
        acc = torch.remainder(acc + 32768, 65536) - 32768
        H = acc.shape[2] // 2
        a, b = acc[..., :H].clamp(0, 127), acc[..., H:].clamp(0, 127)
        x = torch.floor(a * b / 128).reshape(B, -1)  # [stm, opp]
        allf = torch.einsum("bi,soi->bso", x, self.l1w)   # float32 (sums can exceed 2^24: ~1 unit error, irrelevant for a target)
        first = torch.round(allf[torch.arange(B, device=x.device), bucket].double()) + self.l1b[bucket]
        return x, first

    @torch.no_grad()
    def __call__(self, feats, bucket, post_meta):
        B = len(feats)
        x, first = self.x_first(feats, bucket)
        c = first[:, :15].clamp(-32768, 32767)
        sq = torch.clamp(torch.floor(c * c / 2 ** 19), max=127)
        cr = torch.floor(first[:, :15] / 64).clamp(0, 127)
        sin = torch.cat([sq, cr, torch.zeros(B, 2, device=x.device, dtype=sq.dtype)], 1)
        second = torch.einsum("bi,boi->bo", sin, self.l2w[bucket]) + self.l2b[bucket]
        out = self.fb[bucket] + (self.fw[bucket] * torch.floor(second / 64).clamp(0, 127)).sum(1)
        fwd = torch.trunc(first[:, 15] * 9600 / 8128)
        ps = bag(self.psqt, feats.view(B * 2, -1)).view(B, 2, NPSQT)
        ar = torch.arange(B, device=x.device)
        psq = torch.trunc((ps[ar, 0, bucket] - ps[ar, 1, bucket]) / 2)
        if not self.full:
            psq = torch.where(bucket < 8, psq, torch.zeros_like(psq))
        nn_ = torch.trunc((out + fwd + psq) / 16)
        return nn_ * post_meta  # approximately the runtime value (the engine rounds twice more)


# ---------------------------------------------------------------- student
def ste_round(t, scale):
    return t + (torch.round(t * scale) / scale - t).detach()


def ste_floor(t, scale):
    """floor(t * scale) / scale in the forward pass, identity gradient (models the engine's integer truncations)."""
    return t + (torch.floor(t * scale) / scale - t).detach()


class SNN(nn.Module):
    def __init__(self, width, factor=False):
        super().__init__()
        self.qat = False  # emulate the engine's truncations (straight-through), see --qat-from
        assert width % 128 == 0 and 128 <= width <= 1024
        self.W, self.H, self.factor = width, width // 2, factor
        rows = NFEAT + (NVIRT if factor else 0)
        self.ft = nn.Parameter(torch.empty(rows, width + NPSQT))
        vmap = torch.full((NFEAT,), -1, dtype=torch.long)
        for var in (0, R.VARIANT_DIMS):
            for kb in range(6):
                vmap[var + kb * 1260:var + kb * 1260 + NVIRT] = torch.arange(NVIRT)
        self.register_buffer("vmap", vmap, persistent=False)
        self.ft_bias = nn.Parameter(torch.full((width,), 0.5))
        self.l1w = nn.Parameter(torch.empty(NST, 16, width))
        self.l1b = nn.Parameter(torch.zeros(NST, 16))
        self.l2w = nn.Parameter(torch.empty(NST, 32, 30))
        self.l2b = nn.Parameter(torch.zeros(NST, 32))
        self.ow = nn.Parameter(torch.empty(NST, 32))
        self.ob = nn.Parameter(torch.zeros(NST))
        with torch.no_grad():
            self.ft.uniform_(-0.05, 0.05)
            self.ft[:, width:] = 0
            # PSQT starts as material in nn units (own piece +v, opponent -v; both perspectives -> +v in total)
            val = {2: 943, 3: 438, 4: 168, 5: 190, 6: 491, 7: 101}  # ABJchess type code -> AB-JChess runtime value
            base = self.ft[NFEAT:] if factor else None
            for t, v in val.items():
                for opp in (0, 1):
                    plane = R.PLANE[t] + opp
                    sign = -1.0 if opp else 1.0
                    if factor:
                        base[plane * 90:(plane + 1) * 90, width:] = sign * v
                    else:
                        for var in (0, R.VARIANT_DIMS):
                            for kb in range(6):
                                o = var + kb * 1260 + plane * 90
                                self.ft[o:o + 90, width:] = sign * v
            if factor:
                self.ft[:NFEAT].mul_(0.0)
            u1 = 1 / math.sqrt(self.W)
            self.l1w.uniform_(-u1, u1)
            self.l2w.uniform_(-1 / math.sqrt(30), 1 / math.sqrt(30))
            self.ow.uniform_(-50, 50)

    def forward(self, feats, virt, bucket, post):
        B = len(feats)
        idx = feats
        q = self.qat
        ft = self.ft
        if self.factor:  # fold the shared rows in (same gradients as summing both feature sets, exact under qat)
            ft = self.ft[:NFEAT] + self.ft[NFEAT:][self.vmap.clamp(min=0)] * (self.vmap >= 0)[:, None]
            idx = feats
        if q:  # weights as exported
            ft = torch.cat([ste_round(ft[:, :self.W], 127.0), ste_round(ft[:, self.W:], 16.0)], 1)
        a = bag(ft, idx.view(B * 2, -1)).view(B, 2, -1)
        acc, ps = a[..., :self.W] + (ste_round(self.ft_bias, 127.0) if q else self.ft_bias), a[..., self.W:]
        l1w, l1b, l2w, l2b, ow, ob = self.l1w, self.l1b, self.l2w, self.l2b, self.ow, self.ob
        if q:
            l1w, l2w, l1b, l2b = ste_round(l1w, 64.0), ste_round(l2w, 64.0), ste_round(l1b, 8128.0), ste_round(l2b, 8128.0)
            ow, ob = ste_round(ow, 16 / 127), ste_round(ob, 16.0)
        u, v = acc[..., :self.H].clamp(0, 1), acc[..., self.H:].clamp(0, 1)
        x = (u * v * X_C).reshape(B, self.W)
        if self.qat:
            x = ste_floor(x, 127.0)
        f = torch.bmm(l1w[bucket], x.unsqueeze(2)).squeeze(2) + l1b[bucket]
        g = f[:, :15]
        s = torch.cat([torch.clamp(SQR_C * g.clamp(-4.03, 4.03) ** 2, max=1.0), g.clamp(0, 1)], 1)
        if self.qat:
            s = ste_floor(s, 127.0)
        h = (torch.bmm(l2w[bucket], s.unsqueeze(2)).squeeze(2) + l2b[bucket]).clamp(0, 1)
        if self.qat:
            h = ste_floor(h, 127.0)
        ar = torch.arange(B, device=x.device)
        nn_ = (ow[bucket] * h).sum(1) + ob[bucket] + 600.0 * f[:, 15] + 0.5 * (ps[ar, 0, bucket] - ps[ar, 1, bucket])
        return nn_ * post

    @torch.no_grad()
    def clamp_(self):
        self.ft[:, :self.W].clamp_(-7.0, 7.0)
        self.l1w.clamp_(-127 / 64, 127 / 64)
        self.l2w.clamp_(-127 / 64, 127 / 64)
        self.ow.clamp_(-127 * 127 / 16, 127 * 127 / 16)

    @torch.no_grad()
    def folded_ft(self):
        w = self.ft[:NFEAT].detach().clone()
        if self.factor:
            v = self.ft[NFEAT:].detach()
            for var in (0, R.VARIANT_DIMS):
                for kb in range(6):
                    o = var + kb * 1260
                    w[o:o + NVIRT] += v
        return w

    @torch.no_grad()
    def export(self, path):
        W = self.W
        w = self.folded_ft()
        q = lambda t, s, lo, hi, dt: torch.clamp(torch.round(t * s), lo, hi).cpu().numpy().astype(dt)
        with open(path, "wb") as f:
            f.write(R.SNN_MAGIC)
            f.write(struct.pack("<8I", 1, W, NFEAT, NPSQT, NST, 16, 32, 1))
            f.write(q(self.ft_bias, 127, -32767, 32767, "<i2").tobytes())
            f.write(q(w[:, :W], 127, -32767, 32767, "<i2").tobytes())
            f.write(q(w[:, W:], 16, -2**31, 2**31 - 1, "<i4").tobytes())
            l2 = torch.zeros(NST, 32, 32, device=w.device)
            l2[:, :, :30] = self.l2w
            for s in range(NST):
                f.write(q(self.l1b[s], 64 * 127, -2**31, 2**31 - 1, "<i4").tobytes())
                l1 = q(self.l1w[s], 64, -127, 127, np.int8)  # (16, W) -> [W/4][16][4]
                f.write(l1.reshape(16, W // 4, 4).transpose(1, 0, 2).tobytes())
                f.write(q(self.l2b[s], 64 * 127, -2**31, 2**31 - 1, "<i4").tobytes())
                f.write(q(l2[s], 64, -127, 127, np.int8).reshape(32, 8, 4).transpose(1, 0, 2).tobytes())
                f.write(q(self.ob[s:s + 1], 16, -2**31, 2**31 - 1, "<i4").tobytes())
                f.write(q(self.ow[s], 16 / 127, -127, 127, np.int8).tobytes())


@torch.no_grad()
def init_from_abj(net, teacher, feat, sample, lsq=True):
    """Start from the big net: keep the W/2 most important neuron pairs (mean |L1 contribution| over `sample`),
    copy their FT rows, the PSQT and the L2/output layers; L1 is least-squares refitted on the kept inputs to
    reproduce the big net's L1 outputs (or copied with lsq=False)."""
    dev = teacher.ftW.device
    K = net.H
    imp = torch.zeros(2048, device=dev, dtype=torch.float64)
    xs, fs, bs = [], [], []
    for i in range(0, len(sample), 8192):
        raw = sample[i:i + 8192].to(dev)
        feats, _, bucket, _ = feat(raw)
        x, first = teacher.x_first(feats, bucket)
        colabs = teacher.l1w.abs().sum(1)  # (16, 2048)
        imp += (x.double() * colabs[bucket].double()).sum(0)
        xs.append(x); fs.append(first); bs.append(bucket)
    pair = imp[:1024] + imp[1024:]
    keep = torch.sort(torch.argsort(pair, descending=True)[:K]).values
    cols = torch.cat([keep, keep + 1024])
    n = teacher.n
    net.ft.zero_()
    net.ft[:NFEAT, :K] = teacher.ftW[:, keep] / 127
    net.ft[:NFEAT, K:2 * K] = teacher.ftW[:, keep + 1024] / 127
    ps = torch.from_numpy(n.psqt.astype(np.float32)).to(dev) / 16
    if not n.full_psqt:
        ps[:, 8:] = 0  # the shipped runtime ignores the dark-phase PSQT buckets
    net.ft[:NFEAT, net.W:] = ps
    net.ft_bias.copy_(torch.cat([teacher.ftB[keep], teacher.ftB[keep + 1024]]) / 127)
    net.l1w.copy_(teacher.l1w[:, :, cols] / 64)
    net.l1b.copy_((teacher.l1b / (64 * 127)).float())
    if lsq:
        X = torch.cat(xs)[:, cols].double()
        Y = torch.cat(fs)
        Bk = torch.cat(bs)
        for b in range(NST):
            m = Bk == b
            if m.sum() < 4 * len(cols):
                continue
            Xb = torch.cat([X[m], torch.ones(int(m.sum()), 1, device=dev, dtype=torch.float64)], 1)
            A = Xb.T @ Xb
            A += 1e-4 * torch.trace(A) / len(A) * torch.eye(len(A), device=dev, dtype=torch.float64)
            sol = torch.linalg.solve(A, Xb.T @ Y[m])  # (2K+1, 16), integer units
            net.l1w[b] = (sol[:-1].T * 127 / (64 * 127)).float()  # x_float = x_int / 127; w_float = w_int / 64
            net.l1b[b] = (sol[-1] / (64 * 127)).float()
    net.l2w.copy_(teacher.l2w[:, :, :30].float() / 64)
    net.l2b.copy_((teacher.l2b / (64 * 127)).float())
    net.ow.copy_(teacher.fw.float() * 127 / 16)
    net.ob.copy_(teacher.fb.float() / 16)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--val-max", type=int, default=500_000)
    ap.add_argument("--teacher", required=True, help="ABJNNUE package (AB-JChess big net)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--factor", action="store_true", help="train a shared copy of the board features (folded at export)")
    ap.add_argument("--init-abj", action="store_true", help="initialise from the teacher (top W/2 neuron pairs, L1 refit)")
    ap.add_argument("--init", default="", help="initialise from a saved state_dict (.pt)")
    ap.add_argument("--wt", type=float, default=0.6, help="target weight: teacher static value")
    ap.add_argument("--ws", type=float, default=0.3, help="target weight: record search score")
    ap.add_argument("--wr", type=float, default=0.1, help="target weight: game result")
    ap.add_argument("--k", type=float, default=0, help="sigmoid scale; 0 = fit from the data")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--epochs", type=float, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--qat-from", type=float, default=0.75,
                    help="fraction of training after which the engine's truncations are emulated (1 = never)")
    ap.add_argument("--keep-captures", action="store_true")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--data-on-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw, n_all = T.load_records(args.data, args.max_records, args.keep_captures)
    vraw, _ = T.load_records([args.val_file], 0, args.keep_captures)
    vraw = vraw[: args.val_max]
    print(f"train {len(raw)} of {n_all} records, val {len(vraw)} ({time.time()-t0:.0f}s)", flush=True)
    host = torch.from_numpy(raw)
    val = torch.from_numpy(np.ascontiguousarray(vraw)).to(dev)
    feat = Featurizer(dev)
    teacher = Teacher(args.teacher, dev)

    def teach(data):
        outv = torch.empty(len(data), dtype=torch.float32, device=dev)
        for i in range(0, len(data), 32768):
            b = data[i:i + 32768].to(dev)
            fe, _, bu, po = feat(b)
            outv[i:i + len(b)] = teacher(fe, bu, po.double()).float()
        return outv
    tv_train = teach(host)
    tv_val = teach(val)
    print(f"teacher values computed ({time.time()-t0:.0f}s)", flush=True)
    _, _, _, s_all, r_all, _ = T.fields(host[: min(len(host), 2_000_000)].to(dev))
    K = args.k or T.fit_k(s_all, r_all)
    print(f"K = {K}", flush=True)
    train = host.to(dev) if args.data_on_gpu else host
    net = SNN(args.width, args.factor).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev))
    elif args.init_abj:
        keep = init_from_abj(net, teacher, feat, host[torch.randperm(len(host))[:400_000]])
        np.save(out / "abj_pairs.npy", keep.cpu().numpy())
        net.clamp_()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps_per_epoch = max(1, len(train) // args.batch)
    total = int(args.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05,
                                                anneal_strategy="cos", div_factor=10, final_div_factor=100)

    def loss_of(b, tv):
        board_score_res = T.fields(b)
        score, result = board_score_res[3], board_score_res[4]
        fe, vi, bu, po = feat(b)
        y = net(fe, vi, bu, po)
        target = (args.wt * torch.sigmoid(tv / K) + args.ws * torch.sigmoid(score.float() / K)
                  + args.wr * (result.float() + 1) / 2)
        return ((torch.sigmoid(y / K) - target) ** 2).mean(), y

    def evaluate():
        net.eval()  # note: validation follows the current qat mode
        tot, mae, n = 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), 65536):
                b = val[i:i + 65536]
                l, y = loss_of(b, tv_val[i:i + 65536])
                tot += l.item() * len(b)
                mae += (y.clamp(-1500, 1500) - tv_val[i:i + 65536].clamp(-1500, 1500)).abs().sum().item()
                n += len(b)
        net.train()
        return tot / n, mae / n

    v0 = evaluate()
    print(f"init val loss {v0[0]:.6f} teacher-MAE {v0[1]:.1f}", flush=True)
    log, best, step = [], None, 0
    for ep in range(math.ceil(args.epochs)):
        perm = torch.randperm(len(train), device=train.device)
        run, rc = 0.0, 0
        for i in range(steps_per_epoch):
            if step >= total:
                break
            if not net.qat and step >= args.qat_from * total:
                net.qat = True
                print(f"step {step}: quantisation-aware from here", flush=True)
            idx = perm[i * args.batch:(i + 1) * args.batch]
            b = train[idx].to(dev)
            loss, _ = loss_of(b, tv_train[idx.to(dev)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            net.clamp_()
            step += 1
            run += loss.item()
            rc += 1
        vl, vm = evaluate()
        rec = dict(epoch=ep + 1, step=step, train=run / max(rc, 1), val=vl, teacher_mae=vm, lr=sched.get_last_lr()[0],
                   sec=round(time.time() - t0))
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save(net.state_dict(), out / "last.pt")
        net.export(out / "last.snn")
        if best is None or vl < best:
            best = vl
            torch.save(net.state_dict(), out / "best.pt")
            net.export(out / "best.snn")
    (out / "train.json").write_text(json.dumps(dict(args=vars(args), K=K, records=len(raw), log=log), indent=2) + "\n")
    write_refs(out / "best.snn", vraw[:64], out / "refs.json")


def write_refs(path, records, dest):
    """Exact integer reference values of an exported net (snn_ref) (cross-check against the engine)."""
    n = R.Net(str(path))
    d = R.features(records)
    acc, ps = R.accumulate(n, d["feats"])
    x = R.ft_out(acc, d["stm"])
    pos, _ = R.heads(n, x, d["bucket"])
    psq = R.psqt_term(ps, d["stm"], d["bucket"], n.full_psqt)
    rt = R.runtime(d, psq + pos)
    refs = [dict(fen=T.rec_to_fen(r).rsplit(" ", 2)[0] + f" {int(r[111])} 1", psqt=int(a), positional=int(b), runtime=int(c))
            for r, a, b, c in zip(records, psq, pos, rt)]
    Path(dest).write_text(json.dumps(refs, indent=1) + "\n")


if __name__ == "__main__":
    main()
