#!/usr/bin/env python3
# From jqe (https://github.com/devinokeefe/jieqi, engine/train/train.py), unmodified. MIT licence, Copyright (c) 2026 Devin O'Keefe.
"""Train a jqe NNUE (format JQN1) from self-play TrainRecords (engine/src/selfplay.cpp).

Target in win-probability space: lam * sigmoid(score / K) + (1 - lam) * (result + 1) / 2.
Records are kept in host memory (or VRAM with --data-on-gpu); features are computed on the GPU per batch.
With --stream R the data is read in shuffled 1M-record blocks, about R raw records in RAM at a time, so the
training set can exceed memory (requires --val-file).
"""
import argparse
import json
import math
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REC = np.dtype([("board", "u1", (90,)), ("stm", "u1"), ("urn", "i1", (2, 6)), ("score", "<i2"), ("result", "i1"),
                ("flags", "u1"), ("ply", "<u2"), ("move", "<u2"), ("nocap", "u1")])
assert REC.itemsize == 112
NUM_PIECE_FEATURES = 1440
NUM_FEATURES = 1470
PAD = NUM_FEATURES
KING_BUCKETS = 1  # 9 with --kb: piece features are indexed by the perspective's own king square (palace)


FACTOR = False  # with king buckets: also train a shared (unbucketed) copy, folded in at export
TRAIN_FEATURES = NUM_FEATURES


def set_king_buckets(kb, factor=False):
    global KING_BUCKETS, NUM_PIECE_FEATURES, NUM_FEATURES, PAD, FACTOR, TRAIN_FEATURES
    KING_BUCKETS = kb
    FACTOR = factor and kb > 1
    NUM_PIECE_FEATURES = kb * 1440
    NUM_FEATURES = NUM_PIECE_FEATURES + 30
    TRAIN_FEATURES = NUM_FEATURES + (1440 if FACTOR else 0)
    PAD = TRAIN_FEATURES
STACKS = 1  # --stacks: L1/L2/output layer stacks selected by the number of pieces on the board


def stack_index(board):
    """board (B,90) -> stack index from the piece count (both colours, kings and dark pieces included)."""
    n = (board > 0).sum(1)
    return ((n - 1) * STACKS // 32).clamp(0, STACKS - 1)


URN_INIT = [2, 2, 2, 2, 2, 5]
URN_OFFSET = [0, 2, 4, 6, 8, 10]
PIECE_VALUE = [0, 0, 900, 400, 230, 230, 450, 100, 330]  # by type code; dark ~ initial urn mean


def keep_mask(a, keep_captures=False, min_ply=0):
    """Training filter on (N,112) records: no in-check (flag 1), no captures (flag 2) unless asked,
    ply >= min_ply, no mate / known-win scores."""
    flags = a[:, 106]
    ply = a[:, 107].astype(np.int64) | (a[:, 108].astype(np.int64) << 8)
    keep = (flags & 1) == 0
    if not keep_captures:
        keep &= (flags & 2) == 0
    keep &= ply >= min_ply
    score = a[:, 103:105].copy().view("<i2")[:, 0]
    keep &= np.abs(score) < 20000
    # Malformed records (seen once: a page-cache page corrupted in RAM) would index outside the feature table.
    board = a[:, :90]
    urn = a[:, 91:103].view(np.int8)
    keep &= ~(((board & 15) > 8) | ((board >> 4) > 1)).any(1) & (a[:, 90] <= 1)
    keep &= ~((urn < 0) | (urn > np.array(URN_INIT * 2, dtype=np.int8))).any(1)
    return keep


def load_records(paths, max_records=0, keep_captures=False, min_ply=0):
    """Loads and filters record files into one preallocated array (peak memory ~ kept size), in order.
    Returns (records, total records read)."""
    chunk = 4_000_000
    counts, total = [], 0
    for p in paths:  # pass 1: kept counts per file
        m = np.memmap(p, dtype=np.uint8, mode="r")
        m = m[: len(m) // 112 * 112].reshape(-1, 112)
        total += len(m)
        counts.append(sum(int(keep_mask(np.asarray(m[i:i + chunk]), keep_captures, min_ply).sum())
                          for i in range(0, len(m), chunk)))
    out = np.empty((sum(counts), 112), dtype=np.uint8)
    pos = 0
    for p in paths:  # pass 2: fill
        m = np.memmap(p, dtype=np.uint8, mode="r")
        m = m[: len(m) // 112 * 112].reshape(-1, 112)
        for i in range(0, len(m), chunk):
            a = np.asarray(m[i:i + chunk])
            k = a[keep_mask(a, keep_captures, min_ply)]
            out[pos:pos + len(k)] = k
            pos += len(k)
    assert pos == len(out)
    if max_records and len(out) > max_records:
        out = out[-max_records:]
    return out, total


def block_list(paths, block):
    """(path, first, end) record ranges of at most `block` records covering every file."""
    out = []
    for p in paths:
        n = os.path.getsize(p) // 112
        out += [(p, s, min(n, s + block)) for s in range(0, n, block)]
    return out


def load_blocks(blocks, keep_captures=False, min_ply=0):
    """Filtered records of the given blocks, concatenated in the given order."""
    parts = []
    for p, s, e in blocks:
        a = np.fromfile(p, dtype=np.uint8, count=(e - s) * 112, offset=s * 112).reshape(-1, 112)
        parts.append(a[keep_mask(a, keep_captures, min_ply)])
    return np.concatenate(parts) if parts else np.empty((0, 112), dtype=np.uint8)


def count_kept(blocks, keep_captures=False, min_ply=0):
    return sum(int(keep_mask(np.fromfile(p, dtype=np.uint8, count=(e - s) * 112, offset=s * 112).reshape(-1, 112),
                             keep_captures, min_ply).sum()) for p, s, e in blocks)


def fields(raw):
    """raw: (B,112) uint8 tensor on device -> dict of tensors."""
    board = raw[:, :90].long()
    stm = raw[:, 90].long()
    urn = raw[:, 91:103].to(torch.int8).long().view(-1, 2, 6)
    score = (raw[:, 103].long() | (raw[:, 104].long() << 8))
    score = torch.where(score >= 32768, score - 65536, score)
    result = raw[:, 105].to(torch.int8).long()
    flags = raw[:, 106].long()
    return board, stm, urn, score, result, flags


class Featurizer:
    def __init__(self, device):
        sq = torch.arange(90, device=device)
        self.sq = sq
        self.flip_rank = (9 - sq // 9) * 9 + sq % 9
        self.mirror_file = (sq // 9) * 9 + (8 - sq % 9)
        owner, slot, j = [], [], []
        for c in range(2):
            for sl in range(6):
                for jj in range(1, URN_INIT[sl] + 1):
                    owner.append(c)
                    slot.append(sl)
                    j.append(jj)
        self.u_owner = torch.tensor(owner, device=device)
        self.u_slot = torch.tensor(slot, device=device)
        self.u_j = torch.tensor(j, device=device)
        self.u_off = torch.tensor([URN_OFFSET[s] for s in slot], device=device)

    def __call__(self, board, stm, urn, mirror=None):
        if mirror is not None:
            mb = board[:, self.mirror_file]
            board = torch.where(mirror[:, None], mb, board)
        typ = board & 15
        col = board >> 4
        occ = typ > 0
        counts = urn[:, self.u_owner, self.u_slot]  # (B,30)
        present = counts >= self.u_j[None, :]
        out = []
        for persp in (stm, 1 - stm):
            P = persp[:, None]
            osq = torch.where(P == 0, self.sq[None, :], self.flip_rank[None, :])
            rel = (col != P).long() * 8
            f = (rel + typ - 1) * 90 + osq
            if KING_BUCKETS > 1:
                ksq = ((typ == 1) & (col == P)).long().argmax(1, keepdim=True)  # own king square
                ko = torch.gather(osq, 1, ksq)  # oriented: own palace is ranks 0-2, files 3-5
                kb = (ko // 9) * 3 + (ko % 9 - 3) if KING_BUCKETS == 9 else (ko % 9 - 3)
                f = f + kb * 1440
            f = torch.where(occ, f, torch.full_like(f, PAD))
            uf = NUM_PIECE_FEATURES + torch.where(self.u_owner[None, :] == P, 0, 15) + self.u_off[None, :] + self.u_j[None, :] - 1
            uf = torch.where(present, uf, torch.full_like(uf, PAD))
            if FACTOR:
                fs = torch.where(occ, (rel + typ - 1) * 90 + osq + NUM_FEATURES, torch.full_like(f, PAD))
                out.append(torch.cat([f, uf, fs], 1))
            else:
                out.append(torch.cat([f, uf], 1))
        return out


class Net(nn.Module):
    def __init__(self, ft=256, l1=16, l2=32, out_scale=400.0):
        super().__init__()
        self.ftn, self.l1n, self.l2n, self.out_scale = ft, l1, l2, out_scale
        self.ft = nn.EmbeddingBag(TRAIN_FEATURES + 1, ft + 1, mode="sum", padding_idx=PAD)
        self.ft_bias = nn.Parameter(torch.full((ft + 1,), 0.5))
        self.stacks = STACKS
        if STACKS > 1:  # rows of l1 are stack-major: stack s, neuron j -> row s * l1 + j
            self.l1 = nn.Linear(2 * ft, l1 * STACKS)
            u1, u2 = 1 / math.sqrt(l1), 1 / math.sqrt(l2)
            self.l2w = nn.Parameter(torch.empty(STACKS, l2, l1).uniform_(-u1, u1))
            self.l2b = nn.Parameter(torch.empty(STACKS, l2).uniform_(-u1, u1))
            self.outw = nn.Parameter(torch.empty(STACKS, l2).uniform_(-u2, u2) * 0.1)
            self.outb = nn.Parameter(torch.zeros(STACKS))
        else:
            self.l1 = nn.Linear(2 * ft, l1)
            self.l2 = nn.Linear(l1, l2)
            self.out = nn.Linear(l2, 1)
        with torch.no_grad():
            w = self.ft.weight
            w.uniform_(-0.06, 0.06)
            w[PAD].zero_()
            self.ft_bias[ft] = 0.0
            # PSQT column starts as material (own +, opponent -) in output units.
            for rel, sign in ((0, 1.0), (8, -1.0)):
                for t in range(1, 9):
                    k = rel + t - 1
                    w[k * 90:(k + 1) * 90, ft] = sign * PIECE_VALUE[t] / out_scale
            for kb in range(1, KING_BUCKETS):
                w[kb * 1440:(kb + 1) * 1440, ft] = w[:1440, ft]
            w[NUM_PIECE_FEATURES:NUM_FEATURES, ft] = 0.0
            if FACTOR:  # shared copy carries the initial weights; buckets start as zero corrections
                w[NUM_FEATURES:NUM_FEATURES + 1440] = w[:1440]
                w[:NUM_PIECE_FEATURES].zero_()
            if STACKS == 1:
                self.out.weight.mul_(0.1)

    def forward(self, us, them, bucket=None):
        F = self.ftn
        a = self.ft(us) + self.ft_bias
        b = self.ft(them) + self.ft_bias
        x = torch.cat([a[:, :F].clamp(0, 1), b[:, :F].clamp(0, 1)], 1)
        if self.stacks > 1:
            ar = torch.arange(len(x), device=x.device)
            h = self.l1(x).view(len(x), self.stacks, self.l1n)[ar, bucket].clamp(0, 1)
            h = (torch.bmm(self.l2w[bucket], h.unsqueeze(2)).squeeze(2) + self.l2b[bucket]).clamp(0, 1)
            return (self.outw[bucket] * h).sum(1) + self.outb[bucket] + 0.5 * (a[:, F] - b[:, F])
        h = self.l1(x).clamp(0, 1)
        h = self.l2(h).clamp(0, 1)
        return self.out(h)[:, 0] + 0.5 * (a[:, F] - b[:, F])

    def export(self, path):
        with open(path, "wb") as f:
            f.write(b"JQN1")
            # version 1: plain; 2: king buckets (count implied by the feature total); 3: + layer stacks
            version = 3 if self.stacks > 1 else 1 if KING_BUCKETS == 1 else 2
            f.write(struct.pack("<5I", version, NUM_FEATURES, self.ftn, self.l1n, self.l2n))
            f.write(struct.pack("<f", self.out_scale))
            if version == 3:
                f.write(struct.pack("<I", self.stacks))
            dump = lambda t: f.write(t.detach().float().cpu().contiguous().numpy().astype("<f4").tobytes())
            w = self.ft.weight[:NUM_FEATURES].detach().clone()
            if FACTOR:
                shared = self.ft.weight[NUM_FEATURES:NUM_FEATURES + 1440].detach()
                for kb in range(KING_BUCKETS):
                    w[kb * 1440:(kb + 1) * 1440] += shared
            dump(w)
            dump(self.ft_bias)
            dump(self.l1.weight)
            dump(self.l1.bias)
            if self.stacks > 1:
                dump(self.l2w)
                dump(self.l2b)
                dump(self.outw)
                dump(self.outb)
            else:
                dump(self.l2.weight)
                dump(self.l2.bias)
                dump(self.out.weight.view(-1))
                dump(self.out.bias.view(-1))


def fit_k(score, result):
    """Fit P(win) ~ sigmoid(score/K) on decisive-or-draw outcomes (score in cp, |score| < 3000)."""
    m = score.abs() < 3000
    s = score[m].double()
    y = (result[m].double() + 1) / 2
    best = None
    for K in range(100, 1500, 10):
        p = torch.sigmoid(s / K).clamp(1e-6, 1 - 1e-6)
        ll = -(y * p.log() + (1 - y) * (1 - p).log()).mean().item()
        if best is None or ll < best[1]:
            best = (K, ll)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--ft", type=int, default=256)
    ap.add_argument("--l1", type=int, default=16)
    ap.add_argument("--l2", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.7, help="weight of search score vs game result")
    ap.add_argument("--k", type=float, default=0, help="score scale; 0 = fit from data")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--keep-captures", action="store_true")
    ap.add_argument("--min-ply", type=int, default=0)
    ap.add_argument("--init", default="", help="initialise from a saved state_dict (.pt)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--factor", action="store_true", help="with --kb: factorised training (shared + bucket weights)")
    ap.add_argument("--stacks", type=int, default=1, choices=[1, 2, 4, 8],
                    help="layer stacks after the feature transformer, selected by piece count (1 = none)")
    ap.add_argument("--kb", type=int, default=1, choices=[1, 3, 9],
                    help="king buckets for piece features: 1 = none, 3 = own king's file, 9 = palace square")
    ap.add_argument("--val-file", default="",
                    help="validate on this record file (filtered like training data) instead of the last 1%% of the data")
    ap.add_argument("--val-max", type=int, default=1_000_000, help="with --val-file: at most this many records (from the start)")
    ap.add_argument("--data-on-gpu", action="store_true",
                    help="keep the whole dataset in VRAM (faster; default streams batches from host memory)")
    ap.add_argument("--stream", type=int, default=0,
                    help="stream the data: about this many raw records in host memory at a time (0 = load everything)")
    ap.add_argument("--block", type=int, default=1_000_000, help="with --stream: records per shuffled block")
    args = ap.parse_args()
    if args.stream and not args.val_file:
        ap.error("--stream requires --val-file")
    if args.stream and (args.max_records or args.data_on_gpu):
        ap.error("--stream excludes --max-records and --data-on-gpu")
    if args.kb > 1:
        set_king_buckets(args.kb, args.factor)
    global STACKS
    STACKS = args.stacks
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    t0 = time.time()
    if args.stream:
        blocks = block_list(args.data, args.block)
        n_all = sum(e - s for _, s, e in blocks)
        n = count_kept(blocks, args.keep_captures, args.min_ply)
        per_group = max(1, args.stream // args.block)
        print(f"streaming {n_all} records, kept {n}, {len(blocks)} blocks, {per_group} per group ({time.time()-t0:.1f}s)", flush=True)
        host = torch.from_numpy(load_blocks(blocks[:3], args.keep_captures, args.min_ply))  # only for fitting K
    else:
        raw, n_all = load_records(args.data, args.max_records, args.keep_captures, args.min_ply)
        n = len(raw)
        print(f"loaded {n_all} records, kept {n} ({time.time()-t0:.1f}s)", flush=True)
        host = torch.from_numpy(raw)
    if args.val_file:  # external held-out set: all loaded data trains
        vraw, _ = load_records([args.val_file], 0, args.keep_captures, args.min_ply)
        val = torch.from_numpy(np.ascontiguousarray(vraw[: args.val_max])).to(dev)
        train = host.to(dev) if args.data_on_gpu else host
        print(f"validation: {len(val)} records from {args.val_file}", flush=True)
    else:
        nval = max(10000, n // 100)
        if args.data_on_gpu:
            train, val = host[:-nval].to(dev), host[-nval:].to(dev)
        else:  # small VRAM footprint: the GPU may be shared with other jobs
            train, val = host[:-nval], host[-nval:].to(dev)  # batches are index-gathered, so pinning would not help
    _, _, _, s_all, r_all, _ = fields(train[: min(len(train), 2_000_000)].to(dev))
    K = args.k or fit_k(s_all, r_all)
    print(f"score scale K = {K}", flush=True)
    feat = Featurizer(dev)
    net = Net(args.ft, args.l1, args.l2).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, (n if args.stream else len(train)) // args.batch)
    total = int(args.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05,
                                                anneal_strategy="cos", div_factor=10, final_div_factor=100)

    def batch_loss(b, mirror=True):
        board, stm, urn, score, result, _ = fields(b)
        mir = (torch.rand(len(b), device=dev) < 0.5) if mirror else None
        us, them = feat(board, stm, urn, mir)
        y = net(us, them, stack_index(board)) * net.out_scale
        pred = torch.sigmoid(y / K)
        target = args.lam * torch.sigmoid(score.float() / K) + (1 - args.lam) * (result.float() + 1) / 2
        return ((pred - target) ** 2).mean(), pred, result

    def evaluate():
        net.eval()
        tot, cnt, ll = 0.0, 0, 0.0
        with torch.no_grad():
            for i in range(0, len(val), 65536):
                b = val[i:i + 65536]
                loss, pred, result = batch_loss(b, mirror=False)
                tot += loss.item() * len(b)
                cnt += len(b)
                y = (result.float() + 1) / 2
                p = pred.clamp(1e-6, 1 - 1e-6)
                ll += (-(y * p.log() + (1 - y) * (1 - p).log())).sum().item()
        net.train()
        return tot / cnt, ll / cnt

    log = []
    v0 = evaluate()
    print(f"init val loss {v0[0]:.6f} result-logloss {v0[1]:.4f}", flush=True)
    step = 0
    best = None
    if args.stream:
        del train, host
        pool = ThreadPoolExecutor(1)
        brng = np.random.default_rng(args.seed)

        def epoch_groups():
            order = brng.permutation(len(blocks))
            return [[blocks[j] for j in order[i:i + per_group]] for i in range(0, len(order), per_group)]

    def train_on(data, nsteps):
        nonlocal step
        perm = torch.randperm(len(data), device=data.device)
        run, rc = 0.0, 0
        for i in range(nsteps):
            if step >= total:
                break
            b = data[perm[i * args.batch:(i + 1) * args.batch]].to(dev)  # blocking: a pageable temp freed during an async copy can be reused by the prefetch thread
            loss, _, _ = batch_loss(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():  # keep weights inside the quantized inference ranges
                net.l1.weight.clamp_(-127 / 64, 127 / 64)
                net.ft.weight.clamp_(-3.9, 3.9)
            step += 1
            run += loss.item()
            rc += 1
        return run, rc

    for ep in range(math.ceil(args.epochs)):
        if args.stream:
            run, rc = 0.0, 0
            groups = epoch_groups()
            fut = pool.submit(load_blocks, groups[0], args.keep_captures, args.min_ply)
            for gi in range(len(groups)):
                if step >= total:
                    break
                data = torch.from_numpy(fut.result())
                if gi + 1 < len(groups):
                    fut = pool.submit(load_blocks, groups[gi + 1], args.keep_captures, args.min_ply)
                r, c = train_on(data, len(data) // args.batch)
                run, rc = run + r, rc + c
                del data
        else:
            run, rc = train_on(train, steps_per_epoch)
        vl, vll = evaluate()
        rec = dict(epoch=ep + 1, step=step, train=run / max(rc, 1), val=vl, val_result_logloss=vll, lr=sched.get_last_lr()[0],
                   sec=time.time() - t0)
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save(net.state_dict(), out / "last.pt")
        net.export(out / "last.jqn")
        if best is None or vl < best:
            best = vl
            torch.save(net.state_dict(), out / "best.pt")
            net.export(out / "best.jqn")
    (out / "train.json").write_text(json.dumps(dict(args=vars(args), K=K, records=n, log=log), indent=2) + "\n")
    # Reference evaluations for the C++ inference check: first 8 validation records.
    with torch.no_grad():
        net.eval()
        b = val[:8]
        board, stm, urn, *_ = fields(b)
        us, them = feat(board, stm, urn)
        cp = (net(us, them, stack_index(board)) * net.out_scale).tolist()
    refs = [dict(fen=rec_to_fen(r), cp=c) for r, c in zip(b.cpu().numpy(), cp)]
    (out / "refs.json").write_text(json.dumps(refs, indent=1) + "\n")


def rec_to_fen(r):
    chars = ".KRNBACPX"
    board = r[:90]
    rows = []
    for rank in range(9, -1, -1):
        row, empty = "", 0
        for f in range(9):
            p = int(board[rank * 9 + f])
            if not p:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            ch = chars[p & 15]
            row += ch if (p >> 4) == 0 else ch.lower()
        if empty:
            row += str(empty)
        rows.append(row)
    urn = r[91:103].view(np.int8).reshape(2, 6)
    order = [0, 3, 4, 5, 1, 2]  # slots R A C P N B
    letters = "RNBACP"
    pools = ""
    for c in range(2):
        for sl in order:
            if urn[c][sl]:
                ch = letters[sl] if c == 0 else letters[sl].lower()
                pools += ch + str(int(urn[c][sl]))
    side = "w" if r[90] == 0 else "b"
    return "/".join(rows) + f" {side} {pools or '-'} 0 1"


if __name__ == "__main__":
    main()
