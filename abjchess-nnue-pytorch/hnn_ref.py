"""Exact numpy re-implementation of ABJchess's integer inference for HNN small nets (abjchess/src/hnnue.cpp, JQHNNUE1 files):
it014-style inputs (king-bucketed piece-square + urn thresholds) with the SNN/ABJ head (see snn_ref.py, hnnue.h).
Bit-identical to the engine.

  net = Net(path); d = features(records, net.scheme); acc, ps = accumulate(net, d['feats'])
  x = R.ft_out(acc, d['stm']); pos, _ = R.heads(net, x, d['bucket'])
  value = runtime(d, R.psqt_term(ps, d['stm'], d['bucket'], True) + pos)
"""
import struct

import numpy as np
import torch

import snn_ref as R

MAGIC = b"JQHNNUE1"
NPSQT = 16
PIECE_FEATS = 16 * 90
URN_FEATS = 30
URN_OFF = np.array([0, 2, 4, 6, 8, 10])
URN_INIT = np.array([2, 2, 2, 2, 2, 5])
PIECE_VAL = R.PIECE_VAL
KB_OF_SCHEME = [1, 3, 2, 6]
NCOL = 36 + URN_FEATS  # feature slots per perspective (<= 32 pieces + 30 urn thresholds, -1 padded)


def nfeat(scheme):
    return KB_OF_SCHEME[scheme] * PIECE_FEATS + URN_FEATS


def king_tables(scheme):
    """(kb, mirror) per oriented palace point index 3 * rank + file - 3 (rank 0..2, file 3..5)."""
    kb, mir = np.zeros(9, np.int64), np.zeros(9, np.int64)
    for r in range(3):
        for f in range(3, 6):
            i = r * 3 + f - 3
            if scheme == 1:
                kb[i] = f - 3
            elif scheme == 2:
                mir[i], kb[i] = f > 4, f == 4
            elif scheme == 3:
                mir[i], kb[i] = f > 4, 2 * r + (f == 4)
    return kb, mir


def orient(p, s):
    """p: perspective colour (array or int), s: square -> it014 orientation (own home rank = 0)."""
    return np.where(p == 0, s, (9 - s // 9) * 9 + s % 9)


class Net:
    def __init__(self, path):
        b = open(path, "rb").read()
        assert b[:8] == MAGIC, "not a JQHNNUE1 file"
        version, w, nf, scheme, npsqt, nst, l1, l2 = struct.unpack_from("<8I", b, 8)
        assert version == 1 and npsqt == 16 and nst == 16 and l1 == 16 and l2 == 32 and nf == nfeat(scheme), "bad header"
        self.W, self.scheme, self.nfeat, self.full_psqt = w, scheme, nf, True
        o = 40
        self.ftB = np.frombuffer(b, "<i2", w, o).astype(np.int32); o += 2 * w
        self.ftW = np.frombuffer(b, "<i2", nf * w, o).reshape(nf, w); o += 2 * nf * w
        self.psqt = np.frombuffer(b, "<i4", nf * 16, o).reshape(nf, 16); o += 4 * nf * 16
        l1b, l1w, l2b, l2w, fb, fw = [], [], [], [], [], []
        for s in range(16):
            l1b.append(np.frombuffer(b, "<i4", 16, o)); o += 64
            l1w.append(np.frombuffer(b, np.int8, w * 16, o).reshape(w // 4, 16, 4).transpose(1, 0, 2).reshape(16, w)); o += w * 16
            l2b.append(np.frombuffer(b, "<i4", 32, o)); o += 128
            l2w.append(np.frombuffer(b, np.int8, 1024, o).reshape(8, 32, 4).transpose(1, 0, 2).reshape(32, 32)); o += 1024
            fb.append(np.frombuffer(b, "<i4", 1, o)[0]); o += 4
            fw.append(np.frombuffer(b, np.int8, 32, o)); o += 32
        assert o == len(b), "HNN file size"
        self.l1b, self.l1w, self.l2b, self.l2w = np.stack(l1b), np.stack(l1w), np.stack(l2b), np.stack(l2w)
        self.fb, self.fw = np.array(fb), np.stack(fw)
        self.ftW_t = torch.from_numpy(self.ftW.astype(np.float32))
        self.psqt_t = torch.from_numpy(self.psqt.astype(np.float64))


def features(a, scheme):
    """a: (N,112) records -> dict; feats (N,2,66) indexed by absolute colour (0 red, 1 black), -1 padded."""
    N = len(a)
    board = a[:, :90].astype(np.int64)
    typ, col = board & 15, board >> 4
    stm = a[:, 90].astype(np.int64)
    urn = a[:, 91:103].view(np.int8).reshape(N, 2, 6).astype(np.int64)
    nocap = a[:, 111].astype(np.int64)
    occ = typ > 0
    darkTotal = ((typ == 8) & occ).sum(1)
    pieces = occ.sum(1)
    sq = np.arange(90)
    ks = np.stack([np.argmax((typ == 1) & (col == c), 1) for c in range(2)], 1)
    up = np.zeros((N, 2, 6), np.int64)
    for c in range(2):
        for sl in range(6):
            up[:, c, sl] = ((typ == sl + 2) & (col == c)).sum(1)
    kbt, mirt = king_tables(scheme)
    base = KB_OF_SCHEME[scheme] * PIECE_FEATS
    order = np.argsort(~occ, axis=1, kind="stable")[:, :36]
    t_o, c_o = np.take_along_axis(typ, order, 1), np.take_along_axis(col, order, 1)
    feats = np.full((N, 2, NCOL), -1, np.int64)
    for p in range(2):
        ko = orient(p, ks[:, p])
        ki = (ko // 9) * 3 + ko % 9 - 3
        kb, mir = kbt[ki], mirt[ki]
        o = orient(p, order)
        o = np.where(mir[:, None] == 1, (o // 9) * 9 + 8 - o % 9, o)
        f = kb[:, None] * PIECE_FEATS + (np.where(c_o == p, 0, 8) + t_o - 1) * 90 + o
        feats[:, p, :36] = np.where(t_o > 0, f, -1)
        k = 36
        for c in range(2):
            for sl in range(6):
                for j in range(1, URN_INIT[sl] + 1):
                    feats[:, p, k] = np.where(urn[:, c, sl] >= j, base + (0 if c == p else 15) + URN_OFF[sl] + j - 1, -1)
                    k += 1
    bucket = np.minimum((pieces - 1) // 4, 7) + np.where(darkTotal > 0, 8, 0)
    mat = (urn * PIECE_VAL).sum((1, 2)) + (up[:, :, :5] * PIECE_VAL[:5]).sum((1, 2))
    return dict(feats=feats, stm=stm, urn=urn, up=up, nocap=nocap, bucket=bucket, pieces=pieces, darkTotal=darkTotal,
                material=mat, result=a[:, 105].view(np.int8).astype(np.int64),
                score=a[:, 103:105].copy().view("<i2")[:, 0].astype(np.int64))


def accumulate(net, feats):
    """feats (N,2,K) -> acc (N,2,W) int64 (wrapped to int16), psqt (N,2,16) int64 (wrapped to int32)."""
    N, _, K = feats.shape
    f = torch.from_numpy(feats.reshape(N * 2, K))
    valid = f >= 0
    idx = torch.where(valid, f, torch.zeros_like(f))
    acc = torch.nn.functional.embedding_bag(idx, net.ftW_t, per_sample_weights=valid.float(), mode="sum")
    acc = acc.numpy().astype(np.int64) + net.ftB
    acc = ((acc + 32768) % 65536 - 32768).reshape(N, 2, net.W)
    ps = torch.nn.functional.embedding_bag(idx, net.psqt_t, per_sample_weights=valid.double(), mode="sum")
    ps = ps.numpy().astype(np.int64).reshape(N, 2, NPSQT)
    return acc, (ps + 2**31) % 2**32 - 2**31


def runtime(d, nnsum):
    nn = R.tdiv(nnsum, 16)
    ms = 935 + d["material"] * 93 // 5116
    v = R.tdiv(nn * ms, 1024)
    v = R.tdiv(v * (218 - np.minimum(d["nocap"], 120)), 176)
    return np.clip(v, -29507, 29507)


def evaluate(net, records):
    """-> (psqt, positional, runtime) int arrays."""
    d = features(records, net.scheme)
    acc, ps = accumulate(net, d["feats"])
    x = R.ft_out(acc, d["stm"])
    pos, _ = R.heads(net, x, d["bucket"])
    psq = R.psqt_term(ps, d["stm"], d["bucket"], True)
    return psq, pos, runtime(d, psq + pos)


def write_random(path, w, scheme, seed=1):
    """Random JQHNNUE1 file (tests)."""
    rng = np.random.default_rng(seed)
    nf = nfeat(scheme)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<8I", 1, w, nf, scheme, 16, 16, 16, 32))
        f.write(rng.integers(0, 120, w).astype("<i2").tobytes())
        f.write(rng.integers(-40, 41, (nf, w)).astype("<i2").tobytes())
        f.write(rng.integers(-3000, 3000, (nf, 16)).astype("<i4").tobytes())
        for s in range(16):
            f.write(rng.integers(-20000, 20000, 16).astype("<i4").tobytes())
            f.write(rng.integers(-60, 61, w * 16).astype(np.int8).tobytes())
            f.write(rng.integers(-3000, 3000, 32).astype("<i4").tobytes())
            f.write(rng.integers(-60, 61, 1024).astype(np.int8).tobytes())
            f.write(rng.integers(-3000, 3000, 1).astype("<i4").tobytes())
            f.write(rng.integers(-60, 61, 32).astype(np.int8).tobytes())
