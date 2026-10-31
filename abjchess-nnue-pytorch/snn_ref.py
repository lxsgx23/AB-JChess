"""Exact numpy/torch (CPU) re-implementation of ABJchess's integer inference for the AB-JChess big net
(ABJNNUE package) and for SNN small nets with the same architecture (JQSNNUE1 files).
Bit-identical to the engine's integer inference. Records: ABJchess TrainRecords (112 bytes).

  net = Net(path)            # ABJNNUE package or JQSNNUE1 file
  d = features(records); acc, ps = accumulate(net, d['feats'])
  x = ft_out(acc, d['stm']); pos, first = heads(net, x, d['bucket'])
  value = runtime(d, psqt_term(ps, d['stm'], d['bucket'], net.full_psqt) + pos)
"""
import numpy as np, torch, struct

HEADER, METADATA = 24, 5812
W, HALF, NPSQT, NFEAT = 2048, 1024, 16, 15264
HEADS_OFF, HEAD_SIZE = 66070016, 34208
BOARD_DIMS, VARIANT_DIMS = 7560, 7632
REST_TYPE = [0, 2, 3, 4, 1, 5]
PIECE_VAL = np.array([943, 438, 168, 190, 491, 101])
INIT_CNT = np.array([2, 2, 2, 2, 2, 5])
PLANE = np.array([-1, 0, 1, 5, 7, 9, 3, 11, 13])
KB = np.array([0,0,0,1,0,1,0,0,0, 0,0,0,3,2,3,0,0,0, 0,0,0,5,4,5,0,0,0] + [0]*36 +
              [0,0,0,5,4,5,0,0,0, 0,0,0,3,2,3,0,0,0, 0,0,0,1,0,1,0,0,0])
assert len(KB) == 90

SNN_MAGIC = b"JQSNNUE1"


class Net:
    def __init__(self, path):
        b = open(path, 'rb').read()
        if b[:8] == SNN_MAGIC:
            self._load_snn(b)
            return
        self.W, self.full_psqt = W, False
        p = b[HEADER + METADATA:]
        self.ftB = np.frombuffer(p, '<i2', W, 0).astype(np.int32)
        o = W * 2
        self.ftW = np.frombuffer(p, '<i2', NFEAT * W, o).reshape(NFEAT, W)
        o += NFEAT * W * 2
        self.psqt = np.frombuffer(p, '<i4', NFEAT * NPSQT, o).reshape(NFEAT, NPSQT)
        h = np.frombuffer(p, np.uint8, 16 * HEAD_SIZE, HEADS_OFF).reshape(16, HEAD_SIZE)
        i32 = lambda a: a.copy().view('<i4')
        i8 = lambda a: a.copy().view(np.int8)
        self.l1b = np.stack([i32(h[s, 0:64]) for s in range(16)])                        # (16,16)
        # L1W chunk layout: [ch][o][l] -> weight for input ch*4+l, output o
        self.l1w = np.stack([i8(h[s, 0x40:0x40 + 32768]).reshape(512, 16, 4).transpose(1, 0, 2).reshape(16, 2048)
                             for s in range(16)])                                         # (16,16,2048)
        self.l2b = np.stack([i32(h[s, 0x8080:0x8100]) for s in range(16)])               # (16,32)
        self.l2w = np.stack([i8(h[s, 0x8100:0x8500]).reshape(8, 32, 4).transpose(1, 0, 2).reshape(32, 32)
                             for s in range(16)])                                         # (16,32,32) (out,in)
        self.fb = np.array([i32(h[s, 0x8540:0x8544])[0] for s in range(16)])
        self.fw = np.stack([i8(h[s, 0x8580:0x85a0]) for s in range(16)])                 # (16,32)
        self._tensors()

    def _tensors(self):
        self.ftW_t = torch.from_numpy(self.ftW.astype(np.float32))
        self.psqt_t = torch.from_numpy(self.psqt.astype(np.float64))

    def _load_snn(self, b):
        version, w, nfeat, npsqt, nst, l1, l2, full = struct.unpack_from('<8I', b, 8)
        assert version == 1 and nfeat == NFEAT and npsqt == 16 and nst == 16 and l1 == 16 and l2 == 32, 'bad SNN header'
        self.W, self.full_psqt = w, bool(full)
        o = 40
        self.ftB = np.frombuffer(b, '<i2', w, o).astype(np.int32); o += 2 * w
        self.ftW = np.frombuffer(b, '<i2', NFEAT * w, o).reshape(NFEAT, w); o += 2 * NFEAT * w
        self.psqt = np.frombuffer(b, '<i4', NFEAT * 16, o).reshape(NFEAT, 16); o += 4 * NFEAT * 16
        l1b, l1w, l2b, l2w, fb, fw = [], [], [], [], [], []
        for s in range(16):
            l1b.append(np.frombuffer(b, '<i4', 16, o)); o += 64
            l1w.append(np.frombuffer(b, np.int8, w * 16, o).reshape(w // 4, 16, 4).transpose(1, 0, 2).reshape(16, w)); o += w * 16
            l2b.append(np.frombuffer(b, '<i4', 32, o)); o += 128
            l2w.append(np.frombuffer(b, np.int8, 1024, o).reshape(8, 32, 4).transpose(1, 0, 2).reshape(32, 32)); o += 1024
            fb.append(np.frombuffer(b, '<i4', 1, o)[0]); o += 4
            fw.append(np.frombuffer(b, np.int8, 32, o)); o += 32
        assert o == len(b), 'SNN file size'
        self.l1b, self.l1w, self.l2b, self.l2w = np.stack(l1b), np.stack(l1w), np.stack(l2b), np.stack(l2w)
        self.fb, self.fw = np.array(fb), np.stack(fw)
        self._tensors()

def load_records(path, n, offset=0, quiet=True):
    a = np.fromfile(path, np.uint8, count=(n * 3 + offset) * 112).reshape(-1, 112)[offset:]
    if quiet:
        flags = a[:, 106]
        score = a[:, 103:105].copy().view('<i2')[:, 0]
        a = a[((flags & 3) == 0) & (np.abs(score) < 20000)]
    return a[:n]

def features(a):
    """a: (N,112) records -> dict with feature index arrays (N,2,48) padded by -1, meta."""
    N = len(a)
    board = a[:, :90].astype(np.int64)
    typ, col = board & 15, board >> 4
    stm = a[:, 90].astype(np.int64)
    urn = a[:, 91:103].view(np.int8).reshape(N, 2, 6).astype(np.int64)
    nocap = a[:, 111].astype(np.int64)
    occ = typ > 0
    dark = (typ == 8) & occ
    darkTotal = dark.sum(1)
    pieces = occ.sum(1)
    dv = darkTotal > 0
    sq = np.arange(90)
    ks = np.stack([np.argmax((typ == 1) & (col == c), 1) for c in range(2)], 1)  # (N,2)
    up = np.zeros((N, 2, 6), np.int64)
    for c in range(2):
        for sl in range(6):
            up[:, c, sl] = ((typ == sl + 2) & (col == c)).sum(1)
    feats = np.full((N, 2, 48), -1, np.int64)
    kbm = np.zeros((N, 2, 2), np.int64)
    for p in range(2):
        own, opp = ks[:, p], ks[:, 1 - p]
        kb = KB[own]
        mirror = (own % 9 > 4) | ((own % 9 == 4) & (opp % 9 > 4))
        kbm[:, p, 0], kbm[:, p, 1] = kb, mirror
        s = np.where(mirror[:, None], (sq // 9) * 9 + (8 - sq % 9), sq)[:, :] * np.ones((N, 1), np.int64)
        if p == 0:
            s = (9 - s // 9) * 9 + s % 9
        plane = PLANE[typ] + ((typ != 1) & (typ != 8) & (col != p))
        f = np.where(dv[:, None], 0, VARIANT_DIMS) + kb[:, None] * 1260 + plane * 90 + s
        f = np.where(occ, f, -1)
        # compact: occupied first
        order = np.argsort(~occ, axis=1, kind='stable')[:, :36]
        ff = np.take_along_axis(f, order, 1)
        feats[:, p, :36] = ff
        for c in range(2):
            for sl in range(6):
                r = BOARD_DIMS + (0 if c == p else 36) + REST_TYPE[sl] * 6 + urn[:, c, sl]
                feats[:, p, 36 + c * 6 + sl] = np.where(dv, r, -1)
    bucket = np.minimum((pieces - 1) // 4, 7) + np.where(dv, 8, 0)
    return dict(feats=feats, stm=stm, urn=urn, up=up, nocap=nocap, bucket=bucket, dv=dv, pieces=pieces,
                darkTotal=darkTotal, kbm=kbm, result=a[:, 105].view(np.int8).astype(np.int64),
                score=a[:, 103:105].copy().view('<i2')[:, 0].astype(np.int64))

def accumulate(net, feats, rowmask=None):
    """feats (N,2,48) -> acc int32 (N,2,2048) (wrapped to int16), psqt (N,2,16) int64."""
    N = feats.shape[0]
    f = torch.from_numpy(feats.reshape(N * 2, 48))
    valid = f >= 0
    if rowmask is not None:
        valid &= torch.from_numpy(rowmask)[f.clamp(min=0)]
    idx = torch.where(valid, f, torch.zeros_like(f))
    wts = valid.to(torch.float32)
    acc = torch.nn.functional.embedding_bag(idx, net.ftW_t, per_sample_weights=wts, mode='sum')
    acc = acc.numpy().astype(np.int64) + net.ftB
    acc = ((acc + 32768) % 65536 - 32768).reshape(N, 2, net.W)
    ps = torch.nn.functional.embedding_bag(idx, net.psqt_t, per_sample_weights=wts.double(), mode='sum')
    ps = ps.numpy().astype(np.int64).reshape(N, 2, NPSQT)
    ps = (ps + 2**31) % 2**32 - 2**31
    return acc, ps

def tdiv(a, b):  # C integer division (truncate toward zero)
    q = np.abs(a) // abs(b)
    return np.where((a < 0) ^ (b < 0), -q, q)

def ft_out(acc, stm):
    N = len(stm)
    ar = np.arange(N)
    us, them = acc[ar, stm], acc[ar, 1 - stm]
    H = acc.shape[2] // 2
    def pw(v):
        a = np.clip(v[:, :H], 0, 127); b = np.clip(v[:, H:], 0, 127)
        return (a * b) >> 7
    return np.concatenate([pw(us), pw(them)], 1)  # (N,2048) uint8 values

def heads(net, x, bucket):
    """exact integer heads; returns positional (N,), first (N,16)."""
    first = l1(net.l1w, net.l1b, x, bucket)
    return heads_from_first(net, first, bucket), first

def l1(l1w, l1b, x, bucket):
    first = np.zeros((len(x), l1w.shape[1]), np.int64)
    xf = x.astype(np.float64)
    for b in np.unique(bucket):
        m = bucket == b
        first[m] = np.rint(xf[m] @ l1w[b].astype(np.float64).T).astype(np.int64) + l1b[b]
    return first

def heads_from_first(net, first, bucket):
    c = np.clip(first[:, :15], -32768, 32767)
    sq = np.minimum(127, (c * c) >> 19)
    cr = np.clip(first[:, :15] >> 6, 0, 127)
    sin = np.concatenate([sq, cr, np.zeros((len(first), 2), np.int64)], 1)
    second = np.einsum('ni,noi->no', sin, net.l2w[bucket].astype(np.int64)) + net.l2b[bucket]  # small
    out = net.fb[bucket] + (net.fw[bucket].astype(np.int64) * np.clip(second >> 6, 0, 127)).sum(1)
    fwd = tdiv(first[:, 15] * 9600, 8128)
    return out + fwd

def psqt_term(ps, stm, bucket, full=False):
    ar = np.arange(len(stm))
    v = tdiv(ps[ar, stm, bucket] - ps[ar, 1 - stm, bucket], 2)
    return np.where((bucket < 8) | full, v, 0)

def runtime(d, nnsum):
    """nnsum = psqt + positional -> runtime value (AB-JChess runtime_value, optimism 0)."""
    nn = tdiv(nnsum, 16)
    urn, up = d['urn'], d['up']
    revealed = np.maximum(0, INIT_CNT[None, None, :] - urn)
    mat = (urn * PIECE_VAL).sum((1, 2)) + (np.minimum(up, revealed)[:, :, :5] * PIECE_VAL[:5]).sum((1, 2))
    ms = 935 + mat * 93 // 5116
    v = tdiv(nn * ms, 1024)
    v = tdiv(v * (218 - np.minimum(d['nocap'], 120)), 176)
    return np.clip(v, -29507, 29507)
