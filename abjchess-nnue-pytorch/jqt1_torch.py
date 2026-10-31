"""JQT1 threat features on the GPU (batched torch), identical to the engine's thr::active() / jqt1_ref.py.

  tf = ThreatFeaturizer(dev)
  idx0, off0, idx1, off1 = tf(raw_board_uint8 (B,90), persp_abs (B,2) long, mirror (B,2) long)
      -> for each of the 2 bags (k = 0, 1): flat LongTensor of feature indices and (B,) offsets for F.embedding_bag
Candidates = every (attacker piece byte, from, to) in jqt1_ref's empty-board geometry (face-down attackers only on their
start points, moving as darkRole). Active iff board[from] == attacker byte, board[to] is a non-king piece, and the
blocker condition holds (rook: 0 pieces strictly between, cannon: exactly 1, horse: leg empty, elephant: eye empty).
Index table IDX[p][m][cand][victim byte] built from jqt1_ref.E (orientation, relative codes, reverse-dup exclusion = -1).
"""
import numpy as np
import torch
import jqt1_ref as J

TYPE_BYTE = {'K': 1, 'R': 2, 'N': 3, 'B': 4, 'A': 5, 'C': 6, 'P': 7, 'D': 8}
BYTE_TYPE = {v: k for k, v in TYPE_BYTE.items()}


def _between(f, t):
    (ff, fr), (tf, tr) = J.fr(f), J.fr(t)
    if ff == tf:
        return [J.sq(ff, r) for r in range(min(fr, tr) + 1, max(fr, tr))]
    assert fr == tr
    return [J.sq(x, fr) for x in range(min(ff, tf) + 1, max(ff, tf))]


def build():
    cf, ct, cp, ck, cb = [], [], [], [], []   # from, to, attacker byte, block kind, block square / line id
    lines = []                               # between-square lists for rook / cannon candidates
    for c in range(2):
        for kind in J.KINDS:
            for f in range(J.NSQ):
                if kind == 'D' and J.DARK_ROLE[c][f] is None:
                    continue
                mk = J.mkind(c, kind, f)
                horse = dict((tg, leg) for tg, leg in J.horse_moves(f))
                eleph = dict((tg, eye) for tg, eye in J.eleph_moves(f))
                for t in J.attacks(c, kind, f, None):
                    cf.append(f); ct.append(t); cp.append(TYPE_BYTE[kind] | (c << 4))
                    if mk in ('R', 'C'):
                        ck.append(1 if mk == 'R' else 2); cb.append(len(lines)); lines.append(_between(f, t))
                    elif mk == 'N':
                        ck.append(3); cb.append(horse[t])
                    elif mk == 'B':
                        ck.append(4); cb.append(eleph[t])
                    else:
                        ck.append(0); cb.append(0)
    n = len(cf)
    idx = np.full((2, 2, n, 32), -1, np.int64)
    for i in range(n):
        ac, at = cp[i] >> 4, BYTE_TYPE[cp[i] & 15]
        for p in range(2):
            ar = J.KINDS.index(at) + (0 if ac == p else 7)
            for m in range(2):
                fo, to = J.orient(cf[i], p, m), J.orient(ct[i], p, m)
                for vc in range(2):
                    for vt in J.KINDS:
                        vr = J.KINDS.index(vt) + (0 if vc == p else 7)
                        x = J.E.get((ar, fo, to, vr))
                        if x is not None:
                            idx[p, m, i, TYPE_BYTE[vt] | (vc << 4)] = x
    M = np.zeros((J.NSQ, max(1, len(lines))), np.float32)
    for j, l in enumerate(lines):
        M[l, j] = 1.0
    return dict(cf=np.array(cf), ct=np.array(ct), cp=np.array(cp), ck=np.array(ck), cb=np.array(cb), M=M, idx=idx, n=n)


class ThreatFeaturizer:
    def __init__(self, dev):
        d = build()
        self.dev, self.n, self.dims = dev, d['n'], J.DIMS
        t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt, device=dev)
        self.cf, self.ct, self.cb, self.ck = t(d['cf']), t(d['ct']), t(d['cb']), t(d['ck'])
        self.cp = t(d['cp'], torch.uint8)
        self.line = (self.ck == 1) | (self.ck == 2)
        self.li = torch.nonzero(self.line)[:, 0]           # candidate -> column of M
        self.lcol = self.cb[self.li]
        self.leap = torch.nonzero((self.ck == 3) | (self.ck == 4))[:, 0]
        self.M = t(d['M'], torch.float32)
        self.idx = t(d['idx'].reshape(-1), torch.int32)

    @torch.no_grad()
    def active(self, board):
        """board (B,90) uint8 -> (rows, cand) LongTensors of active raw tuples."""
        board = board.to(self.dev)
        typ = board & 15
        occ = (typ > 0)
        ok = (board[:, self.cf] == self.cp[None, :])
        vt = typ[:, self.ct]
        ok &= (vt > 1)                                      # victim present and not a king
        cnt = (occ.float() @ self.M)[:, self.lcol]           # pieces strictly between (exact small ints)
        need = torch.where(self.ck[self.li] == 1, 0.0, 1.0)
        ok[:, self.li] &= (cnt == need[None, :])
        ok[:, self.leap] &= ~occ[:, self.cb[self.leap]]
        r, c = torch.nonzero(ok, as_tuple=True)
        return r, c

    @torch.no_grad()
    def __call__(self, board, persp, mirror):
        """persp, mirror: (B,2) long (bag k uses absolute perspective persp[:,k] and its mirror flag)."""
        board = board.to(self.dev)
        B = len(board)
        r, c = self.active(board)
        vb = board[r, self.ct[c]].long()
        out = []
        for k in range(2):
            p, m = persp[r, k], mirror[r, k]
            x = self.idx[((p * 2 + m) * self.n + c) * 32 + vb].long()
            keep = x >= 0
            xr, xx = r[keep], x[keep]
            cnt = torch.bincount(xr, minlength=B)
            off = torch.cumsum(cnt, 0) - cnt
            out += [xx, off]
        return out
