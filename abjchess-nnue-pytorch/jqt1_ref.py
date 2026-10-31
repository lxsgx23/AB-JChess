#!/usr/bin/env python3
"""JQT1 threat-feature reference (independent of the C++ engine code).

Trainer-side extractor: jqt1_ref.active(board, persp, mirror) -> sorted feature indices, identical to the engine's
thr::active(). Board: list of 90 entries (sq = rank*9 + file, Red home rank 0), each None or (color, type) with
color 0=Red 1=Black, type in 'KRNBACPD' (D = face-down). parse_fen() reads ABJchess FENs ('x'/'X' = face-down).

  python3 jqt1_ref.py                       # print DIMS and the per-attacker-code table
  python3 jqt1_ref.py --check dump.txt      # cross-check against thr_test's dump (fen|mir0|mir1|idx0|idx1)
"""
import sys

NSQ = 90
KINDS = 'RNBACPD'              # kind index 0..6; relative code = kind + (0 own | 7 enemy)
STARTS = [(0, 0, 'R'), (8, 0, 'R'), (1, 0, 'N'), (7, 0, 'N'), (2, 0, 'B'), (6, 0, 'B'), (3, 0, 'A'), (5, 0, 'A'),
          (1, 2, 'C'), (7, 2, 'C'), (0, 3, 'P'), (2, 3, 'P'), (4, 3, 'P'), (6, 3, 'P'), (8, 3, 'P')]
DARK_ROLE = [[None] * NSQ, [None] * NSQ]
for f, r, t in STARTS:
    DARK_ROLE[0][r * 9 + f] = t
    DARK_ROLE[1][(9 - r) * 9 + f] = t

fr = lambda s: (s % 9, s // 9)
sq = lambda f, r: r * 9 + f
onb = lambda f, r: 0 <= f < 9 and 0 <= r < 10
def own_half(c, s): return s // 9 <= 4 if c == 0 else s // 9 >= 5
def in_palace(c, s):
    f, r = fr(s)
    return 3 <= f <= 5 and (r <= 2 if c == 0 else r >= 7)
def mkind(c, t, s): return DARK_ROLE[c][s] if t == 'D' else t

def rays(s):
    f, r = fr(s)
    out = []
    for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ray, nf, nr = [], f + df, r + dr
        while onb(nf, nr):
            ray.append(sq(nf, nr)); nf += df; nr += dr
        out.append(ray)
    return out

def horse_moves(s):  # (target, leg)
    f, r = fr(s)
    out = []
    for df, dr in ((1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)):
        if onb(f + df, r + dr):
            lf, lr = (f, r + dr // 2) if abs(dr) == 2 else (f + df // 2, r)
            out.append((sq(f + df, r + dr), sq(lf, lr)))
    return out

def eleph_moves(s):  # (target, eye)
    f, r = fr(s)
    return [(sq(f + df, r + dr), sq(f + df // 2, r + dr // 2)) for df, dr in ((2, 2), (2, -2), (-2, 2), (-2, -2))
            if onb(f + df, r + dr)]

def adv_moves(s):
    f, r = fr(s)
    return [sq(f + df, r + dr) for df, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)) if onb(f + df, r + dr)]

def attacks(c, t, s, occ):
    """Squares attacked by piece (c, t) on s given occupancy predicate occ(sq) (None = empty-board geometry)."""
    k, dark = mkind(c, t, s), t == 'D'
    o = occ or (lambda x: False)
    res = []
    if k in ('R', 'C'):
        for ray in rays(s):
            if occ is None:
                res += ray[1:] if k == 'C' else ray
                continue
            hits = [x for x in ray if o(x)]
            if k == 'R' and hits: res.append(hits[0])
            if k == 'C' and len(hits) >= 2: res.append(hits[1])
    elif k == 'N':
        res = [tg for tg, leg in horse_moves(s) if not o(leg)]
    elif k == 'B':
        res = [tg for tg, eye in eleph_moves(s) if not o(eye) and (not dark or own_half(c, tg))]
    elif k == 'A':
        res = [tg for tg in adv_moves(s) if not dark or in_palace(c, tg)]
    elif k == 'P':
        f, r = fr(s)
        nr = r + (1 if c == 0 else -1)
        if 0 <= nr < 10: res.append(sq(f, nr))
        if not own_half(c, s):
            if f > 0: res.append(s - 1)
            if f < 8: res.append(s + 1)
    return res

def code_on(code, s):
    c, t = (0 if code < 7 else 1), KINDS[code % 7]
    return t != 'D' or DARK_ROLE[c][s] is not None

def code_geom(code, s):
    return attacks(0 if code < 7 else 1, KINDS[code % 7], s, None)

def code_mkind(code, s):
    return mkind(0 if code < 7 else 1, KINDS[code % 7], s)

def build_table():
    E, cnt = {}, 0
    for a in range(14):
        for frm in range(NSQ):
            if not code_on(a, frm): continue
            ak, geo = code_mkind(a, frm), set(code_geom(a, frm))
            for to in range(NSQ):
                if to not in geo: continue
                for v in range(14):
                    if not code_on(v, to): continue
                    if code_mkind(v, to) == ak and ak != 'N' and frm < to and frm in code_geom(v, to): continue
                    E[(a, frm, to, v)] = cnt; cnt += 1
    return E, cnt

E, DIMS = build_table()

def orient(s, p, m):
    f, r = fr(s)
    if p == 1: r = 9 - r
    if m: f = 8 - f
    return sq(f, r)

def mirror_flag(board, p):
    """HNN scheme-3 mirror: perspective's own king file > 4 (rank orientation does not change files)."""
    for s, pc in enumerate(board):
        if pc and pc[0] == p and pc[1] == 'K':
            return 1 if s % 9 > 4 else 0
    raise ValueError('no king')

def raw_threats(board):
    occ = lambda x: board[x] is not None
    out = []
    for s, pc in enumerate(board):
        if not pc or pc[1] == 'K': continue
        for t in attacks(pc[0], pc[1], s, occ):
            if board[t] and board[t][1] != 'K': out.append((pc, s, board[t], t))
    return out

def rel(pc, p): return KINDS.index(pc[1]) + (0 if pc[0] == p else 7)

def active(board, p, m=None, check=True):
    if m is None: m = mirror_flag(board, p)
    raw = raw_threats(board)
    keys = {(rel(a, p), orient(f, p, m), rel(v, p), orient(t, p, m)) for a, f, v, t in raw}
    idx = []
    for a, f, v, t in keys:
        x = E.get((a, f, t, v))
        if x is None:
            # must be the dropped half of a reverse duplicate whose partner is active
            assert not check or ((v, t, a, f) in keys and E.get((v, t, f, a)) is not None), (a, f, v, t)
            continue
        idx.append(x)
    return sorted(idx)

def parse_fen(fen):
    rows = fen.split()[0].split('/')
    board = [None] * NSQ
    for i, row in enumerate(rows):
        r, f = 9 - i, 0
        for ch in row:
            if ch.isdigit(): f += int(ch); continue
            t = {'H': 'N', 'E': 'B', 'X': 'D'}.get(ch.upper(), ch.upper())
            board[sq(f, r)] = (0 if ch.isupper() else 1, t)
            f += 1
    return board

def parse_hex(h):
    """engine board bytes (type | color << 4; types K=1 R=2 N=3 B=4 A=5 C=6 P=7 D=8)"""
    board = [None] * NSQ
    for s in range(NSQ):
        x = int(h[2 * s:2 * s + 2], 16)
        if x: board[s] = (x >> 4, ' KRNBACPD'[x & 15])
    return board

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] in ('--check', '--checkhex'):
        n = bad = mbad = 0
        for line in open(sys.argv[2]):
            fen, m0, m1, i0, i1 = line.rstrip('\n').split('|')
            b = parse_fen(fen) if sys.argv[1] == '--check' else parse_hex(fen)
            for p, ms, ids in ((0, m0, i0), (1, m1, i1)):
                want = [int(x) for x in ids.split(',')] if ids else []
                if mirror_flag(b, p) != int(ms): mbad += 1
                if active(b, p, int(ms)) != want:
                    bad += 1
                    if bad < 4: print('MISMATCH', fen, p, file=sys.stderr)
            n += 1
        print(f'DIMS {DIMS} positions {n} persp_mismatch {bad} mirror_mismatch {mbad}')
        sys.exit(1 if bad or mbad else 0)
    print('DIMS', DIMS)
    for a in range(14):
        c = sum(1 for k in E if k[0] == a)
        print(f'attacker code {a:2d} ({"own" if a < 7 else "enemy"} {KINDS[a % 7]}): {c}')
