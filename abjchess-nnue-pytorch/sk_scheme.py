"""SK1 layer-stack selection by face-down (dark) piece count. 16 stacks, JQHNNUE1 file format unchanged.
  dark == 0 : 3 stacks by total piece count  (<=9 -> 0, 10..12 -> 1, >=13 -> 2)
  dark >= 1 : 13 stacks by dark count        (1,2,3,4,5,6-7,8-9,10-11,12-13,14-16,17-19,20-23,24-30 -> 3..15)
Old (L3) scheme: min((pieces-1)//4, 7) + (8 if dark > 0 else 0).
The PSQT bucket is the same index (ft columns W..W+15)."""
import numpy as np

DARK_UPPER = [1, 2, 3, 4, 5, 7, 9, 11, 13, 16, 19, 23, 30]   # inclusive upper bound of each dark>=1 stack (stack 3 + i)
PIECE_UPPER0 = [9, 12, 32]                                    # dark==0 stacks 0..2 by pieces


def sk_bucket(dark, pieces):
    dark = np.asarray(dark); pieces = np.asarray(pieces)
    b = np.zeros(np.broadcast(dark, pieces).shape, dtype=np.int64)
    z = dark == 0
    b0 = np.searchsorted(PIECE_UPPER0, pieces, side="left")
    b1 = 3 + np.searchsorted(DARK_UPPER, dark, side="left")
    return np.where(z, b0, b1)


def table():
    """int table [dark 0..31][pieces 0..32] -> stack (the engine would carry the same 32x33 table)."""
    d = np.arange(32)[:, None]; p = np.arange(33)[None, :]
    return sk_bucket(d, p).astype(np.int64)


def old_bucket(dark, pieces):
    return np.minimum((np.asarray(pieces) - 1) // 4, 7) + np.where(np.asarray(dark) > 0, 8, 0)


if __name__ == "__main__":
    t = table()
    print("dark->stack (pieces=32):", [int(t[d, 32]) for d in range(31)])
    print("dark 0 pieces->stack:", [int(t[0, p]) for p in range(33)])
