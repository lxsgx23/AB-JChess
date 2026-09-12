"""AB-JChess V8 feature contract.

The V8 input is the compact HalfKAv2 Jieqi position feature set.  It keeps
the runtime feature table at 31,776 rows (24 king/attack buckets x 1,324
position rows).  Factorized training adds only the 1,324 PSQ rows and the 24
bucket rows; those rows are coalesced before a package is exported.

This module intentionally has no dependency on ``chess`` or Torch.  It is
used by the jqv4 bridge as well as by the Torch-facing code,
so static tools can validate the contract on machines without CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from bisect import bisect_right
from numbers import Integral
from typing import Iterable, Sequence


NUM_SQ = 90
BASE_PS_NB = 14 * NUM_SQ
META_NB = 64
PS_NB = BASE_PS_NB + META_NB
KING_BUCKETS = 6
ATTACK_BUCKETS = 4
NUM_BUCKETS = KING_BUCKETS * ATTACK_BUCKETS
REAL_INPUTS = NUM_BUCKETS * PS_NB

PSQ_FACTOR_INPUTS = PS_NB
BUCKET_FACTOR_INPUTS = NUM_BUCKETS
VIRTUAL_INPUTS = PSQ_FACTOR_INPUTS + BUCKET_FACTOR_INPUTS
TOTAL_INPUTS = REAL_INPUTS + VIRTUAL_INPUTS

NUM_PSQT_BUCKETS = 16
NUM_LAYER_STACKS = 16
FT_DIM = 2048
L1_DIM = 1024
L2_DIM = 15
L3_DIM = 32

# The 32-bit hash is embedded in training checkpoints.  The complete identity is the
# SHA-256 document returned by ``feature_identity_sha256`` below.
FEATURE_HASH = 0xB81A2026

EMPTY = 0xFF

# jqv4 ObservationFieldsV1 piece codes (the on-disk representation).
W_ROOK, W_ADVISOR, W_CANNON, W_PAWN, W_KNIGHT, W_BISHOP, W_KING = range(7)
B_ROOK, B_ADVISOR, B_CANNON, B_PAWN, B_KNIGHT, B_BISHOP, B_KING = range(7, 14)
DARK_WHITE, DARK_BLACK = 14, 15

ROOK, ADVISOR, CANNON, PAWN, KNIGHT, BISHOP, KING = range(1, 8)

PIECE_VALUES = {
    ROOK: 1080,
    ADVISOR: 243,
    CANNON: 769,
    PAWN: 149,
    KNIGHT: 616,
    BISHOP: 218,
}

PS_W_ROOK = 0 * NUM_SQ
PS_B_ROOK = 1 * NUM_SQ
PS_W_CANNON = 2 * NUM_SQ
PS_B_CANNON = 3 * NUM_SQ
PS_W_KNIGHT = 4 * NUM_SQ
PS_B_KNIGHT = 5 * NUM_SQ
PS_W_PAWN = 6 * NUM_SQ
PS_B_PAWN = 7 * NUM_SQ
PS_W_ADVISOR = 8 * NUM_SQ
PS_B_ADVISOR = 9 * NUM_SQ
PS_W_BISHOP = 10 * NUM_SQ
PS_B_BISHOP = 11 * NUM_SQ
PS_WB_KING = 12 * NUM_SQ
PS_DARK = 13 * NUM_SQ
PS_UNKNOWN_OWN = PS_DARK + 15
PS_UNKNOWN_ENEMY = PS_DARK + 78

BALANCE_ENCODING = 0xA4A92A74E989D3A7
UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class Position:
    """Decoded jqv4 observation used by the pure-Python feature encoder."""

    board: Sequence[int]
    rest: Sequence[Sequence[int]]
    side_to_move: int = 0
    # ``None`` is accepted for hand-built fixtures.  A decoded observation
    # supplies the two-side tuple explicitly and can perform the inventory check.
    unknown_loss: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if len(self.board) != NUM_SQ:
            raise ValueError(f"board must contain {NUM_SQ} squares")
        if len(self.rest) != 2 or any(len(row) != 6 for row in self.rest):
            raise ValueError("rest must have shape [2][6]")
        if self.side_to_move not in (0, 1):
            raise ValueError("side_to_move must be 0 or 1")

        for sq, piece in enumerate(self.board):
            if not isinstance(piece, Integral):
                raise ValueError(f"board[{sq}] must be an integer piece code")
            if int(piece) < 0 or int(piece) > 0xFF:
                raise ValueError(f"board[{sq}] is outside the uint8 piece domain")
        for color, row in enumerate(self.rest):
            for ptype, value in enumerate(row):
                if not isinstance(value, Integral) or isinstance(value, bool):
                    raise ValueError(f"rest[{color}][{ptype}] must be a non-negative integer")
                if int(value) < 0:
                    raise ValueError(f"rest[{color}][{ptype}] must be non-negative")

        explicit_unknown_loss = self.unknown_loss is not None
        values = (0, 0) if self.unknown_loss is None else self.unknown_loss
        if len(values) != 2:
            raise ValueError("unknown_loss must have exactly two side values")
        normalized_unknown_loss: list[int] = []
        for color, value in enumerate(values):
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise ValueError(f"unknown_loss[{color}] must be an integer")
            value = int(value)
            # jqv4 stores bounded per-side loss counts.  The feature itself
            # caps at 3+, but values above the wire domain are malformed.
            if value < 0 or value > 15:
                raise ValueError(f"unknown_loss[{color}] must be in 0..15")
            normalized_unknown_loss.append(value)
        object.__setattr__(self, "unknown_loss", tuple(normalized_unknown_loss))

        if explicit_unknown_loss:
            dark_by_color = self.dark_count_by_color
            for color in (0, 1):
                # Code 16 is an uncoloured dark marker.  It has no owner, so
                # an explicit observation cannot pass the
                # per-side inventory identity check when it is present.
                if any(int(piece) == 16 for piece in self.board):
                    raise ValueError(
                        "unknown_loss inventory cannot be validated for uncoloured dark pieces"
                    )
                expected = dark_by_color[color] + normalized_unknown_loss[color]
                if _rest_count(self, color) != expected:
                    raise ValueError(
                        "unknown_loss does not match rest/dark inventory "
                        f"for color {color}: rest={_rest_count(self, color)}, expected={expected}"
                    )

    @property
    def piece_count(self) -> int:
        return sum(int(piece) != EMPTY for piece in self.board)

    @property
    def dark_count(self) -> int:
        return sum(int(piece) in (DARK_WHITE, DARK_BLACK, 16) for piece in self.board)

    @property
    def dark_count_by_color(self) -> tuple[int, int]:
        """Return colored dark-square counts in white/black order.

        Code 16 is intentionally excluded: it marks an occupied dark square
        without preserving its owner and therefore cannot support the
        rest-inventory identity check.
        """

        return (
            sum(int(piece) == DARK_WHITE for piece in self.board),
            sum(int(piece) == DARK_BLACK for piece in self.board),
        )


def square(file: int, rank: int) -> int:
    if not (0 <= file < 9 and 0 <= rank < 10):
        raise ValueError("square is outside the 9x10 board")
    return rank * 9 + file


def file_of(sq: int) -> int:
    return int(sq) % 9


def rank_of(sq: int) -> int:
    return int(sq) // 9


def flip_file(sq: int) -> int:
    return rank_of(sq) * 9 + (8 - file_of(sq))


def flip_rank(sq: int) -> int:
    return (9 - rank_of(sq)) * 9 + file_of(sq)


def _piece_parts(piece: int) -> tuple[int, int] | None:
    """Return ``(color, type)`` for a visible jqv4 piece, or ``None``.

    jqv4's compact codes intentionally overlap the engine's internal 1..15
    piece codes, so accepting both representations here would be ambiguous.
    The Python bridge therefore follows the on-disk jqv4 encoding exactly;
    the native engine has its own encoder.
    """

    piece = int(piece)
    if 0 <= piece <= 6:
        return 0, piece + 1
    if 7 <= piece <= 13:
        return 1, piece - 6
    return None


def _is_dark(piece: int) -> bool:
    return int(piece) in (DARK_WHITE, DARK_BLACK, 16)


def _piece_square_plane(perspective: int, piece: int) -> int:
    parts = _piece_parts(piece)
    if parts is None:
        raise ValueError(f"piece {piece!r} is not a visible Jieqi piece")
    color, ptype = parts
    own = color == perspective
    if ptype == ROOK:
        return PS_W_ROOK if own else PS_B_ROOK
    if ptype == ADVISOR:
        return PS_W_ADVISOR if own else PS_B_ADVISOR
    if ptype == CANNON:
        return PS_W_CANNON if own else PS_B_CANNON
    if ptype == PAWN:
        return PS_W_PAWN if own else PS_B_PAWN
    if ptype == KNIGHT:
        return PS_W_KNIGHT if own else PS_B_KNIGHT
    if ptype == BISHOP:
        return PS_W_BISHOP if own else PS_B_BISHOP
    if ptype == KING:
        return PS_WB_KING
    raise ValueError(f"piece type {ptype!r} is not supported")


def _dark_rest_index(perspective: int, owner: int, ptype: int) -> int:
    if ptype == KING:
        raise ValueError("kings have no dark-rest feature")
    if not 1 <= ptype <= 6:
        raise ValueError(f"invalid dark-rest piece type {ptype}")
    base = 9 if owner == perspective else 72
    return PS_DARK + base + (ptype - 1)


def _piece_type(piece: int) -> int | None:
    parts = _piece_parts(piece)
    return None if parts is None else parts[1]


def _on_board(file: int, rank: int) -> bool:
    return 0 <= file < 9 and 0 <= rank < 10


def _occupied(position: Position, sq: int) -> bool:
    return int(position.board[sq]) != EMPTY


def _visible_piece(position: Position, sq: int) -> tuple[int, int] | None:
    piece = int(position.board[sq])
    if piece == EMPTY or _is_dark(piece):
        return None
    return _piece_parts(piece)


_ORTHOGONAL_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONAL_DELTAS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_KNIGHT_DELTAS = (
    (1, 2), (-1, 2), (1, -2), (-1, -2),
    (2, 1), (2, -1), (-2, 1), (-2, -1),
)
_BISHOP_DELTAS = ((2, 2), (2, -2), (-2, 2), (-2, -2))


def _in_palace(sq: int) -> bool:
    file, rank = file_of(sq), rank_of(sq)
    return 3 <= file <= 5 and (rank <= 2 or rank >= 7)


def _actual_attacks(position: Position, piece: int, sq: int) -> Iterable[int]:
    """Yield occupied-board-aware attack destinations for one visible piece."""

    parts = _piece_parts(piece)
    if parts is None:
        return
    color, ptype = parts
    file, rank = file_of(sq), rank_of(sq)

    if ptype == ROOK:
        for df, dr in _ORTHOGONAL_DELTAS:
            nf, nr = file + df, rank + dr
            while _on_board(nf, nr):
                to = square(nf, nr)
                yield to
                if _occupied(position, to):
                    break
                nf, nr = nf + df, nr + dr
        return

    if ptype == CANNON:
        for df, dr in _ORTHOGONAL_DELTAS:
            nf, nr = file + df, rank + dr
            screen = False
            while _on_board(nf, nr):
                to = square(nf, nr)
                if screen:
                    yield to
                if _occupied(position, to):
                    if not screen:
                        screen = True
                    else:
                        break
                nf, nr = nf + df, nr + dr
        return

    if ptype == KNIGHT:
        for df, dr in _KNIGHT_DELTAS:
            leg_file = file + (df // 2 if abs(df) == 2 else 0)
            leg_rank = rank + (dr // 2 if abs(dr) == 2 else 0)
            if _on_board(file + df, rank + dr) and not _occupied(
                position, square(leg_file, leg_rank)
            ):
                yield square(file + df, rank + dr)
        return

    if ptype == BISHOP:
        for df, dr in _BISHOP_DELTAS:
            nf, nr = file + df, rank + dr
            eye_file, eye_rank = file + df // 2, rank + dr // 2
            if _on_board(nf, nr) and not _occupied(position, square(eye_file, eye_rank)):
                # Xiangqi bishops cannot cross the river.  The source and
                # destination must remain in the same home half.
                if (rank <= 4 and nr <= 4) or (rank >= 5 and nr >= 5):
                    yield square(nf, nr)
        return

    if ptype == ADVISOR:
        if not _in_palace(sq):
            return
        for df, dr in _DIAGONAL_DELTAS:
            nf, nr = file + df, rank + dr
            to = square(nf, nr) if _on_board(nf, nr) else -1
            if to >= 0 and _in_palace(to):
                yield to
        return

    if ptype == KING:
        if not _in_palace(sq):
            return
        for df, dr in _ORTHOGONAL_DELTAS:
            nf, nr = file + df, rank + dr
            to = square(nf, nr) if _on_board(nf, nr) else -1
            if to >= 0 and _in_palace(to):
                yield to
        return

    if ptype == PAWN:
        forward = 1 if color == 0 else -1
        if _on_board(file, rank + forward):
            yield square(file, rank + forward)
        crossed = (color == 0 and rank > 4) or (color == 1 and rank < 5)
        if crossed:
            for df in (-1, 1):
                if _on_board(file + df, rank):
                    yield square(file + df, rank)


def _visible_attack_targets(position: Position, attacker_color: int) -> set[int]:
    targets: set[int] = set()
    for sq, piece in enumerate(position.board):
        parts = _visible_piece(position, sq)
        if parts is None or parts[0] != attacker_color:
            continue
        for to in _actual_attacks(position, int(piece), sq):
            target = _visible_piece(position, to)
            if target is not None and target[0] != attacker_color:
                targets.add(to)
    return targets


def visible_threat_summary(position: Position, perspective: int) -> tuple[int, int]:
    """Count distinct visible enemy and own targets attacked by each side."""

    if perspective not in (0, 1):
        raise ValueError("perspective must be 0 or 1")
    enemy = 1 - perspective
    enemy_targets = _visible_attack_targets(position, perspective)
    own_targets = _visible_attack_targets(position, enemy)
    return min(7, len(enemy_targets)), min(7, len(own_targets))


def attack_bucket(position: Position, color: int) -> int:
    """Return the four-way (rook, minor/cannon) attack bucket."""

    if color not in (0, 1):
        raise ValueError("color must be 0 or 1")
    rooks = knights = cannons = 0
    for piece in position.board:
        parts = _piece_parts(piece)
        if parts is None or parts[0] != color or _is_dark(piece):
            continue
        ptype = parts[1]
        if ptype == ROOK:
            rooks = min(2, rooks + 1)
        elif ptype == CANNON:
            cannons = min(2, cannons + 1)
        elif ptype == KNIGHT:
            knights = min(2, knights + 1)
    return int(bool(rooks)) * 2 + int(bool(knights + cannons))


def _mid_mirror_encoding(piece: int, sq: int) -> int:
    parts = _piece_parts(piece)
    if parts is None:
        return 0
    color, ptype = parts
    if not 1 <= ptype <= 7 or file_of(sq) == 4:
        return 0
    if ptype == KING:
        return 1 << 63
    shifts = ((0, 0), (44, 0), (60, 36), (47, 7), (53, 21), (50, 14), (57, 29), (0, 0))
    rank = rank_of(sq)
    r_norm = rank if color == 0 else 9 - rank
    f = file_of(sq)
    f_norm = f if f < 4 else 8 - f
    s1, s2 = shifts[ptype]
    encoding = (1 << s1) | (((3 - f_norm) * 10 + r_norm) << s2)
    return encoding if f < 4 else (-encoding) & UINT64_MASK


def mid_encoding(position: Position, color: int) -> int:
    if color not in (0, 1):
        raise ValueError("color must be 0 or 1")
    value = BALANCE_ENCODING
    for sq, piece in enumerate(position.board):
        if piece == EMPTY or _is_dark(piece):
            continue
        parts = _piece_parts(piece)
        if parts is not None and parts[0] == color:
            value = (value + _mid_mirror_encoding(piece, sq)) & UINT64_MASK
    return value


def requires_mid_mirror(position: Position, color: int) -> bool:
    own = mid_encoding(position, color)
    enemy = mid_encoding(position, 1 - color)
    return bool((1 << 63) & own & enemy) and (own < BALANCE_ENCODING or (own == BALANCE_ENCODING and enemy < BALANCE_ENCODING))


_KING_BUCKET_TABLE = (
    0, 0, 0, 0, 1, 0x8, 0, 0, 0,
    0, 0, 0, 2, 3, 0xA, 0, 0, 0,
    0, 0, 0, 4, 5, 0xC, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 4, 5, 0xC, 0, 0, 0,
    0, 0, 0, 2, 3, 0xA, 0, 0, 0,
    0, 0, 0, 0, 1, 0x8, 0, 0, 0,
)


def king_bucket(ksq: int, oksq: int, mid_mirror: bool = False) -> tuple[int, bool]:
    if not (0 <= ksq < NUM_SQ and 0 <= oksq < NUM_SQ):
        raise ValueError("king square is outside the board")
    own = _KING_BUCKET_TABLE[ksq]
    enemy = _KING_BUCKET_TABLE[oksq]
    bucket = own & 0x7
    mirror = bool(own >> 3) or bool((bucket & 1) and ((enemy >> 3) or ((enemy & 0x7) & 1 and mid_mirror)))
    return bucket, mirror


def _king_square(position: Position, color: int) -> int:
    wanted = W_KING if color == 0 else B_KING
    # Scan the unambiguous compact wire code first.  Normalized engine codes
    # overlap compact black pieces and dark markers, so a wire-level king must
    # not be shadowed by an earlier normalized match.
    for sq, piece in enumerate(position.board):
        if int(piece) == wanted:
            return sq
    normalized = 7 if color == 0 else 15
    for sq, piece in enumerate(position.board):
        if int(piece) == normalized:
            return sq
    return 4 if color == 0 else 85


def _base_layer_bucket(piece_count: int) -> int:
    table = (-1, -1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
             6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 14, 15)
    return table[max(2, min(32, int(piece_count)))]


def _rest_count(position: Position, color: int) -> int:
    return sum(int(value) for value in position.rest[color])


def _strong_rest_count(position: Position, color: int) -> int:
    # Rook, cannon, and knight are the strong/attacking inventory classes.
    return int(position.rest[color][0]) + int(position.rest[color][2]) + int(position.rest[color][4])


def _strong_rest_profile(position: Position, color: int) -> int:
    return ((1 if int(position.rest[color][0]) > 0 else 0)
            | (2 if int(position.rest[color][2]) > 0 else 0)
            | (4 if int(position.rest[color][4]) > 0 else 0))


def _dark_density_bucket(count: int) -> int:
    if count == 0:
        return 0
    if count <= 2:
        return 1
    if count <= 4:
        return 2
    if count <= 8:
        return 3
    if count <= 12:
        return 4
    if count <= 16:
        return 5
    if count <= 24:
        return 6
    return 7


def unknown_loss_bucket(position: Position, perspective: int) -> int:
    """Return the perspective-relative ``unknown_loss`` 0..3+ bucket."""

    if perspective not in (0, 1):
        raise ValueError("perspective must be 0 or 1")
    return min(3, int(position.unknown_loss[perspective]))


def _uncertainty_phase_bucket(dark_count: int, strong_rest: int) -> int:
    dark_bucket = 4 if dark_count >= 17 else 3 if dark_count >= 9 else 2 if dark_count >= 5 else 1 if dark_count >= 1 else 0
    rest_bucket = 3 if strong_rest >= 9 else 2 if strong_rest >= 5 else 1 if strong_rest >= 2 else 0
    return dark_bucket + rest_bucket


def layer_stack_bucket(position: Position) -> int:
    strong_rest = _strong_rest_count(position, 0) + _strong_rest_count(position, 1)
    phase = _uncertainty_phase_bucket(position.dark_count, strong_rest)
    return max(0, min(15, _base_layer_bucket(position.piece_count) + phase))


# The table maps the clamped piece count to the existing discrete base head.
# Each interval is represented as [lower, upper), making the residual stay in
# [0, 1) even at the last count in a bucket.
_PIECE_BUCKET_INTERVALS = (
    (2, 7), (7, 9), (9, 11), (11, 13), (13, 15), (15, 17),
    (17, 19), (19, 21), (21, 23), (23, 25), (25, 27), (27, 29),
    (29, 31), (31, 32), (32, 33), (33, 34),
)
_DARK_PHASE_THRESHOLDS = (0, 1, 5, 9, 17, 33)
_STRONG_REST_PHASE_THRESHOLDS = (0, 2, 5, 9, 19)


def _interval_residual(value: int, thresholds: Sequence[int]) -> float:
    """Normalize ``value`` within its threshold interval, capped at 1."""

    value = max(0, int(value))
    bucket = bisect_right(thresholds, value) - 1
    if bucket < 0:
        bucket = 0
    if bucket >= len(thresholds) - 1:
        return 0.0
    lower, upper = thresholds[bucket], thresholds[bucket + 1]
    if upper <= lower:
        return 0.0
    return min(upper - lower - 1, max(0, value - lower)) / float(upper - lower)


def _piece_residual(piece_count: int) -> float:
    count = max(2, min(32, int(piece_count)))
    base = _base_layer_bucket(count)
    lower, upper = _PIECE_BUCKET_INTERVALS[base]
    if upper <= lower:
        return 0.0
    return min(upper - lower - 1, max(0, count - lower)) / float(upper - lower)


def _dark_residual(dark_count: int) -> float:
    return _interval_residual(dark_count, _DARK_PHASE_THRESHOLDS)


def _strong_rest_residual(strong_rest: int) -> float:
    return _interval_residual(strong_rest, _STRONG_REST_PHASE_THRESHOLDS)


def layer_stack_selection(position: Position) -> tuple[int, int]:
    """Return the continuous layer-stack selection as ``(floor, blend_q8)``."""

    strong_rest = _strong_rest_count(position, 0) + _strong_rest_count(position, 1)
    floor = max(
        0,
        min(
            15,
            _base_layer_bucket(position.piece_count)
            + _uncertainty_phase_bucket(position.dark_count, strong_rest),
        ),
    )
    fraction = (
        _piece_residual(position.piece_count)
        + _dark_residual(position.dark_count)
        + _strong_rest_residual(strong_rest)
    ) / 3.0
    if floor == 15:
        return 15, 0

    blend_q8 = int(math.floor(fraction * 255.0 + 0.5))
    if blend_q8 >= 255:
        # A rounded carry is represented canonically by the next endpoint.
        return min(15, floor + 1), 0
    return floor, max(0, min(254, blend_q8))


def _map_square(perspective: int, sq: int, mirror: bool) -> int:
    if mirror:
        sq = flip_file(sq)
    if perspective == 1:
        sq = flip_rank(sq)
    return sq


def board_feature_index(perspective: int, sq: int, piece: int, bucket: int, mirror: bool) -> int:
    if not (perspective in (0, 1) and 0 <= sq < NUM_SQ and 0 <= bucket < NUM_BUCKETS):
        raise ValueError("invalid board feature arguments")
    mapped = _map_square(perspective, sq, mirror)
    plane = PS_DARK if _is_dark(piece) else _piece_square_plane(perspective, piece)
    index = bucket * PS_NB + plane + mapped
    if not 0 <= index < REAL_INPUTS:
        raise AssertionError("board feature index escaped the V8 real table")
    return index


def meta_feature_offsets(position: Position, perspective: int) -> tuple[int, ...]:
    if perspective not in (0, 1):
        raise ValueError("perspective must be 0 or 1")
    enemy = 1 - perspective
    enemy_targets, own_targets = visible_threat_summary(position, perspective)
    return (
        min(7, _strong_rest_count(position, perspective)),
        8 + min(7, _strong_rest_count(position, enemy)),
        16 + min(7, _rest_count(position, perspective)),
        24 + min(7, _rest_count(position, enemy)),
        32 + _dark_density_bucket(position.dark_count),
        40 + unknown_loss_bucket(position, perspective),
        44 + unknown_loss_bucket(position, enemy),
        48 + enemy_targets,
        56 + own_targets,
    )


def active_features(position: Position, perspective: int, factorized: bool = False) -> list[int]:
    """Encode one perspective as sparse real (and optional virtual) rows."""

    if perspective not in (0, 1):
        raise ValueError("perspective must be 0 or 1")
    ksq = _king_square(position, perspective)
    oksq = _king_square(position, 1 - perspective)
    king, mirror = king_bucket(ksq, oksq, requires_mid_mirror(position, perspective))
    bucket = king * ATTACK_BUCKETS + attack_bucket(position, perspective)

    rows: list[int] = []
    for sq, piece in enumerate(position.board):
        if int(piece) == EMPTY:
            continue
        rows.append(board_feature_index(perspective, sq, int(piece), bucket, mirror))

    for owner in (0, 1):
        for ptype, count in enumerate(position.rest[owner], start=1):
            if ptype == KING:
                continue
            for _ in range(int(count)):
                rows.append(bucket * PS_NB + _dark_rest_index(perspective, owner, ptype))

    meta_base = bucket * PS_NB + BASE_PS_NB
    rows.extend(meta_base + offset for offset in meta_feature_offsets(position, perspective))

    if factorized:
        virtual: list[int] = []
        for row in rows:
            virtual.append(REAL_INPUTS + (row % PS_NB))
            virtual.append(REAL_INPUTS + PSQ_FACTOR_INPUTS + (row // PS_NB))
        rows.extend(virtual)
    return rows


def real_feature_factors(idx: int) -> list[int]:
    if not 0 <= int(idx) < REAL_INPUTS:
        raise IndexError(idx)
    idx = int(idx)
    return [idx, REAL_INPUTS + (idx % PS_NB), REAL_INPUTS + PSQ_FACTOR_INPUTS + (idx // PS_NB)]


def initial_psqt_features() -> list[int]:
    values = [0] * REAL_INPUTS
    for bucket in range(NUM_BUCKETS):
        base = bucket * PS_NB
        for sq in range(NUM_SQ):
            # Keep this explicit mapping synchronized with the V3 C++ loader
            # and runtime encoder; the plane order is not numeric piece order.
            for plane, value in (
                (PS_W_ROOK, PIECE_VALUES[ROOK]),
                (PS_B_ROOK, -PIECE_VALUES[ROOK]),
                (PS_W_CANNON, PIECE_VALUES[CANNON]),
                (PS_B_CANNON, -PIECE_VALUES[CANNON]),
                (PS_W_KNIGHT, PIECE_VALUES[KNIGHT]),
                (PS_B_KNIGHT, -PIECE_VALUES[KNIGHT]),
                (PS_W_PAWN, PIECE_VALUES[PAWN]),
                (PS_B_PAWN, -PIECE_VALUES[PAWN]),
                (PS_W_ADVISOR, PIECE_VALUES[ADVISOR]),
                (PS_B_ADVISOR, -PIECE_VALUES[ADVISOR]),
                (PS_W_BISHOP, PIECE_VALUES[BISHOP]),
                (PS_B_BISHOP, -PIECE_VALUES[BISHOP]),
            ):
                values[base + plane + sq] = value
    return values


def feature_identity_document(feature_name: str = "HalfKAv2_hm_jieqi_v8") -> dict:
    if feature_name not in FEATURE_NAMES:
        raise ValueError(f"unsupported V8 feature name: {feature_name!r}")
    return {
        "schema": "abjchess-v8.1-feature-identity-v1",
        "name": feature_name,
        "hash": FEATURE_HASH,
        "real_features": REAL_INPUTS,
        "virtual_features": VIRTUAL_INPUTS if feature_name.endswith("^") else 0,
        "total_features": TOTAL_INPUTS if feature_name.endswith("^") else REAL_INPUTS,
        "ps_nb": PS_NB,
        "king_buckets": KING_BUCKETS,
        "attack_buckets": ATTACK_BUCKETS,
        "meta_features": META_NB,
        "mid_mirror_encoding": "pikafish-v2",
        "layer_stack_bucketing": "v3-uncertainty-phase-16",
        "metadata_layout": "unknown-loss-threat-summary-v1",
        "unknown_loss_offsets": [40, 44],
        "threat_summary_offsets": [48, 56],
        "layer_stack_selection": "continuous-q0.8-v1",
        "layer_stack_selection_q_format": "Q0.8",
    }


def feature_identity_sha256(feature_set: object | str) -> str:
    name = feature_set if isinstance(feature_set, str) else getattr(feature_set, "name", None)
    if not isinstance(name, str) or name not in FEATURE_NAMES:
        raise ValueError(f"unsupported V8 feature name: {name!r}")
    encoded = json.dumps(feature_identity_document(name), sort_keys=True, separators=(",", ":")).encode("ascii")
    return sha256(encoded).hexdigest()


class FeatureBlockV8:
    def __init__(self, name: str, factorized: bool = False) -> None:
        self.name = name
        self.hash = FEATURE_HASH
        self.num_real_features = REAL_INPUTS
        self.num_virtual_features = VIRTUAL_INPUTS if factorized else 0
        self.num_features = self.num_real_features + self.num_virtual_features
        self.num_psqt_buckets = NUM_PSQT_BUCKETS
        self.num_ls_buckets = NUM_LAYER_STACKS
        self.ft_dim, self.l1, self.l2, self.l3 = FT_DIM, L1_DIM, L2_DIM, L3_DIM
        self.factorized = factorized

    def get_feature_factors(self, idx: int) -> list[int]:
        return real_feature_factors(idx) if self.factorized else [int(idx)]

    def get_initial_psqt_features(self) -> list[int]:
        values = initial_psqt_features()
        return values + ([0] * VIRTUAL_INPUTS if self.factorized else [])

    def get_active_features(self, board: Position, perspective: int = 0) -> list[int]:
        return active_features(board, perspective, self.factorized)


class FeatureSetV8:
    def __init__(self, block: FeatureBlockV8) -> None:
        self.features = [block]
        self.name = block.name
        self.hash = block.hash
        self.num_real_features = block.num_real_features
        self.num_virtual_features = block.num_virtual_features
        self.num_features = block.num_features
        self.num_psqt_buckets = block.num_psqt_buckets
        self.num_ls_buckets = block.num_ls_buckets
        self.ft_dim, self.l1, self.l2, self.l3 = block.ft_dim, block.l1, block.l2, block.l3
        self.factorized = block.factorized

    def get_virtual_feature_ranges(self) -> list[tuple[int, int]]:
        return [(self.num_real_features, self.num_features)] if self.num_virtual_features else []

    def get_real_feature_ranges(self) -> list[tuple[int, int]]:
        return [(0, self.num_real_features)]

    def get_feature_factors(self, idx: int) -> list[int]:
        return self.features[0].get_feature_factors(idx)

    def get_virtual_to_real_features_gather_indices(self) -> list[list[int]]:
        return [self.get_feature_factors(idx) for idx in range(self.num_real_features)]

    def get_initial_psqt_features(self) -> list[int]:
        return self.features[0].get_initial_psqt_features()

    def get_active_features(self, board: Position, perspective: int = 0) -> list[int]:
        return self.features[0].get_active_features(board, perspective)


FEATURE_NAMES = (
    "HalfKAv2_hm_jieqi_v8",
    "HalfKAv2_hm_jieqi_v8^",
)


def get_feature_set_from_name(name: str) -> FeatureSetV8:
    if name not in FEATURE_NAMES:
        raise KeyError(f"unknown V8 feature set {name!r}; choose one of {FEATURE_NAMES}")
    return FeatureSetV8(FeatureBlockV8(name, name.endswith("^")))


def get_available_feature_blocks_names() -> list[str]:
    return list(FEATURE_NAMES)


def add_argparse_args(parser) -> None:
    parser.add_argument("--features", default=FEATURE_NAMES[1], choices=FEATURE_NAMES,
                        help="V8 feature set (the ^ suffix enables factorized training)")


def feature_contract() -> dict:
    return {
        "feature_names": list(FEATURE_NAMES),
        "real_inputs": REAL_INPUTS,
        "virtual_inputs": VIRTUAL_INPUTS,
        "total_inputs": TOTAL_INPUTS,
        "ps_nb": PS_NB,
        "num_buckets": NUM_BUCKETS,
        "architecture": {"ft_dim": FT_DIM, "l1": L1_DIM, "l2": L2_DIM, "l3": L3_DIM,
                          "psqt_buckets": NUM_PSQT_BUCKETS, "layer_stacks": NUM_LAYER_STACKS},
        "feature_identity_sha256": {name: feature_identity_sha256(name) for name in FEATURE_NAMES},
    }
