/*
  ABJchess (AB JChess), a UCI Jieqi engine.
  Contains code derived from AB-JChess / Pikafish / Stockfish:
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)
  Based on jqe, Copyright (c) 2026 Devin O'Keefe (MIT licence, see abjchess/LICENSE-jqe-MIT.txt).
  See THIRD_PARTY_NOTICES.md at the repository root.

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
// Reference (original mailbox) attack tests and move generation, used only by the JQ_BB_VERIFY build to check
// every bitboard answer against the original code. Verbatim copies of fast2 position.cpp with Position -> RefPos.
#ifdef JQ_BB_VERIFY
#include "position_ref.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
namespace jq {
namespace {
inline void add(Move*& out, int from, int to) { *out++ = make_move(from, to); }
}
namespace {
inline bool same_line(int a, int b) { return file_of(a) == file_of(b) || rank_of(a) == rank_of(b); }
inline bool diag_adjacent(int a, int b) {
    return std::abs(file_of(a) - file_of(b)) == 1 && std::abs(rank_of(a) - rank_of(b)) == 1;
}
inline int cheb(int a, int b) { return std::max(std::abs(file_of(a) - file_of(b)), std::abs(rank_of(a) - rank_of(b))); }
// A non-king move can change attacks on king k only through k's file/rank (sliders, cannon screens,
// facing generals) or k's diagonal neighbours (horse legs and elephant eyes).
inline bool touches_king_lines(int from, int to, int k) {
    return same_line(from, k) || same_line(to, k) || diag_adjacent(from, k);
}
}  // namespace

bool RefPos::ray_attacked(Color c, int k, int d) const {
    const Color e = ~c;
    const int n = T.rayLen[k][d];
    const uint8_t* r = T.ray[k][d];
    int i = 0;
    while (i < n && !board[r[i]]) ++i;
    if (i == n) return false;
    const Piece p = board[r[i]];
    if (color_of(p) == e) {
        const PieceType t = role(r[i]);
        if (t == ROOK) return true;
        if (t == KING && d < 2) return true;  // facing generals
    }
    for (++i; i < n; ++i) {
        if (!board[r[i]]) continue;
        return color_of(board[r[i]]) == e && role(r[i]) == CANNON;
    }
    return false;
}

bool RefPos::king_attacked(Color c, int k) const {
    const Color e = ~c;
    for (int d = 0; d < 4; ++d)
        if (ray_attacked(c, k, d)) return true;
    for (int j = 0; j < T.horseFromN[k]; ++j) {
        const int p = T.horseFrom[k][j];
        if (board[p] && color_of(board[p]) == e && role(p) == HORSE && !board[T.horseFromLeg[k][j]]) return true;
    }
    // Enemy soldiers attack forward (towards our home) and sideways once across the river.
    const int back = c == RED ? 9 : -9;  // square "in front" of our king from the enemy soldier's view
    const int s1 = k + back;
    if (s1 >= 0 && s1 < NUM_SQ && board[s1] && color_of(board[s1]) == e && role(s1) == SOLDIER) return true;
    if (file_of(k) > 0) {
        const int s = k - 1;
        if (board[s] && color_of(board[s]) == e && role(s) == SOLDIER && !own_half(e, s)) return true;
    }
    if (file_of(k) < 8) {
        const int s = k + 1;
        if (board[s] && color_of(board[s]) == e && role(s) == SOLDIER && !own_half(e, s)) return true;
    }
    // Revealed advisors and elephants roam freely in Jieqi.
    for (int j = 0; j < T.advN[k]; ++j) {
        const int s = T.advTo[k][j];
        if (board[s] && color_of(board[s]) == e && role(s) == ADVISOR) {
            if (type_of(board[s]) != DARK || in_palace(e, k)) return true;
        }
    }
    for (int j = 0; j < T.elephN[k]; ++j) {
        const int s = T.elephTo[k][j];
        if (board[s] && color_of(board[s]) == e && role(s) == ELEPHANT && !board[T.elephEye[k][j]]) {
            if (type_of(board[s]) != DARK || own_half(e, k)) return true;
        }
    }
    return false;
}

template <bool CAP>
int RefPos::gen(Move* list) const {
    Move* out = list;
    const Color us = stm;
    for (int i = 0; i < pcount[us]; ++i) {
        const int s = plist[us][i];
        const Piece pc = board[s];
        const bool dark = type_of(pc) == DARK;
        const PieceType t = role(s);
        auto target_ok = [&](int to) {
            const Piece q = board[to];
            return !q || (color_of(q) != us && type_of(q) != KING);
        };
        switch (t) {
        case KING:
            for (int j = 0; j < T.kingN[s]; ++j)
                if (target_ok(T.kingTo[s][j]) && (!CAP || board[T.kingTo[s][j]])) add(out, s, T.kingTo[s][j]);
            break;
        case ROOK:
            for (int d = 0; d < 4; ++d)
                for (int j = 0; j < T.rayLen[s][d]; ++j) {
                    const int to = T.ray[s][d][j];
                    if (!board[to]) { if (!CAP) add(out, s, to); continue; }
                    if (target_ok(to)) add(out, s, to);
                    break;
                }
            break;
        case CANNON:
            for (int d = 0; d < 4; ++d) {
                int j = 0;
                const int n = T.rayLen[s][d];
                for (; j < n; ++j) {
                    const int to = T.ray[s][d][j];
                    if (board[to]) break;
                    if (!CAP) add(out, s, to);
                }
                for (++j; j < n; ++j) {
                    const int to = T.ray[s][d][j];
                    if (!board[to]) continue;
                    if (target_ok(to)) add(out, s, to);
                    break;
                }
            }
            break;
        case HORSE:
            for (int j = 0; j < T.horseN[s]; ++j)
                if (!board[T.horseLeg[s][j]] && target_ok(T.horseTo[s][j]) && (!CAP || board[T.horseTo[s][j]])) add(out, s, T.horseTo[s][j]);
            break;
        case ELEPHANT:
            for (int j = 0; j < T.elephN[s]; ++j) {
                const int to = T.elephTo[s][j];
                if (board[T.elephEye[s][j]] || !target_ok(to) || (CAP && !board[to])) continue;
                if (dark && !own_half(us, to)) continue;
                add(out, s, to);
            }
            break;
        case ADVISOR:
            for (int j = 0; j < T.advN[s]; ++j) {
                const int to = T.advTo[s][j];
                if (!target_ok(to) || (CAP && !board[to]) || (dark && !in_palace(us, to))) continue;
                add(out, s, to);
            }
            break;
        case SOLDIER: {
            const int fwd = us == RED ? 9 : -9;
            const int to = s + fwd;
            if (to >= 0 && to < NUM_SQ && target_ok(to) && (!CAP || board[to])) add(out, s, to);
            if (!own_half(us, s)) {
                if (file_of(s) > 0 && target_ok(s - 1) && (!CAP || board[s - 1])) add(out, s, s - 1);
                if (file_of(s) < 8 && target_ok(s + 1) && (!CAP || board[s + 1])) add(out, s, s + 1);
            }
            break;
        }
        default: break;
        }
    }
    return int(out - list);
}

bool RefPos::legal(Move m) const {
    const int from = from_sq(m), to = to_sq(m);
    if (type_of(board[from]) != KING && !checkHist.back() && !touches_king_lines(from, to, kingSq[stm])) return true;
    auto& b = const_cast<Piece(&)[NUM_SQ]>(board);
    const Piece moving = b[from], captured = b[to];
    b[to] = type_of(moving) == DARK ? make_piece(stm, SOLDIER) : moving;  // identity irrelevant for own safety
    b[from] = NO_PIECE;
    bool ok;
    const int k0 = kingSq[stm];
    if (type_of(moving) == KING || checkHist.back() || diag_adjacent(from, k0))
        ok = !king_attacked(stm, type_of(moving) == KING ? to : k0);
    else {
        // Not in check and the king stays: only rays through from (discovery) or to (new cannon
        // screen) can carry a new attack.
        const int df = T.dirTo[k0][from], dt = T.dirTo[k0][to];
        ok = !((df >= 0 && ray_attacked(stm, k0, df)) || (dt >= 0 && dt != df && ray_attacked(stm, k0, dt)));
    }
#ifdef JQ_VERIFY_LEGAL
    if (ok != !king_attacked(stm, type_of(moving) == KING ? to : k0)) { std::fprintf(stderr, "legal mismatch %s\n", fen().c_str()); std::abort(); }
#endif
    b[from] = moving;
    b[to] = captured;
    return ok;
}

bool RefPos::reveal_may_check(int to) const {
    const int k = kingSq[~stm];
    return file_of(to) == file_of(k) || rank_of(to) == rank_of(k) || cheb(to, k) <= 2;
}

bool RefPos::reveal_gives_check(int from, int to, PieceType t) const {
    auto& b = const_cast<Piece(&)[NUM_SQ]>(board);
    const Piece moving = b[from], captured = b[to];
    b[to] = make_piece(stm, t);
    b[from] = NO_PIECE;
    const bool check = king_attacked(~stm, kingSq[~stm]);
    b[from] = moving;
    b[to] = captured;
    return check;
}

bool RefPos::check_after(int from, int to, PieceType t) const {
    const int k = kingSq[stm];
    if (!touches_king_lines(from, to, k) && (t == ROOK || t == CANNON || t == KING || cheb(to, k) > 2)) return false;
    if (diag_adjacent(from, k) || (t != ROOK && t != CANNON && t != KING && cheb(to, k) <= 2)) return king_attacked(stm, k);
    const int df = T.dirTo[k][from], dt = T.dirTo[k][to];
    return (df >= 0 && ray_attacked(stm, k, df)) || (dt >= 0 && dt != df && ray_attacked(stm, k, dt));
}

bool RefPos::gives_check(Move m) const {
    const int from = from_sq(m), to = to_sq(m);
    const Piece moving = board[from];
    if (type_of(moving) == DARK) return false;
    {
        const int k = kingSq[~stm];
        const PieceType t = type_of(moving);
        // Off the enemy king's lines a rook/cannon/king cannot attack it, and the others need to be close.
        if (!touches_king_lines(from, to, k) && (t == ROOK || t == CANNON || t == KING || cheb(to, k) > 2)) return false;
    }
    auto& b = const_cast<Piece(&)[NUM_SQ]>(board);
    const Piece captured = b[to];
    b[to] = moving;
    b[from] = NO_PIECE;
    const int k = kingSq[~stm];
    const PieceType t = type_of(moving);
    bool check;
    if (diag_adjacent(from, k) || (t != ROOK && t != CANNON && t != KING && cheb(to, k) <= 2))
        check = king_attacked(~stm, k);
    else {
        // The mover can only check along a ray through to; discoveries come through from's ray.
        const int df = T.dirTo[k][from], dt = T.dirTo[k][to];
        check = (df >= 0 && ray_attacked(~stm, k, df)) || (dt >= 0 && dt != df && ray_attacked(~stm, k, dt));
    }
#ifdef JQ_VERIFY_LEGAL
    if (check != king_attacked(~stm, k)) { std::fprintf(stderr, "check mismatch %s\n", fen().c_str()); std::abort(); }
#endif
    b[from] = moving;
    b[to] = captured;
    return check;
}

template int RefPos::gen<true>(Move*) const;
template int RefPos::gen<false>(Move*) const;
}  // namespace jq
#endif
