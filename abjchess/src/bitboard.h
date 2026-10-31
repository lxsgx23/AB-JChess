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
// 128-bit bitboards for the 9x10 Jieqi board (Pikafish-style layout: square s = rank * 9 + file is bit s,
// rank 0 = Red's home rank, file 0 = file 'a'). Shared, engine-independent layer: geometry masks and ray /
// leaper attack helpers only; Position keeps byColor / byRole / dark sets on top of it.
//
// Jieqi rules encoded here (identical to ABJchess's mailbox movegen in position.cpp):
//   * a face-down (dark) piece moves by the role of the start square it stands on (T.darkRole);
//   * a face-down advisor may only step into its own palace, a face-down elephant only within its own half;
//   * revealed advisors / elephants roam the whole board (no palace / river limit);
//   * soldiers move forward, sideways once across the river; cannons capture over exactly one screen;
//   * horse legs and elephant eyes block.
#pragma once
#include "tables.h"
#include "types.h"

#include <cstdint>
#include "jqu128.h"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace jq {

using Bitboard = jq_u128;

constexpr Bitboard BB_ONE = 1;
constexpr Bitboard BB_ALL = (BB_ONE << NUM_SQ) - 1;
// Square masks by table (a variable 128-bit shift costs a shld/shl/cmov sequence); valid for 0 <= s < 128.
struct SquareTable {
    Bitboard b[128];
    constexpr SquareTable() : b() {
        for (int s = 0; s < 128; ++s) b[s] = BB_ONE << s;
    }
};
inline constexpr SquareTable SQUARE_BB{};
inline Bitboard sq_bb(int s) { return SQUARE_BB.b[s]; }

inline int popcount(Bitboard b) { return __builtin_popcountll(uint64_t(b)) + __builtin_popcountll(uint64_t(b >> 64)); }
inline int lsb(Bitboard b) {  // b != 0
    const uint64_t lo = uint64_t(b);
    return lo ? __builtin_ctzll(lo) : 64 + __builtin_ctzll(uint64_t(b >> 64));
}
inline int msb(Bitboard b) {  // b != 0
    const uint64_t hi = uint64_t(b >> 64);
    return hi ? 127 - __builtin_clzll(hi) : 63 - __builtin_clzll(uint64_t(b));
}
inline int pop_lsb(Bitboard& b) {
    const int s = lsb(b);
    b &= b - 1;
    return s;
}
inline bool more_than_one(Bitboard b) { return (b & (b - 1)) != 0; }
inline bool test(Bitboard b, int s) { return (b & SQUARE_BB.b[s]) != 0; }
// Calls f(square) for every set bit, ascending.
template <class F>
inline void for_each_sq(Bitboard b, F&& f) {
    uint64_t lo = uint64_t(b), hi = uint64_t(b >> 64);
    while (lo) { f(__builtin_ctzll(lo)); lo &= lo - 1; }
    while (hi) { f(64 + __builtin_ctzll(hi)); hi &= hi - 1; }
}

namespace BB {

// Directions as in tables.h: NORTH = +9 (towards rank 9), SOUTH = -9, EAST = +1, WEST = -1.
// Ray[s][d]: squares strictly beyond s along d (same squares as T.ray[s][d]).
extern Bitboard Ray[NUM_SQ][4];
extern Bitboard FileBB[9], RankBB[10];
extern Bitboard Half[2];     // own half of colour c (ranks 0-4 for Red)
extern Bitboard Palace[2];
// Leaper geometry (no blocking): destinations from s.
extern Bitboard HorseTo[NUM_SQ];    // = T.horseTo[s][*]
extern Bitboard HorseFrom[NUM_SQ];  // squares from which a horse reaches s (= T.horseFrom[s][*])
extern Bitboard ElephTo[NUM_SQ];    // = T.elephTo (symmetric: also the elephant squares that reach s)
extern Bitboard AdvTo[NUM_SQ];      // = T.advTo (symmetric)
extern Bitboard KingTo[NUM_SQ];     // = T.kingTo (palace steps)
// Horse leg masks: HorseByLeg[s][i]: targets of a horse on s whose leg is the i-th orthogonal neighbour
// (i = direction N,S,E,W); HorseAttByLeg[s][i]: horses that attack s through the i-th diagonal neighbour of s
// (diagonals NE, NW, SE, SW) being empty. Elephant eye masks: ElephByEye[s][i] (diagonal i).
extern Bitboard HorseByLeg[NUM_SQ][4], HorseAttByLeg[NUM_SQ][4], ElephByEye[NUM_SQ][4];
extern int8_t OrthoNb[NUM_SQ][4], DiagNb[NUM_SQ][4];  // neighbour squares, -1 off board
// Soldiers: SoldierTo[c][s] = targets of a colour-c soldier on s; SoldierFrom[c][s] = squares from which a
// colour-c soldier attacks s.
extern Bitboard SoldierTo[2][NUM_SQ], SoldierFrom[2][NUM_SQ];
// Start squares of colour c whose face-down role is t (t = ROOK..SOLDIER), and all start squares.
extern Bitboard DarkRoleBB[2][9], DarkStart[2];
// Leg / eye square for a (from, to) leaper pair, 0xFF if not a horse / elephant step.
extern uint8_t Leg[NUM_SQ][NUM_SQ], Eye[NUM_SQ][NUM_SQ];

void init();  // idempotent; calls init_tables()

// first / second blockers of all four rays of s in one pass (file / rank occupancy,
// tzcnt / lzcnt), -1 if none; directions N, S, E, W as Dir. Same squares as blockers(s, d, occ, ..) for each d.
// The file occupancy uses PEXT on the tiers where it is fast (JQ_USE_PEXT: bmi2 and up), else the exact multiply
// gather of threats.cpp; tiers without BMI1 / LZCNT get nobmi.h's exact replacements (force-included).
extern uint64_t FileLo[9], FileHi[9];
extern int FileNLo[9];
// REL[a][b]: bit 0 same file or rank, bit 1 diagonal neighbours, bit 2 Chebyshev distance <= 2
extern uint8_t REL[NUM_SQ][NUM_SQ];

inline bool ascending(int d) { return d == NORTH || d == EAST; }
// Nearest occupied square from s along d (-1 if none).
inline int first_blocker(int s, int d, Bitboard occ) {
    const Bitboard m = Ray[s][d] & occ;
    if (!m) return -1;
    return ascending(d) ? lsb(m) : msb(m);
}
// First (f1) and second (f2) occupied squares from s along d (-1 if none).
inline void blockers(int s, int d, Bitboard occ, int& f1, int& f2) {
    Bitboard m = Ray[s][d] & occ;
    if (!m) { f1 = f2 = -1; return; }
    if (ascending(d)) {
        f1 = lsb(m);
        m &= m - 1;
        f2 = m ? lsb(m) : -1;
    } else {
        f1 = msb(m);
        m ^= sq_bb(f1);
        f2 = m ? msb(m) : -1;
    }
}

inline void scan4(int s, Bitboard occ, int (&f1)[4], int (&f2)[4]) {
    const int f = file_of(s), r = rank_of(s);
    const uint32_t ro = uint32_t(occ >> (9 * r)) & 0x1FFu & ~(1u << f);
#if defined(JQ_USE_PEXT)
    const uint64_t lo = uint64_t(occ), hi = uint64_t(occ >> 64);
    const uint32_t fo = (uint32_t(_pext_u64(lo, FileLo[f])) | (uint32_t(_pext_u64(hi, FileHi[f])) << FileNLo[f])) & ~(1u << r);
#else  // same bits by one multiply: file f = bits f + 9 i of occ; i < 8 in the low word, 8 and 9 above
    const Bitboard sh = occ >> f;
    const uint64_t l8 = ((uint64_t(sh) & 0x8040201008040201ull) * 0x0101010101010101ull) >> 56;
    const uint64_t h = uint64_t(sh >> 64);
    const uint32_t fo = (uint32_t(l8) | uint32_t((h >> 8) & 1) << 8 | uint32_t((h >> 17) & 1) << 9) & ~(1u << r);
#endif
    {
        uint32_t u = fo >> (r + 1);
        const int t1 = int(_tzcnt_u32(u)); u = _blsr_u32(u); const int t2 = int(_tzcnt_u32(u));
        f1[NORTH] = t1 < 32 ? (r + 1 + t1) * 9 + f : -1; f2[NORTH] = t2 < 32 ? (r + 1 + t2) * 9 + f : -1;
        const uint32_t dn = fo & ((1u << r) - 1);
        const int d1 = 31 - int(_lzcnt_u32(dn)); const int d2 = 31 - int(_lzcnt_u32(_bzhi_u32(dn, uint32_t(d1) & 255u)));
        f1[SOUTH] = d1 >= 0 ? d1 * 9 + f : -1; f2[SOUTH] = d2 >= 0 ? d2 * 9 + f : -1;
    }
    {
        uint32_t u = ro >> (f + 1);
        const int t1 = int(_tzcnt_u32(u)); u = _blsr_u32(u); const int t2 = int(_tzcnt_u32(u));
        f1[EAST] = t1 < 32 ? 9 * r + f + 1 + t1 : -1; f2[EAST] = t2 < 32 ? 9 * r + f + 1 + t2 : -1;
        const uint32_t dn = ro & ((1u << f) - 1);
        const int d1 = 31 - int(_lzcnt_u32(dn)); const int d2 = 31 - int(_lzcnt_u32(_bzhi_u32(dn, uint32_t(d1) & 255u)));
        f1[WEST] = d1 >= 0 ? 9 * r + d1 : -1; f2[WEST] = d2 >= 0 ? 9 * r + d2 : -1;
    }
}

// Pikafish-style attack sets (computed from the rays; used by attackers_to-style queries and by the att
// threat prototype). rook_attacks: empty squares up to and including the first blocker on each ray;
// cannon_attacks: the second blocker on each ray (capture targets); cannon quiet moves = rook_attacks & ~occ.
inline Bitboard rook_attacks(int s, Bitboard occ) {
    Bitboard a = 0;
    for (int d = 0; d < 4; ++d) {
        const int b = first_blocker(s, d, occ);
        a |= b < 0 ? Ray[s][d] : Ray[s][d] & ~Ray[b][d];
    }
    return a;
}
inline Bitboard cannon_attacks(int s, Bitboard occ) {
    Bitboard a = 0;
    for (int d = 0; d < 4; ++d) {
        int f1, f2;
        blockers(s, d, occ, f1, f2);
        if (f2 >= 0) a |= sq_bb(f2);
    }
    return a;
}
inline Bitboard horse_attacks(int s, Bitboard occ) {
    Bitboard a = 0;
    for (int i = 0; i < 4; ++i)
        if (OrthoNb[s][i] >= 0 && !test(occ, OrthoNb[s][i])) a |= HorseByLeg[s][i];
    return a;
}
// Squares holding (any) piece from which a horse would attack s given occupancy occ (legs checked).
inline Bitboard horse_attackers_geom(int s, Bitboard occ) {
    Bitboard a = 0;
    for (int i = 0; i < 4; ++i)
        if (DiagNb[s][i] >= 0 && !test(occ, DiagNb[s][i])) a |= HorseAttByLeg[s][i];
    return a;
}
inline Bitboard elephant_attacks(int s, Bitboard occ) {
    Bitboard a = 0;
    for (int i = 0; i < 4; ++i)
        if (DiagNb[s][i] >= 0 && !test(occ, DiagNb[s][i])) a |= ElephByEye[s][i];
    return a;
}

}  // namespace BB
}  // namespace jq
