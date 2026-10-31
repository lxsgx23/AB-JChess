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
#include "bitboard.h"

#include <cstdlib>
#include <cstring>

namespace jq::BB {

Bitboard Ray[NUM_SQ][4];
Bitboard FileBB[9], RankBB[10];
Bitboard Half[2], Palace[2];
Bitboard HorseTo[NUM_SQ], HorseFrom[NUM_SQ], ElephTo[NUM_SQ], AdvTo[NUM_SQ], KingTo[NUM_SQ];
Bitboard HorseByLeg[NUM_SQ][4], HorseAttByLeg[NUM_SQ][4], ElephByEye[NUM_SQ][4];
int8_t OrthoNb[NUM_SQ][4], DiagNb[NUM_SQ][4];
Bitboard SoldierTo[2][NUM_SQ], SoldierFrom[2][NUM_SQ];
Bitboard DarkRoleBB[2][9], DarkStart[2];
uint8_t Leg[NUM_SQ][NUM_SQ], Eye[NUM_SQ][NUM_SQ];
uint64_t FileLo[9], FileHi[9];
int FileNLo[9];
uint8_t REL[NUM_SQ][NUM_SQ];

namespace {
bool on(int f, int r) { return f >= 0 && f < 9 && r >= 0 && r < 10; }
}

void init() {
    static bool done = false;
    if (done) return;
    done = true;
    init_tables();
    for (int f = 0; f < 9; ++f) {  // scan4 / REL tables
        FileLo[f] = FileHi[f] = 0;
        for (int r = 0; r < 10; ++r) { const int s = r * 9 + f; if (s < 64) FileLo[f] |= 1ull << s; else FileHi[f] |= 1ull << (s - 64); }
        FileNLo[f] = __builtin_popcountll(FileLo[f]);
    }
    for (int a = 0; a < NUM_SQ; ++a)
        for (int b = 0; b < NUM_SQ; ++b) {
            const int df = std::abs(file_of(a) - file_of(b)), dr = std::abs(rank_of(a) - rank_of(b));
            REL[a][b] = uint8_t((file_of(a) == file_of(b) || rank_of(a) == rank_of(b) ? 1 : 0) | (df == 1 && dr == 1 ? 2 : 0) |
                                ((df > dr ? df : dr) <= 2 ? 4 : 0));
        }
    std::memset(Leg, 0xFF, sizeof Leg);
    std::memset(Eye, 0xFF, sizeof Eye);
    for (int f = 0; f < 9; ++f) FileBB[f] = 0;
    for (int r = 0; r < 10; ++r) RankBB[r] = 0;
    Half[0] = Half[1] = Palace[0] = Palace[1] = 0;
    for (int c = 0; c < 2; ++c) {
        DarkStart[c] = 0;
        for (int t = 0; t < 9; ++t) DarkRoleBB[c][t] = 0;
    }
    const int odf[4] = {0, 0, 1, -1}, odr[4] = {1, -1, 0, 0};    // N S E W
    const int ddf[4] = {1, -1, 1, -1}, ddr[4] = {1, 1, -1, -1};  // NE NW SE SW
    for (int s = 0; s < NUM_SQ; ++s) {
        const int f = file_of(s), r = rank_of(s);
        FileBB[f] |= sq_bb(s);
        RankBB[r] |= sq_bb(s);
        for (int c = 0; c < 2; ++c) {
            if (own_half(Color(c), s)) Half[c] |= sq_bb(s);
            if (in_palace(Color(c), s)) Palace[c] |= sq_bb(s);
            if (T.darkRole[c][s]) {
                DarkStart[c] |= sq_bb(s);
                DarkRoleBB[c][T.darkRole[c][s]] |= sq_bb(s);
            }
        }
        for (int d = 0; d < 4; ++d) {
            Ray[s][d] = 0;
            for (int j = 0; j < T.rayLen[s][d]; ++j) Ray[s][d] |= sq_bb(T.ray[s][d][j]);
            OrthoNb[s][d] = int8_t(on(f + odf[d], r + odr[d]) ? make_sq(f + odf[d], r + odr[d]) : -1);
            DiagNb[s][d] = int8_t(on(f + ddf[d], r + ddr[d]) ? make_sq(f + ddf[d], r + ddr[d]) : -1);
            HorseByLeg[s][d] = HorseAttByLeg[s][d] = ElephByEye[s][d] = 0;
        }
        HorseTo[s] = HorseFrom[s] = ElephTo[s] = AdvTo[s] = KingTo[s] = 0;
        for (int j = 0; j < T.horseN[s]; ++j) {
            const int t = T.horseTo[s][j], l = T.horseLeg[s][j];
            HorseTo[s] |= sq_bb(t);
            Leg[s][t] = uint8_t(l);
            for (int d = 0; d < 4; ++d)
                if (OrthoNb[s][d] == l) HorseByLeg[s][d] |= sq_bb(t);
        }
        for (int j = 0; j < T.horseFromN[s]; ++j) HorseFrom[s] |= sq_bb(T.horseFrom[s][j]);
        for (int j = 0; j < T.elephN[s]; ++j) {
            const int t = T.elephTo[s][j], e = T.elephEye[s][j];
            ElephTo[s] |= sq_bb(t);
            Eye[s][t] = uint8_t(e);
            for (int d = 0; d < 4; ++d)
                if (DiagNb[s][d] == e) ElephByEye[s][d] |= sq_bb(t);
        }
        for (int j = 0; j < T.advN[s]; ++j) AdvTo[s] |= sq_bb(T.advTo[s][j]);
        for (int j = 0; j < T.kingN[s]; ++j) KingTo[s] |= sq_bb(T.kingTo[s][j]);
    }
    for (int s = 0; s < NUM_SQ; ++s) {
        // a horse on p attacks s through leg L = Leg[p][s]; L is a diagonal neighbour of s
        for (int j = 0; j < T.horseFromN[s]; ++j) {
            const int p = T.horseFrom[s][j], l = T.horseFromLeg[s][j];
            for (int d = 0; d < 4; ++d)
                if (DiagNb[s][d] == l) HorseAttByLeg[s][d] |= sq_bb(p);
        }
        for (int c = 0; c < 2; ++c) {
            SoldierTo[c][s] = SoldierFrom[c][s] = 0;
            const int fwd = c == RED ? 9 : -9;
            if (s + fwd >= 0 && s + fwd < NUM_SQ) SoldierTo[c][s] |= sq_bb(s + fwd);
            if (!own_half(Color(c), s)) {
                if (file_of(s) > 0) SoldierTo[c][s] |= sq_bb(s - 1);
                if (file_of(s) < 8) SoldierTo[c][s] |= sq_bb(s + 1);
            }
        }
    }
    for (int c = 0; c < 2; ++c)
        for (int p = 0; p < NUM_SQ; ++p)
            for_each_sq(SoldierTo[c][p], [&](int s) { SoldierFrom[c][s] |= sq_bb(p); });
}

}  // namespace jq::BB
