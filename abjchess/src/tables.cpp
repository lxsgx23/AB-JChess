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
#include "tables.h"

#include <cstdlib>

namespace jq {

Tables T;

namespace {
bool on_board(int f, int r) { return f >= 0 && f < 9 && r >= 0 && r < 10; }

uint64_t splitmix(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
}  // namespace

void init_tables() {
    static bool done = false;
    if (done) return;
    done = true;
    const int df[4] = {0, 0, 1, -1}, dr[4] = {1, -1, 0, 0};
    for (int s = 0; s < NUM_SQ; ++s) {
        const int f = file_of(s), r = rank_of(s);
        for (int t = 0; t < NUM_SQ; ++t) T.dirTo[s][t] = -1;
        for (int d = 0; d < 4; ++d) {
            int n = 0;
            for (int k = 1;; ++k) {
                const int nf = f + df[d] * k, nr = r + dr[d] * k;
                if (!on_board(nf, nr)) break;
                T.dirTo[s][make_sq(nf, nr)] = int8_t(d);
                T.ray[s][d][n++] = uint8_t(make_sq(nf, nr));
            }
            T.rayLen[s][d] = uint8_t(n);
        }
        // Horse: (df, dr) with leg one step along the long axis.
        const int hf[8] = {1, 2, 2, 1, -1, -2, -2, -1}, hr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
        T.horseN[s] = T.horseFromN[s] = 0;
        for (int k = 0; k < 8; ++k) {
            const int tf = f + hf[k], tr = r + hr[k];
            if (!on_board(tf, tr)) continue;
            const int lf = f + (std::abs(hf[k]) == 2 ? hf[k] / 2 : 0);
            const int lr = r + (std::abs(hr[k]) == 2 ? hr[k] / 2 : 0);
            T.horseTo[s][T.horseN[s]] = uint8_t(make_sq(tf, tr));
            T.horseLeg[s][T.horseN[s]++] = uint8_t(make_sq(lf, lr));
        }
        for (int k = 0; k < 8; ++k) {
            // Attacker at p = s - delta moving by +delta; its leg is p + longaxis/2.
            const int pf = f - hf[k], pr = r - hr[k];
            if (!on_board(pf, pr)) continue;
            const int lf = pf + (std::abs(hf[k]) == 2 ? hf[k] / 2 : 0);
            const int lr = pr + (std::abs(hr[k]) == 2 ? hr[k] / 2 : 0);
            T.horseFrom[s][T.horseFromN[s]] = uint8_t(make_sq(pf, pr));
            T.horseFromLeg[s][T.horseFromN[s]++] = uint8_t(make_sq(lf, lr));
        }
        T.elephN[s] = T.advN[s] = T.kingN[s] = 0;
        const int ef[4] = {2, 2, -2, -2}, er[4] = {2, -2, 2, -2};
        for (int k = 0; k < 4; ++k) {
            const int tf = f + ef[k], tr = r + er[k];
            if (on_board(tf, tr)) {
                T.elephTo[s][T.elephN[s]] = uint8_t(make_sq(tf, tr));
                T.elephEye[s][T.elephN[s]++] = uint8_t(make_sq(f + ef[k] / 2, r + er[k] / 2));
            }
            const int af = f + ef[k] / 2, ar = r + er[k] / 2;
            if (on_board(af, ar)) T.advTo[s][T.advN[s]++] = uint8_t(make_sq(af, ar));
        }
        for (int d = 0; d < 4; ++d) {
            const int tf = f + df[d], tr = r + dr[d];
            if (!on_board(tf, tr)) continue;
            const int t = make_sq(tf, tr);
            if ((in_palace(RED, s) && in_palace(RED, t)) || (in_palace(BLACK, s) && in_palace(BLACK, t)))
                T.kingTo[s][T.kingN[s]++] = uint8_t(t);
        }
        T.darkRole[RED][s] = T.darkRole[BLACK][s] = NO_TYPE;
    }
    // Red start points; Black's are mirrored by rank.
    const struct { int f, r; PieceType t; } starts[] = {
        {0, 0, ROOK}, {8, 0, ROOK}, {1, 0, HORSE}, {7, 0, HORSE}, {2, 0, ELEPHANT}, {6, 0, ELEPHANT},
        {3, 0, ADVISOR}, {5, 0, ADVISOR}, {1, 2, CANNON}, {7, 2, CANNON},
        {0, 3, SOLDIER}, {2, 3, SOLDIER}, {4, 3, SOLDIER}, {6, 3, SOLDIER}, {8, 3, SOLDIER}};
    for (const auto& st : starts) {
        T.darkRole[RED][make_sq(st.f, st.r)] = uint8_t(st.t);
        T.darkRole[BLACK][make_sq(st.f, 9 - st.r)] = uint8_t(st.t);
    }
    uint64_t seed = 0x6A6965716931ull;  // fixed for reproducible hashing
    for (auto& row : T.zPiece)
        for (auto& z : row) z = splitmix(seed);
    T.zSide = splitmix(seed);
    for (auto& c : T.zUrn)
        for (auto& sl : c)
            for (auto& z : sl) z = splitmix(seed);
}

}  // namespace jq
