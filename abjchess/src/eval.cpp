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
#include "eval.h"

#include "nnue.h"

namespace jq {

int PieceValue[9] = {0, 0, 900, 400, 230, 230, 450, 100, 0};


namespace {
int mobility_ray(const Position& pos, int s) {
    int n = 0;
    for (int d = 0; d < 4; ++d)
        for (int j = 0; j < T.rayLen[s][d]; ++j) {
            if (pos.board[T.ray[s][d][j]]) break;
            ++n;
        }
    return n;
}

int side_score(const Position& pos, Color c) {
    int score = 0;
    const int darkValue = urn_mean_value(pos, c);
    const int enemyKing = pos.kingSq[~c];
    for (int i = 0; i < pos.pcount[c]; ++i) {
        const int s = pos.plist[c][i];
        const PieceType t = type_of(pos.board[s]);
        const int rel = c == RED ? rank_of(s) : 9 - rank_of(s);  // 0 = own back rank
        const int centre = 4 - (file_of(s) > 4 ? file_of(s) - 4 : 4 - file_of(s));
        switch (t) {
        case DARK: score += darkValue; break;
        case ROOK: score += PieceValue[ROOK] + 6 * mobility_ray(pos, s); break;
        case CANNON: score += PieceValue[CANNON] + 2 * mobility_ray(pos, s) + (file_of(s) == file_of(enemyKing) ? 25 : 0); break;
        case HORSE: score += PieceValue[HORSE] + 8 * centre + (rel >= 3 && rel <= 7 ? 20 : 0); break;
        case SOLDIER:
            if (rel >= 5) score += 2 * PieceValue[SOLDIER] + (rel <= 8 ? 15 * (rel - 4) : 0) + 6 * centre;
            else score += PieceValue[SOLDIER];
            break;
        case ADVISOR: case ELEPHANT: score += PieceValue[t]; break;
        default: break;
        }
    }
    return score;
}
}  // namespace

Value classical_eval(const Position& pos) {
    const int v = side_score(pos, RED) - side_score(pos, BLACK);
    return (pos.stm == RED ? v : -v) + 20;
}

Value evaluate(const Position& pos) {
    if (nnue::loaded()) return nnue::evaluate(pos);
    return classical_eval(pos);
}

}  // namespace jq
