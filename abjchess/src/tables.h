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
// Precomputed geometry for the 9x10 board.
#pragma once
#include "types.h"

namespace jq {

enum Dir : int { NORTH = 0, SOUTH = 1, EAST = 2, WEST = 3 };  // NORTH = towards rank 9

struct Tables {
    uint8_t ray[NUM_SQ][4][9];
    uint8_t rayLen[NUM_SQ][4];
    // Horse moves from a square: destination and blocking "leg" square.
    uint8_t horseTo[NUM_SQ][8], horseLeg[NUM_SQ][8], horseN[NUM_SQ];
    // Squares from which a horse attacks this square, with the attacker's leg square.
    uint8_t horseFrom[NUM_SQ][8], horseFromLeg[NUM_SQ][8], horseFromN[NUM_SQ];
    uint8_t elephTo[NUM_SQ][4], elephEye[NUM_SQ][4], elephN[NUM_SQ];
    uint8_t advTo[NUM_SQ][4], advN[NUM_SQ];
    uint8_t kingTo[NUM_SQ][4], kingN[NUM_SQ];   // orthogonal steps staying inside the same palace
    uint8_t darkRole[2][NUM_SQ];                // PieceType role of a face-down piece, 0 if not a start point
    int8_t dirTo[NUM_SQ][NUM_SQ];               // ray direction from the first square to the second, -1 if none
    uint64_t zPiece[PIECE_CODES][NUM_SQ];
    uint64_t zSide;
    uint64_t zUrn[2][NUM_SLOTS][8];
};

extern Tables T;
void init_tables();

}  // namespace jq
