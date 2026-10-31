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
// Core types for the ABJchess Jieqi engine (derived from jqe by Devin O'Keefe, MIT licence, see ../LICENSE-jqe-MIT.txt; ABJchess as a whole is GPLv3, see ../Copying.txt).
#pragma once
#include <cstdint>
#include <cstddef>

namespace jq {

enum Color : int { RED = 0, BLACK = 1 };
constexpr Color operator~(Color c) { return Color(c ^ 1); }

// DARK is a face-down piece; its movement role is fixed by the square it stands on
// (dark pieces never leave their starting point without being revealed).
enum PieceType : int {
    NO_TYPE = 0, KING = 1, ROOK = 2, HORSE = 3, ELEPHANT = 4, ADVISOR = 5, CANNON = 6, SOLDIER = 7, DARK = 8
};
// Hidden identities ("urn slots") are the six revealable kinds: slot = type - ROOK.
constexpr int NUM_SLOTS = 6;
constexpr int URN_INIT[NUM_SLOTS] = {2, 2, 2, 2, 2, 5};  // R N B A C P
constexpr PieceType slot_type(int slot) { return PieceType(slot + ROOK); }
constexpr int type_slot(PieceType t) { return int(t) - ROOK; }

using Piece = uint8_t;  // 0 = empty, otherwise type | color << 4
constexpr Piece NO_PIECE = 0;
constexpr Piece make_piece(Color c, PieceType t) { return Piece(int(t) | (int(c) << 4)); }
constexpr PieceType type_of(Piece p) { return PieceType(p & 15); }
constexpr Color color_of(Piece p) { return Color(p >> 4); }
constexpr int PIECE_CODES = 32;

constexpr int NUM_SQ = 90;
constexpr int file_of(int s) { return s % 9; }
constexpr int rank_of(int s) { return s / 9; }
constexpr int make_sq(int f, int r) { return r * 9 + f; }
// Red's home rank is 0; Red soldiers move towards rank 9.
constexpr bool own_half(Color c, int s) { return c == RED ? rank_of(s) <= 4 : rank_of(s) >= 5; }
constexpr bool in_palace(Color c, int s) {
    return file_of(s) >= 3 && file_of(s) <= 5 && (c == RED ? rank_of(s) <= 2 : rank_of(s) >= 7);
}

// Move: from (7 bits) | to (7 bits) | reveal type (4 bits, only for dark movers inside search).
using Move = uint32_t;
constexpr Move MOVE_NONE = 0;
constexpr Move MOVE_NULL = 0x3FFFu;  // from = to = 127
constexpr Move make_move(int from, int to, int reveal = 0) { return Move(from | (to << 7) | (reveal << 14)); }
constexpr int from_sq(Move m) { return int(m & 127); }
constexpr int to_sq(Move m) { return int((m >> 7) & 127); }
constexpr PieceType reveal_of(Move m) { return PieceType((m >> 14) & 15); }
constexpr Move base_move(Move m) { return m & 0x3FFFu; }

using Value = int;
constexpr Value VALUE_ZERO = 0;
constexpr Value VALUE_DRAW = 0;
constexpr int MAX_PLY = 120;
constexpr Value VALUE_MATE = 32000;
constexpr Value VALUE_INFINITE = 32001;
constexpr Value VALUE_NONE = 32002;
constexpr Value VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY;
constexpr Value VALUE_MATED_IN_MAX_PLY = -VALUE_MATE_IN_MAX_PLY;
constexpr Value VALUE_KNOWN_WIN = 20000;   // cap for aggregated chance values / repetition wins
constexpr Value mate_in(int ply) { return VALUE_MATE - ply; }
constexpr Value mated_in(int ply) { return -VALUE_MATE + ply; }

// ChaseRule UCI values. The numeric values preserve the legacy spin option:
// 0 was the disabled mode and 1 was the old referee implementation.
enum ChaseRule : int8_t {
    CHASE_RULE_NONE = 0,
    CHASE_RULE_DEFAULT = 1,
    CHASE_RULE_SKYRULE = 2,
};

enum SkyRuleAction : uint8_t {
    SKYRULE_NONE = 0,
    SKYRULE_CHECK = 1,
    SKYRULE_CHASE = 2,
};

enum Bound : uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

}  // namespace jq
