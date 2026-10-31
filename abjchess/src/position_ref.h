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
#pragma once
#ifdef JQ_BB_VERIFY
#include "position.h"
#include <cstring>
namespace jq {
// Snapshot of a Position's mailbox state on which the original (pre-bitboard) code runs.
struct RefPos {
    Piece board[NUM_SQ];
    Color stm;
    int8_t kingSq[2];
    uint8_t plist[2][16];
    int8_t pcount[2];
    std::vector<uint8_t> checkHist;
    explicit RefPos(const Position& p) : stm(p.stm) {
        std::memcpy(board, p.board, sizeof board);
        std::memcpy(kingSq, p.kingSq, sizeof kingSq);
        std::memcpy(plist, p.plist, sizeof plist);
        std::memcpy(pcount, p.pcount, sizeof pcount);
        checkHist.push_back(p.checkHist.empty() ? 0 : p.checkHist.back());
    }
    PieceType role(int s) const {
        const Piece p = board[s];
        return type_of(p) == DARK ? PieceType(T.darkRole[color_of(p)][s]) : type_of(p);
    }
    std::string fen() const { return "?"; }
    bool ray_attacked(Color c, int k, int d) const;
    bool king_attacked(Color c, int k) const;
    template <bool CAP> int gen(Move* list) const;
    bool legal(Move m) const;
    bool reveal_may_check(int to) const;
    bool reveal_gives_check(int from, int to, PieceType t) const;
    bool check_after(int from, int to, PieceType t) const;
    bool gives_check(Move m) const;
};
}  // namespace jq
#endif
