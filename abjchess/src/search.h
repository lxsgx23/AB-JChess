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
// Alpha-beta search with chance nodes for face-down movers.
#pragma once
#include "position.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace jq {

struct Limits {
    int depth = 0;
    int64_t nodes = 0;
    int movetime = 0;          // ms
    int time[2] = {0, 0};      // ms remaining per colour
    int inc[2] = {0, 0};
    int movestogo = 0;
    int mate = 0;
    bool infinite = false;
    bool ponder = false;
    std::vector<Move> searchMoves;  // root restricted to these moves when not empty
};

struct SearchResult {
    Move best = MOVE_NONE;
    Value score = VALUE_NONE;
    int depth = 0;
    uint64_t nodes = 0;
    int timeMs = 0;
    std::vector<Move> pv;
};

// Tunable integer parameters exposed as UCI spin options (for SPSA later).
struct Param {
    const char* name;
    int* value;
    int lo, hi;
};
std::vector<Param>& params();

class Searcher {
public:
    explicit Searcher(int threads = 1, std::size_t hashMb = 16);
    ~Searcher();
    Searcher(const Searcher&) = delete;
    Searcher& operator=(const Searcher&) = delete;
    void set_threads(int n);
    int threads() const;
    void resize_hash(std::size_t mb);
    void clear();  // histories and transposition table
    void stop();   // may be called from another thread
    void ponderhit();
    void prepare(bool pondering = false);  // called before asynchronous go() to preserve early stop/ponderhit commands
    // Runs synchronously; `info` receives UCI info lines from the main thread. With resetStop = false the
    // caller must have called prepare().
    SearchResult go(const Position& root, const Limits& limits, const std::function<void(const std::string&)>& info,
                    bool resetStop = true);
    int hashfull() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

namespace search {

void init();  // tables and LMR (call again after changing LMR parameters)
uint64_t perft(Position& pos, int depth);   // base moves (one representative reveal per dark move)
double perft_chance(Position& pos, int depth, uint64_t& leaves);  // expands reveals, returns probability mass
// Static exchange evaluation of a pseudo-legal base move: does the side to move net >= threshold on its
// target square? Face-down movers and victims count at their side's expected identity value.
bool see_ge(const Position& pos, Move m, int threshold);
int see_value(const Position& pos, Move m);  // exact SEE value (search over thresholds)
// Face-down piece of the side to move on s under opponent attack: 0 = not attacked, 1 = attacked, 2 = ThreatRev
// trigger; *loss = probability-weighted opponent SEE gain over the mover's urn identities (cp).
int threat_reveal(const Position& pos, int s, int* loss, const int* darkV = nullptr);  // darkV: urn means (RED, BLACK) if known

}  // namespace search
}  // namespace jq
