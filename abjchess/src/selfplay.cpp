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
// Native self-play data generation. Each game uses a random hidden deal; every side searches its
// own observation (reveals are public, captured face-down identities are seen by the capturer only).
// Output: fixed 112-byte TrainRecord per searched position (read by the trainer in abjchess-nnue-pytorch/).
#include "nnue.h"
#include "search.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace jq {

#pragma pack(push, 1)
struct TrainRecord {
    uint8_t board[NUM_SQ];  // Piece codes (type | colour << 4)
    uint8_t stm;
    int8_t urn[2][NUM_SLOTS];  // as seen by the side to move
    int16_t score;             // search score, side-to-move perspective
    int8_t result;             // game result, side-to-move perspective: 1 / 0 / -1
    uint8_t flags;             // 1 in check, 2 best move captures, 4 best move reveals, bits 3-7 depth (max 31)
    uint16_t ply;
    uint16_t move;             // base move of the search's choice
    uint8_t noCapture;
};
#pragma pack(pop)
static_assert(sizeof(TrainRecord) == 112, "record layout");

namespace {

struct Config {
    int games = 100;
    int64_t nodes = 5000;
    int depth = 0;
    int threads = 1;
    uint64_t seed = 1;
    int randomPlies = 6;
    int hashMb = 16;
    int maxPly = 600;
    std::string out = "selfplay.bin";
};

uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

struct Stats {
    std::atomic<int> games{0}, redWins{0}, blackWins{0}, draws{0};
    std::atomic<uint64_t> records{0}, plies{0};
};

// Plays one game; returns result from Red's perspective and appends records.
int play_one(const Config& cfg, uint64_t gameSeed, Searcher* S[2], std::vector<TrainRecord>& recs) {
    std::mt19937_64 rng(gameSeed);
    Position view[2];
    view[RED].set_fen(START_FEN);
    view[BLACK] = view[RED];
    int deal[2][NUM_SQ];
    for (int c = 0; c < 2; ++c) {
        std::vector<int> bag;
        for (int sl = 0; sl < NUM_SLOTS; ++sl)
            for (int k = 0; k < URN_INIT[sl]; ++k) bag.push_back(sl);
        std::shuffle(bag.begin(), bag.end(), rng);
        int k = 0;
        for (int s = 0; s < NUM_SQ; ++s) {
            const Piece p = view[RED].board[s];
            deal[c][s] = (p && type_of(p) == DARK && color_of(p) == c) ? bag[k++] : -1;
        }
    }
    S[0]->clear();
    S[1]->clear();
    const int randomPlies = cfg.randomPlies > 0 ? int(rng() % uint64_t(cfg.randomPlies + 1)) : 0;
    const std::size_t first = recs.size();
    int result = 0;
    Limits lim;
    lim.nodes = cfg.nodes;
    lim.depth = cfg.depth;
    const std::function<void(const std::string&)> quiet = [](const std::string&) {};
    for (int ply = 0;; ++ply) {
        const Color side = view[RED].stm;
        Position& v = view[side];
        Move legal[MAX_MOVES];
        const int n = v.gen_legal(legal);
        if (n == 0) { result = side == RED ? -1 : 1; break; }
        const int rs = ply > 0 ? v.repetition_status() : 0;
        if (rs == 1) { result = 0; break; }
        if (rs == 2) { result = side == RED ? 1 : -1; break; }
        if (rs == 3) { result = side == RED ? -1 : 1; break; }
        if (v.noCapture >= NO_CAPTURE_LIMIT || ply >= cfg.maxPly) { result = 0; break; }
        Move m;
        if (ply < randomPlies) {
            m = legal[rng() % uint64_t(n)];
        } else {
            const SearchResult r = S[side]->go(v, lim, quiet);
            m = base_move(r.best);
            TrainRecord rec{};
            for (int s = 0; s < NUM_SQ; ++s) rec.board[s] = v.board[s];
            rec.stm = uint8_t(side);
            for (int c = 0; c < 2; ++c)
                for (int sl = 0; sl < NUM_SLOTS; ++sl) rec.urn[c][sl] = v.urn[c][sl];
            rec.score = int16_t(std::clamp(r.score, -32000, 32000));
            rec.flags = uint8_t((v.checkHist.back() ? 1 : 0) | (v.board[to_sq(m)] ? 2 : 0) | (v.is_dark_move(m) ? 4 : 0) |
                                (std::min(r.depth, 31) << 3));
            rec.ply = uint16_t(ply);
            rec.move = uint16_t(m);
            rec.noCapture = uint8_t(std::min<int>(v.noCapture, 255));
            recs.push_back(rec);
        }
        const int from = from_sq(m), to = to_sq(m);
        const PieceType reveal = v.is_dark_move(m) ? slot_type(deal[side][from]) : NO_TYPE;
        const Piece victim = v.board[to];
        const PieceType capId = victim && type_of(victim) == DARK ? slot_type(deal[~side][to]) : NO_TYPE;
        std::string err;
        if (!view[side].apply_observed(m, reveal, capId, &err) || !view[~side].apply_observed(m, reveal, NO_TYPE, &err)) {
            std::fprintf(stderr, "selfplay: apply failed: %s\n", err.c_str());
            result = 0;
            break;
        }
    }
    for (std::size_t i = first; i < recs.size(); ++i)
        recs[i].result = int8_t(recs[i].stm == RED ? result : -result);
    return result;
}

}  // namespace

void selfplay(std::istringstream& is) {
    Config cfg;
    std::string key;
    while (is >> key) {
        if (key == "games") is >> cfg.games;
        else if (key == "nodes") is >> cfg.nodes;
        else if (key == "depth") is >> cfg.depth;
        else if (key == "threads") is >> cfg.threads;
        else if (key == "seed") is >> cfg.seed;
        else if (key == "random") is >> cfg.randomPlies;
        else if (key == "hash") is >> cfg.hashMb;
        else if (key == "maxply") is >> cfg.maxPly;
        else if (key == "out") is >> cfg.out;
        else if (key == "quant") {
            int on = 0;
            is >> on;
            nnue::set_quantized(on != 0);
        } else if (key == "evalfile") {
            std::string path, err;
            is >> path;
            if (!nnue::load(path, &err)) {
                std::printf("info string selfplay: cannot load %s: %s\n", path.c_str(), err.c_str());
                return;
            }
            std::printf("info string selfplay: loaded %s\n", path.c_str());
        }
    }
    FILE* f = std::fopen(cfg.out.c_str(), "ab");
    if (!f) {
        std::printf("info string cannot open %s\n", cfg.out.c_str());
        return;
    }
    Stats st;
    std::atomic<int> next{0};
    std::mutex fileMutex;
    const auto t0 = std::chrono::steady_clock::now();
    auto work = [&] {
        Searcher a(1, std::size_t(cfg.hashMb)), b(1, std::size_t(cfg.hashMb));
        Searcher* S[2] = {&a, &b};
        std::vector<TrainRecord> recs;
        while (true) {
            const int g = next.fetch_add(1);
            if (g >= cfg.games) break;
            recs.clear();
            const int r = play_one(cfg, mix(cfg.seed * 0x100000001B3ull + uint64_t(g)), S, recs);
            {
                std::lock_guard<std::mutex> lock(fileMutex);
                std::fwrite(recs.data(), sizeof(TrainRecord), recs.size(), f);
            }
            st.records += recs.size();
            (r > 0 ? st.redWins : r < 0 ? st.blackWins : st.draws)++;
            const int done = ++st.games;
            if (done % 100 == 0 || done == cfg.games) {
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("info string selfplay games %d red %d black %d draw %d records %llu (%.0f rec/s) %.0fs\n", done,
                            st.redWins.load(), st.blackWins.load(), st.draws.load(),
                            (unsigned long long)st.records.load(), st.records.load() / std::max(sec, 1e-9), sec);
                std::fflush(stdout);
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < cfg.threads; ++i) pool.emplace_back(work);
    for (auto& t : pool) t.join();
    std::fclose(f);
    std::printf("info string selfplay done %d games, %llu records -> %s\n", st.games.load(),
                (unsigned long long)st.records.load(), cfg.out.c_str());
    std::fflush(stdout);
}

}  // namespace jq
