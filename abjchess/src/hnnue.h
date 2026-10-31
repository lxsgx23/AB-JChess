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
// Fourth evaluator: "HNN", a hybrid small net: the cheap inputs of the it014 JQN1 nets with the ABJ-style head of
// the SNN nets (snnue.h). Trained by the trainer in abjchess-nnue-pytorch/ (train_hnn.py).
//
// Inputs per perspective P: KB king buckets x 16 piece kinds (own/opponent x K R N B A C P face-down) x 90 points,
// oriented so that P's home rank is rank 0 (nnue::orient) and mirrored left-right when the bucket scheme says so,
// plus 30 urn thresholds "urn[c][slot] >= j" (own colour first; nnue::urn_feature layout). No face-down variant
// doubling: a move changes at most 2 + 1 + 1 inputs (from, captured, to, one urn threshold on a reveal); only
// a king move that changes the king's bucket or mirror state refreshes that side (per-thread cache).
// King bucket schemes (own king on the oriented palace, rank r = 0..2, file f = 3..5):
//   0: none (KB 1)   1: by file, no mirror (KB 3)   2: mirror f > 4, bucket f == 4 (KB 2)
//   3: mirror f > 4, bucket 2 r + (f == 4) (KB 6)
// FT: W int16 + 16 int32 PSQT; head exactly as SNN: pairwise product of the halves -> W uint8, 16 layer stacks
// (bucket = min((pieces - 1) / 4, 7) + 8 if any face-down piece), W -> 16 -> [sqr, crelu] -> 32 -> 1 + skip,
// nn = (psqt + positional) / 16, then the AB-JChess post-processing with material = sum urn * value + face-up
// R N B A C on the board * value (no min with the revealed count) and the 80-ply damping.
//
// File "JQHNNUE1": 8-byte magic, <8I version(1) W nfeat kbScheme npsqt(16) nstacks(16) l1(16) l2(32),
// then int16 ftB[W], int16 ftW[nfeat][W], int32 psqt[nfeat][16], then 16 stacks exactly as JQSNNUE1.
#pragma once
#include "position.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace jq::hnn {

constexpr int MAX_W = 1024, NPSQT = 16;

struct Meta {
    int16_t material = 0, noCapture = 0;
    int8_t pieces = 0, dark = 0;
    int8_t kb[2] = {0, 0}, mirror[2] = {0, 0};
};

constexpr int MAX_TW = 256;  // threat width up to which the threat part is kept separately (t)

struct alignas(64) Acc {
    int16_t v[2][MAX_W];
    int32_t ps[2][NPSQT];
    Meta m;
    // (the threat-row sum Acc::t lives in a separate big-net-only side table, hnnue.cpp tacc_of(): the small-net
    // accumulator layout and size are unchanged.)
};

// Feature changes of one move (computed by make_delta from the position before the move).
struct Delta {
    bool valid = false;            // false: null move (copy, noCapture + 1)
    bool refP[2] = {false, false}; // this perspective's king bucket / mirror changed: refresh it from the cache
    int8_t nAdd = 0, nRem = 0;
    uint16_t add[2][1], rem[2][3];
    int8_t nFull[2] = {0, 0};
    uint16_t full[2][64];          // refP[p]: all features of p after the move
    Meta m;
};

// Threat features: on = 0 off (baseline behaviour), 1 on; init = 0 zero weights, 1 random int8 in [-amp, amp].
// width: threat rows feed accumulator lanes [0, width) only (0 = all W lanes); multiple of 128.
void set_threats(int on, int init, int amp, uint64_t seed, int width = 0);
bool threats_on();
int threat_width();  // threat row width Wt (0 when off)
bool tacc();         // separate threat accumulator (side table) in use
// Big-net side table: accumulators first + i * stride (i < n) map to slot i (the search stack); any other Acc gets one of
// a few extra slots on first use. Only allocated / used when tacc() is on.
void bind_acc_stack(const Acc* first, std::size_t stride, int n);
// Full recompute without the Finny cache (verification).
void refresh_nocache(const Position& pos, Acc& acc);
// Statistics (JQ_THR_STATS builds): makes, updates, tuples generated, rows applied.
struct ThrStats { uint64_t makes, gens, tuples, rows, refreshes; };
ThrStats thr_stats();

bool is_hnn_file(const std::string& path);
bool load(const std::string& path, std::string* error);  // raw .hnn or a JQHNNL01/JQHNNZ01 compressed net
bool load_buffer(const uint8_t* data, std::size_t size, std::string* error);  // same, from memory
// Last successful load: container ("raw", "JQHNNL01", "JQHNNZ01"), file bytes, decoded bytes, decode time (ms).
struct LoadInfo { std::string fmt = "raw"; std::size_t fileBytes = 0, rawBytes = 0; double decodeMs = 0; std::string rawMd5; };  // rawMd5: of the decoded bytes
const LoadInfo& last_load();
void unload();
bool loaded();
int width();
int kb_scheme();
std::string head_desc();  // 'v1 L1 16 L2 32' or 'v3 L1 .. L2 .. [sq2]'
extern int g_scale;  // permille

void refresh(const Position& pos, Acc& acc);
void make_delta(const Position& pos, Move m, Delta& d);
void update(const Acc& prev, Acc& next, const Delta& d);
int runtime_value(const Acc& acc, Color stm, int* psqtRaw = nullptr, int* positionalRaw = nullptr);
Value evaluate(const Acc& acc, Color stm);

}  // namespace jq::hnn
