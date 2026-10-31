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
// NNUE evaluation (original design, float inference).
//
// Inputs per perspective P (1470): piece-square features for 16 kinds (own/opponent x K R N B A C
// P face-down) on 90 squares, oriented so that P's home rank is rank 0; plus urn thresholds
// "urn[c][t] >= j" (15 per colour, own colour first). Feature transformer 1470 -> FT (+1 PSQT
// column), shared by both perspectives; then [us, them] -> L1 -> L2 -> 1 with clipped ReLU.
#pragma once
#include "hnnue.h"
#include "position.h"

#include <cstddef>
#include <string>

namespace jq::nnue {

constexpr int NUM_PIECE_FEATURES = 16 * NUM_SQ;          // 1440 (per king bucket)
constexpr int NUM_URN_FEATURES = 30;
constexpr int NUM_FEATURES = NUM_PIECE_FEATURES + NUM_URN_FEATURES;  // 1470 (version-1 nets)
constexpr int MAX_FT = 1024;
constexpr int URN_OFFSET[NUM_SLOTS] = {0, 2, 4, 6, 8, 10};  // cumulative URN_INIT

inline int orient(Color persp, int s) { return persp == RED ? s : (9 - rank_of(s)) * 9 + file_of(s); }
// Version-2 nets replicate the piece features per bucket of the perspective's own king square
// (3 = king file, 9 = palace square); urn features follow all buckets. Set by load().
extern int g_kingBuckets;
extern int g_urnBase;  // g_kingBuckets * NUM_PIECE_FEATURES
inline int king_bucket_offset(Color persp, int ksq) {
    if (g_kingBuckets == 1) return 0;
    const int ko = orient(persp, ksq);
    const int b = g_kingBuckets == 9 ? (ko / 9) * 3 + (ko % 9 - 3) : ko % 9 - 3;
    return b * NUM_PIECE_FEATURES;
}
inline int piece_feature(Color persp, Piece p, int s, int kbOff) {
    const int rel = color_of(p) == persp ? 0 : 8;
    return kbOff + (rel + int(type_of(p)) - 1) * NUM_SQ + orient(persp, s);
}
// Feature for "urn[owner][slot] >= j", j in 1..URN_INIT[slot].
inline int urn_feature(Color persp, Color owner, int slot, int j) {
    return g_urnBase + (owner == persp ? 0 : 15) + URN_OFFSET[slot] + j - 1;
}

// Small-net-only release build (nnue_small.cpp): the HNN small net is the only evaluator (no big / dual / legacy nets).
struct alignas(64) Accumulator {
    hnn::Acc hnn;
};
struct Dirty {
    int8_t refresh = -1;   // always -1 (king-bucket changes are handled inside hnn::update)
    bool bigSkip = false;  // always false (no dual mode)
    hnn::Delta hd;
};

// EvalFile: a JQN1 net or a JQSNNUE1 (SNN) net (the "small" net), or an ABJNNUE package (pure big-net mode,
// small net inactive).
bool load(const std::string& path, std::string* error = nullptr);
// Embedded-net hooks (no net is compiled into release builds; nets ship as separate files).
bool has_embedded();
std::size_t embedded_size();
bool export_embedded(const std::string& path, std::string* error = nullptr);  // decoded .hnn bytes
bool load_embedded(std::string* error = nullptr);
// Any net file (raw .hnn or JQHNNL01 / JQHNNZ01 compressed, CRC-32 checked) -> the raw .hnn bytes written to out.
bool decode_net_file(const std::string& in, const std::string& out, std::string* error = nullptr);
void unload();
// EvalFileBig: an ABJNNUE package loaded alongside the small EvalFile net. With both loaded ("dual"),
// both accumulators are maintained incrementally and evaluate() uses the big net; evaluate_small() gives
// the small net's value scaled by g_smallScale / 1000 (to the big net's scale).
bool load_big(const std::string& path, std::string* error = nullptr);
void unload_big();
bool dual();
extern int g_smallScale;  // permille
Value evaluate_small(const Accumulator& acc, Color stm);
void set_flush_denormals();  // call once per thread that evaluates (denormals are very slow)
// Quantized inference (int16 accumulators, int8 L1) is used when available and enabled.
void set_quantized(bool on);
bool quantized();
bool loaded();
bool is_abj();  // evaluation uses an AB-JChess ABJNNUE package (EvalFile or EvalFileBig)
bool is_snn();  // the small net is an SNN (JQSNNUE1) net
bool is_hnn();  // the small net is an HNN (JQHNNUE1) net
int ft_size();  // FT width including the PSQT column

// Net masks for refresh / update: NET_MAIN = the net evaluate() uses (the big net if one is active, else the
// small net); NET_SMALL = the small net in dual mode. Tools pass NET_ALL.
constexpr unsigned NET_MAIN = 1, NET_SMALL = 2, NET_ALL = 3;
// dual HNN (not used by the release build): EvalFileSmall = a second HNN net
// (no threats) kept in Accumulator::hs (mask NET_SMALL) next to the EvalFile HNN net (NET_MAIN). The search picks the
// net per node by DualMode (search.cpp). Not loaded = bit-identical to the single-net engine.
bool load_hsm(const std::string& path, std::string* error = nullptr);
void unload_hsm();
bool hsm_loaded();
extern uint64_t g_dualEv[3];  // evals by big / small / small computed then rejected (DualMode 6); T1 diagnostics, not atomic
// dual skip: make_dirty skips the big net's delta when the child is predicted to be evaluated by the small net
// (mode/threshold mirror DualMode; margin on the material prediction; mode 0 = never skip). Set per search.
void set_dual_skip(int mode, int thr, int hid, int margin);
extern int g_hsmScale;   // permille applied to the small net's value
Value evaluate_hsm(const Accumulator& acc, Color stm);
int simple_material(const Position& pos);  // side-to-move relative, PieceValue units, face-down = urn mean
inline int hidden_count(const Position& pos) { return pos.darkCount[0] + pos.darkCount[1]; }
void refresh(const Position& pos, Accumulator& acc, unsigned mask = NET_ALL);
void make_dirty(const Position& pos, Move m, Dirty& d);  // m carries the reveal for a dark mover
// Incremental update; not valid when d.refresh >= 0 (refresh from the new position instead).
void update(const Accumulator& prev, Accumulator& next, const Dirty& d, unsigned mask = NET_ALL);
Value evaluate(const Accumulator& acc, Color stm);
Value evaluate(const Position& pos);  // full refresh; for tools and the root

}  // namespace jq::nnue
