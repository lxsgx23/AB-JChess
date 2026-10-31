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
// Small-net-only evaluator facade (release builds, -DJQ_SMALL_ONLY): replaces nnue.cpp + abjnnue.cpp + snnue.cpp +
// hnnue_small.cpp + threats.cpp. The only evaluator is the HNN small net (hnnue.cpp compiled without the JQT1 threat
// feature code); EvalFile accepts only a small HNN net (raw JQHNNUE1 or JQHNNL01 / JQHNNZ01 compressed, detected by
// magic): no threat trailer, width <= 512. Big nets (threat features), ABJNNUE packages, SNN and JQN1 nets are
// refused with an error and the active net stays. Search results are identical to nnue.cpp with an HNN net loaded
// (same calls in the same order: refresh / make_dirty / update / evaluate of jq::hnn).
#include "nnue.h"
#include "eval.h"
#include "jqnz.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <xmmintrin.h>
#endif

namespace jq::nnue {

int g_kingBuckets = 1;
int g_urnBase = NUM_PIECE_FEATURES;
int g_smallScale = 875;
int g_hsmScale = 1000;
uint64_t g_dualEv[3] = {0, 0, 0};

namespace {
bool g_loaded = false;
constexpr int SMALL_MAX_W = 512;
bool fail(std::string* error, const std::string& m) { if (error) *error = m; return false; }
}  // namespace

bool load(const std::string& path, std::string* error) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(error, "cannot open file");
    char m[8] = {};
    const bool rd = std::fread(m, 1, 8, f) == 8;
    std::fclose(f);
    if (!rd) return fail(error, "file too short");
    if (!hnn::is_hnn_file(path)) {
        if (!std::memcmp(m, "ABJNNUE", 7) || !std::memcmp(m, "JQN1", 4) || !std::memcmp(m, "JQSNNUE1", 8))
            return fail(error, "not a small net: this engine supports only the small HNN net (abjchess-20261030.nnue format)");
        return fail(error, "unknown net format (expected a small HNN net, raw JQHNNUE1 or JQHNNL01 / JQHNNZ01 compressed)");
    }
    if (!hnn::load(path, error)) return false;
    if (hnn::width() > SMALL_MAX_W) {  // a wide net without threat features: not a small net either
        hnn::unload();
        g_loaded = false;
        return fail(error, "HNN: width " + std::to_string(hnn::width()) + " > " + std::to_string(SMALL_MAX_W) + ": a big net; refused, this engine supports only the small net");
    }
    g_loaded = true;
    return true;
}
bool has_embedded() { return false; }
std::size_t embedded_size() { return 0; }
bool export_embedded(const std::string&, std::string* error) { return fail(error, "no embedded net in this build"); }
bool load_embedded(std::string* error) { return fail(error, "no embedded net in this build"); }

bool decode_net_file(const std::string& in, const std::string& outPath, std::string* error) {
    FILE* f = std::fopen(in.c_str(), "rb");
    if (!f) return fail(error, "cannot open " + in);
    std::vector<uint8_t> buf;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    buf.resize(sz > 0 ? std::size_t(sz) : 0);
    const bool rd = !buf.empty() && std::fread(buf.data(), 1, buf.size(), f) == buf.size();
    std::fclose(f);
    if (!rd) return fail(error, "read error " + in);
    std::vector<uint8_t> raw;
    if (jqnz::is_compressed(buf.data(), buf.size())) {
        if (!jqnz::decode(buf.data(), buf.size(), raw, error)) return false;
    } else raw.swap(buf);
    FILE* o = std::fopen(outPath.c_str(), "wb");
    const bool ok = o && std::fwrite(raw.data(), 1, raw.size(), o) == raw.size();
    if (o && std::fclose(o) != 0) return fail(error, "cannot write " + outPath);
    return ok ? true : fail(error, "cannot write " + outPath);
}

void unload() { g_loaded = false; }
bool load_big(const std::string&, std::string* error) { return fail(error, "big nets are not supported by this small-net-only engine"); }
void unload_big() {}
bool dual() { return false; }
Value evaluate_small(const Accumulator& acc, Color stm) {
    const int v = int(hnn::evaluate(acc.hnn, stm));
    const int s = v * g_smallScale / 1000;
    return Value(std::clamp(s, -int(VALUE_KNOWN_WIN - 1), int(VALUE_KNOWN_WIN - 1)));
}
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
void set_flush_denormals() { _mm_setcsr(_mm_getcsr() | 0x8040); }  // FTZ | DAZ for this thread
#elif defined(__aarch64__)
void set_flush_denormals() {
    uint64_t r; __asm__ volatile("mrs %0, fpcr" : "=r"(r)); r |= uint64_t(1) << 24; __asm__ volatile("msr fpcr, %0" : : "r"(r));
}
#elif defined(__arm__)
void set_flush_denormals() {
    uint32_t r; __asm__ volatile("vmrs %0, fpscr" : "=r"(r)); r |= uint32_t(1) << 24; __asm__ volatile("vmsr fpscr, %0" : : "r"(r));
}
#else
void set_flush_denormals() {}
#endif
void set_quantized(bool) {}
bool quantized() { return false; }
bool loaded() { return g_loaded; }
bool is_abj() { return false; }
bool is_snn() { return false; }
bool is_hnn() { return g_loaded; }
int ft_size() { return 1; }  // the JQN1 float net's FT width + PSQT column; no JQN1 net in this build (nnue.cpp: net.ft 0 + 1)

bool load_hsm(const std::string&, std::string* error) { return fail(error, "EvalFileSmall (dual HNN) is not supported by this build"); }
void unload_hsm() {}
bool hsm_loaded() { return false; }
void set_dual_skip(int, int, int, int) {}
Value evaluate_hsm(const Accumulator& acc, Color stm) { return hnn::evaluate(acc.hnn, stm); }  // never called (no dual mode)
int simple_material(const Position& pos) {
    int v[2] = {0, 0};
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < pos.pcount[c]; ++i) {
            const PieceType t = type_of(pos.board[pos.plist[c][i]]);
            v[c] += t == DARK ? pos.urnMean[c] : PieceValue[t];
        }
    return pos.stm == RED ? v[0] - v[1] : v[1] - v[0];
}

// As nnue.cpp with an HNN net loaded (g_small, g_hnnSmall; no big / dual net): the small net is NET_MAIN.
void refresh(const Position& pos, Accumulator& acc, unsigned mask) {
    if (g_loaded && (mask & NET_MAIN)) hnn::refresh(pos, acc.hnn);
}
void make_dirty(const Position& pos, Move m, Dirty& d) {
    d.bigSkip = false;
    hnn::make_delta(pos, m, d.hd);
    d.refresh = -1;
}
void update(const Accumulator& prev, Accumulator& next, const Dirty& d, unsigned mask) {
    if (g_loaded && (mask & NET_MAIN)) hnn::update(prev.hnn, next.hnn, d.hd);
}
Value evaluate(const Accumulator& acc, Color stm) { return hnn::evaluate(acc.hnn, stm); }
Value evaluate(const Position& pos) {
    static thread_local Accumulator acc;
    refresh(pos, acc);
    return evaluate(acc, pos.stm);
}

}  // namespace jq::nnue
