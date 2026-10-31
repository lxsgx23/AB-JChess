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
#include "hnnue.h"
#include "jqnz.h"
#include "jqmd5.h"
#include <chrono>

#include "mem.h"
#include "nnkern.h"
#include "headkern.h"
#include "headfast.h"
#ifndef JQ_THR_PF_LINES
#define JQ_THR_PF_LINES 2
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace jq::hnn {

int g_scale = 1000;

namespace {

constexpr int NSTACKS = 16, L1 = nnk::L1, L2 = nnk::L2;
constexpr int PIECE_FEATS = 16 * NUM_SQ;  // per king bucket
constexpr int URN_FEATS = 30;
constexpr int URN_OFF[NUM_SLOTS] = {0, 2, 4, 6, 8, 10};
constexpr int PIECE_VAL[NUM_SLOTS] = {943, 438, 168, 190, 491, 101};
constexpr char MAGIC[8] = {'J', 'Q', 'H', 'N', 'N', 'U', 'E', '1'};
constexpr int KB_OF_SCHEME[4] = {1, 3, 2, 6};

struct FreeDel { void operator()(void* p) const { jq_aligned_free(p); } };
int g_W = 0, g_scheme = 0, g_nfeat = 0, g_urnBase = 0;
bool g_loaded = false;
std::atomic<unsigned> g_gen{0};
std::unique_ptr<int16_t[], FreeDel> g_ftW;
std::unique_ptr<int32_t[], FreeDel> g_psqt;
alignas(64) int16_t g_ftB[MAX_W];
std::unique_ptr<uint8_t[], FreeDel> g_l1w;
#define THRST(x, n) ((void)0)
}  // namespace
void bind_acc_stack(const Acc*, std::size_t, int) {}  // small-net-only build: no threat side table

namespace {
alignas(64) int32_t g_l1b[NSTACKS][L1];
alignas(64) int32_t g_l2b[NSTACKS][L2];
alignas(64) int8_t g_l2w[NSTACKS][L2 * L2];
int32_t g_fb[NSTACKS];
alignas(64) int8_t g_fw[NSTACKS][L2];
// version-3 head (configurable L1/L2/sq2); g_head = 0 for version-1 nets (original arrays and kernel above)
int g_head = 0, g_hL1 = 16, g_hL2 = 32, g_hSq2 = 0;
alignas(64) int32_t g3_l1b[NSTACKS][32];
alignas(64) int32_t g3_l2b[NSTACKS][64];
alignas(64) int8_t g3_l2w[NSTACKS][64 * 64];
int32_t g3_fb[NSTACKS];
alignas(64) int8_t g3_fw[NSTACKS][128];
// re-laid-out stacks for the AVX2 fast kernel (headfast.h); g_fast = L1 (16 or 32) when the net's head is
// W -> L1 -> 32 -> 1 without sq2 (all version-1 nets, version-3 L2 32 no sq2), else 0 (generic kernels above)
int g_fast = 0;
alignas(64) nnk::FastHead g_fh[NSTACKS];
// Per scheme and oriented palace point (rank 0..2, file 3..5 -> 3 r + f - 3): bucket and mirror flag.
int8_t g_kbTab[9], g_mirTab[9];

inline int orient(Color persp, int s) { return persp == RED ? s : (9 - rank_of(s)) * 9 + file_of(s); }

void set_scheme(int scheme) {
    for (int r = 0; r < 3; ++r)
        for (int f = 3; f <= 5; ++f) {
            const int i = r * 3 + f - 3;
            int b = 0, m = 0;
            if (scheme == 1) b = f - 3;
            else if (scheme == 2) { m = f > 4; b = f == 4; }
            else if (scheme == 3) { m = f > 4; b = 2 * r + (f == 4); }
            g_kbTab[i] = int8_t(b);
            g_mirTab[i] = int8_t(m);
        }
}

inline void king_state(Color persp, int ksq, int8_t& kb, int8_t& mir) {
    const int o = orient(persp, ksq);
    const int i = rank_of(o) * 3 + file_of(o) - 3;  // the king never leaves its palace
    kb = g_kbTab[i];
    mir = g_mirTab[i];
}

inline int piece_feat(Color persp, Piece p, int s, int kb, int mir) {
    int o = orient(persp, s);
    if (mir) o = rank_of(o) * 9 + 8 - file_of(o);
    return kb * PIECE_FEATS + ((color_of(p) == persp ? 0 : 8) + int(type_of(p)) - 1) * NUM_SQ + o;
}
inline int urn_feat(Color persp, Color owner, int slot, int j) {  // urn[owner][slot] >= j, j >= 1
    return g_urnBase + (owner == persp ? 0 : 15) + URN_OFF[slot] + j - 1;
}

int features(const Position& pos, Color persp, int kb, int mir, uint16_t* f) {
    int n = 0;
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < pos.pcount[c]; ++i) {
            const int s = pos.plist[c][i];
            f[n++] = uint16_t(piece_feat(persp, pos.board[s], s, kb, mir));
        }
    for (int c = 0; c < 2; ++c)
        for (int sl = 0; sl < NUM_SLOTS; ++sl)
            for (int j = 1; j <= pos.urn[c][sl]; ++j) f[n++] = uint16_t(urn_feat(persp, Color(c), sl, j));
    return n;
}

inline void apply(int16_t* dst, int32_t* pdst, const int16_t* src, const int32_t* psrc, const uint16_t* add, int nAdd,
                  const uint16_t* rem, int nRem) {
    nnk::apply(g_ftW.get(), g_psqt.get(), g_W, dst, pdst, src, psrc, add, nAdd, rem, nRem);
}

struct FinnyEntry {
    alignas(64) int16_t v[MAX_W];
    int32_t ps[NPSQT];
    uint16_t f[64];
    int n = -1;
    unsigned gen = ~0u;
};
// Refresh of perspective p from the per-thread cache entry of its (bucket, mirror): only the difference between
// the cached and the new sorted feature lists is applied.
void finny_refresh(int p, int kb, int mir, uint16_t* f, int n, int16_t* out, int32_t* pout) {
    static thread_local std::unique_ptr<FinnyEntry[]> tab;
    if (!tab) tab.reset(new FinnyEntry[2 * 6 * 2]);
    FinnyEntry* e = &tab[(p * 6 + kb) * 2 + mir];
    const unsigned gen = g_gen.load(std::memory_order_relaxed);
    if (e->gen != gen) { e->n = -1; e->gen = gen; }
    std::sort(f, f + n);
    alignas(64) static const int32_t zero[NPSQT] = {};
    if (e->n < 0) {
        apply(e->v, e->ps, g_ftB, zero, f, n, nullptr, 0);
    } else {
        uint16_t add[64], rem[64];
        int na = 0, nr = 0, i = 0, j = 0;
        while (i < n || j < e->n) {
            if (j >= e->n || (i < n && f[i] < e->f[j])) add[na++] = f[i++];
            else if (i >= n || e->f[j] < f[i]) rem[nr++] = e->f[j++];
            else { ++i; ++j; }
        }
        if (na || nr) apply(e->v, e->ps, e->v, e->ps, add, na, rem, nr);
    }
    std::memcpy(e->f, f, sizeof(uint16_t) * std::size_t(n));
    e->n = n;
    std::memcpy(out, e->v, sizeof(int16_t) * std::size_t(g_W));
    std::memcpy(pout, e->ps, sizeof(e->ps));
}

}  // namespace

bool is_hnn_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char m[8] = {};
    const bool ok = std::fread(m, 1, 8, f) == 8 &&
                    (std::memcmp(m, MAGIC, 8) == 0 || jqnz::is_compressed(reinterpret_cast<const uint8_t*>(m), 8));
    std::fclose(f);
    return ok;
}

bool load(const std::string& path, std::string* error) {
    auto fail = [&](const char* msg) { if (error) *error = msg; return false; };
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail("cannot open file");
    std::vector<uint8_t> buf;
    {
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        buf.resize(sz > 0 ? std::size_t(sz) : 0);
        const bool ok = !buf.empty() && std::fread(buf.data(), 1, buf.size(), f) == buf.size();
        std::fclose(f);
        if (!ok) return fail("HNN: read error");
    }
    return load_buffer(buf.data(), buf.size(), error);
}

namespace { LoadInfo g_li, g_liPending; }
const LoadInfo& last_load() { return g_li; }
bool load_buffer(const uint8_t* data, std::size_t size, std::string* error) {
    auto fail = [&](const char* msg) { if (error) *error = msg; return false; };
    std::vector<uint8_t> dec;
    g_liPending = LoadInfo{};
    g_liPending.fileBytes = size;
    if (jqnz::is_compressed(data, size)) {  // LEB128 / range-coded container: decode to the exact original bytes (CRC-32 checked)
        g_liPending.fmt = std::string(reinterpret_cast<const char*>(data), 8);
        const auto t0 = std::chrono::steady_clock::now();
        if (!jqnz::decode(data, size, dec, error)) return false;
        g_liPending.decodeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        data = dec.data();
        size = dec.size();
    }
    g_liPending.rawBytes = size;
    g_liPending.rawMd5 = md5_hex(data, size);
    struct View {
        const uint8_t* p; std::size_t n;
        std::size_t size() const { return n; }
        const uint8_t* data() const { return p; }
    } buf{data, size};
    if (buf.size() < 40 || std::memcmp(buf.data(), MAGIC, 8) != 0) return fail("HNN: bad magic");
    uint32_t h[8];
    std::memcpy(h, buf.data() + 8, sizeof h);
    const int W = int(h[1]), scheme = int(h[3]);
    const bool v3 = h[0] == 3;
    uint32_t ext[4] = {0, 0, 0, 0};
    if (v3) { if (buf.size() < 56) return fail("HNN: short v3 header"); std::memcpy(ext, buf.data() + 40, sizeof ext); }
    const int hL1 = int(h[6]), hL2 = int(h[7]), hSq2 = int(ext[0] & 1);
    if (!(h[0] == 1 || v3) || W < 128 || W > MAX_W || W % 128 || h[3] > 3 || h[4] != NPSQT || h[5] != NSTACKS ||
        (!v3 && (hL1 != L1 || hL2 != L2)) || (v3 && ((hL1 != 16 && hL1 != 32) || (hL2 != 32 && hL2 != 64) || (ext[0] & ~1u) || ext[1] || ext[2] || ext[3])))
        return fail("HNN: unsupported header (version 1: 16/16/16/32; version 3: L1 16|32, L2 32|64, flags sq2; W multiple of 128 up to 1024, scheme 0..3)");
    const int hL2IN = (2 * (hL1 - 1) + 31) / 32 * 32, hNO = hL2 * (hSq2 ? 2 : 1);
    const int nfeat = KB_OF_SCHEME[scheme] * PIECE_FEATS + URN_FEATS;
    if (int(h[2]) != nfeat) return fail("HNN: feature count does not match the king bucket scheme");
    const std::size_t headSz = v3 ? 4 * hL1 + std::size_t(W) * hL1 + 4 * hL2 + std::size_t(hL2) * hL2IN + 4 + hNO
                                  : 64 + std::size_t(W) * L1 + 128 + 1024 + 4 + 32;
    const std::size_t need = (v3 ? 56 : 40) + std::size_t(W) * 2 + std::size_t(nfeat) * W * 2 + std::size_t(nfeat) * NPSQT * 4 + NSTACKS * headSz;
    if (buf.size() >= need + 8 && std::memcmp(buf.data() + need, "JQT1THR1", 8) == 0)
        return fail("HNN: this is a big net (JQT1 threat features); refused: this engine supports only the small net");
    if (buf.size() != need) return fail("HNN: unexpected file size");
    g_ftW.reset(static_cast<int16_t*>(alloc_large(std::size_t(nfeat) * W * sizeof(int16_t))));
    g_psqt.reset(static_cast<int32_t*>(alloc_large(std::size_t(nfeat) * NPSQT * sizeof(int32_t))));
    g_l1w.reset(static_cast<uint8_t*>(jq_aligned_alloc(64, std::size_t(NSTACKS) * W * (v3 ? hL1 : L1))));
    if (!g_ftW || !g_psqt || !g_l1w) return fail("HNN: out of memory");
    const uint8_t* p = buf.data() + (v3 ? 56 : 40);
    std::memset(g_ftB, 0, sizeof g_ftB);
    std::memcpy(g_ftB, p, std::size_t(W) * 2); p += std::size_t(W) * 2;
    std::memcpy(g_ftW.get(), p, std::size_t(nfeat) * W * 2); p += std::size_t(nfeat) * W * 2;
    std::memcpy(g_psqt.get(), p, std::size_t(nfeat) * NPSQT * 4); p += std::size_t(nfeat) * NPSQT * 4;
    if (v3) {
        std::memset(g3_l2w, 0, sizeof g3_l2w);
        for (int s = 0; s < NSTACKS; ++s) {
            std::memcpy(g3_l1b[s], p, 4 * hL1); p += 4 * hL1;
            std::memcpy(g_l1w.get() + std::size_t(s) * W * hL1, p, std::size_t(W) * hL1); p += std::size_t(W) * hL1;
            std::memcpy(g3_l2b[s], p, 4 * hL2); p += 4 * hL2;
            std::memcpy(g3_l2w[s], p, std::size_t(hL2) * hL2IN); p += std::size_t(hL2) * hL2IN;
            std::memcpy(&g3_fb[s], p, 4); p += 4;
            std::memcpy(g3_fw[s], p, hNO); p += hNO;
        }
    }
    for (int s = 0; s < NSTACKS && !v3; ++s) {
        std::memcpy(g_l1b[s], p, 64); p += 64;
        std::memcpy(g_l1w.get() + std::size_t(s) * W * L1, p, std::size_t(W) * L1); p += std::size_t(W) * L1;
        std::memcpy(g_l2b[s], p, 128); p += 128;
        std::memcpy(g_l2w[s], p, 1024); p += 1024;
        std::memcpy(&g_fb[s], p, 4); p += 4;
        std::memcpy(g_fw[s], p, 32); p += 32;
    }
    g_head = v3 ? (hL1 == 32 ? 2 : 0) + (hL2 == 64 ? 4 : 0) + (hSq2 ? 1 : 0) + 8 : 0;
    g_hL1 = v3 ? hL1 : L1; g_hL2 = v3 ? hL2 : L2; g_hSq2 = hSq2;
    g_fast = 0;
#ifdef JQ_PAIRWISE_NNZ
    if (!std::getenv("ABJ_NO_FASTHEAD") && (!v3 || (hL2 == 32 && !hSq2))) {
        for (int s = 0; s < NSTACKS; ++s) {
            if (!v3) nnk::fast_head_init(g_fh[s], L1, W, g_l1b[s], g_l1w.get() + std::size_t(s) * W * L1, g_l2b[s], g_l2w[s], g_fb[s], g_fw[s]);
            else nnk::fast_head_init(g_fh[s], hL1, W, g3_l1b[s], g_l1w.get() + std::size_t(s) * W * hL1, g3_l2b[s], g3_l2w[s], g3_fb[s], g3_fw[s]);
        }
        g_fast = v3 ? hL1 : L1;
    }
#endif
    g_W = W;
    g_scheme = scheme;
    g_nfeat = nfeat;
    g_urnBase = KB_OF_SCHEME[scheme] * PIECE_FEATS;
    set_scheme(scheme);
    g_gen.fetch_add(1);
    g_loaded = true;
    g_li = g_liPending;
    return true;
}

void unload() { g_loaded = false; }
bool loaded() { return g_loaded; }
int width() { return g_W; }
int kb_scheme() { return g_scheme; }
std::string head_desc() { return g_head ? "v3 L1 " + std::to_string(g_hL1) + " L2 " + std::to_string(g_hL2) + (g_hSq2 ? " sq2" : "") : "v1 L1 16 L2 32"; }

void set_threats(int, int, int, uint64_t, int) {}
bool threats_on() { return false; }
int threat_width() { return 0; }
bool tacc() { return false; }
ThrStats thr_stats() { return {0, 0, 0, 0, 0}; }

void refresh(const Position& pos, Acc& acc) {
    Meta& m = acc.m;
    int material = 0;
    for (int c = 0; c < 2; ++c) {
        for (int sl = 0; sl < NUM_SLOTS; ++sl) material += pos.urn[c][sl] * PIECE_VAL[sl];
        for (int i = 0; i < pos.pcount[c]; ++i) {
            const PieceType t = type_of(pos.board[pos.plist[c][i]]);
            if (t >= ROOK && t <= CANNON) material += PIECE_VAL[type_slot(t)];
        }
    }
    m.material = int16_t(material);
    m.noCapture = pos.noCapture;
    m.pieces = int8_t(pos.pcount[0] + pos.pcount[1]);
    m.dark = int8_t(pos.darkCount[0] + pos.darkCount[1]);
    for (int p = 0; p < 2; ++p) {
        king_state(Color(p), pos.kingSq[p], m.kb[p], m.mirror[p]);
        uint16_t f[64];
        const int n = features(pos, Color(p), m.kb[p], m.mirror[p], f);
        finny_refresh(p, m.kb[p], m.mirror[p], f, n, acc.v[p], acc.ps[p]);
    }
}

void refresh_nocache(const Position& pos, Acc& acc) {
    Meta& m = acc.m;
    refresh(pos, acc);  // meta (and a cached result we overwrite below)
    alignas(64) static const int32_t zero[NPSQT] = {};
    for (int p = 0; p < 2; ++p) {
        uint16_t f[64];
        const int n = features(pos, Color(p), m.kb[p], m.mirror[p], f);
        apply(acc.v[p], acc.ps[p], g_ftB, zero, f, n, nullptr, 0);
    }
}

void make_delta(const Position& pos, Move mv, Delta& d) {
    const int from = from_sq(mv), to = to_sq(mv);
    const Piece pc = pos.board[from], cap = pos.board[to];
    const Color us = color_of(pc);
    Meta& m = d.m;
    int matD = 0;
    Piece np = pc;
    int revSlot = -1;
    if (type_of(pc) == DARK) {
        np = make_piece(us, reveal_of(mv));
        revSlot = type_slot(reveal_of(mv));
        if (revSlot == NUM_SLOTS - 1) matD -= PIECE_VAL[revSlot];  // urn value leaves, a soldier is not counted
    }
    if (cap && type_of(cap) >= ROOK && type_of(cap) <= CANNON) matD -= PIECE_VAL[type_slot(type_of(cap))];
    m.material = int16_t(matD);  // a delta here; update() adds the parent's material
    m.noCapture = cap ? 0 : int16_t(pos.noCapture + 1);
    m.pieces = int8_t(pos.pcount[0] + pos.pcount[1] - (cap ? 1 : 0));
    m.dark = int8_t(pos.darkCount[0] + pos.darkCount[1] - (type_of(pc) == DARK) - (cap && type_of(cap) == DARK));
    d.valid = true;
    for (int p = 0; p < 2; ++p) {
        const Color persp = Color(p);
        king_state(persp, type_of(pc) == KING && us == persp ? to : pos.kingSq[p], m.kb[p], m.mirror[p]);
        int8_t kb0, mir0;
        king_state(persp, pos.kingSq[p], kb0, mir0);
        d.refP[p] = kb0 != m.kb[p] || mir0 != m.mirror[p];
        if (d.refP[p]) {  // only a king move: no capture of a face-down piece, no reveal
            uint16_t* f = d.full[p];
            int n = 0;
            for (int c = 0; c < 2; ++c)
                for (int i = 0; i < pos.pcount[c]; ++i) {
                    const int s = pos.plist[c][i];
                    if (s == from || s == to) continue;
                    f[n++] = uint16_t(piece_feat(persp, pos.board[s], s, m.kb[p], m.mirror[p]));
                }
            f[n++] = uint16_t(piece_feat(persp, np, to, m.kb[p], m.mirror[p]));
            for (int c = 0; c < 2; ++c)
                for (int sl = 0; sl < NUM_SLOTS; ++sl)
                    for (int j = 1; j <= pos.urn[c][sl]; ++j) f[n++] = uint16_t(urn_feat(persp, Color(c), sl, j));
            d.nFull[p] = int8_t(n);
            continue;
        }
        int nr = 0;
        d.rem[p][nr++] = uint16_t(piece_feat(persp, pc, from, m.kb[p], m.mirror[p]));
        if (cap) d.rem[p][nr++] = uint16_t(piece_feat(persp, cap, to, m.kb[p], m.mirror[p]));
        if (revSlot >= 0) d.rem[p][nr++] = uint16_t(urn_feat(persp, us, revSlot, pos.urn[us][revSlot]));
        d.add[p][0] = uint16_t(piece_feat(persp, np, to, m.kb[p], m.mirror[p]));
        d.nRem = int8_t(nr);
        d.nAdd = 1;
#if defined(JQ_PF_FT) && JQ_PF_FT > 0
        {   // prefetch the FT rows this child's update will read (JQ_PF_FT 1: first 2 lines, 2: whole row)
            const int lines = JQ_PF_FT >= 2 ? g_W * 2 / 64 : 2;
            auto pf = [&](int f) { const char* r = reinterpret_cast<const char*>(g_ftW.get() + std::size_t(f) * std::size_t(g_W));
                                   for (int l = 0; l < lines; ++l) _mm_prefetch(r + 64 * l, _MM_HINT_T1); };
            pf(d.add[p][0]);
            for (int i = 0; i < nr; ++i) pf(d.rem[p][i]);
        }
#endif
    }
}

#ifdef ABL_NOROWS
volatile uint32_t sinkx;
#endif
void update(const Acc& prev, Acc& next, const Delta& d) {
    if (!d.valid) {
        std::memcpy(next.v, prev.v, sizeof(int16_t) * std::size_t(g_W));
        std::memcpy(next.v[1], prev.v[1], sizeof(int16_t) * std::size_t(g_W));
        std::memcpy(next.ps, prev.ps, sizeof prev.ps);
        next.m = prev.m;
        ++next.m.noCapture;
        return;
    }
    for (int p = 0; p < 2; ++p) {
        if (d.refP[p]) {
            uint16_t f[64];
            std::memcpy(f, d.full[p], sizeof(uint16_t) * std::size_t(d.nFull[p]));
            finny_refresh(p, d.m.kb[p], d.m.mirror[p], f, d.nFull[p], next.v[p], next.ps[p]);
        } else
            apply(next.v[p], next.ps[p], prev.v[p], prev.ps[p], d.add[p], d.nAdd, d.rem[p], d.nRem);
    }
    next.m = d.m;
    next.m.material = int16_t(prev.m.material + d.m.material);
}

int runtime_value(const Acc& acc, Color stm, int* psqtRaw, int* positionalRaw) {
    const Meta& m = acc.m;
    const int b = std::min((m.pieces - 1) / 4, 7) + (m.dark > 0 ? 8 : 0);
    alignas(64) uint8_t x[MAX_W];
#ifdef JQ_PAIRWISE_NNZ
    alignas(32) uint16_t nnz[MAX_W / 4 + 16];
    const int nnzN = nnk::pairwise_nnz(acc.v[stm], acc.v[~stm], g_W, x, nnz);
#else
    const uint16_t* nnz = nullptr;
    const int nnzN = 0;
    nnk::pairwise(acc.v[stm], acc.v[~stm], g_W, x);
#endif
    const int psqt = (acc.ps[stm][b] - acc.ps[~stm][b]) / 2;
    int positional;
#ifdef JQ_PAIRWISE_NNZ
    if (g_fast == 32) positional = nnk::propagate_fast<32>(g_fh[b], g_l1w.get() + std::size_t(b) * std::size_t(g_W) * 32, x, nnz, nnzN);
    else if (g_fast == 16) positional = nnk::propagate_fast<16>(g_fh[b], g_l1w.get() + std::size_t(b) * std::size_t(g_W) * 16, x, nnz, nnzN);
    else
#endif
    if (!g_head)
        positional = nnk::propagate(g_W, g_l1b[b], g_l1w.get() + std::size_t(b) * std::size_t(g_W) * L1, g_l2b[b],
                                    g_l2w[b], g_fb[b], g_fw[b], x, nnz, nnzN);
    else {
        const uint8_t* w1 = g_l1w.get() + std::size_t(b) * std::size_t(g_W) * std::size_t(g_hL1);
#define JQ_H3(A, B, Q) nnk::propagate_g<A, B, Q>(g_W, g3_l1b[b], w1, g3_l2b[b], g3_l2w[b], g3_fb[b], g3_fw[b], x, nnz, nnzN)
        switch (g_head & 7) {
            case 0: positional = JQ_H3(16, 32, false); break;
            case 1: positional = JQ_H3(16, 32, true); break;
            case 2: positional = JQ_H3(32, 32, false); break;
            case 3: positional = JQ_H3(32, 32, true); break;
            case 4: positional = JQ_H3(16, 64, false); break;
            case 5: positional = JQ_H3(16, 64, true); break;
            case 6: positional = JQ_H3(32, 64, false); break;
            default: positional = JQ_H3(32, 64, true); break;
        }
#undef JQ_H3
    }
    if (psqtRaw) *psqtRaw = psqt;
    if (positionalRaw) *positionalRaw = positional;
    const int nn = (psqt + positional) / 16;
    const int materialScale = 935 + m.material * 93 / 5116;
    int value = (nn * materialScale) / 1024;
    value = value * (218 - std::min<int>(m.noCapture, 120)) / 176;
    return std::clamp(value, -29507, 29507);
}

Value evaluate(const Acc& acc, Color stm) {
    const int64_t v = int64_t(runtime_value(acc, stm)) * g_scale;
    const int64_t s = v >= 0 ? (v + 500) / 1000 : -((-v + 500) / 1000);
    return Value(std::clamp<int64_t>(s, -(VALUE_KNOWN_WIN - 1), VALUE_KNOWN_WIN - 1));
}

}  // namespace jq::hnn
