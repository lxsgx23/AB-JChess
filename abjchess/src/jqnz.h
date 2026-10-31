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
// jqnz.h - self-contained lossless compression for JQHNNUE1 (.hnn) nets (no external libraries).
//
// Two container formats, both decode to the *exact* original .hnn bytes (size + CRC32 checked):
//   "JQHNNL01"  Stockfish/Pikafish-style signed LEB128 of the FT biases, FT weights and PSQT
//               (each block "COMPRESSED_LEB128" + u32 byte count + LEB128 stream), rest stored raw.
//   "JQHNNZ01"  context-modelled adaptive binary range coding (LZMA-style coder, two-rate counters) of every
//               section: FT weights are predicted from the same channel of the other king bucket and of the
//               neighbouring squares of the same piece plane (per-plane integer coefficients fitted by the
//               packer), residuals coded with contexts from the channel's running scale and local activity.
//
// Decoding: jqnz::decode(data, size, out, &err). Encoding (packer only): jqnz::encode_leb / jqnz::encode_z.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#if defined(__GNUC__)
#define JQNZ_NOINLINE __attribute__((noinline))
#else
#define JQNZ_NOINLINE
#endif

namespace jqnz {

inline uint32_t crc32(const uint8_t* p, size_t n) {  // standard CRC-32 (zlib.crc32)
    struct Table {
        uint32_t t[256];
        Table() {
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                t[i] = c;
            }
        }
    };
    static const Table tab;  // thread-safe one-time init
    const uint32_t* T = tab.t;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = T[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

// ------------------------------------------------------------------------------------------------------------------
// .hnn layout (mirrors hnnue.cpp)
// ------------------------------------------------------------------------------------------------------------------
struct Layout {
    int W = 0, nfeat = 0, scheme = 0, kb = 0;
    bool v3 = false;
    size_t hdr = 0;              // 40 (v1) or 56 (v3)
    size_t ftB = 0, ftW = 0, psqt = 0, stacks = 0, stacksEnd = 0;  // byte offsets
    size_t stackSz = 0;
    int hL1 = 16, hL2 = 32, hNO = 32, hL2IN = 32;
};
constexpr int NSQ = 90, PIECE_FEATS = 16 * NSQ, URN_FEATS = 30, NPSQT = 16, NSTACKS = 16;

JQNZ_NOINLINE inline bool layout(const uint8_t* b, size_t n, Layout& L, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    if (n < 40 || std::memcmp(b, "JQHNNUE1", 8) != 0) return fail("jqnz: not a JQHNNUE1 net");
    uint32_t h[8];
    std::memcpy(h, b + 8, sizeof h);
    static const int KB[4] = {1, 3, 2, 6};
    if ((h[0] != 1 && h[0] != 3) || h[3] > 3 || h[4] != NPSQT || h[5] != NSTACKS || h[1] < 32 || h[1] > 4096)
        return fail("jqnz: unsupported header");
    L.v3 = h[0] == 3;
    L.W = int(h[1]);
    L.scheme = int(h[3]);
    L.kb = KB[L.scheme];
    L.nfeat = int(h[2]);
    if (L.nfeat != L.kb * PIECE_FEATS + URN_FEATS) return fail("jqnz: feature count mismatch");
    L.hdr = L.v3 ? 56 : 40;
    if (n < L.hdr) return fail("jqnz: short header");
    if (L.v3) {
        uint32_t ext[4];
        std::memcpy(ext, b + 40, sizeof ext);
        L.hL1 = int(h[6]);
        L.hL2 = int(h[7]);
        L.hNO = L.hL2 * ((ext[0] & 1) ? 2 : 1);
        L.hL2IN = (2 * (L.hL1 - 1) + 31) / 32 * 32;
        L.stackSz = 4 * L.hL1 + size_t(L.W) * L.hL1 + 4 * L.hL2 + size_t(L.hL2) * L.hL2IN + 4 + L.hNO;
    } else {
        L.stackSz = 64 + size_t(L.W) * 16 + 128 + 1024 + 4 + 32;
    }
    L.ftB = L.hdr;
    L.ftW = L.ftB + 2 * size_t(L.W);
    L.psqt = L.ftW + 2 * size_t(L.nfeat) * L.W;
    L.stacks = L.psqt + 4 * size_t(L.nfeat) * NPSQT;
    L.stacksEnd = L.stacks + NSTACKS * L.stackSz;
    if (n < L.stacksEnd) return fail("jqnz: file too short");
    return true;
}

// ------------------------------------------------------------------------------------------------------------------
// LEB128 (Stockfish nnue_common.h read_leb_128 / write_leb_128)
// ------------------------------------------------------------------------------------------------------------------
static const char LEB_MAGIC[] = "COMPRESSED_LEB128";
constexpr size_t LEB_MAGIC_LEN = sizeof(LEB_MAGIC) - 1;

template <typename T>
inline void leb_write(std::vector<uint8_t>& out, const uint8_t* raw, size_t count) {
    out.insert(out.end(), LEB_MAGIC, LEB_MAGIC + LEB_MAGIC_LEN);
    const size_t cntPos = out.size();
    out.resize(out.size() + 4);
    const size_t start = out.size();
    for (size_t i = 0; i < count; ++i) {
        T tv;
        std::memcpy(&tv, raw + i * sizeof(T), sizeof(T));
        int64_t v = tv;
        while (true) {
            uint8_t byte = uint8_t(v & 0x7f);
            v >>= 7;
            if ((byte & 0x40) == 0 ? v == 0 : v == -1) { out.push_back(byte); break; }
            out.push_back(byte | 0x80);
        }
    }
    const uint32_t nb = uint32_t(out.size() - start);
    std::memcpy(out.data() + cntPos, &nb, 4);
}

template <typename T>
JQNZ_NOINLINE inline bool leb_read(const uint8_t*& p, const uint8_t* end, uint8_t* raw, size_t count) {
    if (size_t(end - p) < LEB_MAGIC_LEN + 4 || std::memcmp(p, LEB_MAGIC, LEB_MAGIC_LEN) != 0) return false;
    p += LEB_MAGIC_LEN;
    uint32_t nb;
    std::memcpy(&nb, p, 4);
    p += 4;
    if (size_t(end - p) < nb) return false;
    const uint8_t* q = p;
    const uint8_t* qe = p + nb;
    for (size_t i = 0; i < count; ++i) {
        int64_t result = 0;
        int shift = 0;
        uint8_t byte;
        do {
            if (q >= qe || shift > 63) return false;
            byte = *q++;
            result |= int64_t(byte & 0x7f) << shift;
            shift += 7;
        } while (byte & 0x80);
        if (shift < 64 && (byte & 0x40)) result |= -(int64_t(1) << shift);
        const T tv = T(result);
        std::memcpy(raw + i * sizeof(T), &tv, sizeof(T));
    }
    if (q != qe) return false;
    p = qe;
    return true;
}

// ------------------------------------------------------------------------------------------------------------------
// Binary range coder (LZMA style) with two-rate adaptive probabilities
// ------------------------------------------------------------------------------------------------------------------
#ifndef JQNZ_FAST
#define JQNZ_FAST 5
#endif
#ifndef JQNZ_SLOW
#define JQNZ_SLOW 8
#endif
struct Prob {  // P(bit == 0) in 1/65536: fast + slow estimate, the slow one starts with count-based (1/n) rates
    uint16_t a, b;  // set by IntCtx() (a loop, not per-member initialisers: keeps the decoder small)
    uint8_t n;
    void reset() { a = b = 1 << 15; n = 0; }
    uint32_t p() const { uint32_t q = (uint32_t(a) + b) >> 6; return q < 31 ? 31 : q > 2017 ? 2017 : q; }  // 11-bit
    void upd(int bit) {
        const int sb = n < JQNZ_SLOW ? n + 1 : JQNZ_SLOW;
        const int sa = sb < JQNZ_FAST ? sb : JQNZ_FAST;
        if (n < 255) ++n;
        if (bit) { a -= a >> sa; b -= b >> sb; }
        else { a += (65536 - a) >> sa; b += (65536 - b) >> sb; }
    }
};

struct Enc {
    std::vector<uint8_t>& out;
    uint64_t low = 0;
    uint32_t range = 0xFFFFFFFFu;
    uint8_t cache = 0;
    uint64_t cacheSize = 1;
    explicit Enc(std::vector<uint8_t>& o) : out(o) {}
    void shift_low() {
        if (uint32_t(low) < 0xFF000000u || (low >> 32) != 0) {
            uint8_t carry = uint8_t(low >> 32);
            uint8_t temp = cache;
            do {
                out.push_back(uint8_t(temp + carry));
                temp = 0xFF;
            } while (--cacheSize != 0);
            cache = uint8_t(uint32_t(low) >> 24);
        }
        ++cacheSize;
        low = uint32_t(low) << 8;
    }
    void bit(Prob& pr, int b) {
        const uint32_t bound = (range >> 11) * pr.p();
        if (!b) range = bound;
        else { low += bound; range -= bound; }
        pr.upd(b);
        while (range < (1u << 24)) { range <<= 8; shift_low(); }
    }
    void flush() { for (int i = 0; i < 5; ++i) shift_low(); }
};

struct Dec {
    const uint8_t* p;
    const uint8_t* end;
    uint32_t range = 0xFFFFFFFFu, code = 0;
    Dec(const uint8_t* b, const uint8_t* e) : p(b), end(e) {
        for (int i = 0; i < 5; ++i) code = (code << 8) | next();
    }
    uint8_t next() { return p < end ? *p++ : 0; }
    int bit(Prob& pr) {
        const uint32_t bound = (range >> 11) * pr.p();
        int b;
        if (code < bound) { range = bound; b = 0; }
        else { code -= bound; range -= bound; b = 1; }
        pr.upd(b);
        while (range < (1u << 24)) { range <<= 8; code = (code << 8) | next(); }
        return b;
    }
};

// Integer model: zero flag, sign, Elias-gamma length (unary), mantissa bits (all adaptive per context).
struct IntCtx {
    Prob zero, sign[3], len[33], mant[33][3], low[33][32];
    IntCtx() { Prob* q = &zero; for (size_t i = 0; i < sizeof(IntCtx) / sizeof(Prob); ++i) q[i].reset(); }
};
static_assert(sizeof(IntCtx) == 1192 * sizeof(Prob), "IntCtx must be a plain array of Prob");

template <class Coder>
struct IntCoder;

template <>
struct IntCoder<Enc> {
    JQNZ_NOINLINE static void code(Enc& e, IntCtx& c, int64_t v, int signCtx) {
        e.bit(c.zero, v != 0);
        if (!v) return;
        e.bit(c.sign[signCtx], v < 0);
        uint64_t a = uint64_t(v < 0 ? -v : v);  // >= 1
        int n = 63 - __builtin_clzll(a);         // a in [2^n, 2^(n+1))
        for (int i = 0; i < n; ++i) e.bit(c.len[i], 1);
        if (n < 32) e.bit(c.len[n], 0);
        // mantissa: top bit tree of depth 2 (adaptive), remaining bits adaptive per (n, position)
        int node = 1;
        for (int k = n - 1; k >= 0; --k) {
            const int bit = int((a >> k) & 1);
            if (n - 1 - k < 2) { e.bit(c.mant[n][node - 1], bit); node = node * 2 + bit; if (node > 3) node = 3; }
            else e.bit(c.low[n][k], bit);
        }
    }
};

template <>
struct IntCoder<Dec> {
    JQNZ_NOINLINE static int64_t code(Dec& d, IntCtx& c, int signCtx) {
        if (!d.bit(c.zero)) return 0;
        const int neg = d.bit(c.sign[signCtx]);
        int n = 0;
        while (n < 32 && d.bit(c.len[n])) ++n;
        uint64_t a = 1;
        int node = 1;
        for (int k = n - 1; k >= 0; --k) {
            int bit;
            if (n - 1 - k < 2) { bit = d.bit(c.mant[n][node - 1]); node = node * 2 + bit; if (node > 3) node = 3; }
            else bit = d.bit(c.low[n][k]);
            a = (a << 1) | uint64_t(bit);
        }
        return neg ? -int64_t(a) : int64_t(a);
    }
};

inline int qlog(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }  // 0..32

// ------------------------------------------------------------------------------------------------------------------
// The JQHNNZ01 model. Encoder and decoder run the same code (template on the coder); `Io` either reads the original
// values (encoder) or writes them (decoder).
// ------------------------------------------------------------------------------------------------------------------
constexpr int SCALE_Q = 14, ACT_Q = 12;     // context quantisation levels
constexpr int PRED_SHIFT = 6;               // predictor coefficients in 1/64

constexpr int NTERMS = 7;  // predictor terms: other bucket, left, down, down-left, down-right, down 2, left 2
struct Coefs {
    int8_t c[NTERMS];
};
// Neighbour rows of feature f used by the predictor (-1 = not available). Piece features are laid out as
// [bucket][16 piece kinds][rank 0..9][file 0..8]; urn features only use the previous urn row as "left".
// Rows whose PSQT row is all zero are features the trainer never saw (unreachable king / dark-piece squares):
// their FT weights are untrained init noise, so they get their own group and are not used to predict live rows.
inline void neighbours_geo(const Layout& L, int f, int nb[NTERMS], int& group);
inline int ngroups(const Layout& L) { return (L.kb + 1) * 16 + 1; }
inline void neighbours(const Layout& L, int f, const uint8_t* dead, int nb[NTERMS], int& group) {
    neighbours_geo(L, f, nb, group);
    if (dead[f]) {
        group = ngroups(L) - 1;
        for (int t = 1; t < NTERMS; ++t) nb[t] = -1;
        if (nb[0] >= 0 && !dead[nb[0]]) nb[0] = -1;
    } else {
        for (int t = 0; t < NTERMS; ++t) if (nb[t] >= 0 && dead[nb[t]]) nb[t] = -1;
    }
}
inline void neighbours_geo(const Layout& L, int f, int nb[NTERMS], int& group) {
    for (int t = 0; t < NTERMS; ++t) nb[t] = -1;
    if (f < L.kb * PIECE_FEATS) {
        const int kb = f / PIECE_FEATS, pc = (f % PIECE_FEATS) / NSQ, sq = f % NSQ, r = sq / 9, fi = sq % 9;
        group = kb * 16 + pc;
        if (kb > 0) nb[0] = f - PIECE_FEATS;
        if (fi > 0) nb[1] = f - 1;
        if (r > 0) nb[2] = f - 9;
        if (r > 0 && fi > 0) nb[3] = f - 10;
        if (r > 0 && fi < 8) nb[4] = f - 8;
        if (r > 1) nb[5] = f - 18;
        if (fi > 1) nb[6] = f - 2;
    } else {
        group = L.kb * 16;
        if (f > L.kb * PIECE_FEATS) nb[1] = f - 1;
    }
}

struct Model {
    std::vector<IntCtx> ftw;   // [2][SCALE_Q * ACT_Q] (live, dead rows)
    IntCtx ftb, psqtB0[8], psqtD[8], i32, i8[6][8], l1c[2][16];
    JQNZ_NOINLINE Model() : ftw(2 * SCALE_Q * ACT_Q) {}
};

inline std::vector<uint8_t> dead_rows(const Layout& L, const int32_t* p) {
    std::vector<uint8_t> d(size_t(L.nfeat), 0);
    for (int f = 0; f < L.nfeat; ++f) {
        bool z = true;
        for (int k = 0; k < NPSQT; ++k) z = z && p[size_t(f) * NPSQT + k] == 0;
        d[size_t(f)] = z;
    }
    return d;
}

template <class Coder>
struct ZCodec {
    Coder& cd;
    Model m;
    JQNZ_NOINLINE explicit ZCodec(Coder& c) : cd(c) {}

    int64_t val(IntCtx& c, int64_t v, int sc) {
        if constexpr (std::is_same_v<Coder, Enc>) { IntCoder<Enc>::code(cd, c, v, sc); return v; }
        else { (void)v; return IntCoder<Dec>::code(cd, c, sc); }
    }

    static int sctx(int64_t pred) { return pred > 0 ? 1 : pred < 0 ? 2 : 0; }

    // FT weights: int16 [nfeat][W]
    JQNZ_NOINLINE void ft_weights(const Layout& L, int16_t* w, const std::vector<Coefs>& cf, const uint8_t* dead) {
        const int W = L.W;
        std::vector<uint32_t> scale(size_t(W) * 18, 16u << 4);   // running mean |residual| per channel, x16
        std::vector<int32_t> res(size_t(L.nfeat) * W, 0);   // residuals (for activity contexts)
        for (int f = 0; f < L.nfeat; ++f) {
            int nb[NTERMS], g;
            neighbours(L, f, dead, nb, g);
            IntCtx* cbase = &m.ftw[dead[f] ? size_t(SCALE_Q) * ACT_Q : 0];
            const Coefs& c = cf[size_t(g)];
            uint32_t* scl = &scale[size_t(dead[f] ? 17 : (g & 15)) * W];
            for (int j = 0; j < W; ++j) {
                const size_t i = size_t(f) * W + j;
                int64_t acc = 1 << (PRED_SHIFT - 1);
                uint32_t act = 0;
                int nact = 0;
                for (int t = 0; t < NTERMS; ++t) {
                    if (nb[t] < 0) continue;
                    acc += int64_t(c.c[t]) * w[size_t(nb[t]) * W + j];
                    if (t < 3) { act += uint32_t(std::abs(res[size_t(nb[t]) * W + j])); ++nact; }
                }
                const int64_t pred = acc >> PRED_SHIFT;
                const int aq = nact ? std::min(ACT_Q - 1, 1 + qlog(act * 2 / uint32_t(nact))) : 0;
                const int sq_ = std::min(SCALE_Q - 1, qlog(scl[j] >> 2));
                IntCtx& ctx = cbase[size_t(sq_) * ACT_Q + aq];
                const int64_t rv = val(ctx, int64_t(w[i]) - pred, sctx(pred));
                w[i] = int16_t(rv + pred);
                res[i] = int32_t(rv);
                const uint32_t a = uint32_t(rv < 0 ? -rv : rv) << 4;
                scl[j] = scl[j] + (a >> 3) - (scl[j] >> 3);
            }
        }
    }

    // PSQT: int32 [nfeat][16]; bucket b>0 rows predicted from bucket b-1, bucket 0 from the previous column
    JQNZ_NOINLINE void psqt(const Layout& L, int32_t* p) {
        for (int f = 0; f < L.nfeat; ++f) {
            const bool piece = f < L.kb * PIECE_FEATS;
            const int kb = piece ? f / PIECE_FEATS : 0;
            for (int k = 0; k < NPSQT; ++k) {
                const size_t i = size_t(f) * NPSQT + k;
                if (piece && kb > 0) {
                    const int64_t pred = p[i - size_t(PIECE_FEATS) * NPSQT];
                    const int cq = std::min(7, k / 2);
                    p[i] = int32_t(val(m.psqtD[cq], int64_t(p[i]) - pred, 0) + pred);
                } else {
                    const int64_t pred = k > 0 ? p[i - 1] : 0;
                    const int cq = k == 0 ? 0 : std::min(7, 1 + k / 3);
                    p[i] = int32_t(val(m.psqtB0[cq], int64_t(p[i]) - pred, sctx(pred)) + pred);
                }
            }
        }
    }

    JQNZ_NOINLINE void i16s(int16_t* v, size_t n, IntCtx& c) { for (size_t i = 0; i < n; ++i) v[i] = int16_t(val(c, v[i], 0)); }
    JQNZ_NOINLINE void i32s(int32_t* v, size_t n, IntCtx& c) { for (size_t i = 0; i < n; ++i) v[i] = int32_t(val(c, v[i], 0)); }
    JQNZ_NOINLINE void i8s(int8_t* v, size_t n, IntCtx* c8, int stride) {  // context: |previous value| (same row) quantised
        int prev = 0;
        for (size_t i = 0; i < n; ++i) {
            if (stride && i % size_t(stride) == 0) prev = 0;
            const int q = std::min(7, qlog(uint32_t(std::abs(prev))));
            v[i] = int8_t(val(c8[q], v[i], 0));
            prev = v[i];
        }
    }

    // int8 matrix [rows][cols]; context = running mean |v| of the column (shared across stacks)
    JQNZ_NOINLINE void i8cols(int8_t* v, int rows, int cols, std::vector<uint32_t>& sc, IntCtx* c16) {
        for (int r = 0; r < rows; ++r) {
            for (int j = 0; j < cols; ++j) {
                const int a = std::min(15, int(sc[size_t(j)] >> 5));
                int8_t& x = v[size_t(r) * cols + j];
                x = int8_t(val(c16[a], x, 0));
                const uint32_t m = uint32_t(std::abs(int(x))) << 6;
                sc[size_t(j)] = sc[size_t(j)] + (m >> 4) - (sc[size_t(j)] >> 4);
            }
        }
    }

    JQNZ_NOINLINE void stacks(const Layout& L, uint8_t* s) {
        const int W = L.W, h1 = L.v3 ? L.hL1 : 16, h2 = L.v3 ? L.hL2 : 32, h2in = L.v3 ? L.hL2IN : 32;
        std::vector<uint32_t> sc1(size_t(W), 4u << 6), sc2(size_t(h2in), 4u << 6);
        for (int st = 0; st < NSTACKS; ++st) {
            uint8_t* q = s + size_t(st) * L.stackSz;
            auto I32 = [&](size_t n) { std::vector<int32_t> t(n); std::memcpy(t.data(), q, 4 * n); i32s(t.data(), n, m.i32); std::memcpy(q, t.data(), 4 * n); q += 4 * n; };
            auto I8M = [&](int r, int c, std::vector<uint32_t>& sc, IntCtx* cx) { i8cols(reinterpret_cast<int8_t*>(q), r, c, sc, cx); q += size_t(r) * c; };
            auto I8 = [&](size_t n, int sec, int stride) { i8s(reinterpret_cast<int8_t*>(q), n, m.i8[sec], stride); q += n; };
            if (L.v3) {
                I32(size_t(L.hL1)); I8M(h1, W, sc1, m.l1c[0]);
                I32(size_t(L.hL2)); I8M(h2, h2in, sc2, m.l1c[1]);
                I32(1); I8(size_t(L.hNO), 2, 0);
            } else {
                I32(16); I8M(h1, W, sc1, m.l1c[0]);
                I32(32); I8M(h2, h2in, sc2, m.l1c[1]);
                I32(1); I8(32, 2, 0);
            }
        }
    }

    JQNZ_NOINLINE void bytes(uint8_t* b, size_t n) {  // trailer / unknown data: order-0 int8 model
        i8s(reinterpret_cast<int8_t*>(b), n, m.i8[3], 0);
    }
};

// Fit the per-plane predictor coefficients (least squares, rounded to int8 in 1/64).
inline std::vector<Coefs> fit_coefs(const Layout& L, const int16_t* w, const uint8_t* dead) {
    const int G = ngroups(L), W = L.W, K = NTERMS;
    std::vector<double> A(size_t(G) * K * K, 0.0), B(size_t(G) * K, 0.0);
    for (int f = 0; f < L.nfeat; ++f) {
        int nb[NTERMS], g;
        neighbours(L, f, dead, nb, g);
        double* a = &A[size_t(g) * K * K];
        double* bb = &B[size_t(g) * K];
        for (int j = 0; j < W; ++j) {
            double x[NTERMS];
            for (int t = 0; t < K; ++t) x[t] = nb[t] >= 0 ? double(w[size_t(nb[t]) * W + j]) : 0.0;
            const double y = w[size_t(f) * W + j];
            for (int t = 0; t < K; ++t) { bb[t] += x[t] * y; for (int u = 0; u < K; ++u) a[t * K + u] += x[t] * x[u]; }
        }
    }
    std::vector<Coefs> cf(static_cast<size_t>(G));
    for (int g = 0; g < G; ++g) {
        double M[NTERMS][NTERMS + 1];
        for (int t = 0; t < K; ++t) {
            for (int u = 0; u < K; ++u) M[t][u] = A[size_t(g) * K * K + t * K + u];
            M[t][t] += 1e-3 + 1e-6 * M[t][t];
            M[t][K] = B[size_t(g) * K + t];
        }
        for (int t = 0; t < K; ++t) {
            int piv = t;
            for (int u = t + 1; u < K; ++u) if (std::fabs(M[u][t]) > std::fabs(M[piv][t])) piv = u;
            for (int k = 0; k <= K; ++k) std::swap(M[t][k], M[piv][k]);
            if (std::fabs(M[t][t]) < 1e-9) continue;
            for (int u = 0; u < K; ++u) {
                if (u == t) continue;
                const double q = M[u][t] / M[t][t];
                for (int k = 0; k <= K; ++k) M[u][k] -= q * M[t][k];
            }
        }
        for (int t = 0; t < K; ++t) {
            const double x = std::fabs(M[t][t]) < 1e-9 ? 0.0 : M[t][K] / M[t][t];
            const long q = std::lround(x * (1 << PRED_SHIFT));
            cf[size_t(g)].c[t] = int8_t(q < -127 ? -127 : q > 127 ? 127 : q);
        }
    }
    return cf;
}

// ------------------------------------------------------------------------------------------------------------------
// Containers
// ------------------------------------------------------------------------------------------------------------------
inline void put32(std::vector<uint8_t>& o, uint32_t v) { for (int i = 0; i < 4; ++i) o.push_back(uint8_t(v >> (8 * i))); }
inline uint32_t get32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }

inline bool encode_leb(const std::vector<uint8_t>& raw, std::vector<uint8_t>& out, std::string* err) {
    Layout L;
    if (!layout(raw.data(), raw.size(), L, err)) return false;
    out.clear();
    out.insert(out.end(), {'J', 'Q', 'H', 'N', 'N', 'L', '0', '1'});
    put32(out, uint32_t(raw.size()));
    put32(out, crc32(raw.data(), raw.size()));
    out.insert(out.end(), raw.begin(), raw.begin() + long(L.hdr));
    leb_write<int16_t>(out, raw.data() + L.ftB, size_t(L.W));
    leb_write<int16_t>(out, raw.data() + L.ftW, size_t(L.nfeat) * L.W);
    leb_write<int32_t>(out, raw.data() + L.psqt, size_t(L.nfeat) * NPSQT);
    out.insert(out.end(), raw.begin() + long(L.stacks), raw.end());
    return true;
}

inline bool encode_z(const std::vector<uint8_t>& raw, std::vector<uint8_t>& out, std::string* err,
                     std::vector<size_t>* sections = nullptr) {
    Layout L;
    if (!layout(raw.data(), raw.size(), L, err)) return false;
    std::vector<uint8_t> tmp(raw);  // the coder works in place
    std::vector<uint8_t> dead;
    {
        std::vector<int32_t> pp(size_t(L.nfeat) * NPSQT);
        std::memcpy(pp.data(), raw.data() + L.psqt, 4 * pp.size());
        dead = dead_rows(L, pp.data());
    }
    std::vector<int16_t> w0(size_t(L.nfeat) * L.W);
    std::memcpy(w0.data(), raw.data() + L.ftW, 2 * w0.size());
    std::vector<Coefs> cf = fit_coefs(L, w0.data(), dead.data());
    out.clear();
    out.insert(out.end(), {'J', 'Q', 'H', 'N', 'N', 'Z', '0', '1'});
    put32(out, uint32_t(raw.size()));
    put32(out, crc32(raw.data(), raw.size()));
    out.insert(out.end(), raw.begin(), raw.begin() + long(L.hdr));
    out.push_back(uint8_t(cf.size()));
    for (const Coefs& c : cf) for (int t = 0; t < NTERMS; ++t) out.push_back(uint8_t(c.c[t]));
    Enc e(out);
    ZCodec<Enc> z(e);
    std::vector<int16_t> b16(size_t(L.W)), w16(size_t(L.nfeat) * L.W);
    std::vector<int32_t> p32(size_t(L.nfeat) * NPSQT);
    std::memcpy(b16.data(), tmp.data() + L.ftB, 2 * b16.size());
    std::memcpy(w16.data(), tmp.data() + L.ftW, 2 * w16.size());
    std::memcpy(p32.data(), tmp.data() + L.psqt, 4 * p32.size());
    auto mark = [&] { if (sections) sections->push_back(out.size()); };
    mark();
    z.i16s(b16.data(), b16.size(), z.m.ftb);
    mark();
    z.psqt(L, p32.data());
    mark();
    z.ft_weights(L, w16.data(), cf, dead.data());
    mark();
    z.stacks(L, tmp.data() + L.stacks);
    mark();
    z.bytes(tmp.data() + L.stacksEnd, raw.size() - L.stacksEnd);
    e.flush();
    mark();
    return true;
}

inline bool is_compressed(const uint8_t* p, size_t n) {
    return n >= 8 && (std::memcmp(p, "JQHNNL01", 8) == 0 || std::memcmp(p, "JQHNNZ01", 8) == 0);
}

// Decode a JQHNNL01 / JQHNNZ01 container back to the original .hnn bytes.
inline bool decode(const uint8_t* in, size_t n, std::vector<uint8_t>& raw, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    if (n < 16 + 40 || !is_compressed(in, n)) return fail("jqnz: not a compressed net");
    const bool z = in[5] == 'Z';
    const uint32_t size = get32(in + 8), crc = get32(in + 12);
    if (size < 40 || size > (1u << 30)) return fail("jqnz: bad size");
    raw.assign(size, 0);
    const uint8_t* p = in + 16;
    const uint8_t* end = in + n;
    // header: v1 = 40 bytes, v3 = 56 bytes (version field at offset 8)
    const size_t hdr = get32(p + 8) == 3 ? 56 : 40;
    if (size_t(end - p) < hdr) return fail("jqnz: truncated");
    std::memcpy(raw.data(), p, hdr);
    p += hdr;
    Layout L;
    if (!layout(raw.data(), raw.size(), L, err)) return false;
    if (!z) {
        if (!leb_read<int16_t>(p, end, raw.data() + L.ftB, size_t(L.W)) ||
            !leb_read<int16_t>(p, end, raw.data() + L.ftW, size_t(L.nfeat) * L.W) ||
            !leb_read<int32_t>(p, end, raw.data() + L.psqt, size_t(L.nfeat) * NPSQT))
            return fail("jqnz: bad LEB128 block");
        if (size_t(end - p) != size - L.stacks) return fail("jqnz: bad tail size");
        std::memcpy(raw.data() + L.stacks, p, size - L.stacks);
    } else {
        if (end - p < 1) return fail("jqnz: truncated");
        const size_t ncf = *p++;
        if (ncf != size_t(ngroups(L)) || size_t(end - p) < NTERMS * ncf) return fail("jqnz: bad coefficient table");
        std::vector<Coefs> cf(ncf);
        for (size_t i = 0; i < ncf; ++i) for (int t = 0; t < NTERMS; ++t) cf[i].c[t] = int8_t(*p++);
        Dec d(p, end);
        ZCodec<Dec> zc(d);
        std::vector<int16_t> b16(size_t(L.W)), w16(size_t(L.nfeat) * L.W);
        std::vector<int32_t> p32(size_t(L.nfeat) * NPSQT);
        zc.i16s(b16.data(), b16.size(), zc.m.ftb);
        zc.psqt(L, p32.data());
        const std::vector<uint8_t> dead = dead_rows(L, p32.data());
        zc.ft_weights(L, w16.data(), cf, dead.data());
        std::memcpy(raw.data() + L.ftB, b16.data(), 2 * b16.size());
        std::memcpy(raw.data() + L.ftW, w16.data(), 2 * w16.size());
        std::memcpy(raw.data() + L.psqt, p32.data(), 4 * p32.size());
        zc.stacks(L, raw.data() + L.stacks);
        zc.bytes(raw.data() + L.stacksEnd, size - L.stacksEnd);
    }
    if (crc32(raw.data(), raw.size()) != crc) return fail("jqnz: CRC mismatch after decoding");
    return true;
}

}  // namespace jqnz
