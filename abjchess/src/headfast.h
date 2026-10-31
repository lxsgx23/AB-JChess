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
// Fast AVX2 layer stack for the two shipped head shapes: W -> L1 (16 or 32; L1-1 + skip) ->
// [sqr, crelu] -> 32 -> crelu -> 1 + skip, no sq2. Integer results identical to nnk::propagate (L1 16) and
// nnk::propagate_g<32, 32, false>:
//  * L1/L2 int8 dot products: maddubs pair sums are exact (|x| <= 127, |w| <= 128); when the load-time bound
//    G * max_pair(127 * (|w_a| + |w_b|)) <= 32767 holds for a stack, G = 2 or 4 chunks are summed in int16
//    before one madd (no saturation, no wrap -> same int32 sums); otherwise G = 1.
//  * activations vectorised: packs_epi32 = clamp to int16, sqr via mulhi_epu16(|c|, |c|) >> 3 = c^2 >> 19,
//    crelu = clamp(f >> 6, 0, 127); the L2 input order is [sqr(f0..f(L1-1)), crelu(f0..f(L1-1))] with zero
//    weights on the two skip-neuron slots (load-time re-layout of the L2 weights; int32 sums order-free).
//  * L2 output crelu packed to int16 and dotted with int16 fw by madd (fw re-laid out at load).
#pragma once
#include "nnkern.h"

namespace jq::nnk {

struct alignas(64) FastHead {
    int32_t l1b[32];
    int32_t l2b[32];
    int8_t l2w[16 * 128];  // [L2 input chunk (2 * L1 / 4)][32 outputs][4]
    int16_t fw[32];        // packs order: [o0-3, o8-11, o4-7, o12-15], [o16-19, o24-27, o20-23, o28-31]
    int32_t fb;
    int8_t fwn[32];        // fw in the original (natural) order, for the 128-bit kernels
    int g1, g2;            // int16 grouping of L1 / L2 chunks (1, 2 or 4) valid for every chunk combination
    uint8_t l2plan[4];     // per group of 4 consecutive L2 input chunks: 4 = one int16 sum of 4, 2 = two pair sums,
                           // bit0/bit1 set = pair 0 / pair 1 summed in int16, 0 = singles (all proven at load)
#if defined(JQ_SNN_SSE) && !defined(__SSSE3__)
#define JQ_V128_W16 1
    // SSE2 (no pmaddubsw): int16 copies of the weights for pmaddwd, [chunk][input pair 0/1][output][2]
    int16_t* l1w16;        // W / 4 chunks x 2 x L1 x 2 (heap, owned)
    alignas(16) int16_t l2w16[16 * 2 * 32 * 2];
#endif
};

// Build a FastHead from the original arrays. l2w: [L2IN/4][32][4] in the original input order
// (s[0..N1-1] = sqr, s[N1..2N1-1] = crelu, rest 0), l1w: [W/4][L1][4].
inline void fast_head_init(FastHead& H, int L1_, int W, const int32_t* l1b, const uint8_t* l1w, const int32_t* l2b,
                           const int8_t* l2w, int32_t fb, const int8_t* fw) {
#if defined(JQ_V128_W16)
    delete[] H.l1w16;
#endif
    std::memset(&H, 0, sizeof H);
    const int N1 = L1_ - 1, L2IN = (2 * N1 + 31) / 32 * 32;
    for (int o = 0; o < L1_; ++o) H.l1b[o] = l1b[o];
    for (int o = 0; o < 32; ++o) H.l2b[o] = l2b[o];
    // new input j: j < L1_ -> sqr of f_j (old j if j < N1), j >= L1_ -> crelu of f_(j - L1_) (old N1 + j - L1_)
    for (int j = 0; j < 2 * L1_; ++j) {
        const int k = j < L1_ ? j : j - L1_;
        if (k >= N1) continue;
        const int old = j < L1_ ? k : N1 + k;
        if (old >= L2IN) continue;
        for (int o = 0; o < 32; ++o) H.l2w[(j / 4 * 32 + o) * 4 + j % 4] = l2w[(old / 4 * 32 + o) * 4 + old % 4];
    }
    static const int ord[16] = {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15};
    for (int i = 0; i < 16; ++i) { H.fw[i] = fw[ord[i]]; H.fw[16 + i] = fw[16 + ord[i]]; }
    H.fb = fb;
    for (int i = 0; i < 32; ++i) H.fwn[i] = fw[i];
#if defined(JQ_V128_W16)
    {
        const int8_t* w8 = reinterpret_cast<const int8_t*>(l1w);
        H.l1w16 = new int16_t[std::size_t(W / 4) * 2 * L1_ * 2];
        for (int c = 0; c < W / 4; ++c)
            for (int p = 0; p < 2; ++p)
                for (int o = 0; o < L1_; ++o)
                    for (int l = 0; l < 2; ++l)
                        H.l1w16[((std::size_t(c) * 2 + p) * L1_ + o) * 2 + l] = w8[(std::size_t(c) * L1_ + o) * 4 + 2 * p + l];
        for (int c = 0; c < 2 * L1_ / 4; ++c)
            for (int p = 0; p < 2; ++p)
                for (int o = 0; o < 32; ++o)
                    for (int l = 0; l < 2; ++l)
                        H.l2w16[((c * 2 + p) * 32 + o) * 2 + l] = H.l2w[(c * 32 + o) * 4 + 2 * p + l];
    }
#endif
    auto bound = [](const int8_t* w, int nch, int nout) {
        int m = 0;
        for (int c = 0; c < nch; ++c)
            for (int o = 0; o < nout; ++o)
                for (int q = 0; q < 2; ++q) {
                    const int8_t* p = w + (std::size_t(c) * nout + o) * 4 + 2 * q;
                    m = std::max(m, 127 * (std::abs(int(p[0])) + std::abs(int(p[1]))));
                }
        return m;
    };
    auto group = [](int b) { return 4 * b <= 32767 ? 4 : 2 * b <= 32767 ? 2 : 1; };
    H.g1 = group(bound(reinterpret_cast<const int8_t*>(l1w), W / 4, L1_));
    H.g2 = group(bound(H.l2w, 2 * L1_ / 4, 32));
    for (int q = 0; q < 2 * L1_ / 16; ++q) {
        int b[4];
        for (int c = 0; c < 4; ++c) b[c] = bound(H.l2w + (4 * q + c) * 128, 1, 32);
        H.l2plan[q] = b[0] + b[1] + b[2] + b[3] <= 32767 ? 4 : uint8_t((b[0] + b[1] <= 32767 ? 1 : 0) | (b[2] + b[3] <= 32767 ? 2 : 0));
    }
}

#if defined(JQ_SNN_AVX2)
#define JQ_MUB(x, w) _mm256_maddubs_epi16(x, _mm256_load_si256((const __m256i*)(w)))

template <int L1_>
inline int32_t propagate_fast(const FastHead& H, const uint8_t* l1w, const uint8_t* x, const uint16_t* nnz, int n) {
    constexpr int NV = L1_ / 8, CB = L1_ * 4;  // ymm per L1 output vector, weight bytes per input chunk
    const int32_t* in32 = reinterpret_cast<const int32_t*>(x);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i a[NV];
    int i = 0;
#if defined(JQ_SNN_AVX512) || defined(JQ_SNN_DP256)
    if (true) {
#if defined(JQ_SNN_AVX512)
        // isa 2026-10-30: zmm accumulators (L1 16: one zmm per input, L1 32: two); VNNI: dpbusd = maddubs + madd exactly
        constexpr int NZ = L1_ / 16;
        __m512i z0[NZ], z1[NZ];
        for (int k = 0; k < NZ; ++k) { z0[k] = _mm512_load_si512(H.l1b + 16 * k); z1[k] = _mm512_setzero_si512(); }
#if defined(JQ_SNN_VNNI)
        for (; i + 2 <= n; i += 2) {
            const __m512i x0 = _mm512_set1_epi32(in32[nnz[i]]), x1 = _mm512_set1_epi32(in32[nnz[i + 1]]);
            const uint8_t *w0 = l1w + std::size_t(nnz[i]) * CB, *w1 = l1w + std::size_t(nnz[i + 1]) * CB;
            for (int k = 0; k < NZ; ++k) {
                z0[k] = _mm512_dpbusd_epi32(z0[k], x0, _mm512_loadu_si512(w0 + 64 * k));
                z1[k] = _mm512_dpbusd_epi32(z1[k], x1, _mm512_loadu_si512(w1 + 64 * k));
            }
        }
        for (; i < n; ++i) {
            const __m512i x0 = _mm512_set1_epi32(in32[nnz[i]]);
            const uint8_t* w0 = l1w + std::size_t(nnz[i]) * CB;
            for (int k = 0; k < NZ; ++k) z0[k] = _mm512_dpbusd_epi32(z0[k], x0, _mm512_loadu_si512(w0 + 64 * k));
        }
#else
        const __m512i ones5 = _mm512_set1_epi16(1);
#define JQ_MUB5(x, w) _mm512_maddubs_epi16(x, _mm512_loadu_si512(w))
        if (H.g1 >= 4)
            for (; i + 4 <= n; i += 4) {
                const __m512i x0 = _mm512_set1_epi32(in32[nnz[i]]), x1 = _mm512_set1_epi32(in32[nnz[i + 1]]),
                              x2 = _mm512_set1_epi32(in32[nnz[i + 2]]), x3 = _mm512_set1_epi32(in32[nnz[i + 3]]);
                const uint8_t *w0 = l1w + std::size_t(nnz[i]) * CB, *w1 = l1w + std::size_t(nnz[i + 1]) * CB,
                              *w2 = l1w + std::size_t(nnz[i + 2]) * CB, *w3 = l1w + std::size_t(nnz[i + 3]) * CB;
                for (int k = 0; k < NZ; ++k) {
                    const __m512i t = _mm512_add_epi16(_mm512_add_epi16(JQ_MUB5(x0, w0 + 64 * k), JQ_MUB5(x1, w1 + 64 * k)),
                                                       _mm512_add_epi16(JQ_MUB5(x2, w2 + 64 * k), JQ_MUB5(x3, w3 + 64 * k)));
                    z0[k] = _mm512_add_epi32(z0[k], _mm512_madd_epi16(t, ones5));
                }
            }
        if (H.g1 >= 2)
            for (; i + 2 <= n; i += 2) {
                const __m512i x0 = _mm512_set1_epi32(in32[nnz[i]]), x1 = _mm512_set1_epi32(in32[nnz[i + 1]]);
                const uint8_t *w0 = l1w + std::size_t(nnz[i]) * CB, *w1 = l1w + std::size_t(nnz[i + 1]) * CB;
                for (int k = 0; k < NZ; ++k)
                    z0[k] = _mm512_add_epi32(z0[k], _mm512_madd_epi16(_mm512_add_epi16(JQ_MUB5(x0, w0 + 64 * k), JQ_MUB5(x1, w1 + 64 * k)), ones5));
            }
        for (; i < n; ++i) {
            const __m512i x0 = _mm512_set1_epi32(in32[nnz[i]]);
            const uint8_t* w0 = l1w + std::size_t(nnz[i]) * CB;
            for (int k = 0; k < NZ; ++k) z1[k] = _mm512_add_epi32(z1[k], _mm512_madd_epi16(JQ_MUB5(x0, w0 + 64 * k), ones5));
        }
#undef JQ_MUB5
#endif
        for (int k = 0; k < NZ; ++k) {
            const __m512i t = _mm512_add_epi32(z0[k], z1[k]);
            a[2 * k] = _mm512_castsi512_si256(t);
            a[2 * k + 1] = _mm512_extracti64x4_epi64(t, 1);
        }
#else   // AVX-VNNI (ymm dpbusd)
        __m256i b[NV];
        for (int k = 0; k < NV; ++k) { a[k] = _mm256_load_si256((const __m256i*)(H.l1b + 8 * k)); b[k] = _mm256_setzero_si256(); }
        for (; i + 2 <= n; i += 2) {
            const __m256i x0 = _mm256_set1_epi32(in32[nnz[i]]), x1 = _mm256_set1_epi32(in32[nnz[i + 1]]);
            const uint8_t *w0 = l1w + std::size_t(nnz[i]) * CB, *w1 = l1w + std::size_t(nnz[i + 1]) * CB;
            for (int k = 0; k < NV; ++k) {
                a[k] = JQ_DP256(a[k], x0, _mm256_load_si256((const __m256i*)(w0 + 32 * k)));
                b[k] = JQ_DP256(b[k], x1, _mm256_load_si256((const __m256i*)(w1 + 32 * k)));
            }
        }
        for (; i < n; ++i) {
            const __m256i x0 = _mm256_set1_epi32(in32[nnz[i]]);
            const uint8_t* w0 = l1w + std::size_t(nnz[i]) * CB;
            for (int k = 0; k < NV; ++k) a[k] = JQ_DP256(a[k], x0, _mm256_load_si256((const __m256i*)(w0 + 32 * k)));
        }
        for (int k = 0; k < NV; ++k) a[k] = _mm256_add_epi32(a[k], b[k]);
#endif
    } else
#endif
    {
    for (int k = 0; k < NV; ++k) a[k] = _mm256_load_si256((const __m256i*)(H.l1b + 8 * k));
    if (H.g1 >= 4)
        for (; i + 4 <= n; i += 4) {
            const __m256i x0 = _mm256_set1_epi32(in32[nnz[i]]), x1 = _mm256_set1_epi32(in32[nnz[i + 1]]),
                          x2 = _mm256_set1_epi32(in32[nnz[i + 2]]), x3 = _mm256_set1_epi32(in32[nnz[i + 3]]);
            const uint8_t *w0 = l1w + std::size_t(nnz[i]) * CB, *w1 = l1w + std::size_t(nnz[i + 1]) * CB,
                          *w2 = l1w + std::size_t(nnz[i + 2]) * CB, *w3 = l1w + std::size_t(nnz[i + 3]) * CB;
            for (int k = 0; k < NV; ++k) {
                const __m256i t = _mm256_add_epi16(_mm256_add_epi16(JQ_MUB(x0, w0 + 32 * k), JQ_MUB(x1, w1 + 32 * k)),
                                                   _mm256_add_epi16(JQ_MUB(x2, w2 + 32 * k), JQ_MUB(x3, w3 + 32 * k)));
                a[k] = _mm256_add_epi32(a[k], _mm256_madd_epi16(t, ones));
            }
        }
    if (H.g1 >= 2)
        for (; i + 2 <= n; i += 2) {
            const __m256i x0 = _mm256_set1_epi32(in32[nnz[i]]), x1 = _mm256_set1_epi32(in32[nnz[i + 1]]);
            const uint8_t *w0 = l1w + std::size_t(nnz[i]) * CB, *w1 = l1w + std::size_t(nnz[i + 1]) * CB;
            for (int k = 0; k < NV; ++k)
                a[k] = _mm256_add_epi32(a[k], _mm256_madd_epi16(_mm256_add_epi16(JQ_MUB(x0, w0 + 32 * k), JQ_MUB(x1, w1 + 32 * k)), ones));
        }
    for (; i < n; ++i) {
        const __m256i x0 = _mm256_set1_epi32(in32[nnz[i]]);
        const uint8_t* w0 = l1w + std::size_t(nnz[i]) * CB;
        for (int k = 0; k < NV; ++k) a[k] = _mm256_add_epi32(a[k], _mm256_madd_epi16(JQ_MUB(x0, w0 + 32 * k), ones));
    }
    }
    const int32_t skip = _mm256_extract_epi32(a[NV - 1], 7);
    // activations -> s (2 * L1_ bytes): [sqr(f0..), crelu(f0..)], dword order fixed by one permute per vector
    const __m256i z = _mm256_setzero_si256(), m127 = _mm256_set1_epi16(127), perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    auto sqr16 = [&](__m256i p) { const __m256i u = _mm256_abs_epi16(p); return _mm256_min_epu16(_mm256_srli_epi16(_mm256_mulhi_epu16(u, u), 3), m127); };
    auto cr16 = [&](__m256i f0, __m256i f1) {
        return _mm256_min_epi16(_mm256_max_epi16(_mm256_packs_epi32(_mm256_srai_epi32(f0, 6), _mm256_srai_epi32(f1, 6)), z), m127); };
    alignas(32) uint8_t s[2 * L1_];
    if constexpr (L1_ == 16) {
        const __m256i v = _mm256_packus_epi16(sqr16(_mm256_packs_epi32(a[0], a[1])), cr16(a[0], a[1]));
        _mm256_store_si256((__m256i*)s, _mm256_permutevar8x32_epi32(v, perm));
    } else {
        const __m256i sq = _mm256_packus_epi16(sqr16(_mm256_packs_epi32(a[0], a[1])), sqr16(_mm256_packs_epi32(a[2], a[3])));
        const __m256i cr = _mm256_packus_epi16(cr16(a[0], a[1]), cr16(a[2], a[3]));
        _mm256_store_si256((__m256i*)s, _mm256_permutevar8x32_epi32(sq, perm));
        _mm256_store_si256((__m256i*)(s + 32), _mm256_permutevar8x32_epi32(cr, perm));
    }
    // L2: 2 * L1_ / 4 input chunks x 32 outputs, nonzero chunks only (bit list), grouped by g2
    const int32_t* s32 = reinterpret_cast<const int32_t*>(s);
    __m256i acc[4];
    for (int k = 0; k < 4; ++k) acc[k] = _mm256_load_si256((const __m256i*)(H.l2b + 8 * k));
    constexpr int NQ = 2 * L1_ / 16;  // groups of 4 L2 input chunks
#if defined(JQ_SNN_VNNI)
    {
        __m512i c0 = _mm512_load_si512(H.l2b), c1 = _mm512_load_si512(H.l2b + 16);
        for (int ch = 0; ch < 4 * NQ; ++ch) {
            if (!s32[ch]) continue;
            const __m512i xv = _mm512_set1_epi32(s32[ch]);
            const int8_t* w = H.l2w + ch * 128;
            c0 = _mm512_dpbusd_epi32(c0, xv, _mm512_load_si512(w));
            c1 = _mm512_dpbusd_epi32(c1, xv, _mm512_load_si512(w + 64));
        }
        acc[0] = _mm512_castsi512_si256(c0); acc[1] = _mm512_extracti64x4_epi64(c0, 1);
        acc[2] = _mm512_castsi512_si256(c1); acc[3] = _mm512_extracti64x4_epi64(c1, 1);
    }
    if (false)
#elif defined(JQ_SNN_DP256)
    for (int ch = 0; ch < 4 * NQ; ++ch) {
        if (!s32[ch]) continue;
        const __m256i xv = _mm256_set1_epi32(s32[ch]);
        const int8_t* w = H.l2w + ch * 128;
        for (int k = 0; k < 4; ++k) acc[k] = JQ_DP256(acc[k], xv, _mm256_load_si256((const __m256i*)(w + 32 * k)));
    }
    if (false)
#endif
    for (int q = 0; q < NQ; ++q) {
        const int8_t* w = H.l2w + q * 512;
        const __m256i x0 = _mm256_set1_epi32(s32[4 * q]), x1 = _mm256_set1_epi32(s32[4 * q + 1]),
                      x2 = _mm256_set1_epi32(s32[4 * q + 2]), x3 = _mm256_set1_epi32(s32[4 * q + 3]);
        const int pl = H.l2plan[q];
        if (pl == 4) {
            for (int k = 0; k < 4; ++k) {
                const __m256i t = _mm256_add_epi16(_mm256_add_epi16(JQ_MUB(x0, w + 32 * k), JQ_MUB(x1, w + 128 + 32 * k)),
                                                   _mm256_add_epi16(JQ_MUB(x2, w + 256 + 32 * k), JQ_MUB(x3, w + 384 + 32 * k)));
                acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(t, ones));
            }
        } else {
            const __m256i xs[4] = {x0, x1, x2, x3};
            for (int p = 0; p < 2; ++p) {
                const int8_t* wp = w + p * 256;
                if (pl >> p & 1) {
                    for (int k = 0; k < 4; ++k)
                        acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(_mm256_add_epi16(JQ_MUB(xs[2 * p], wp + 32 * k), JQ_MUB(xs[2 * p + 1], wp + 128 + 32 * k)), ones));
                } else {
                    for (int k = 0; k < 4; ++k) {
                        acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(JQ_MUB(xs[2 * p], wp + 32 * k), ones));
                        acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(JQ_MUB(xs[2 * p + 1], wp + 128 + 32 * k), ones));
                    }
                }
            }
        }
    }
    // crelu(second >> 6) as int16 (packs saturation then clamp = clamp), dot with int16 fw
    const __m256i h0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_packs_epi32(_mm256_srai_epi32(acc[0], 6), _mm256_srai_epi32(acc[1], 6)), z), m127);
    const __m256i h1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_packs_epi32(_mm256_srai_epi32(acc[2], 6), _mm256_srai_epi32(acc[3], 6)), z), m127);
    const __m256i o = _mm256_add_epi32(_mm256_madd_epi16(h0, _mm256_load_si256((const __m256i*)H.fw)),
                                       _mm256_madd_epi16(h1, _mm256_load_si256((const __m256i*)(H.fw + 16))));
    __m128i r = _mm_add_epi32(_mm256_castsi256_si128(o), _mm256_extracti128_si256(o, 1));
    r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 0x4E));
    r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 0xB1));
    const int32_t out = H.fb + _mm_cvtsi128_si32(r);
    const int64_t forward = int64_t(skip) * (600 * 16) / (127 * 64);
    return int32_t(int64_t(out) + forward);
}
#undef JQ_MUB
#elif defined(JQ_SNN_V128)
// 128-bit version (SSE2 / SSSE3 / NEON / dotprod): exact 4-byte dot products per output, natural output order.
template <int L1_>
inline int32_t propagate_fast(const FastHead& H, const uint8_t* l1w, const uint8_t* x, const uint16_t* nnz, int n) {
    using namespace v128;
    constexpr int NV = L1_ / 4, CB = L1_ * 4;
    const int32_t* in32 = reinterpret_cast<const int32_t*>(x);
    I32 a[NV], b[NV];
    for (int k = 0; k < NV; ++k) { a[k] = ld32(H.l1b + 4 * k); b[k] = zero32(); }
    int i = 0;
#if defined(JQ_V128_W16)
    // SSE2: per input chunk two pmaddwd per 4 outputs on the int16 weight copy (x pairs broadcast as int16)
    for (; i < n; ++i) {
        const uint32_t v = uint32_t(in32[nnz[i]]);
        const __m128i p0 = _mm_set1_epi32(int32_t((v & 0xFFu) | (v & 0xFF00u) << 8)), p1 = _mm_set1_epi32(int32_t((v >> 16 & 0xFFu) | (v >> 24) << 16));
        const int16_t* w = H.l1w16 + std::size_t(nnz[i]) * (4 * L1_);
        I32* acc = (i & 1) ? b : a;
        for (int k = 0; k < NV; ++k)
            acc[k] = _mm_add_epi32(acc[k], _mm_add_epi32(_mm_madd_epi16(p0, _mm_loadu_si128((const __m128i*)(w + 8 * k))),
                                                         _mm_madd_epi16(p1, _mm_loadu_si128((const __m128i*)(w + 2 * L1_ + 8 * k)))));
    }
#endif
#if defined(JQ_V128_MUB)
    // int16 grouping of 4 / 2 inputs (load-time proof H.g1, as the AVX2 kernel), one widening add per group
    if (H.g1 >= 4)
        for (; i + 4 <= n; i += 4) {
            const X x0 = bx(in32[nnz[i]]), x1 = bx(in32[nnz[i + 1]]), x2 = bx(in32[nnz[i + 2]]), x3 = bx(in32[nnz[i + 3]]);
            const uint8_t *w0 = l1w + std::size_t(nnz[i]) * CB, *w1 = l1w + std::size_t(nnz[i + 1]) * CB,
                          *w2 = l1w + std::size_t(nnz[i + 2]) * CB, *w3 = l1w + std::size_t(nnz[i + 3]) * CB;
            for (int k = 0; k < NV; ++k)
                a[k] = widen_add(a[k], add16(add16(mub(x0, w0 + 16 * k), mub(x1, w1 + 16 * k)), add16(mub(x2, w2 + 16 * k), mub(x3, w3 + 16 * k))));
        }
    if (H.g1 >= 2)
        for (; i + 2 <= n; i += 2) {
            const X x0 = bx(in32[nnz[i]]), x1 = bx(in32[nnz[i + 1]]);
            const uint8_t *w0 = l1w + std::size_t(nnz[i]) * CB, *w1 = l1w + std::size_t(nnz[i + 1]) * CB;
            for (int k = 0; k < NV; ++k) a[k] = widen_add(a[k], add16(mub(x0, w0 + 16 * k), mub(x1, w1 + 16 * k)));
        }
    for (; i < n; ++i) {
        const X x0 = bx(in32[nnz[i]]);
        const uint8_t* w0 = l1w + std::size_t(nnz[i]) * CB;
        for (int k = 0; k < NV; ++k) b[k] = widen_add(b[k], mub(x0, w0 + 16 * k));
    }
#endif
    for (; i + 2 <= n; i += 2) {
        const X x0 = bx(in32[nnz[i]]), x1 = bx(in32[nnz[i + 1]]);
        const uint8_t *w0 = l1w + std::size_t(nnz[i]) * CB, *w1 = l1w + std::size_t(nnz[i + 1]) * CB;
        for (int k = 0; k < NV; ++k) { a[k] = dot4(a[k], x0, w0 + 16 * k); b[k] = dot4(b[k], x1, w1 + 16 * k); }
    }
    if (i < n) {
        const X x0 = bx(in32[nnz[i]]);
        const uint8_t* w0 = l1w + std::size_t(nnz[i]) * CB;
        for (int k = 0; k < NV; ++k) a[k] = dot4(a[k], x0, w0 + 16 * k);
    }
    for (int k = 0; k < NV; ++k) a[k] = add32(a[k], b[k]);
    const int32_t skip = lane3(a[NV - 1]);
    alignas(16) uint8_t s[2 * L1_];
    for (int m = 0; m < NV / 4; ++m) {  // 16 outputs per step
        const I16 f0 = packs(a[4 * m], a[4 * m + 1]), f1 = packs(a[4 * m + 2], a[4 * m + 3]);
        stu8(s + 16 * m, narrow(sqr16(f0), sqr16(f1)));
        stu8(s + L1_ + 16 * m, narrow(clamp0_127(packs(sra6(a[4 * m]), sra6(a[4 * m + 1]))),
                                      clamp0_127(packs(sra6(a[4 * m + 2]), sra6(a[4 * m + 3])))));
    }
    const int32_t* s32 = reinterpret_cast<const int32_t*>(s);
    I32 c[8];
    for (int k = 0; k < 8; ++k) c[k] = ld32(H.l2b + 4 * k);
#if defined(JQ_V128_MUB)
    for (int q = 0; q < 2 * L1_ / 16; ++q) {  // groups of 4 L2 input chunks, int16 grouping per H.l2plan (load-time proof)
        if (!(s32[4 * q] | s32[4 * q + 1] | s32[4 * q + 2] | s32[4 * q + 3])) continue;
        const int8_t* w = H.l2w + q * 512;
        const X xs[4] = {bx(s32[4 * q]), bx(s32[4 * q + 1]), bx(s32[4 * q + 2]), bx(s32[4 * q + 3])};
        const int pl = H.l2plan[q];
        if (pl == 4) {
            for (int k = 0; k < 8; ++k)
                c[k] = widen_add(c[k], add16(add16(mub(xs[0], w + 16 * k), mub(xs[1], w + 128 + 16 * k)),
                                             add16(mub(xs[2], w + 256 + 16 * k), mub(xs[3], w + 384 + 16 * k))));
        } else {
            for (int p = 0; p < 2; ++p) {
                const int8_t* wp = w + p * 256;
                if (pl >> p & 1) {
                    for (int k = 0; k < 8; ++k) c[k] = widen_add(c[k], add16(mub(xs[2 * p], wp + 16 * k), mub(xs[2 * p + 1], wp + 128 + 16 * k)));
                } else {
                    for (int k = 0; k < 8; ++k) {
                        c[k] = widen_add(c[k], mub(xs[2 * p], wp + 16 * k));
                        c[k] = widen_add(c[k], mub(xs[2 * p + 1], wp + 128 + 16 * k));
                    }
                }
            }
        }
    }
#elif defined(JQ_V128_W16)
    for (int ch = 0; ch < 2 * L1_ / 4; ++ch) {
        const uint32_t v = uint32_t(s32[ch]);
        if (!v) continue;
        const __m128i p0 = _mm_set1_epi32(int32_t((v & 0xFFu) | (v & 0xFF00u) << 8)), p1 = _mm_set1_epi32(int32_t((v >> 16 & 0xFFu) | (v >> 24) << 16));
        const int16_t* w = H.l2w16 + ch * 128;
        for (int k = 0; k < 8; ++k)
            c[k] = _mm_add_epi32(c[k], _mm_add_epi32(_mm_madd_epi16(p0, _mm_load_si128((const __m128i*)(w + 8 * k))),
                                                     _mm_madd_epi16(p1, _mm_load_si128((const __m128i*)(w + 64 + 8 * k)))));
    }
#else
    for (int ch = 0; ch < 2 * L1_ / 4; ++ch) {
        if (!s32[ch]) continue;
        const X xv = bx(s32[ch]);
        const int8_t* w = H.l2w + ch * 128;
        for (int k = 0; k < 8; ++k) c[k] = dot4(c[k], xv, w + 16 * k);
    }
#endif
    alignas(16) int32_t sec[32];
    for (int k = 0; k < 8; ++k) st32(sec + 4 * k, c[k]);
    int32_t out = H.fb;
    for (int o = 0; o < 32; ++o) out += int32_t(H.fwn[o]) * std::clamp(sec[o] >> 6, 0, 127);
    const int64_t forward = int64_t(skip) * (600 * 16) / (127 * 64);
    return int32_t(int64_t(out) + forward);
}
#endif

}  // namespace jq::nnk
