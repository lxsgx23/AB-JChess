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
// Shared integer kernels of the ABJ-style small nets (snnue.cpp, hnnue.cpp): accumulator row updates (AVX-512 /
// AVX2 / scalar), the pairwise FT activation, and one layer stack (W -> 16 -> [sqr, crelu] 30 -> 32 -> 1 + skip),
// with exactly the integer semantics of abjnnue.cpp.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#define JQ_SNN_AVX2 1
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && !defined(JQ_NO_ZMM)
#define JQ_SNN_AVX512 1
#endif
#if defined(JQ_SNN_AVX512) && defined(__AVX512VNNI__)
#define JQ_SNN_VNNI 1
#endif
#if defined(JQ_SNN_AVX2) && defined(__AVXVNNI__) && !defined(JQ_SNN_VNNI)
#define JQ_SNN_DP256 1
#define JQ_DP256(a, x, w) _mm256_dpbusd_avx_epi32(a, x, w)
#endif
// isa 2026-10-30: 128-bit kernels for the pre-AVX2 x86 tiers (SSE2 / SSSE3 / SSE4.1) and ARM NEON (armv7-neon, armv8,
// armv8.2 dotprod). Integer results identical to the scalar / AVX2 code (exact int8 x uint8 dot products, wrapping
// int16 / int32 adds, same clamps and shifts).
#if !defined(JQ_SNN_AVX2) && (defined(__SSE2__) || defined(_M_X64))
#include <immintrin.h>
#define JQ_SNN_V128 1
#define JQ_SNN_SSE 1
#elif !defined(JQ_SNN_AVX2) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define JQ_SNN_V128 1
#define JQ_SNN_NEON 1
#endif

namespace jq::nnk {

constexpr int NPSQT = 16, L1 = 16, L2 = 32, MAX_W = 1024;

#if defined(JQ_SNN_V128)
namespace v128 {
#if defined(JQ_SNN_SSE)
using I16 = __m128i; using I32 = __m128i; using U8 = __m128i;
inline I16 ld16(const int16_t* p) { return _mm_load_si128((const __m128i*)p); }
inline void st16(int16_t* p, I16 v) { _mm_store_si128((__m128i*)p, v); }
inline I32 ld32(const int32_t* p) { return _mm_loadu_si128((const __m128i*)p); }
inline void st32(int32_t* p, I32 v) { _mm_storeu_si128((__m128i*)p, v); }
inline I16 add16(I16 a, I16 b) { return _mm_add_epi16(a, b); }
inline I16 sub16(I16 a, I16 b) { return _mm_sub_epi16(a, b); }
inline I32 add32(I32 a, I32 b) { return _mm_add_epi32(a, b); }
inline I32 sub32(I32 a, I32 b) { return _mm_sub_epi32(a, b); }
inline I32 zero32() { return _mm_setzero_si128(); }
inline I16 zero16() { return _mm_setzero_si128(); }
inline int32_t lane3(I32 v) { return _mm_cvtsi128_si32(_mm_shuffle_epi32(v, 0xFF)); }
// Broadcast 4 input bytes (each <= 127). SSE2 keeps the zero-extended int16 copy for madd.
#if defined(__SSSE3__)
struct X { __m128i b; };
inline X bx(int32_t v) { return X{_mm_set1_epi32(v)}; }
// acc[j] += sum_l w[4j + l] * x[l] (exact: maddubs pairs <= 2 * 127 * 128 < 32767, madd exact)
inline I32 dot4(I32 acc, X x, const void* w) {
    return _mm_add_epi32(acc, _mm_madd_epi16(_mm_maddubs_epi16(x.b, _mm_load_si128((const __m128i*)w)), _mm_set1_epi16(1)));
}
#else
struct X { __m128i h; };
inline X bx(int32_t v) { return X{_mm_unpacklo_epi8(_mm_set1_epi32(v), _mm_setzero_si128())}; }
inline I32 dot4(I32 acc, X x, const void* w) {
    const __m128i wv = _mm_load_si128((const __m128i*)w);
    const __m128i p0 = _mm_madd_epi16(x.h, _mm_srai_epi16(_mm_unpacklo_epi8(wv, wv), 8));
    const __m128i p1 = _mm_madd_epi16(x.h, _mm_srai_epi16(_mm_unpackhi_epi8(wv, wv), 8));
    const __m128 a = _mm_castsi128_ps(p0), b = _mm_castsi128_ps(p1);
    return _mm_add_epi32(acc, _mm_add_epi32(_mm_castps_si128(_mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0))),
                                            _mm_castps_si128(_mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1)))));
}
#endif
#if defined(__SSSE3__)
// maddubs layout: 8 int16 pair sums (exact, <= 2 * 127 * 128); widen_add: acc[j] += t[2j] + t[2j + 1]
#define JQ_V128_MUB 1
inline I16 mub(X x, const void* w) { return _mm_maddubs_epi16(x.b, _mm_load_si128((const __m128i*)w)); }
inline I32 widen_add(I32 acc, I16 t) { return _mm_add_epi32(acc, _mm_madd_epi16(t, _mm_set1_epi16(1))); }
#endif
// 8 int32 -> 8 int16 (saturating) / 16 int16 -> 16 uint8 (values already in [0, 127])
inline I16 packs(I32 a, I32 b) { return _mm_packs_epi32(a, b); }
inline U8 narrow(I16 a, I16 b) { return _mm_packus_epi16(a, b); }
inline I32 sra6(I32 a) { return _mm_srai_epi32(a, 6); }
inline I16 clamp0_127(I16 a) { return _mm_min_epi16(_mm_max_epi16(a, _mm_setzero_si128()), _mm_set1_epi16(127)); }
// min(127, clamp16(f)^2 >> 19) per int16 lane
inline I16 sqr16(I16 p) {
#if defined(__SSSE3__)
    const __m128i u = _mm_abs_epi16(p);
#else
    const __m128i u = _mm_max_epi16(p, _mm_sub_epi16(_mm_setzero_si128(), p));  // -32768 -> 0x8000 like pabsw
#endif
    return _mm_min_epi16(_mm_srli_epi16(_mm_mulhi_epu16(u, u), 3), _mm_set1_epi16(127));  // <= 8191: signed min ok
}
inline void stu8(uint8_t* p, U8 v) { _mm_store_si128((__m128i*)p, v); }
// 8 int8 -> 8 int16 (sign extension)
inline I16 sx8(const int8_t* p) {
#if defined(__SSE4_1__)
    return _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)p));
#else
    const __m128i v = _mm_loadl_epi64((const __m128i*)p);
    return _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 8);
#endif
}
// pairwise product clamp(a,0,127) * clamp(b,0,127) >> 7 on 8 lanes
inline I16 pw(I16 a, I16 b) { return _mm_srli_epi16(_mm_mullo_epi16(clamp0_127(a), clamp0_127(b)), 7); }
// bit j set = 4-byte chunk j of v nonzero (4 bits)
inline unsigned nz4(U8 v) { return ~unsigned(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(v, _mm_setzero_si128())))) & 0xFu; }
inline void lut_store(uint16_t* dst, const uint16_t* lut, int base) {
    _mm_storeu_si128((__m128i*)dst, _mm_add_epi16(_mm_load_si128((const __m128i*)lut), _mm_set1_epi16(int16_t(base))));
}
#else  // NEON
using I16 = int16x8_t; using I32 = int32x4_t; using U8 = uint8x16_t;
inline I16 ld16(const int16_t* p) { return vld1q_s16(p); }
inline void st16(int16_t* p, I16 v) { vst1q_s16(p, v); }
inline I32 ld32(const int32_t* p) { return vld1q_s32(p); }
inline void st32(int32_t* p, I32 v) { vst1q_s32(p, v); }
inline I16 add16(I16 a, I16 b) { return vaddq_s16(a, b); }
inline I16 sub16(I16 a, I16 b) { return vsubq_s16(a, b); }
inline I32 add32(I32 a, I32 b) { return vaddq_s32(a, b); }
inline I32 sub32(I32 a, I32 b) { return vsubq_s32(a, b); }
inline I32 zero32() { return vdupq_n_s32(0); }
inline I16 zero16() { return vdupq_n_s16(0); }
inline int32_t lane3(I32 v) { return vgetq_lane_s32(v, 3); }
struct X { int8x16_t b; };
inline X bx(int32_t v) { return X{vreinterpretq_s8_s32(vdupq_n_s32(v))}; }  // bytes <= 127: same as int8
inline I32 dot4(I32 acc, X x, const void* w) {
    const int8x16_t wv = vld1q_s8(reinterpret_cast<const int8_t*>(w));
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, wv, x.b);
#else
    const int32x4_t p0 = vpaddlq_s16(vmull_s8(vget_low_s8(wv), vget_low_s8(x.b)));
    const int32x4_t p1 = vpaddlq_s16(vmull_s8(vget_high_s8(wv), vget_high_s8(x.b)));
#if defined(__aarch64__)
    return vaddq_s32(acc, vpaddq_s32(p0, p1));
#else
    return vaddq_s32(acc, vcombine_s32(vpadd_s32(vget_low_s32(p0), vget_high_s32(p0)), vpadd_s32(vget_low_s32(p1), vget_high_s32(p1))));
#endif
#endif
}
#if defined(__aarch64__) && !defined(__ARM_FEATURE_DOTPROD)
// maddubs layout via widening multiplies + pairwise add (pair sums exact in int16), vpadal = madd(t, 1) + add
#define JQ_V128_MUB 1
inline I16 mub(X x, const void* w) {
    const int8x16_t wv = vld1q_s8(reinterpret_cast<const int8_t*>(w));
    return vpaddq_s16(vmull_s8(vget_low_s8(wv), vget_low_s8(x.b)), vmull_high_s8(wv, x.b));
}
inline I32 widen_add(I32 acc, I16 t) { return vpadalq_s16(acc, t); }
#endif
inline I16 packs(I32 a, I32 b) { return vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)); }
inline U8 narrow(I16 a, I16 b) { return vcombine_u8(vmovn_u16(vreinterpretq_u16_s16(a)), vmovn_u16(vreinterpretq_u16_s16(b))); }
inline I32 sra6(I32 a) { return vshrq_n_s32(a, 6); }
inline I16 clamp0_127(I16 a) { return vminq_s16(vmaxq_s16(a, vdupq_n_s16(0)), vdupq_n_s16(127)); }
inline I16 sqr16(I16 p) {
    const uint16x8_t u = vreinterpretq_u16_s16(vabsq_s16(p));  // wrapping abs: -32768 -> 0x8000
    const uint16x8_t hi = vcombine_u16(vshrn_n_u32(vmull_u16(vget_low_u16(u), vget_low_u16(u)), 16),
                                       vshrn_n_u32(vmull_u16(vget_high_u16(u), vget_high_u16(u)), 16));
    return vreinterpretq_s16_u16(vminq_u16(vshrq_n_u16(hi, 3), vdupq_n_u16(127)));
}
inline void stu8(uint8_t* p, U8 v) { vst1q_u8(p, v); }
inline I16 sx8(const int8_t* p) { return vmovl_s8(vld1_s8(p)); }
inline I16 pw(I16 a, I16 b) {
    return vreinterpretq_s16_u16(vshrq_n_u16(vreinterpretq_u16_s16(vmulq_s16(clamp0_127(a), clamp0_127(b))), 7));
}
inline unsigned nz4(U8 v) {
    static const uint32_t bits[4] = {1, 2, 4, 8};
    const uint32x4_t m = vandq_u32(vtstq_u32(vreinterpretq_u32_u8(v), vreinterpretq_u32_u8(v)), vld1q_u32(bits));
#if defined(__aarch64__)
    return vaddvq_u32(m);
#else
    const uint32x2_t t = vpadd_u32(vget_low_u32(m), vget_high_u32(m));
    return vget_lane_u32(vpadd_u32(t, t), 0);
#endif
}
inline void lut_store(uint16_t* dst, const uint16_t* lut, int base) {
    vst1q_u16(dst, vaddq_u16(vld1q_u16(lut), vdupq_n_u16(uint16_t(base))));
}
#endif
}  // namespace v128
#endif

// dst = src + sum(add rows) - sum(rem rows) over W int16 columns (wrapping, like the engine's int16 adds) and the
// NPSQT int32 PSQT columns. Rows: ftW + f * W, psqt + f * NPSQT.
inline void apply(const int16_t* ftW, const int32_t* psqt, int W, int16_t* dst, int32_t* pdst, const int16_t* src,
                  const int32_t* psrc, const uint16_t* add, int nAdd, const uint16_t* rem, int nRem) {
    auto wrow = [&](int f) { return ftW + std::size_t(f) * std::size_t(W); };
    auto prow = [&](int f) { return psqt + std::size_t(f) * NPSQT; };
#if defined(JQ_SNN_AVX512)
    for (int c = 0; c < W; c += 128) {
        __m512i r0 = _mm512_load_si512(src + c), r1 = _mm512_load_si512(src + c + 32),
                r2 = _mm512_load_si512(src + c + 64), r3 = _mm512_load_si512(src + c + 96);
        for (int i = 0; i < nAdd; ++i) {
            const int16_t* w = wrow(add[i]) + c;
            r0 = _mm512_add_epi16(r0, _mm512_load_si512(w));
            r1 = _mm512_add_epi16(r1, _mm512_load_si512(w + 32));
            r2 = _mm512_add_epi16(r2, _mm512_load_si512(w + 64));
            r3 = _mm512_add_epi16(r3, _mm512_load_si512(w + 96));
        }
        for (int i = 0; i < nRem; ++i) {
            const int16_t* w = wrow(rem[i]) + c;
            r0 = _mm512_sub_epi16(r0, _mm512_load_si512(w));
            r1 = _mm512_sub_epi16(r1, _mm512_load_si512(w + 32));
            r2 = _mm512_sub_epi16(r2, _mm512_load_si512(w + 64));
            r3 = _mm512_sub_epi16(r3, _mm512_load_si512(w + 96));
        }
        _mm512_store_si512(dst + c, r0);
        _mm512_store_si512(dst + c + 32, r1);
        _mm512_store_si512(dst + c + 64, r2);
        _mm512_store_si512(dst + c + 96, r3);
    }
    __m512i p = _mm512_loadu_si512(psrc);
    for (int i = 0; i < nAdd; ++i) p = _mm512_add_epi32(p, _mm512_loadu_si512(prow(add[i])));
    for (int i = 0; i < nRem; ++i) p = _mm512_sub_epi32(p, _mm512_loadu_si512(prow(rem[i])));
    _mm512_storeu_si512(pdst, p);
#elif defined(JQ_SNN_AVX2)
    for (int c = 0; c < W; c += 64) {
        __m256i r0 = _mm256_load_si256((const __m256i*)(src + c));
        __m256i r1 = _mm256_load_si256((const __m256i*)(src + c + 16));
        __m256i r2 = _mm256_load_si256((const __m256i*)(src + c + 32));
        __m256i r3 = _mm256_load_si256((const __m256i*)(src + c + 48));
        for (int i = 0; i < nAdd; ++i) {
            const int16_t* w = wrow(add[i]) + c;
            r0 = _mm256_add_epi16(r0, _mm256_load_si256((const __m256i*)w));
            r1 = _mm256_add_epi16(r1, _mm256_load_si256((const __m256i*)(w + 16)));
            r2 = _mm256_add_epi16(r2, _mm256_load_si256((const __m256i*)(w + 32)));
            r3 = _mm256_add_epi16(r3, _mm256_load_si256((const __m256i*)(w + 48)));
        }
        for (int i = 0; i < nRem; ++i) {
            const int16_t* w = wrow(rem[i]) + c;
            r0 = _mm256_sub_epi16(r0, _mm256_load_si256((const __m256i*)w));
            r1 = _mm256_sub_epi16(r1, _mm256_load_si256((const __m256i*)(w + 16)));
            r2 = _mm256_sub_epi16(r2, _mm256_load_si256((const __m256i*)(w + 32)));
            r3 = _mm256_sub_epi16(r3, _mm256_load_si256((const __m256i*)(w + 48)));
        }
        _mm256_store_si256((__m256i*)(dst + c), r0);
        _mm256_store_si256((__m256i*)(dst + c + 16), r1);
        _mm256_store_si256((__m256i*)(dst + c + 32), r2);
        _mm256_store_si256((__m256i*)(dst + c + 48), r3);
    }
    for (int h = 0; h < NPSQT; h += 8) {
        __m256i p = _mm256_loadu_si256((const __m256i*)(psrc + h));
        for (int i = 0; i < nAdd; ++i) p = _mm256_add_epi32(p, _mm256_loadu_si256((const __m256i*)(prow(add[i]) + h)));
        for (int i = 0; i < nRem; ++i) p = _mm256_sub_epi32(p, _mm256_loadu_si256((const __m256i*)(prow(rem[i]) + h)));
        _mm256_storeu_si256((__m256i*)(pdst + h), p);
    }
#elif defined(JQ_SNN_V128)
    for (int c = 0; c < W; c += 32) {
        v128::I16 r0 = v128::ld16(src + c), r1 = v128::ld16(src + c + 8), r2 = v128::ld16(src + c + 16), r3 = v128::ld16(src + c + 24);
        for (int i = 0; i < nAdd; ++i) {
            const int16_t* w = wrow(add[i]) + c;
            r0 = v128::add16(r0, v128::ld16(w)); r1 = v128::add16(r1, v128::ld16(w + 8));
            r2 = v128::add16(r2, v128::ld16(w + 16)); r3 = v128::add16(r3, v128::ld16(w + 24));
        }
        for (int i = 0; i < nRem; ++i) {
            const int16_t* w = wrow(rem[i]) + c;
            r0 = v128::sub16(r0, v128::ld16(w)); r1 = v128::sub16(r1, v128::ld16(w + 8));
            r2 = v128::sub16(r2, v128::ld16(w + 16)); r3 = v128::sub16(r3, v128::ld16(w + 24));
        }
        v128::st16(dst + c, r0); v128::st16(dst + c + 8, r1); v128::st16(dst + c + 16, r2); v128::st16(dst + c + 24, r3);
    }
    for (int h = 0; h < NPSQT; h += 4) {
        v128::I32 p = v128::ld32(psrc + h);
        for (int i = 0; i < nAdd; ++i) p = v128::add32(p, v128::ld32(prow(add[i]) + h));
        for (int i = 0; i < nRem; ++i) p = v128::sub32(p, v128::ld32(prow(rem[i]) + h));
        v128::st32(pdst + h, p);
    }
#else
    for (int c = 0; c < W; ++c) {
        uint16_t v = uint16_t(src[c]);
        for (int i = 0; i < nAdd; ++i) v = uint16_t(v + uint16_t(wrow(add[i])[c]));
        for (int i = 0; i < nRem; ++i) v = uint16_t(v - uint16_t(wrow(rem[i])[c]));
        dst[c] = int16_t(v);
    }
    for (int b = 0; b < NPSQT; ++b) {
        uint32_t v = uint32_t(psrc[b]);
        for (int i = 0; i < nAdd; ++i) v += uint32_t(prow(add[i])[b]);
        for (int i = 0; i < nRem; ++i) v -= uint32_t(prow(rem[i])[b]);
        pdst[b] = int32_t(v);
    }
#endif
}

// Same as apply() plus int8 threat rows (thrW + f * W, sign-extended; no PSQT columns).
inline void apply_thr(const int16_t* ftW, const int32_t* psqt, int W, int16_t* dst, int32_t* pdst, const int16_t* src,
                      const int32_t* psrc, const uint16_t* add, int nAdd, const uint16_t* rem, int nRem,
                      const int8_t* thrW, int Wt, const uint32_t* tAdd, int ntA, const uint32_t* tRem, int ntR) {
    auto wrow = [&](int f) { return ftW + std::size_t(f) * std::size_t(W); };
    auto trow = [&](uint32_t f) { return thrW + std::size_t(f) * std::size_t(Wt); };  // threat rows cover lanes [0, Wt)
    auto prow = [&](int f) { return psqt + std::size_t(f) * NPSQT; };
#if defined(JQ_SNN_AVX512)
    for (int c = 0; c < W; c += 128) {
        __m512i r0 = _mm512_load_si512(src + c), r1 = _mm512_load_si512(src + c + 32),
                r2 = _mm512_load_si512(src + c + 64), r3 = _mm512_load_si512(src + c + 96);
        for (int i = 0; i < nAdd; ++i) {
            const int16_t* w = wrow(add[i]) + c;
            r0 = _mm512_add_epi16(r0, _mm512_load_si512(w)); r1 = _mm512_add_epi16(r1, _mm512_load_si512(w + 32));
            r2 = _mm512_add_epi16(r2, _mm512_load_si512(w + 64)); r3 = _mm512_add_epi16(r3, _mm512_load_si512(w + 96));
        }
        for (int i = 0; i < nRem; ++i) {
            const int16_t* w = wrow(rem[i]) + c;
            r0 = _mm512_sub_epi16(r0, _mm512_load_si512(w)); r1 = _mm512_sub_epi16(r1, _mm512_load_si512(w + 32));
            r2 = _mm512_sub_epi16(r2, _mm512_load_si512(w + 64)); r3 = _mm512_sub_epi16(r3, _mm512_load_si512(w + 96));
        }
        // a 128-lane chunk may extend past Wt (Wt % 32 == 0 but Wt % 128 != 0, e.g. Wt 32/64/96):
        // only the 32-lane registers inside [0, Wt) take threat rows; the old code read the NEXT rows' bytes into lanes >= Wt.
        if (c + 128 <= Wt) {
            for (int i = 0; i < ntA; ++i) {
                const int8_t* w = trow(tAdd[i]) + c;
                r0 = _mm512_add_epi16(r0, _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)w)));
                r1 = _mm512_add_epi16(r1, _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)(w + 32))));
                r2 = _mm512_add_epi16(r2, _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)(w + 64))));
                r3 = _mm512_add_epi16(r3, _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)(w + 96))));
            }
            for (int i = 0; i < ntR; ++i) {
                const int8_t* w = trow(tRem[i]) + c;
                r0 = _mm512_sub_epi16(r0, _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)w)));
                r1 = _mm512_sub_epi16(r1, _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)(w + 32))));
                r2 = _mm512_sub_epi16(r2, _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)(w + 64))));
                r3 = _mm512_sub_epi16(r3, _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)(w + 96))));
            }
        } else if (c < Wt) {
            const int nr = (Wt - c) / 32;   // 1..3 registers inside [0, Wt)
            for (int i = 0; i < ntA; ++i) {
                const int8_t* w = trow(tAdd[i]) + c;
                r0 = _mm512_add_epi16(r0, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)w)));
                if (nr > 1) r1 = _mm512_add_epi16(r1, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(w + 32))));
                if (nr > 2) r2 = _mm512_add_epi16(r2, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(w + 64))));
            }
            for (int i = 0; i < ntR; ++i) {
                const int8_t* w = trow(tRem[i]) + c;
                r0 = _mm512_sub_epi16(r0, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)w)));
                if (nr > 1) r1 = _mm512_sub_epi16(r1, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(w + 32))));
                if (nr > 2) r2 = _mm512_sub_epi16(r2, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(w + 64))));
            }
        }
        _mm512_store_si512(dst + c, r0); _mm512_store_si512(dst + c + 32, r1);
        _mm512_store_si512(dst + c + 64, r2); _mm512_store_si512(dst + c + 96, r3);
    }
    __m512i p = _mm512_loadu_si512(psrc);
    for (int i = 0; i < nAdd; ++i) p = _mm512_add_epi32(p, _mm512_loadu_si512(prow(add[i])));
    for (int i = 0; i < nRem; ++i) p = _mm512_sub_epi32(p, _mm512_loadu_si512(prow(rem[i])));
    _mm512_storeu_si512(pdst, p);
#elif defined(JQ_SNN_AVX2)
    for (int c = 0; c < W; c += 64) {
        __m256i r0 = _mm256_load_si256((const __m256i*)(src + c)), r1 = _mm256_load_si256((const __m256i*)(src + c + 16)),
                r2 = _mm256_load_si256((const __m256i*)(src + c + 32)), r3 = _mm256_load_si256((const __m256i*)(src + c + 48));
        for (int i = 0; i < nAdd; ++i) {
            const int16_t* w = wrow(add[i]) + c;
            r0 = _mm256_add_epi16(r0, _mm256_load_si256((const __m256i*)w));
            r1 = _mm256_add_epi16(r1, _mm256_load_si256((const __m256i*)(w + 16)));
            r2 = _mm256_add_epi16(r2, _mm256_load_si256((const __m256i*)(w + 32)));
            r3 = _mm256_add_epi16(r3, _mm256_load_si256((const __m256i*)(w + 48)));
        }
        for (int i = 0; i < nRem; ++i) {
            const int16_t* w = wrow(rem[i]) + c;
            r0 = _mm256_sub_epi16(r0, _mm256_load_si256((const __m256i*)w));
            r1 = _mm256_sub_epi16(r1, _mm256_load_si256((const __m256i*)(w + 16)));
            r2 = _mm256_sub_epi16(r2, _mm256_load_si256((const __m256i*)(w + 32)));
            r3 = _mm256_sub_epi16(r3, _mm256_load_si256((const __m256i*)(w + 48)));
        }
        // a 64-lane chunk may extend past Wt (Wt 32, 96, ...): lanes >= Wt must get no threat
        // rows. The old code loaded 64 bytes from row f at offset c for every chunk with c < Wt, i.e. for Wt 32 lanes 32..63
        // received row f+1's lanes 0..31. Wt % 64 == 0: unchanged path.
        if (c + 64 <= Wt) {
            for (int i = 0; i < ntA; ++i) {
                const int8_t* w = trow(tAdd[i]) + c;
                r0 = _mm256_add_epi16(r0, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)w)));
                r1 = _mm256_add_epi16(r1, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(w + 16))));
                r2 = _mm256_add_epi16(r2, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(w + 32))));
                r3 = _mm256_add_epi16(r3, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(w + 48))));
            }
            for (int i = 0; i < ntR; ++i) {
                const int8_t* w = trow(tRem[i]) + c;
                r0 = _mm256_sub_epi16(r0, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)w)));
                r1 = _mm256_sub_epi16(r1, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(w + 16))));
                r2 = _mm256_sub_epi16(r2, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(w + 32))));
                r3 = _mm256_sub_epi16(r3, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(w + 48))));
            }
        } else if (c < Wt) {   // Wt % 32 == 0 -> exactly lanes [c, c + 32) = r0, r1
            for (int i = 0; i < ntA; ++i) {
                const int8_t* w = trow(tAdd[i]) + c;
                r0 = _mm256_add_epi16(r0, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)w)));
                r1 = _mm256_add_epi16(r1, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(w + 16))));
            }
            for (int i = 0; i < ntR; ++i) {
                const int8_t* w = trow(tRem[i]) + c;
                r0 = _mm256_sub_epi16(r0, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)w)));
                r1 = _mm256_sub_epi16(r1, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(w + 16))));
            }
        }
        _mm256_store_si256((__m256i*)(dst + c), r0); _mm256_store_si256((__m256i*)(dst + c + 16), r1);
        _mm256_store_si256((__m256i*)(dst + c + 32), r2); _mm256_store_si256((__m256i*)(dst + c + 48), r3);
    }
    for (int h = 0; h < NPSQT; h += 8) {
        __m256i p = _mm256_loadu_si256((const __m256i*)(psrc + h));
        for (int i = 0; i < nAdd; ++i) p = _mm256_add_epi32(p, _mm256_loadu_si256((const __m256i*)(prow(add[i]) + h)));
        for (int i = 0; i < nRem; ++i) p = _mm256_sub_epi32(p, _mm256_loadu_si256((const __m256i*)(prow(rem[i]) + h)));
        _mm256_storeu_si256((__m256i*)(pdst + h), p);
    }
#elif defined(JQ_SNN_V128)
    apply(ftW, psqt, W, dst, pdst, src, psrc, add, nAdd, rem, nRem);
    for (int c = 0; c < Wt && c < W; c += 32) {
        v128::I16 r[4];
        for (int j = 0; j < 4; ++j) r[j] = v128::ld16(dst + c + 8 * j);
        for (int i = 0; i < ntA; ++i) { const int8_t* w = trow(tAdd[i]) + c; for (int j = 0; j < 4; ++j) r[j] = v128::add16(r[j], v128::sx8(w + 8 * j)); }
        for (int i = 0; i < ntR; ++i) { const int8_t* w = trow(tRem[i]) + c; for (int j = 0; j < 4; ++j) r[j] = v128::sub16(r[j], v128::sx8(w + 8 * j)); }
        for (int j = 0; j < 4; ++j) v128::st16(dst + c + 8 * j, r[j]);
    }
#else
    for (int c = 0; c < W; ++c) {
        uint16_t v = uint16_t(src[c]);
        for (int i = 0; i < nAdd; ++i) v = uint16_t(v + uint16_t(wrow(add[i])[c]));
        for (int i = 0; i < nRem; ++i) v = uint16_t(v - uint16_t(wrow(rem[i])[c]));
        if (c < Wt) for (int i = 0; i < ntA; ++i) v = uint16_t(v + uint16_t(int16_t(trow(tAdd[i])[c])));
        if (c < Wt) for (int i = 0; i < ntR; ++i) v = uint16_t(v - uint16_t(int16_t(trow(tRem[i])[c])));
        dst[c] = int16_t(v);
    }
    for (int b = 0; b < NPSQT; ++b) {
        uint32_t v = uint32_t(psrc[b]);
        for (int i = 0; i < nAdd; ++i) v += uint32_t(prow(add[i])[b]);
        for (int i = 0; i < nRem; ++i) v -= uint32_t(prow(rem[i])[b]);
        pdst[b] = int32_t(v);
    }
#endif
}


// threat part kept separately per accumulator (lanes [0, Wt), Wt % 32 == 0, Wt <= MAX_TW).
// D = sum(tAdd rows) - sum(tRem rows) (int8 rows sign-extended, int16 wrapping); tdst = tsrc + D (tsrc nullptr: 0 + D);
// then dst = src + sum(add rows) - sum(rem rows) over all W lanes, plus E on lanes [0, Wt) with E = tdst (addT) or D.
// All sums are modulo 2^16 per lane, so the result equals apply_thr's (only the order of the wrapping adds differs).
constexpr int MAX_TW = 256;
inline void thr_delta(const int8_t* thrW, int Wt, const uint32_t* tAdd, int ntA, const uint32_t* tRem, int ntR,
                      int16_t* D) {
    auto trow = [&](uint32_t f) { return thrW + std::size_t(f) * std::size_t(Wt); };
#if defined(JQ_SNN_AVX512)
    int c = 0;
    for (; c + 128 <= Wt; c += 128) {
        __m512i r0 = _mm512_setzero_si512(), r1 = r0, r2 = r0, r3 = r0;
        for (int i = 0; i < ntA; ++i) {
            const int8_t* w = trow(tAdd[i]) + c;
            r0 = _mm512_add_epi16(r0, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)w)));
            r1 = _mm512_add_epi16(r1, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(w + 32))));
            r2 = _mm512_add_epi16(r2, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(w + 64))));
            r3 = _mm512_add_epi16(r3, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(w + 96))));
        }
        for (int i = 0; i < ntR; ++i) {
            const int8_t* w = trow(tRem[i]) + c;
            r0 = _mm512_sub_epi16(r0, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)w)));
            r1 = _mm512_sub_epi16(r1, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(w + 32))));
            r2 = _mm512_sub_epi16(r2, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(w + 64))));
            r3 = _mm512_sub_epi16(r3, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(w + 96))));
        }
        _mm512_store_si512(D + c, r0); _mm512_store_si512(D + c + 32, r1);
        _mm512_store_si512(D + c + 64, r2); _mm512_store_si512(D + c + 96, r3);
    }
    for (; c < Wt; c += 32) {
        __m512i r0 = _mm512_setzero_si512();
        for (int i = 0; i < ntA; ++i) r0 = _mm512_add_epi16(r0, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(trow(tAdd[i]) + c))));
        for (int i = 0; i < ntR; ++i) r0 = _mm512_sub_epi16(r0, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(trow(tRem[i]) + c))));
        _mm512_store_si512(D + c, r0);
    }
#elif defined(JQ_SNN_AVX2)
    for (int c = 0; c < Wt; c += 32) {
        __m256i r0 = _mm256_setzero_si256(), r1 = r0;
        for (int i = 0; i < ntA; ++i) {
            const int8_t* w = trow(tAdd[i]) + c;
            r0 = _mm256_add_epi16(r0, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)w)));
            r1 = _mm256_add_epi16(r1, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w + 16))));
        }
        for (int i = 0; i < ntR; ++i) {
            const int8_t* w = trow(tRem[i]) + c;
            r0 = _mm256_sub_epi16(r0, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)w)));
            r1 = _mm256_sub_epi16(r1, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w + 16))));
        }
        _mm256_store_si256((__m256i*)(D + c), r0); _mm256_store_si256((__m256i*)(D + c + 16), r1);
    }
#elif defined(JQ_SNN_V128)
    for (int c = 0; c < Wt; c += 32) {
        v128::I16 r[4];
        for (int j = 0; j < 4; ++j) r[j] = v128::zero16();
        for (int i = 0; i < ntA; ++i) { const int8_t* w = trow(tAdd[i]) + c; for (int j = 0; j < 4; ++j) r[j] = v128::add16(r[j], v128::sx8(w + 8 * j)); }
        for (int i = 0; i < ntR; ++i) { const int8_t* w = trow(tRem[i]) + c; for (int j = 0; j < 4; ++j) r[j] = v128::sub16(r[j], v128::sx8(w + 8 * j)); }
        for (int j = 0; j < 4; ++j) v128::st16(D + c + 8 * j, r[j]);
    }
#else
    for (int c = 0; c < Wt; ++c) {
        uint16_t v = 0;
        for (int i = 0; i < ntA; ++i) v = uint16_t(v + uint16_t(int16_t(trow(tAdd[i])[c])));
        for (int i = 0; i < ntR; ++i) v = uint16_t(v - uint16_t(int16_t(trow(tRem[i])[c])));
        D[c] = int16_t(v);
    }
#endif
}
// a[c] = b[c] + e[c] for c < n (n % 32 == 0); b may alias a; b nullptr: a = e
inline void add16(int16_t* a, const int16_t* b, const int16_t* e, int n) {
#if defined(JQ_SNN_AVX512)
    for (int c = 0; c < n; c += 32)
        _mm512_store_si512(a + c, b ? _mm512_add_epi16(_mm512_load_si512(b + c), _mm512_load_si512(e + c)) : _mm512_load_si512(e + c));
#elif defined(JQ_SNN_AVX2)
    for (int c = 0; c < n; c += 16)
        _mm256_store_si256((__m256i*)(a + c), b ? _mm256_add_epi16(_mm256_load_si256((const __m256i*)(b + c)), _mm256_load_si256((const __m256i*)(e + c)))
                                                : _mm256_load_si256((const __m256i*)(e + c)));
#elif defined(JQ_SNN_V128)
    for (int c = 0; c < n; c += 8) v128::st16(a + c, b ? v128::add16(v128::ld16(b + c), v128::ld16(e + c)) : v128::ld16(e + c));
#else
    for (int c = 0; c < n; ++c) a[c] = int16_t(uint16_t((b ? uint16_t(b[c]) : 0) + uint16_t(e[c])));
#endif
}
// dst = src + piece rows over W lanes + E over lanes [0, Wt) (E aligned, Wt % 32 == 0); PSQT as apply().
inline void apply_e(const int16_t* ftW, const int32_t* psqt, int W, int16_t* dst, int32_t* pdst, const int16_t* src,
                    const int32_t* psrc, const uint16_t* add, int nAdd, const uint16_t* rem, int nRem, const int16_t* E, int Wt) {
    auto wrow = [&](int f) { return ftW + std::size_t(f) * std::size_t(W); };
    auto prow = [&](int f) { return psqt + std::size_t(f) * NPSQT; };
#if defined(JQ_SNN_AVX512)
    for (int c = 0; c < W; c += 128) {
        __m512i r0 = _mm512_load_si512(src + c), r1 = _mm512_load_si512(src + c + 32),
                r2 = _mm512_load_si512(src + c + 64), r3 = _mm512_load_si512(src + c + 96);
        for (int i = 0; i < nAdd; ++i) {
            const int16_t* w = wrow(add[i]) + c;
            r0 = _mm512_add_epi16(r0, _mm512_load_si512(w)); r1 = _mm512_add_epi16(r1, _mm512_load_si512(w + 32));
            r2 = _mm512_add_epi16(r2, _mm512_load_si512(w + 64)); r3 = _mm512_add_epi16(r3, _mm512_load_si512(w + 96));
        }
        for (int i = 0; i < nRem; ++i) {
            const int16_t* w = wrow(rem[i]) + c;
            r0 = _mm512_sub_epi16(r0, _mm512_load_si512(w)); r1 = _mm512_sub_epi16(r1, _mm512_load_si512(w + 32));
            r2 = _mm512_sub_epi16(r2, _mm512_load_si512(w + 64)); r3 = _mm512_sub_epi16(r3, _mm512_load_si512(w + 96));
        }
        if (c < Wt) {
            r0 = _mm512_add_epi16(r0, _mm512_load_si512(E + c));
            if (c + 32 < Wt) r1 = _mm512_add_epi16(r1, _mm512_load_si512(E + c + 32));
            if (c + 64 < Wt) r2 = _mm512_add_epi16(r2, _mm512_load_si512(E + c + 64));
            if (c + 96 < Wt) r3 = _mm512_add_epi16(r3, _mm512_load_si512(E + c + 96));
        }
        _mm512_store_si512(dst + c, r0); _mm512_store_si512(dst + c + 32, r1);
        _mm512_store_si512(dst + c + 64, r2); _mm512_store_si512(dst + c + 96, r3);
    }
    __m512i p = _mm512_loadu_si512(psrc);
    for (int i = 0; i < nAdd; ++i) p = _mm512_add_epi32(p, _mm512_loadu_si512(prow(add[i])));
    for (int i = 0; i < nRem; ++i) p = _mm512_sub_epi32(p, _mm512_loadu_si512(prow(rem[i])));
    _mm512_storeu_si512(pdst, p);
#elif defined(JQ_SNN_AVX2)
    for (int c = 0; c < W; c += 64) {
        __m256i r0 = _mm256_load_si256((const __m256i*)(src + c)), r1 = _mm256_load_si256((const __m256i*)(src + c + 16)),
                r2 = _mm256_load_si256((const __m256i*)(src + c + 32)), r3 = _mm256_load_si256((const __m256i*)(src + c + 48));
        for (int i = 0; i < nAdd; ++i) {
            const int16_t* w = wrow(add[i]) + c;
            r0 = _mm256_add_epi16(r0, _mm256_load_si256((const __m256i*)w)); r1 = _mm256_add_epi16(r1, _mm256_load_si256((const __m256i*)(w + 16)));
            r2 = _mm256_add_epi16(r2, _mm256_load_si256((const __m256i*)(w + 32))); r3 = _mm256_add_epi16(r3, _mm256_load_si256((const __m256i*)(w + 48)));
        }
        for (int i = 0; i < nRem; ++i) {
            const int16_t* w = wrow(rem[i]) + c;
            r0 = _mm256_sub_epi16(r0, _mm256_load_si256((const __m256i*)w)); r1 = _mm256_sub_epi16(r1, _mm256_load_si256((const __m256i*)(w + 16)));
            r2 = _mm256_sub_epi16(r2, _mm256_load_si256((const __m256i*)(w + 32))); r3 = _mm256_sub_epi16(r3, _mm256_load_si256((const __m256i*)(w + 48)));
        }
        if (c < Wt) {
            r0 = _mm256_add_epi16(r0, _mm256_load_si256((const __m256i*)(E + c)));
            r1 = _mm256_add_epi16(r1, _mm256_load_si256((const __m256i*)(E + c + 16)));
            if (c + 32 < Wt) {
                r2 = _mm256_add_epi16(r2, _mm256_load_si256((const __m256i*)(E + c + 32)));
                r3 = _mm256_add_epi16(r3, _mm256_load_si256((const __m256i*)(E + c + 48)));
            }
        }
        _mm256_store_si256((__m256i*)(dst + c), r0); _mm256_store_si256((__m256i*)(dst + c + 16), r1);
        _mm256_store_si256((__m256i*)(dst + c + 32), r2); _mm256_store_si256((__m256i*)(dst + c + 48), r3);
    }
    for (int h = 0; h < NPSQT; h += 8) {
        __m256i p = _mm256_loadu_si256((const __m256i*)(psrc + h));
        for (int i = 0; i < nAdd; ++i) p = _mm256_add_epi32(p, _mm256_loadu_si256((const __m256i*)(prow(add[i]) + h)));
        for (int i = 0; i < nRem; ++i) p = _mm256_sub_epi32(p, _mm256_loadu_si256((const __m256i*)(prow(rem[i]) + h)));
        _mm256_storeu_si256((__m256i*)(pdst + h), p);
    }
#elif defined(JQ_SNN_V128)
    apply(ftW, psqt, W, dst, pdst, src, psrc, add, nAdd, rem, nRem);
    for (int c = 0; c < Wt; c += 8) v128::st16(dst + c, v128::add16(v128::ld16(dst + c), v128::ld16(E + c)));
#else
    for (int c = 0; c < W; ++c) {
        uint16_t v = uint16_t(src[c]);
        for (int i = 0; i < nAdd; ++i) v = uint16_t(v + uint16_t(wrow(add[i])[c]));
        for (int i = 0; i < nRem; ++i) v = uint16_t(v - uint16_t(wrow(rem[i])[c]));
        if (c < Wt) v = uint16_t(v + uint16_t(E[c]));
        dst[c] = int16_t(v);
    }
    for (int b = 0; b < NPSQT; ++b) {
        uint32_t v = uint32_t(psrc[b]);
        for (int i = 0; i < nAdd; ++i) v += uint32_t(prow(add[i])[b]);
        for (int i = 0; i < nRem; ++i) v -= uint32_t(prow(rem[i])[b]);
        pdst[b] = int32_t(v);
    }
#endif
}

#if defined(JQ_SNN_AVX2) || defined(JQ_SNN_V128)
// kNnzLut.v[m]: the indices of the set bits of the 8-bit mask m, ascending, padded with zeros.
struct NnzLut {
    alignas(16) uint16_t v[256][8];
    constexpr NnzLut() : v{} {
        for (int m = 0; m < 256; ++m) {
            int k = 0;
            for (int b = 0; b < 8; ++b)
                if (m >> b & 1) v[m][k++] = uint16_t(b);
        }
    }
};
inline constexpr NnzLut kNnzLut{};
#endif

// W -> 16 over the nonzero 4-byte input chunks (exact: inputs <= 127, so maddubs pairs cannot saturate).
inline void affine_l1(const uint8_t* in, int W, const int32_t* bias, const uint8_t* wb, int32_t* out,
                      const uint16_t* preNnz = nullptr, int preN = 0) {
    const int32_t* in32 = reinterpret_cast<const int32_t*>(in);
#if defined(JQ_SNN_VNNI)
    __m512i a0 = _mm512_loadu_si512(bias), a1 = _mm512_setzero_si512(), a2 = _mm512_setzero_si512(),
            a3 = _mm512_setzero_si512();
    alignas(64) uint16_t nnz[MAX_W / 4];
    int n = 0;
    for (int blk = 0; blk < W; blk += 64) {
        uint32_t mk = _mm512_test_epi32_mask(_mm512_loadu_si512(in + blk), _mm512_loadu_si512(in + blk));
        while (mk) { nnz[n++] = uint16_t(blk / 4 + __builtin_ctz(mk)); mk &= mk - 1; }
    }
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32(in32[nnz[i]]), _mm512_load_si512(wb + std::size_t(nnz[i]) * 64));
        a1 = _mm512_dpbusd_epi32(a1, _mm512_set1_epi32(in32[nnz[i + 1]]), _mm512_load_si512(wb + std::size_t(nnz[i + 1]) * 64));
        a2 = _mm512_dpbusd_epi32(a2, _mm512_set1_epi32(in32[nnz[i + 2]]), _mm512_load_si512(wb + std::size_t(nnz[i + 2]) * 64));
        a3 = _mm512_dpbusd_epi32(a3, _mm512_set1_epi32(in32[nnz[i + 3]]), _mm512_load_si512(wb + std::size_t(nnz[i + 3]) * 64));
    }
    for (; i < n; ++i)
        a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32(in32[nnz[i]]), _mm512_load_si512(wb + std::size_t(nnz[i]) * 64));
    _mm512_storeu_si512(out, _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3)));
#elif defined(JQ_SNN_AVX2)
    // Sparse: list the nonzero 4-byte chunks branch-free (8 per 32-byte block via a lookup table), then accumulate
    // only those. Exact: every per-chunk product is the same; int32 sums are order-independent (wrapping).
    alignas(32) uint16_t nnzBuf[MAX_W / 4 + 16];
    const uint16_t* nnz = preNnz;
    int n = preN;
    if (!nnz) {
        nnz = nnzBuf;
        uint16_t* nb = nnzBuf;
        const __m256i zero = _mm256_setzero_si256();
        const __m128i eight = _mm_set1_epi16(8);
        __m128i base = _mm_setzero_si128();
        for (int blk = 0; blk < W; blk += 32) {
            const __m256i v = _mm256_loadu_si256((const __m256i*)(in + blk));
            const unsigned mk = ~unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, zero)))) & 0xFFu;
            _mm_storeu_si128((__m128i*)(nb + n), _mm_add_epi16(_mm_load_si128((const __m128i*)kNnzLut.v[mk]), base));
            n += __builtin_popcount(mk);
            base = _mm_add_epi16(base, eight);
        }
    }
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i a0 = _mm256_loadu_si256((const __m256i*)bias), a1 = _mm256_loadu_si256((const __m256i*)(bias + 8));
    __m256i b0 = _mm256_setzero_si256(), b1 = _mm256_setzero_si256();
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        const int c0 = nnz[i], c1 = nnz[i + 1];
        const __m256i x0 = _mm256_set1_epi32(in32[c0]), x1 = _mm256_set1_epi32(in32[c1]);
        const uint8_t* w0 = wb + std::size_t(c0) * 64;
        const uint8_t* w1 = wb + std::size_t(c1) * 64;
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, _mm256_load_si256((const __m256i*)w0)), ones));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, _mm256_load_si256((const __m256i*)(w0 + 32))), ones));
        b0 = _mm256_add_epi32(b0, _mm256_madd_epi16(_mm256_maddubs_epi16(x1, _mm256_load_si256((const __m256i*)w1)), ones));
        b1 = _mm256_add_epi32(b1, _mm256_madd_epi16(_mm256_maddubs_epi16(x1, _mm256_load_si256((const __m256i*)(w1 + 32))), ones));
    }
    if (i < n) {
        const int c0 = nnz[i];
        const __m256i x0 = _mm256_set1_epi32(in32[c0]);
        const uint8_t* w0 = wb + std::size_t(c0) * 64;
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, _mm256_load_si256((const __m256i*)w0)), ones));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, _mm256_load_si256((const __m256i*)(w0 + 32))), ones));
    }
    _mm256_storeu_si256((__m256i*)out, _mm256_add_epi32(a0, b0));
    _mm256_storeu_si256((__m256i*)(out + 8), _mm256_add_epi32(a1, b1));
#else
    for (int o = 0; o < L1; ++o) out[o] = bias[o];
    for (int ch = 0; ch < W / 4; ++ch) {
        if (!in32[ch]) continue;
        const int8_t* w = reinterpret_cast<const int8_t*>(wb + std::size_t(ch) * 64);
        for (int o = 0; o < L1; ++o)
            for (int l = 0; l < 4; ++l) out[o] += int32_t(w[o * 4 + l]) * in[ch * 4 + l];
    }
#endif
}

// One layer stack on the W uint8 inputs x: returns out + skip (the "positional" term, x16 units of nn).
inline int32_t propagate(int W, const int32_t* l1b, const uint8_t* l1w, const int32_t* l2b, const int8_t* w2, int32_t fb,
                         const int8_t* fw, const uint8_t* x, const uint16_t* preNnz = nullptr, int preN = 0) {
    alignas(64) int32_t first[L1];
    affine_l1(x, W, l1b, l1w, first, preNnz, preN);
    alignas(32) uint8_t s[32];
    int32_t out = fb;
#ifdef JQ_SNN_AVX2
    {
        // activations: s[0..14] = min(127, clamp(f)^2 >> 19), s[15..29] = clamp(f >> 6, 0, 127), s[30..31] = 0
        alignas(32) int32_t sq[16], cr[16];
        const __m256i lo = _mm256_set1_epi32(-32768), hi = _mm256_set1_epi32(32767), m127 = _mm256_set1_epi32(127),
                      z = _mm256_setzero_si256();
        for (int h = 0; h < 16; h += 8) {
            const __m256i f = _mm256_load_si256((const __m256i*)(first + h));
            const __m256i c = _mm256_min_epi32(_mm256_max_epi32(f, lo), hi);
            _mm256_store_si256((__m256i*)(sq + h), _mm256_min_epi32(_mm256_srli_epi32(_mm256_mullo_epi32(c, c), 19), m127));
            _mm256_store_si256((__m256i*)(cr + h), _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(f, 6), z), m127));
        }
        for (int i = 0; i < 15; ++i) { s[i] = uint8_t(sq[i]); s[15 + i] = uint8_t(cr[i]); }
        s[30] = s[31] = 0;
        // L2 32 -> 32: [8 chunks][32 outputs][4] layout, maddubs exact (inputs <= 127)
        const int32_t* s32 = reinterpret_cast<const int32_t*>(s);
        const __m256i ones = _mm256_set1_epi16(1);
        __m256i acc[4];
        for (int k = 0; k < 4; ++k) acc[k] = _mm256_load_si256((const __m256i*)(l2b + 8 * k));
        for (int ch = 0; ch < 8; ++ch) {
            if (!s32[ch]) continue;
            const __m256i xv = _mm256_set1_epi32(s32[ch]);
            const int8_t* w = w2 + ch * 128;
            for (int k = 0; k < 4; ++k)
                acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_load_si256((const __m256i*)(w + 32 * k))), ones));
        }
        __m256i o = _mm256_setzero_si256();
        for (int k = 0; k < 4; ++k) {
            const __m256i h = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(acc[k], 6), z), m127);
            const __m256i fwv = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(fw + 8 * k)));
            o = _mm256_add_epi32(o, _mm256_mullo_epi32(h, fwv));
        }
        __m128i r = _mm_add_epi32(_mm256_castsi256_si128(o), _mm256_extracti128_si256(o, 1));
        r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 0x4E));
        r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 0xB1));
        out += _mm_cvtsi128_si32(r);
    }
#else
    std::memset(s, 0, sizeof s);
    for (int i = 0; i < 15; ++i) {
        const int64_t c = std::clamp(first[i], -32768, 32767);
        s[i] = uint8_t(std::min<int64_t>(127, (c * c) >> 19));
        s[15 + i] = uint8_t(std::clamp(first[i] >> 6, 0, 127));
    }
    int32_t second[L2];
    for (int o = 0; o < L2; ++o) second[o] = l2b[o];
    for (int ch = 0; ch < 8; ++ch)
        for (int o = 0; o < L2; ++o)
            for (int l = 0; l < 4; ++l) second[o] += int32_t(w2[(ch * L2 + o) * 4 + l]) * s[ch * 4 + l];
    for (int i = 0; i < L2; ++i) out += int32_t(fw[i]) * std::clamp(second[i] >> 6, 0, 127);
#endif
    const int64_t forward = int64_t(first[15]) * (600 * 16) / (127 * 64);
    return int32_t(int64_t(out) + forward);
}


// x[k*W/2 + i] = clamp(v_k[i], 0, 127) * clamp(v_k[i + W/2], 0, 127) >> 7 for k = 0 (us), 1 (them).
inline void pairwise(const int16_t* us, const int16_t* them, int W, uint8_t* x) {
    const int H = W / 2;
    for (int k = 0; k < 2; ++k) {
        const int16_t* v = k ? them : us;
        uint8_t* o = x + k * H;
#ifdef JQ_SNN_AVX2
        const __m256i zero = _mm256_setzero_si256(), mx = _mm256_set1_epi16(127);
        for (int i = 0; i < H; i += 32) {
            __m256i a0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)(v + i)), zero), mx);
            __m256i a1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)(v + i + 16)), zero), mx);
            __m256i b0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)(v + H + i)), zero), mx);
            __m256i b1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)(v + H + i + 16)), zero), mx);
            __m256i p0 = _mm256_srli_epi16(_mm256_mullo_epi16(a0, b0), 7);
            __m256i p1 = _mm256_srli_epi16(_mm256_mullo_epi16(a1, b1), 7);
            _mm256_store_si256((__m256i*)(o + i), _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), 0xD8));
        }
#else
        for (int i = 0; i < H; ++i) {
            const int a = std::clamp<int>(v[i], 0, 127), b = std::clamp<int>(v[i + H], 0, 127);
            o[i] = uint8_t((a * b) >> 7);
        }
#endif
    }
}

#if defined(JQ_SNN_AVX2) || defined(JQ_SNN_V128)
#define JQ_PAIRWISE_NNZ 1
// pairwise() that also lists the nonzero 4-byte output chunks (ascending), as affine_l1 would; returns the count.
inline int pairwise_nnz(const int16_t* us, const int16_t* them, int W, uint8_t* x, uint16_t* nnz) {
    const int H = W / 2;
#if defined(JQ_SNN_V128)
    int n = 0, base = 0;
    for (int k = 0; k < 2; ++k) {
        const int16_t* v = k ? them : us;
        uint8_t* o = x + k * H;
        for (int i = 0; i < H; i += 32) {
            const v128::U8 r0 = v128::narrow(v128::pw(v128::ld16(v + i), v128::ld16(v + H + i)),
                                             v128::pw(v128::ld16(v + i + 8), v128::ld16(v + H + i + 8)));
            const v128::U8 r1 = v128::narrow(v128::pw(v128::ld16(v + i + 16), v128::ld16(v + H + i + 16)),
                                             v128::pw(v128::ld16(v + i + 24), v128::ld16(v + H + i + 24)));
            v128::stu8(o + i, r0);
            v128::stu8(o + i + 16, r1);
            const unsigned mk = v128::nz4(r0) | v128::nz4(r1) << 4;
            v128::lut_store(nnz + n, kNnzLut.v[mk], base);
            n += __builtin_popcount(mk);
            base += 8;
        }
    }
    return n;
#elif defined(JQ_SNN_AVX512)
    // isa 2026-10-30: 64 output bytes per step; nonzero 4-byte chunks from one 16-bit test mask (VBMI2: compress the
    // chunk indices, else two 8-bit table lookups). Same bytes, same ascending list as the AVX2 code.
    const __m512i z = _mm512_setzero_si512(), mx = _mm512_set1_epi16(127), perm = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
#if defined(__AVX512VBMI2__)
    const __m256i iota = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#endif
    int n = 0, base = 0;
    for (int k = 0; k < 2; ++k) {
        const int16_t* v = k ? them : us;
        uint8_t* o = x + k * H;
        for (int i = 0; i < H; i += 64) {
            const __m512i a0 = _mm512_min_epi16(_mm512_max_epi16(_mm512_loadu_si512(v + i), z), mx);
            const __m512i a1 = _mm512_min_epi16(_mm512_max_epi16(_mm512_loadu_si512(v + i + 32), z), mx);
            const __m512i b0 = _mm512_min_epi16(_mm512_max_epi16(_mm512_loadu_si512(v + H + i), z), mx);
            const __m512i b1 = _mm512_min_epi16(_mm512_max_epi16(_mm512_loadu_si512(v + H + i + 32), z), mx);
            const __m512i p0 = _mm512_srli_epi16(_mm512_mullo_epi16(a0, b0), 7);
            const __m512i p1 = _mm512_srli_epi16(_mm512_mullo_epi16(a1, b1), 7);
            const __m512i r = _mm512_permutexvar_epi64(perm, _mm512_packus_epi16(p0, p1));
            _mm512_storeu_si512(o + i, r);
            const unsigned mk = _mm512_test_epi32_mask(r, r);
#if defined(__AVX512VBMI2__)
            _mm256_storeu_si256((__m256i*)(nnz + n), _mm256_maskz_compress_epi16(__mmask16(mk), _mm256_add_epi16(iota, _mm256_set1_epi16(int16_t(base)))));
#else
            const __m128i bv = _mm_set1_epi16(int16_t(base));
            _mm_storeu_si128((__m128i*)(nnz + n), _mm_add_epi16(_mm_load_si128((const __m128i*)kNnzLut.v[mk & 0xFF]), bv));
            const int n0 = __builtin_popcount(mk & 0xFF);
            _mm_storeu_si128((__m128i*)(nnz + n + n0), _mm_add_epi16(_mm_load_si128((const __m128i*)kNnzLut.v[mk >> 8]), _mm_add_epi16(bv, _mm_set1_epi16(8))));
#endif
            n += __builtin_popcount(mk);
            base += 16;
        }
    }
    return n;
#else
    const __m256i zero = _mm256_setzero_si256(), mx = _mm256_set1_epi16(127);
    const __m128i eight = _mm_set1_epi16(8);
    __m128i base = _mm_setzero_si128();
    int n = 0;
    for (int k = 0; k < 2; ++k) {
        const int16_t* v = k ? them : us;
        uint8_t* o = x + k * H;
        for (int i = 0; i < H; i += 32) {
            __m256i a0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)(v + i)), zero), mx);
            __m256i a1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)(v + i + 16)), zero), mx);
            __m256i b0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)(v + H + i)), zero), mx);
            __m256i b1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)(v + H + i + 16)), zero), mx);
            __m256i p0 = _mm256_srli_epi16(_mm256_mullo_epi16(a0, b0), 7);
            __m256i p1 = _mm256_srli_epi16(_mm256_mullo_epi16(a1, b1), 7);
            const __m256i r = _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), 0xD8);
            _mm256_store_si256((__m256i*)(o + i), r);
            const unsigned mk = ~unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(r, zero)))) & 0xFFu;
            _mm_storeu_si128((__m128i*)(nnz + n), _mm_add_epi16(_mm_load_si128((const __m128i*)kNnzLut.v[mk]), base));
            n += __builtin_popcount(mk);
            base = _mm_add_epi16(base, eight);
        }
    }
    return n;
#endif
}
#endif

}  // namespace jq::nnk
