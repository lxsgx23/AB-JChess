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
// Generic small head for JQHNNUE1 version-3 nets (arch experiments 2/3, 2026-10-28): W -> L1 (L1-1 + skip) ->
// [sqr, crelu] (padded to L2IN) -> L2 -> crelu [, sqr = (h*h)>>7 if SQ2] -> 1 + skip. Integer semantics identical to
// nnkern::propagate for (16, 32, false); used only for version-3 files (version-1 nets keep the original kernel).
#pragma once
#include "nnkern.h"

namespace jq::nnk {

template <int L1_, int L2_, bool SQ2>
inline int32_t propagate_g(int W, const int32_t* l1b, const uint8_t* l1w, const int32_t* l2b, const int8_t* w2, int32_t fb,
                           const int8_t* fw, const uint8_t* x, const uint16_t* preNnz, int preN) {
    constexpr int N1 = L1_ - 1, L2IN = (2 * N1 + 31) / 32 * 32;
    alignas(64) int32_t first[L1_];
    const int32_t* in32 = reinterpret_cast<const int32_t*>(x);
#if defined(JQ_SNN_AVX2)
    {
        alignas(32) uint16_t nnzBuf[MAX_W / 4 + 16];
        const uint16_t* nnz = preNnz;
        int n = preN;
        if (!nnz) {
            n = 0;
            for (int ch = 0; ch < W / 4; ++ch) if (in32[ch]) nnzBuf[n++] = uint16_t(ch);
            nnz = nnzBuf;
        }
        constexpr int NV = L1_ / 8;  // ymm accumulators of 8 int32 outputs
        const __m256i ones = _mm256_set1_epi16(1);
        __m256i a[NV];
        for (int k = 0; k < NV; ++k) a[k] = _mm256_loadu_si256((const __m256i*)(l1b + 8 * k));
        for (int i = 0; i < n; ++i) {
            const int c = nnz[i];
            const __m256i xv = _mm256_set1_epi32(in32[c]);
            const uint8_t* w = l1w + std::size_t(c) * (L1_ * 4);
            for (int k = 0; k < NV; ++k)
                a[k] = _mm256_add_epi32(a[k], _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_load_si256((const __m256i*)(w + 32 * k))), ones));
        }
        for (int k = 0; k < NV; ++k) _mm256_store_si256((__m256i*)(first + 8 * k), a[k]);
    }
#else
    for (int o = 0; o < L1_; ++o) first[o] = l1b[o];
    for (int ch = 0; ch < W / 4; ++ch) {
        if (!in32[ch]) continue;
        const int8_t* w = reinterpret_cast<const int8_t*>(l1w + std::size_t(ch) * (L1_ * 4));
        for (int o = 0; o < L1_; ++o)
            for (int l = 0; l < 4; ++l) first[o] += int32_t(w[o * 4 + l]) * x[ch * 4 + l];
    }
#endif
    alignas(32) uint8_t s[L2IN] = {};
    for (int i = 0; i < N1; ++i) {
        const int64_t c = std::clamp(first[i], -32768, 32767);
        s[i] = uint8_t(std::min<int64_t>(127, (c * c) >> 19));
        s[N1 + i] = uint8_t(std::clamp(first[i] >> 6, 0, 127));
    }
    alignas(32) int32_t second[L2_];
#if defined(JQ_SNN_AVX2)
    {
        const int32_t* s32 = reinterpret_cast<const int32_t*>(s);
        const __m256i ones = _mm256_set1_epi16(1);
        constexpr int NV = L2_ / 8;
        __m256i acc[NV];
        for (int k = 0; k < NV; ++k) acc[k] = _mm256_loadu_si256((const __m256i*)(l2b + 8 * k));
        for (int ch = 0; ch < L2IN / 4; ++ch) {
            if (!s32[ch]) continue;
            const __m256i xv = _mm256_set1_epi32(s32[ch]);
            const int8_t* w = w2 + ch * (L2_ * 4);
            for (int k = 0; k < NV; ++k)
                acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((const __m256i*)(w + 32 * k))), ones));
        }
        for (int k = 0; k < NV; ++k) _mm256_store_si256((__m256i*)(second + 8 * k), acc[k]);
    }
#else
    for (int o = 0; o < L2_; ++o) second[o] = l2b[o];
    for (int ch = 0; ch < L2IN / 4; ++ch)
        for (int o = 0; o < L2_; ++o)
            for (int l = 0; l < 4; ++l) second[o] += int32_t(w2[(ch * L2_ + o) * 4 + l]) * s[ch * 4 + l];
#endif
    int32_t out = fb;
    for (int i = 0; i < L2_; ++i) {
        const int32_t h = std::clamp(second[i] >> 6, 0, 127);
        out += int32_t(fw[i]) * h;
        if constexpr (SQ2) out += int32_t(fw[L2_ + i]) * ((h * h) >> 7);
    }
    const int64_t forward = int64_t(first[N1]) * (600 * 16) / (127 * 64);
    return int32_t(int64_t(out) + forward);
}

}  // namespace jq::nnk
