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
// Portable (non-BMI/LZCNT) replacements for the BMI1/BMI2/LZCNT intrinsics used by ABJchess, force-included
// (-include nobmi.h) by the tiers without them. Same results as the instructions (incl. zero inputs).
// JQ_NOBMI2_ONLY: the CPU has BMI1 + LZCNT (AVX2 tier), only the BMI2 ones (pext, bzhi) are replaced.
#pragma once
#include <cstdint>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <immintrin.h>
#endif
static inline unsigned jq_tzcnt_u32(unsigned x) { return x ? unsigned(__builtin_ctz(x)) : 32u; }
static inline unsigned jq_lzcnt_u32(unsigned x) { return x ? unsigned(__builtin_clz(x)) : 32u; }
static inline unsigned jq_blsr_u32(unsigned x) { return x & (x - 1u); }
static inline unsigned jq_bzhi_u32(unsigned x, unsigned n) { n &= 0xFFu; return n >= 32u ? x : (x & ((1u << n) - 1u)); }
static inline unsigned long long jq_pext_u64(unsigned long long x, unsigned long long m) {
    unsigned long long r = 0, bit = 1;
    while (m) { const unsigned long long low = m & (0ull - m); if (x & low) r |= bit; bit <<= 1; m &= m - 1; }
    return r;
}
#ifndef JQ_NOBMI2_ONLY
#define _tzcnt_u32 jq_tzcnt_u32
#define _lzcnt_u32 jq_lzcnt_u32
#define _blsr_u32 jq_blsr_u32
#endif
#define _bzhi_u32 jq_bzhi_u32
#define _pext_u64 jq_pext_u64
