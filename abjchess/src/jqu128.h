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
// jq_u128: unsigned __int128 where the compiler has it; otherwise (32-bit ARM, armeabi-v7a) an exact two-word
// emulation with the same modular semantics, so every bitboard result is bit-identical across targets.
#pragma once
#include <cstdint>
#if defined(__SIZEOF_INT128__)
typedef unsigned __int128 jq_u128;
#else
struct jq_u128 {
    uint64_t lo, hi;
    constexpr jq_u128() : lo(0), hi(0) {}
    constexpr jq_u128(uint64_t l) : lo(l), hi(0) {}
    constexpr jq_u128(int l) : lo(uint64_t(int64_t(l))), hi(l < 0 ? ~0ull : 0) {}
    constexpr jq_u128(unsigned l) : lo(l), hi(0) {}
    constexpr jq_u128(long l) : lo(uint64_t(int64_t(l))), hi(l < 0 ? ~0ull : 0) {}
    constexpr jq_u128(unsigned long l) : lo(l), hi(0) {}
    constexpr jq_u128(long long l) : lo(uint64_t(l)), hi(l < 0 ? ~0ull : 0) {}
    constexpr jq_u128(uint64_t h, uint64_t l, int) : lo(l), hi(h) {}
    constexpr explicit operator bool() const { return (lo | hi) != 0; }
    constexpr explicit operator uint64_t() const { return lo; }
    constexpr explicit operator uint32_t() const { return uint32_t(lo); }
    constexpr explicit operator int() const { return int(lo); }
    constexpr explicit operator uint16_t() const { return uint16_t(lo); }
    constexpr explicit operator uint8_t() const { return uint8_t(lo); }
    friend constexpr jq_u128 operator&(jq_u128 a, jq_u128 b) { return jq_u128(a.hi & b.hi, a.lo & b.lo, 0); }
    friend constexpr jq_u128 operator|(jq_u128 a, jq_u128 b) { return jq_u128(a.hi | b.hi, a.lo | b.lo, 0); }
    friend constexpr jq_u128 operator^(jq_u128 a, jq_u128 b) { return jq_u128(a.hi ^ b.hi, a.lo ^ b.lo, 0); }
    constexpr jq_u128 operator~() const { return jq_u128(~hi, ~lo, 0); }
    constexpr jq_u128 operator-() const { return jq_u128(0) - *this; }
    friend constexpr jq_u128 operator+(jq_u128 a, jq_u128 b) { uint64_t l = a.lo + b.lo; return jq_u128(a.hi + b.hi + (l < a.lo), l, 0); }
    friend constexpr jq_u128 operator-(jq_u128 a, jq_u128 b) { uint64_t l = a.lo - b.lo; return jq_u128(a.hi - b.hi - (a.lo < b.lo), l, 0); }
    friend constexpr jq_u128 operator*(jq_u128 a, jq_u128 b) {  // low 128 bits
        uint64_t a0 = uint32_t(a.lo), a1 = a.lo >> 32, b0 = uint32_t(b.lo), b1 = b.lo >> 32;
        uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
        uint64_t mid = (p00 >> 32) + uint32_t(p01) + uint32_t(p10);
        uint64_t lo = (mid << 32) | uint32_t(p00);
        uint64_t hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32) + a.hi * b.lo + a.lo * b.hi;
        return jq_u128(hi, lo, 0);
    }
    friend constexpr jq_u128 operator<<(jq_u128 a, int s) {
        s &= 127; if (!s) return a;
        if (s >= 64) return jq_u128(a.lo << (s - 64), 0, 0);
        return jq_u128((a.hi << s) | (a.lo >> (64 - s)), a.lo << s, 0);
    }
    friend constexpr jq_u128 operator>>(jq_u128 a, int s) {
        s &= 127; if (!s) return a;
        if (s >= 64) return jq_u128(0, a.hi >> (s - 64), 0);
        return jq_u128(a.hi >> s, (a.lo >> s) | (a.hi << (64 - s)), 0);
    }
    friend constexpr bool operator==(jq_u128 a, jq_u128 b) { return a.lo == b.lo && a.hi == b.hi; }
    friend constexpr bool operator!=(jq_u128 a, jq_u128 b) { return !(a == b); }
    friend constexpr bool operator<(jq_u128 a, jq_u128 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
    constexpr bool operator!() const { return (lo | hi) == 0; }
    constexpr jq_u128& operator&=(jq_u128 b) { return *this = *this & b; }
    constexpr jq_u128& operator|=(jq_u128 b) { return *this = *this | b; }
    constexpr jq_u128& operator^=(jq_u128 b) { return *this = *this ^ b; }
    constexpr jq_u128& operator+=(jq_u128 b) { return *this = *this + b; }
    constexpr jq_u128& operator-=(jq_u128 b) { return *this = *this - b; }
    constexpr jq_u128& operator<<=(int s) { return *this = *this << s; }
    constexpr jq_u128& operator>>=(int s) { return *this = *this >> s; }
};
#endif
