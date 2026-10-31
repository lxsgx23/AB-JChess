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
// Minimal MD5 (RFC 1321) for reporting the identity of the loaded net (decoded raw bytes).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace jq {
inline std::string md5_hex(const uint8_t* data, std::size_t n) {
    static const uint32_t K[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af,
        0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
        0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244, 0x432aff97,
        0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
    static const int R[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
                              5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21,
                              6, 10, 15, 21, 6, 10, 15, 21};
    uint32_t h[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    auto block = [&](const uint8_t* b) {
        uint32_t w[16];
        for (int i = 0; i < 16; ++i) w[i] = uint32_t(b[4 * i]) | uint32_t(b[4 * i + 1]) << 8 | uint32_t(b[4 * i + 2]) << 16 | uint32_t(b[4 * i + 3]) << 24;
        uint32_t a = h[0], bb = h[1], c = h[2], d = h[3];
        for (int i = 0; i < 64; ++i) {
            uint32_t f; int g;
            if (i < 16) { f = (bb & c) | (~bb & d); g = i; }
            else if (i < 32) { f = (d & bb) | (~d & c); g = (5 * i + 1) % 16; }
            else if (i < 48) { f = bb ^ c ^ d; g = (3 * i + 5) % 16; }
            else { f = c ^ (bb | ~d); g = (7 * i) % 16; }
            const uint32_t t = d; d = c; c = bb;
            const uint32_t x = a + f + K[i] + w[g];
            bb = bb + ((x << R[i]) | (x >> (32 - R[i])));
            a = t;
        }
        h[0] += a; h[1] += bb; h[2] += c; h[3] += d;
    };
    std::size_t i = 0;
    for (; i + 64 <= n; i += 64) block(data + i);
    uint8_t tail[128] = {};
    const std::size_t r = n - i;
    std::memcpy(tail, data + i, r);
    tail[r] = 0x80;
    const std::size_t tl = r < 56 ? 64 : 128;
    const uint64_t bits = uint64_t(n) * 8;
    for (int k = 0; k < 8; ++k) tail[tl - 8 + k] = uint8_t(bits >> (8 * k));
    block(tail);
    if (tl == 128) block(tail + 64);
    static const char* hx = "0123456789abcdef";
    std::string out;
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 4; ++j) { const uint8_t v = uint8_t(h[k] >> (8 * j)); out += hx[v >> 4]; out += hx[v & 15]; }
    return out;
}
}  // namespace jq
