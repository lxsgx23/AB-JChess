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
// Aligned allocation for the large tables.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <sys/mman.h>
#endif

#if !defined(_WIN32)  // Windows gets these from wincompat.h (_aligned_malloc / _aligned_free)
// size rounded up to a multiple of align: C11 requires it and macOS's aligned_alloc returns NULL otherwise.
inline void* jq_aligned_alloc(std::size_t align, std::size_t size) {
    return std::aligned_alloc(align, (size ? size + align - 1 : align) / align * align);
}
inline void jq_aligned_free(void* p) { std::free(p); }
#endif

namespace jq {

// Zeroed block of at least `bytes`, aligned to 64. A block of half a megabyte or more is aligned to 2 MB, padded to a
// multiple of 2 MB and offered to the kernel as transparent huge pages (Linux keeps them in "madvise" mode by
// default): a probe into a table of hundreds of megabytes then costs a cache miss, not a cache miss plus a page walk.
// Release with std::free.
inline void* alloc_large(std::size_t bytes) {
    constexpr std::size_t HUGE_PAGE = std::size_t(2) << 20;
    const bool big = bytes >= HUGE_PAGE / 4;
    const std::size_t align = big ? HUGE_PAGE : 64;
    const std::size_t size = ((bytes ? bytes : 1) + align - 1) / align * align;
    void* p = jq_aligned_alloc(align, size);
    if (!p) return nullptr;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (big) madvise(p, size, MADV_HUGEPAGE);
#endif
    std::memset(p, 0, size);
    return p;
}

}  // namespace jq
