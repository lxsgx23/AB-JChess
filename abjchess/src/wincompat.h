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
// Windows (mingw-w64) build shim, force-included with -include wincompat.h (Windows release build only).
// MSVCRT/UCRT have no aligned_alloc: memory from _aligned_malloc must be released with _aligned_free.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <malloc.h>
inline void* jq_aligned_alloc(std::size_t align, std::size_t size) { return _aligned_malloc(size ? size : 1, align); }
inline void jq_aligned_free(void* p) { _aligned_free(p); }
// Paths (EvalFile) arrive as bytes on stdin: UTF-8 (most GUIs) or the ANSI code page (e.g. GBK). The narrow CRT fopen
// would interpret them in the ANSI code page only, so non-ASCII UTF-8 paths (Chinese folders) could not be opened.
// jq_win_fopen: valid UTF-8 -> _wfopen; otherwise the ANSI code page; plain ASCII unchanged.
#include <cstdio>
#include <cwchar>
#include <string>
// declared here instead of <windows.h> (its macros min/max/ERROR/near/far would leak into every source file)
extern "C" __declspec(dllimport) int __stdcall MultiByteToWideChar(unsigned int, unsigned long, const char*, int, wchar_t*, int);
inline FILE* jq_win_fopen(const char* path, const char* mode) {
    auto wide = [](const char* s, unsigned cp, unsigned long flags, std::wstring& w) {
        const int n = MultiByteToWideChar(cp, flags, s, -1, nullptr, 0);
        if (n <= 0) return false;
        w.assign(std::size_t(n), L'\0');
        return MultiByteToWideChar(cp, flags, s, -1, &w[0], n) == n;
    };
    std::wstring wp, wm;
    if (!wide(mode, 65001u, 0, wm)) return std::fopen(path, mode);
    if (wide(path, 65001u, 8ul, wp) || wide(path, 0u, 0, wp)) return _wfopen(wp.c_str(), wm.c_str());
    return std::fopen(path, mode);
}
namespace std { using ::jq_win_fopen; }
#define fopen jq_win_fopen
