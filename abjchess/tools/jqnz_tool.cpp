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
// jqnz_tool: pack / unpack / verify compressed .hnn nets.
//   jqnz_tool leb  in.hnn out.jqnl      (Stockfish-style LEB128 container)
//   jqnz_tool z    in.hnn out.jqnz      (context-modelled range coding)
//   jqnz_tool d    in.(jqnl|jqnz) out.hnn
//   jqnz_tool test in.hnn               (both formats, round trip, sizes, timings)
#include "jqnz.h"  // from ../src (build: g++ -std=c++20 -O2 -I../src jqnz_tool.cpp)
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
static std::vector<uint8_t> rd(const char* p) { std::ifstream f(p, std::ios::binary); return {std::istreambuf_iterator<char>(f), {}}; }
static void wr(const char* p, const std::vector<uint8_t>& v) { std::ofstream f(p, std::ios::binary); f.write((const char*)v.data(), long(v.size())); }
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: jqnz_tool leb|z|d|test in [out]\n"); return 2; }
    const std::string mode = argv[1];
    std::vector<uint8_t> in = rd(argv[2]), out;
    std::string err;
    if (in.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    if (mode == "sections") {
        std::vector<size_t> sec;
        jqnz::encode_z(in, out, &err, &sec);
        const char* names[] = {"header+coefs", "ftB", "psqt", "ftW", "stacks", "trailer+flush"};
        size_t prev = 0;
        for (size_t i = 0; i < sec.size(); ++i) { std::printf("%-14s %8zu\n", names[i], sec[i] - prev); prev = sec[i]; }
        std::printf("total %zu\n", out.size());
        return 0;
    }
    if (mode == "test") {
        for (int k = 0; k < 2; ++k) {
            auto t0 = now();
            bool ok = k ? jqnz::encode_z(in, out, &err) : jqnz::encode_leb(in, out, &err);
            auto t1 = now();
            std::vector<uint8_t> back;
            ok = ok && jqnz::decode(out.data(), out.size(), back, &err);
            auto t2 = now();
            std::printf("%s: %zu -> %zu bytes (%.2f%%, ratio %.3f) encode %.1f ms decode %.1f ms roundtrip %s %s\n",
                        k ? "JQHNNZ01" : "JQHNNL01", in.size(), out.size(), 100.0 * out.size() / in.size(),
                        double(in.size()) / out.size(), ms(t0, t1), ms(t1, t2), ok && back == in ? "IDENTICAL" : "FAIL", err.c_str());
        }
        return 0;
    }
    if (argc < 4) return 2;
    bool ok = mode == "leb" ? jqnz::encode_leb(in, out, &err) : mode == "z" ? jqnz::encode_z(in, out, &err)
            : mode == "d" ? jqnz::decode(in.data(), in.size(), out, &err) : false;
    if (!ok) { std::fprintf(stderr, "failed: %s\n", err.c_str()); return 1; }
    if (mode != "d") {  // verify
        std::vector<uint8_t> back;
        if (!jqnz::decode(out.data(), out.size(), back, &err) || back != in) { std::fprintf(stderr, "verify failed %s\n", err.c_str()); return 1; }
    }
    wr(argv[3], out);
    std::printf("%s -> %s: %zu -> %zu bytes\n", argv[2], argv[3], in.size(), out.size());
    return 0;
}
