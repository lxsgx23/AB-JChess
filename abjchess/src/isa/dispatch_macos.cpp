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
// macOS universal entry (isa, 2026-10-30): one executable per architecture containing every tier of that architecture
// (each tier's objects merged with 'ld -r', everything but its abj_main_<id> made local, its static initializers moved
// by isa/macho_initsec.py from __mod_init_func into the plain section __DATA,__jqi<k>), merged with lipo into one fat
// binary (x86_64 slice: x86-64-bmi2 + x86-64-avx2 only; a CPU without AVX2 gets a clear error, exit 2).
// At start: pick the best tier the CPU supports (x86_64: CPUID + XGETBV; arm64: sysctl FEAT_DotProd), run that
// tier's initializers, then its main. Same interface as the Linux dispatcher: ABJ_TIER=<tier> forces a tier (if the CPU
// supports it), --print-tier prints the choice.
#include <mach-o/getsect.h>
#include <mach-o/ldsyms.h>
#include <sys/sysctl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__)
#include <cpuid.h>
int abj_main_x86_64_bmi2(int, char**);
int abj_main_x86_64_avx2(int, char**);
struct Tier { const char* name; const char* sect; int (*main)(int, char**); };
// macOS x86_64: only the AVX2 tiers (the baseline x86-64 tier is not built for macOS; CPUs without AVX2 get an error)
static const Tier kTiers[] = {{"x86-64-bmi2", "__jqi0", abj_main_x86_64_bmi2},
                              {"x86-64-avx2", "__jqi1", abj_main_x86_64_avx2}};
static bool tier_ok(int i, bool verbose) {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
    const bool osxsave = c & (1u << 27), avx = c & (1u << 28), fma = c & (1u << 12), popcnt = c & (1u << 23);
    unsigned xa = 0, xd = 0;
    if (osxsave) __asm__ volatile("xgetbv" : "=a"(xa), "=d"(xd) : "c"(0));
    unsigned b7 = 0, c7 = 0, d7 = 0, a7 = 0;
    __get_cpuid_count(7, 0, &a7, &b7, &c7, &d7);
    const bool avx2 = b7 & (1u << 5), bmi1 = b7 & (1u << 3), bmi2 = b7 & (1u << 8);
    unsigned ea, eb, ec = 0, ed;
    const bool lzcnt = __get_cpuid(0x80000001u, &ea, &eb, &ec, &ed) && (ec & (1u << 5));
    if (verbose && i == 0)
        std::printf("osxsave %d xcr0 %#x avx %d fma %d popcnt %d avx2 %d bmi1 %d bmi2 %d lzcnt %d\n", osxsave, xa, avx, fma, popcnt, avx2, bmi1, bmi2, lzcnt);
    const bool base = osxsave && avx && fma && popcnt && (xa & 6u) == 6u && avx2 && bmi1 && lzcnt;
    return base && (i == 1 || bmi2);
}
#else
int abj_main_armv8_dotprod(int, char**);
int abj_main_armv8(int, char**);
struct Tier { const char* name; const char* sect; int (*main)(int, char**); };
static const Tier kTiers[] = {{"armv8-dotprod", "__jqi0", abj_main_armv8_dotprod}, {"armv8", "__jqi1", abj_main_armv8}};
static bool tier_ok(int i, bool verbose) {
    int v = 0; size_t n = sizeof v;
    const bool dp = sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &n, nullptr, 0) == 0 && v == 1;
    if (verbose && i == 0) std::printf("FEAT_DotProd %d\n", int(dp));
    return i == 1 || dp;
}
#endif
static const int kN = int(sizeof kTiers / sizeof kTiers[0]);

int main(int argc, char** argv, char** envp) {
    const bool print = argc > 1 && !std::strcmp(argv[1], "--print-tier");
    int best = -1;
    for (int i = 0; i < kN; ++i) if (tier_ok(i, print && i == 0)) { best = i; break; }
    if (best < 0) {  // x86_64 without AVX2 (pre-Haswell Intel Mac): no tier can run
        std::fprintf(stderr, "ERROR: this engine needs Apple Silicon or an Intel Mac with AVX2/FMA/BMI1/LZCNT (Haswell, 2013 or "
                             "newer); this CPU lacks them. No search is possible.\n");
        if (print) std::printf("selected none (CPU unsupported)\n");
        return 2;
    }
    int t = best;
    if (const char* e = std::getenv("ABJ_TIER"); e && *e) {
        int k = -1;
        for (int i = 0; i < kN; ++i) if (!std::strcmp(e, kTiers[i].name)) k = i;
        if (k < 0) std::fprintf(stderr, "ABJ_TIER=%s: unknown tier, using %s\n", e, kTiers[t].name);
        else if (!tier_ok(k, false)) std::fprintf(stderr, "ABJ_TIER=%s: not supported by this CPU, using %s\n", e, kTiers[t].name);
        else t = k;
    }
    if (print) {
        std::printf("selected %s (best supported %s)\ntiers:", kTiers[t].name, kTiers[best].name);
        for (int i = 0; i < kN; ++i) std::printf(" %s%s", kTiers[i].name, tier_ok(i, false) ? "" : "(n/a)");
        std::printf("\n");
        return 0;
    }
    // the selected tier's static initializers (dyld ran none of the tiers' ones), in object order
    unsigned long size = 0;
    uint8_t* p = getsectiondata(&_mh_execute_header, "__DATA", kTiers[t].sect, &size);
    if (!p) p = getsectiondata(&_mh_execute_header, "__DATA_CONST", kTiers[t].sect, &size);
    using Init = void (*)(int, char**, char**);
    for (unsigned long o = 0; p && o + sizeof(Init) <= size; o += sizeof(Init)) {
        Init f;
        std::memcpy(&f, p + o, sizeof f);
        f(argc, argv, envp);
    }
    return kTiers[t].main(argc, argv);
}
