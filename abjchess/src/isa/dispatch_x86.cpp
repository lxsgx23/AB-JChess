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
// Universal x86-64 entry (isa 2026-10-30): picks the best compiled tier at run time from CPUID + OS state (XCR0), then
// runs that tier's static constructors and main. Every tier is a complete, internalised copy of the engine (one LTO
// partition each), so no code of a higher tier can execute on a CPU that lacks its instructions.
// Override: environment ABJ_TIER=<tier name>; "--print-tier" prints the selected tier and the list.
#include <cpuid.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define JQ_TIERS(X) X(x86_64_avx512icl, "x86-64-avx512icl") X(x86_64_vnni512, "x86-64-vnni512") \
    X(x86_64_avx512, "x86-64-avx512") X(x86_64_avxvnni, "x86-64-avxvnni") X(x86_64_bmi2, "x86-64-bmi2") \
    X(x86_64_avx2, "x86-64-avx2") X(x86_64_sse41_popcnt, "x86-64-sse41-popcnt") X(x86_64_ssse3, "x86-64-ssse3") \
    X(x86_64_sse3_popcnt, "x86-64-sse3-popcnt") X(x86_64, "x86-64")

typedef void (*InitFn)(int, char**, char**);
#if defined(_WIN32)
// Windows (PE): each tier's .ctors was renamed to .data$jqi_<id>_m; ld sorts .data$* by name, so the pointers lie
// between the two null markers below (_a < _m < _z). GCC's .ctors run in reverse order.
#define DECL(id, name) extern "C" int abj_main_##id(int, char**); \
    __attribute__((section(".data$jqi_" #id "_a"), used, aligned(8))) InitFn jqi_b_##id[1] = {nullptr}; \
    __attribute__((section(".data$jqi_" #id "_z"), used, aligned(8))) InitFn jqi_e_##id[1] = {nullptr};
#define ROW(id, name) {name, abj_main_##id, jqi_b_##id + 1, jqi_e_##id},
#else
// Linux (ELF): each tier's .init_array was renamed to jqinit_<id>; the linker defines __start_/__stop_ for it.
#define DECL(id, name) extern "C" int abj_main_##id(int, char**); \
    extern "C" InitFn __start_jqinit_##id[] __attribute__((weak)); extern "C" InitFn __stop_jqinit_##id[] __attribute__((weak));
#define ROW(id, name) {name, abj_main_##id, __start_jqinit_##id, __stop_jqinit_##id},
#endif
JQ_TIERS(DECL)

struct Tier { const char* name; int (*main)(int, char**); InitFn* ib; InitFn* ie; };
static const Tier kTiers[] = {JQ_TIERS(ROW)};
static const int kN = sizeof kTiers / sizeof kTiers[0];

static uint64_t xgetbv0() { uint32_t a, d; __asm__ volatile("xgetbv" : "=a"(a), "=d"(d) : "c"(0)); return (uint64_t(d) << 32) | a; }

// Highest usable tier index into kTiers (kN - 1 = baseline x86-64).
static int detect(char* why, size_t whyN) {
    unsigned a, b, c, d;
    unsigned c1 = 0, d1 = 0, b7 = 0, c7 = 0, a71 = 0, ce1 = 0, maxl = 0, fam = 0, vendorAmd = 0;
    if (__get_cpuid(0, &maxl, &b, &c, &d)) vendorAmd = (b == 0x68747541);  // "Auth"enticAMD
    if (maxl >= 1 && __get_cpuid(1, &a, &b, &c1, &d1)) fam = ((a >> 8) & 0xF) + (((a >> 8) & 0xF) == 0xF ? ((a >> 20) & 0xFF) : 0);
    if (maxl >= 7) { __cpuid_count(7, 0, a, b7, c7, d); __cpuid_count(7, 1, a71, b, c, d); }
    unsigned maxe = __get_cpuid_max(0x80000000, nullptr);
    if (maxe >= 0x80000001) __get_cpuid(0x80000001, &a, &b, &ce1, &d);
    auto bit = [](unsigned r, int n) { return (r >> n & 1u) != 0; };
    const bool osx = bit(c1, 27);
    const uint64_t xcr0 = osx ? xgetbv0() : 0;
    const bool osAvx = osx && (xcr0 & 0x6) == 0x6, osZmm = osx && (xcr0 & 0xE6) == 0xE6;
    const bool sse2 = bit(d1, 26), sse3 = bit(c1, 0), ssse3 = bit(c1, 9), sse41 = bit(c1, 19), sse42 = bit(c1, 20),
               popcnt = bit(c1, 23), fma = bit(c1, 12), avx = bit(c1, 28) && osAvx, lzcnt = bit(ce1, 5);
    const bool bmi1 = bit(b7, 3), avx2 = bit(b7, 5) && avx, bmi2 = bit(b7, 8);
    const bool f = bit(b7, 16) && osZmm, dq = bit(b7, 17), ifma = bit(b7, 21), cd = bit(b7, 28), bw = bit(b7, 30), vl = bit(b7, 31);
    const bool vbmi = bit(c7, 1), vbmi2 = bit(c7, 6), gfni = bit(c7, 8), vaes = bit(c7, 9), vpclmul = bit(c7, 10),
               vnni = bit(c7, 11), bitalg = bit(c7, 12), vpopcntdq = bit(c7, 14), avxvnni = bit(a71, 4) && avx2;
    // pdep/pext are microcoded (very slow) on AMD before Zen 3 (families 0x15..0x17)
    const bool slowBmi2 = vendorAmd && fam >= 0x15 && fam <= 0x18;
    const bool tAvx2 = avx2 && fma && bmi1 && lzcnt && popcnt && sse42;
    const bool tBmi2 = tAvx2 && bmi2 && !slowBmi2;
    const bool tA512 = tBmi2 && f && bw && dq && vl && cd;
    const bool tVnni512 = tA512 && vnni;
    const bool tIcl = tVnni512 && ifma && vbmi && vbmi2 && vpopcntdq && bitalg && vpclmul && gfni && vaes;
    const bool ok[] = {tIcl, tVnni512, tA512, tBmi2 && avxvnni, tBmi2, tAvx2, sse41 && popcnt && ssse3 && sse3 && sse2,
                       ssse3 && sse3 && sse2, sse3 && popcnt && sse2, true};
    if (why) std::snprintf(why, whyN, "osxsave %d xcr0 0x%llx avx2 %d bmi2 %d slowBmi2 %d avx512f/bw/dq/vl/cd %d%d%d%d%d vnni %d avxvnni %d",
                           int(osx), (unsigned long long)xcr0, int(avx2), int(bmi2), int(slowBmi2), int(f), int(bw), int(dq), int(vl), int(cd), int(vnni), int(avxvnni));
    for (int i = 0; i < kN; ++i) if (ok[i]) return i;
    return kN - 1;
}

int main(int argc, char** argv, char** envp) {
    char why[256];
    const int best = detect(why, sizeof why);
    int t = best;
    if (const char* e = std::getenv("ABJ_TIER")) {
        int k = -1;
        for (int i = 0; i < kN; ++i) if (!std::strcmp(e, kTiers[i].name)) k = i;
        if (k < 0) std::fprintf(stderr, "ABJ_TIER=%s: unknown tier, using %s\n", e, kTiers[t].name);
        else if (k < best) std::fprintf(stderr, "ABJ_TIER=%s: not supported by this CPU/OS, using %s\n", e, kTiers[t].name);
        else t = k;
    }
    if (argc > 1 && !std::strcmp(argv[1], "--print-tier")) {
        std::printf("selected %s (best supported %s)\n%s\ntiers:", kTiers[t].name, kTiers[best].name, why);
        for (int i = 0; i < kN; ++i) std::printf(" %s%s", kTiers[i].name, i >= best ? "" : "(n/a)");
        std::printf("\n");
        return 0;
    }
#if defined(_WIN32)
    for (InitFn* p = kTiers[t].ie; p != kTiers[t].ib;) if (InitFn f = *--p) reinterpret_cast<void (*)()>(f)();
#else
    for (InitFn* p = kTiers[t].ib; p && p != kTiers[t].ie; ++p) (*p)(argc, argv, envp);
#endif
    return kTiers[t].main(argc, argv);
}
