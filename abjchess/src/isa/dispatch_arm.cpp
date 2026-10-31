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
// Universal ARM entry (isa 2026-10-30), Android / Linux: picks the tier from getauxval(AT_HWCAP) (kernel-reported CPU
// features), runs that tier's static constructors, then its main. Override: ABJ_TIER=<name>; "--print-tier".
#include <sys/auxv.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__aarch64__)
#define JQ_TIERS(X) X(armv8_dotprod, "armv8-dotprod") X(armv8, "armv8")
static bool tier_ok(int i) {
    const unsigned long hw = getauxval(AT_HWCAP);
    if (i == 0) return (hw & (1ul << 20)) != 0 && (hw & (1ul << 1)) != 0;  // HWCAP_ASIMDDP, HWCAP_ASIMD
    return true;
}
#else
#define JQ_TIERS(X) X(armv7_neon, "armv7-neon") X(armv7, "armv7")
static bool tier_ok(int i) {
    const unsigned long hw = getauxval(AT_HWCAP);
    if (i == 0) return (hw & (1ul << 12)) != 0;  // HWCAP_NEON
    return true;
}
#endif

typedef void (*InitFn)(int, char**, char**);
#define DECL(id, name) extern "C" int abj_main_##id(int, char**); \
    extern "C" InitFn __start_jqinit_##id[] __attribute__((weak)); extern "C" InitFn __stop_jqinit_##id[] __attribute__((weak));
#define ROW(id, name) {name, abj_main_##id, __start_jqinit_##id, __stop_jqinit_##id},
JQ_TIERS(DECL)
struct Tier { const char* name; int (*main)(int, char**); InitFn* ib; InitFn* ie; };
static const Tier kTiers[] = {JQ_TIERS(ROW)};
static const int kN = sizeof kTiers / sizeof kTiers[0];

int main(int argc, char** argv, char** envp) {
    int best = kN - 1;
    for (int i = 0; i < kN; ++i) if (tier_ok(i)) { best = i; break; }
    int t = best;
    if (const char* e = std::getenv("ABJ_TIER")) {
        int k = -1;
        for (int i = 0; i < kN; ++i) if (!std::strcmp(e, kTiers[i].name)) k = i;
        if (k < 0) std::fprintf(stderr, "ABJ_TIER=%s: unknown tier, using %s\n", e, kTiers[t].name);
        else if (k < best) std::fprintf(stderr, "ABJ_TIER=%s: not supported by this CPU, using %s\n", e, kTiers[t].name);
        else t = k;
    }
    if (argc > 1 && !std::strcmp(argv[1], "--print-tier")) {
        std::printf("selected %s (best supported %s) hwcap 0x%lx\ntiers:", kTiers[t].name, kTiers[best].name, getauxval(AT_HWCAP));
        for (int i = 0; i < kN; ++i) std::printf(" %s%s", kTiers[i].name, i >= best ? "" : "(n/a)");
        std::printf("\n");
        return 0;
    }
    for (InitFn* p = kTiers[t].ib; p && p != kTiers[t].ie; ++p) (*p)(argc, argv, envp);
    return kTiers[t].main(argc, argv);
}
