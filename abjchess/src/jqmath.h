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
// jqmath.h (ISA/universal build): platform-independent, bit-exact double math for the search.
// The reference build (user zip, g++ -O3 -mavx2 -mfma, static glibc 2.41) evaluates the chance-node arithmetic with
// GCC-contracted FMAs and calls glibc's FMA variants of exp/log (__ieee754_exp_fma / __ieee754_log_fma, selected by
// ifunc on every AVX2 CPU). To give every ISA tier / OS (SSE2..AVX-512, Windows msvcrt/ucrt, Android bionic, NEON)
// exactly the same doubles:
//   * jqm::fma: hardware FMA where the tier has it (x86 -mfma, AArch64), else an exact software FMA (port of musl
//     src/math/fma.c, MIT licence) -- an FMA is correctly rounded by definition, so both give identical bits;
//   * jqm::log / jqm::exp: instruction-by-instruction transcriptions of glibc 2.41 __ieee754_log_fma /
//     __ieee754_exp_fma (algorithm and tables: ARM optimized-routines, MIT / Apache-2.0 WITH LLVM-exception), every
//     FMA of the compiled reference written as jqm::fma, compiled with -ffp-contract=off. Verified against glibc
//     (tools: jqmath_test.cpp).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace jqm {

inline uint64_t asu(double x) { uint64_t u; std::memcpy(&u, &x, 8); return u; }
inline double asd(uint64_t u) { double x; std::memcpy(&x, &u, 8); return x; }

#if defined(__FMA__) || defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_FEATURE_FMA)
#define JQM_HW_FMA 1
inline double fma(double x, double y, double z) { return __builtin_fma(x, y, z); }
#else
#define JQM_HW_FMA 0
namespace sw {
struct num { uint64_t m; int e; int sign; };
inline num normalize(double x) {
    uint64_t ix = asu(x);
    int e = int(ix >> 52);
    const int sign = e & 0x800;
    e &= 0x7ff;
    if (!e) { ix = asu(x * 0x1p63); e = int(ix >> 52 & 0x7ff); e = e ? e - 63 : 0x800; }
    ix &= (1ull << 52) - 1; ix |= 1ull << 52; ix <<= 1;
    e -= 0x3ff + 52 + 1;
    return {ix, e, sign};
}
inline void mul(uint64_t* hi, uint64_t* lo, uint64_t x, uint64_t y) {
    const uint64_t xlo = uint32_t(x), xhi = x >> 32, ylo = uint32_t(y), yhi = y >> 32;
    const uint64_t t1 = xlo * ylo, t2 = xlo * yhi + xhi * ylo, t3 = xhi * yhi;
    *lo = t1 + (t2 << 32);
    *hi = t3 + (t2 >> 32) + (t1 > *lo);
}
inline int clz64(uint64_t x) { return __builtin_clzll(x); }
}  // namespace sw
// Exact fused multiply-add (round to nearest), musl fma.c (Szabolcs Nagy), MIT licence.
__attribute__((noinline)) inline double fma_musl(double x, double y, double z) {
    constexpr int ZEROINFNAN = 0x7ff - 0x3ff - 52 - 1;
    sw::num nx = sw::normalize(x), ny = sw::normalize(y), nz = sw::normalize(z);
    if (nx.e >= ZEROINFNAN || ny.e >= ZEROINFNAN) return x * y + z;
    if (nz.e >= ZEROINFNAN) { if (nz.e > ZEROINFNAN) return x * y; return z; }
    uint64_t rhi, rlo, zhi, zlo;
    sw::mul(&rhi, &rlo, nx.m, ny.m);
    int e = nx.e + ny.e;
    int d = nz.e - e;
    if (d > 0) {
        if (d < 64) { zlo = nz.m << d; zhi = nz.m >> (64 - d); }
        else {
            zlo = 0; zhi = nz.m; e = nz.e - 64; d -= 64;
            if (d == 0) {}
            else if (d < 64) { rlo = rhi << (64 - d) | rlo >> d | uint64_t(!!(rlo << (64 - d))); rhi = rhi >> d; }
            else { rlo = 1; rhi = 0; }
        }
    } else {
        zhi = 0; d = -d;
        if (d == 0) zlo = nz.m;
        else if (d < 64) zlo = nz.m >> d | uint64_t(!!(nz.m << (64 - d)));
        else zlo = 1;
    }
    int sign = nx.sign ^ ny.sign;
    const int samesign = !(sign ^ nz.sign);
    int nonzero = 1;
    if (samesign) { rlo += zlo; rhi += zhi + (rlo < zlo); }
    else {
        const uint64_t t = rlo;
        rlo -= zlo; rhi = rhi - zhi - (t < rlo);
        if (rhi >> 63) { rlo = 0 - rlo; rhi = 0 - rhi - uint64_t(!!rlo); sign = !sign; }
        nonzero = !!rhi;
    }
    if (nonzero) { e += 64; d = sw::clz64(rhi) - 1; rhi = rhi << d | rlo >> (64 - d) | uint64_t(!!(rlo << d)); }
    else if (rlo) { d = sw::clz64(rlo) - 1; if (d < 0) rhi = rlo >> 1 | (rlo & 1); else rhi = rlo << d; }
    else return x * y + z;
    e -= d;
    int64_t i = int64_t(rhi);
    if (sign) i = -i;
    double r = double(i);
    if (e < -1022 - 62) {
        if (e == -1022 - 63) {
            double c = 0x1p63;
            if (sign) c = -c;
            if (r == c) { const float fltmin = float(0x0.ffffff8p-63 * 1.17549435082228750797e-38 * r); return 2.2250738585072014e-308 / 1.17549435082228750797e-38 * double(fltmin); }
            if (rhi << 53) {
                i = int64_t(rhi >> 1 | (rhi & 1) | 1ull << 62);
                if (sign) i = -i;
                r = double(i); r = 2 * r - c;
            }
        } else {
            d = 10;
            i = int64_t((rhi >> d | uint64_t(!!(rhi << (64 - d)))) << d);
            if (sign) i = -i;
            r = double(i);
        }
    }
    return std::scalbn(r, e);
}
// Fast exact FMA (isa 2026-10-30) for the usual exponent range, else fma_musl. Boldo & Melquiond, "Emulation of FMA and
// correctly rounded sums: proved algorithms using rounding to odd", IEEE TC 57(4), 2008: with (uh, ul) = a*b exactly
// (Dekker / Veltkamp), (th, tl) = c + uh exactly (TwoSum), RN(a*b + c) = RN(th + RO(tl + ul)), RO = round to odd
// (emulated: RN sum, TwoSum error, step one ulp toward the error when the RN result is even and inexact). Range guard:
// |a|, |b| in [2^-400, 2^401), c = 0 or |c| in [2^-600, 2^601): no overflow, no subnormal intermediate (also under
// FTZ / DAZ). a or b = 0 (finite): exactly c + a*b. Verified against hardware FMA (jqmath_test.cpp, fmatest).
__attribute__((always_inline)) inline double fma(double a, double b, double c) {
    const uint64_t ua = asu(a), ub = asu(b), uc = asu(c);
    const uint32_t ea = uint32_t(ua >> 52 & 0x7ff), eb = uint32_t(ub >> 52 & 0x7ff), ec = uint32_t(uc >> 52 & 0x7ff);
    if (__builtin_expect(ea - (0x3ffu - 400) <= 800u && eb - (0x3ffu - 400) <= 800u &&
                         (ec - (0x3ffu - 600) <= 1200u || (uc << 1) == 0), 1)) {
        const double C = 134217729.0;  // 2^27 + 1
        const double ta = C * a, ah = ta - (ta - a), al = a - ah;
        const double tb = C * b, bh = tb - (tb - b), bl = b - bh;
        const double uh = a * b;
        const double ul = ((ah * bh - uh) + ah * bl + al * bh) + al * bl;
        const double th = c + uh, z = th - c, tl = (c - (th - z)) + (uh - z);
        double v = tl + ul;
        const double w = v - tl, err = (tl - (v - w)) + (ul - w);
        if (err != 0.0) {
            uint64_t uv = asu(v);
            if (!(uv & 1)) { uv += ((uv ^ asu(err)) >> 63) ? ~uint64_t(0) : uint64_t(1); v = asd(uv); }
        }
        return th + v;
    }
    if (((ua << 1) == 0 && eb != 0x7ff) || ((ub << 1) == 0 && ea != 0x7ff)) return c + a * b;
    return fma_musl(a, b, c);
}
#endif

namespace tbl {
#include "jqmath_tables.inc"
}

// glibc 2.41 __ieee754_log_fma (sysdeps/ieee754/dbl-64/e_log.c built with -mfma -mavx2), every FMA explicit.
inline double log(double x) {
    const uint64_t* D = tbl::kLogData;
    auto c = [&](int i) { return asd(D[i]); };
    const double Ln2hi = c(0), Ln2lo = c(1);
    auto A = [&](int i) { return c(2 + i); };
    auto B = [&](int i) { return c(7 + i); };
    uint64_t ix = asu(x);
    const uint32_t top = uint32_t(ix >> 48);
    if (ix - 0x3fee000000000000ull < 0x3ff1090000000000ull - 0x3fee000000000000ull) {
        if (ix == 0x3ff0000000000000ull) return 0;
        const double r = x - 1.0;
        const double r2 = r * r, r3 = r * r2;
        const double p1 = fma(r2, B(3), fma(r, B(2), B(1)));
        const double p2 = fma(r2, B(6), fma(r, B(5), B(4)));
        double p3 = fma(r2, B(9), fma(r, B(8), B(7)));
        p3 = fma(r3, B(10), p3);
        const double P = fma(fma(p3, r3, p2), r3, p1);
        const double t = fma(r, 0x1p27, r);
        const double rhi = fma(-r, 0x1p27, t);  // vfnmadd: -(2^27 * r) + t
        const double rlo = r - rhi;
        const double rh2 = rhi * rhi;
        const double hi = fma(rh2, B(0), r);
        double lo = fma(rh2, B(0), r - hi);
        lo = fma(B(0) * rlo, rhi + r, lo);
        const double y = fma(P, r3, lo);
        return hi + y;
    }
    if (top - 0x0010u >= 0x7ff0u - 0x0010u) {
        if ((ix << 1) == 0 || ix == 0x7ff0000000000000ull || (top & 0x8000) || (top & 0x7ff0) == 0x7ff0) return std::log(x);
        ix = asu(x * 0x1p52);
        ix -= 52ull << 52;
    }
    const uint64_t tmp = ix - 0x3fe6000000000000ull;
    const int i = int((tmp >> 45) % 128);
    const int k = int(int64_t(tmp) >> 52);
    const uint64_t iz = ix - (tmp & 0xfffull << 52);
    const double invc = c(18 + 2 * i), logc = c(19 + 2 * i);
    const double z = asd(iz);
    const double kd = double(k);
    const double w = fma(kd, Ln2hi, logc);
    const double r = fma(z, invc, -1.0);
    const double q = fma(r, A(2), A(1));
    const double hi = r + w;
    const double r2 = r * r;
    const double lo = fma(kd, Ln2lo, (w - hi) + r);
    const double rr2 = r * r2;
    const double q2 = fma(fma(r, A(4), A(3)), r2, q);
    const double l2 = fma(r2, A(0), lo);
    return fma(rr2, q2, l2) + hi;
}

// glibc 2.41 __ieee754_exp_fma (sysdeps/ieee754/dbl-64/e_exp.c built with -mfma -mavx2), every FMA explicit.
// |x| >= 512 (outside every use in the engine) falls back to the platform exp.
inline double exp(double x) {
    const uint64_t* H = tbl::kExpHead;
    const uint32_t abstop = uint32_t(asu(x) >> 52) & 0x7ff;
    if (abstop - 0x3c9u >= 0x408u - 0x3c9u) {
        if (abstop - 0x3c9u >= 0x80000000u) return 1.0 + x;
        return std::exp(x);
    }
    const double InvLn2N = asd(H[0]), NegLn2hiN = asd(H[1]), NegLn2loN = asd(H[2]);
    const double C2 = asd(H[3]), C3 = asd(H[4]), C4 = asd(H[5]), C5 = asd(H[6]), Shift = asd(H[7]);
    double kd = fma(x, InvLn2N, Shift);
    const uint64_t ki = asu(kd);
    kd -= Shift;
    const double r = fma(kd, NegLn2loN, fma(kd, NegLn2hiN, x));
    const uint64_t idx = 2 * (ki % 128);
    const uint64_t top = ki << 45;
    const double tail = asd(tbl::kExpTab[idx]);
    const uint64_t sbits = tbl::kExpTab[idx + 1] + top;
    const double r2 = r * r;
    const double t1 = fma(fma(r, C3, C2), r2, r + tail);
    const double tmp = fma(r2 * r2, fma(r, C5, C4), t1);
    const double scale = asd(sbits);
    return fma(scale, tmp, scale);
}

}  // namespace jqm
