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
#include <atomic>
#include "hnnue.h"
#include "search.h"

#include "eval.h"
#include "nnue.h"
#include "tt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include "jqthread.h"
#include "jqmath.h"  // ISA build: explicit FMAs = the reference build's GCC contractions

namespace jq {
#ifdef JQ_VERIFY_THR
struct VThr {
    std::atomic<uint64_t> checks{0}, bad{0};
    ~VThr() { std::fprintf(stderr, "verify_thr checks %llu mismatches %llu\n", (unsigned long long)checks.load(), (unsigned long long)bad.load()); }
} g_vthr;
#endif

int g_ttWays = 1;  // TT entries per bucket (tt.h)
int g_ttWaysMT = 4;  // smpfix: TT bucket size used when Threads > 1 (shared-TT capacity for Lazy SMP)

namespace {

// ---- tunables ----------------------------------------------------------------------------
int P_ChanceK = 430;     // logistic scale used to average chance outcomes in expected-score space
int P_RfpMargin = 90;
int P_RfpDepth = 7;
int P_NullBase = 3;
int P_NullDiv = 200;
int P_FutBase = 100;
int P_FutMul = 120;
int P_LmpBase = 3;
int P_LmrBase = 50;      // /100
int P_LmrDiv = 220;      // /100
int P_AspDelta = 30;
int P_QsDarkPly = 4;     // deepest quiescence ply at which face-down movers may capture
int P_HistDiv = 8192;
int P_DeltaMargin = 200;
int P_BadCapMargin = 50;
int P_ChanceMode = 1;        // 0 = exact Star1 windows, 1 = parent-window probes averaged
int P_MaxChanceLayers = 1;   // full chance expansions allowed along one line (then representative)
int P_ChanceTail = 1;         // beyond the cap: 0 = cap layer resolved by qsearch (lines end),
                             // 1 = representative reveal at full depth from the cap layer on,
                             // 2 = cap layer expanded at half depth, then representative at full depth
int P_ChanceRed = 0;          // depth reduction for the non-representative outcomes of an expansion
int P_SeDepth = 0;            // singular extension of the TT move from this depth (0 = off)
int P_SeMargin = 16;          // singular beta = ttValue - SeMargin * depth / 8
int P_HistPrune = 0;          // prune late quiets at depth <= 3 with history < -HistPrune * depth (0 = off)
int P_NullDepthDiv = 4;       // null-move reduction grows by depth / NullDepthDiv
int P_PruneDepth = 6;         // move-count / futility pruning of quiets up to this depth
int P_RazorDepth = 0;         // razoring: drop to quiescence when eval + RazorMargin * depth < alpha (0 = off)
int P_RazorMargin = 250;
int P_SmpSkip = 1;            // helper depth skipping: 0 = none, 1 = skip (d + id) % 3 == 0, 2 = skip blocks by thread
int P_SmpDefer = 0;           // ABDADA-style deferral of moves other threads are searching, at depth >= SmpDefer (0 = off)
int P_SmpVote = 0;            // Lazy SMP: pick the move by depth/score-weighted votes over threads (0 = main thread)
int P_ContHist = 1;          // continuation history (1- and 2-ply) for quiet ordering / LMR
int P_CapHist = 1;           // capture history for capture ordering
int P_Improving = 1;         // "improving" adjustments to LMP / LMR / RFP
int P_QsTT = 1;              // transposition table in quiescence
int P_ProbCut = 0;            // ProbCut margin above beta (0 = off)
int P_ProbCutDepth = 5;       // ProbCut from this depth; verification at depth - 4
int P_QsTTMove = 0;          // quiescence stores its best move and tries the TT move first
int P_CheckExt = 1;          // check extension: 0 = off, 1 = while ply < 2*rootDepth+4, 2 = while ply < rootDepth
int P_QsSee = 1;             // quiescence skips captures with SEE < -QsSeeMargin (0 = attacker-value test)
int P_QsSeeMargin = 0;
int P_SeeOrder = 1;          // main search orders captures with SEE < -BadCapMargin last (0 = attacker-value test)
int P_SeeCap = 100;            // prune captures with SEE < -SeeCap * depth at depth <= SeePruneDepth (0 = off)
int P_SeeQuiet = 30;          // prune quiet and face-down moves with SEE < -SeeQuiet * depth^2 (0 = off)
int P_SeePruneDepth = 6;
int P_CapLmr = 0;             // reduce SEE-losing captures by this many plies at depth >= 3 (0 = off)
int P_CorrHist = 0;           // weight (%) of the structure correction history added to static evals (0 = off)
int P_RevealCheck = 0;        // beyond the chance cap: split off identities whose reveal checks differently
                             // from the representative, at most this many times along one line (0 = off)
int P_RevealCheckQs = 0;      // RevealCheck also in quiescence
int P_ChanceCVProbe = 0;      // ChanceCV probe depth min(k, depth-1) when ChanceCVGap is 0 (0 = depth 0)
int P_RootDarkPen = 0;        // root: face-down moves are compared at value - RootDarkPen (cp); negative = reveal bonus
int P_RootDarkHid = 0;        // RootDarkPen only when the mover has at most this many face-down pieces (0 = always)
// Threatened reveals: a face-down piece of the side to move on a square the opponent can win material on
// (search::threat_reveal). Trigger: undefended (opponent captures for free) always; otherwise ThreatRevMode
// 0 = probability-weighted opponent SEE gain over the mover's urn identities > ThreatRevSee cp, 1 = P(opponent
// SEE gain > 0) >= ThreatRevProb %. Computed only when ThreatRevRoot/Scale (root), ThreatRevLmr or ThreatRevOrder is on.
int P_ThreatRevRoot = 0;      // root: threatened face-down moves compared at value + ThreatRevRoot (cp)
int P_ThreatRevScale = 0;     // root: if > 0, bonus = min(300, expected loss * ThreatRevScale / 100) instead of ThreatRevRoot
int P_ThreatRevLmr = 100;     // LMR of threatened face-down quiet moves in percent (replaces DarkLmr for them when != 100)
int P_ThreatRevOrder = 0;     // quiet-ordering bonus of threatened face-down moves
int P_ThreatRevSee = 0;       // mode 0 threshold (cp) on the expected opponent SEE gain
int P_ThreatRevMode = 0;
int P_ThreatRevProb = 50;     // mode 1 threshold (%)
int P_ChanceCVLow = 0;        // ChanceCV correction from static evals at depth >= ChanceCVLow below ChanceCV (0 = off)
int P_ChanceCVStatic = 0;     // ChanceCV probes: 0 = depth-0 searches (or ChanceCVGap/Probe depth), 1 = static evals
int P_TmMtg = 20;             // clock: moves-to-go estimate when none is given
int P_TmInc = 75;             // clock: percent of the increment in the per-move optimum
int P_TmSoft = 100;           // clock: start a new iteration only before TmSoft% of the optimum
int P_TmStab = 100;           // clock: soft-limit scaling by decayed best-move instability (0 = off)
int P_TmFall = 0;             // clock: soft-limit scaling by the score drop between iterations (0 = off)
int P_TmMtgMin = 0;           // clock: moves-to-go falls by TmMtgDec/10 per own move played, down to TmMtgMin (0 = off)
int P_TmMtgDec = 0;
int P_NodesTime = 0;          // testing: the clock runs on nodes, NodesTime nodes per millisecond (0 = wall clock)
int P_ChanceLateDark = 0;     // root with at most this many face-down pieces (both sides): chance layer cap is
int P_ChanceLateAdd = 1;      // MaxChanceLayers + ChanceLateAdd for that search (ChanceLateDark 0 = off)
int P_ChancePvW = 0;          // probe mode, PV chance nodes: outcomes are searched with the window widened by this
                              // much on both sides, so that outcome values outside the parent's window enter the
                              // average as values rather than as bounds (0 = parent's window)
int P_ChanceTailFix = 0;      // probe mode: an outcome that fails outside the window is also searched with an open
                              // window at depth - ChanceTailR, and that value replaces the bound when it lies further
                              // out (1 = PV chance nodes, 2 = all expanded chance nodes; 0 = off)
int P_ChanceTailR = 6;
int P_ChanceCVRho = 100;      // ChanceCV: E ~ mean of probes + c (deep(rep) - probe(rep)), c = p_rep + (1 - p_rep) Rho/100
int P_ChanceMix = 6;          // expanded chance nodes: every outcome gets an open-window search at depth - ChanceMix,
                              // the most numerous identity a full-depth search; combined as in ChanceCV with
                              // ChanceMixRho (0 = probe mode)
int P_ChanceMixRho = 0;
int P_ChanceMixMin = 1;       // ChanceMix only at depth >= ChanceMixMin
int P_ChanceMixPv = 0;        // 1 = ChanceMix only at PV chance nodes (others keep probe mode)
int P_ChanceMixRho0 = -1;     // ChanceMix: weight (percent) when the probes are quiescence searches; -1 = ChanceMixRho
int P_ChanceFut = 1;          // ChanceMix with quiescence probes at a null-window node: 1 = fail low from static values when the
                              // full-depth identity's child would be cut by reverse futility anyway (saves the probes)
int P_ChanceFutPre = 20;      // ChanceFut is tried only when the node's static value + ChanceFutPre * depth <= alpha
int P_ChanceNW = 1;           // CV / mix at a null-window node: 1 = full-depth identity searched with a null window at the decision threshold
int P_ChanceMixTT = 1;        // ChanceMix probes: 1 = an exact TT entry of sufficient depth replaces the probe search
int P_ChanceMixTop = 1;       // ChanceMix: number of identities (most numerous first) searched at full depth
int P_ChanceMixSkip = 0;      // ChanceMix with ChanceMixRho 0: 1 = no probe for the full-depth identity (its probe value cancels)
int P_ChanceMixProbe = 1;     // ChanceMix probes: 1 = searched as PV nodes, 0 = with the node's own PV flag
int P_ChanceMixDiv = 0;       // probes at depth - ChanceMix - depth / ChanceMixDiv (0 = no depth term)
int P_ChanceMixPvR = 0;       // open-window (PV) mix nodes: probes at depth - ChanceMixPvR instead (0 = ChanceMix)
int P_ChanceMixLow = 0;       // ChanceMix probes: identities below this probability (%) are probed ChanceMixLowR plies
                              // shallower (0 = off; never the full-depth identities)
int P_ChanceMixLowR = 1;      // ChanceMixLow reduction in plies (clamped at depth 0 = quiescence; 99 = always quiescence)
int P_ChanceProbeSmall = 0;    // dual nets (EvalFileBig): ChanceMix probe subtrees (and ChanceFut static values) use the small net
// dual HNN (EvalFile = big HNN, EvalFileSmall = small HNN): per-node net choice.
// DualMode 0 off (big only; ChanceProbeSmall=1 still puts ChanceMix probe subtrees on the small net), 1 |simple material| > DualThreshold
// -> small, 2 reverse (|mat| > T -> big, else small), 3 hidden (face-down) pieces <= DualHidden -> small, 4 hidden <= H -> big (else
// small), 5 (|mat| > T or hidden <= H) -> small, 6 SF-style: small net value |v| > T -> keep it, else big, 7 (|mat| > T or hidden > H) -> small.
int P_DualMode = 0, P_DualThr = 100, P_DualHid = 16, P_DualSkipMargin = -1;   // DualSkipMargin -1: never skip big deltas
using nnue::g_dualEv;
int P_SmallTTSalt = 1;         // ChanceProbeSmall: probe subtrees use TT keys salted with a constant (no score mixing)
int P_ChanceMixPvPly = 0;     // ChanceMixPvR only for moves made at a ply below this (1 = root moves; 0 = any ply)
int P_ChanceXDepth = 1;      // chance nodes with depth >= ChanceXDepth are fully expanded at any layer (0 = off)
int P_ChanceRedLow = 0;       // expanded layers: non-representative identities below this probability (%) one ply shallower
int P_RepMode = 0;            // representative reveal: 0 = identity closest to the urn mean value, 1 = most numerous,
                              // 2 = as 0, but at ChanceCV nodes the identity whose depth-0 value is closest to their mean
int P_ChanceCVGap = 0;        // ChanceCV probes at depth - gap (0 = depth-0 probes)
int P_DarkLmr = 100;          // LMR of face-down moves in percent of the quiet reduction
int P_ChanceCV = 3;           // beyond the chance cap at depth >= ChanceCV: representative value + probability-
                             // weighted depth-0 differences between identities (control variate; 0 = off)
int P_SeeQuietOrder = 0;     // quiet / face-down moves with SEE < -SeeQuietOrder are ordered after all other quiets (0 = off)

constexpr Value SAT = 10000;  // chance aggregation saturates here (mates collapse to +/-SAT)

using Clock = std::chrono::steady_clock;

struct Shared {
    std::atomic<bool> stop{false};
    std::atomic<bool> pondering{false};
    std::atomic<bool> ponderArmed{false};
    std::atomic<bool> ponderHitPending{false};
    std::atomic<int64_t> ponderOffsetMs{0};
    Clock::time_point start;
    int64_t optimumMs = 0, maximumMs = 0;
    bool timed = false;
    bool clockMode = false;  // wtime/btime (soft limit is adjustable)
    int64_t nodeLimit = 0;
    int depthLimit = 0;
    int mateLimit = 0;
    int nThreads = 1;
    std::vector<Move> rootMoves;  // go searchmoves
    int64_t raw_elapsed() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    }
    int64_t elapsed() const {
        return std::max<int64_t>(0, raw_elapsed() - ponderOffsetMs.load(std::memory_order_relaxed));
    }
};

int lmrTable[64][64];

// Moves being searched, keyed by position and move (SmpDefer). Racy by design: a stale or overwritten
// entry only changes the move order.
constexpr int BUSY_BITS = 16;
std::atomic<uint64_t> g_busy[1 << BUSY_BITS];
constexpr int DEFER_SCORE = -(1 << 30);
inline uint64_t busy_key(uint64_t key, Move m) { return (key ^ (uint64_t(m) * 0x9E3779B97F4A7C15ULL)) | 1; }
inline std::atomic<uint64_t>& busy_slot(uint64_t bk) { return g_busy[bk >> (64 - BUSY_BITS)]; }

inline int pidx16(Piece p) { return (int(color_of(p)) << 3) | (int(type_of(p)) - 1); }

Value value_to_tt(Value v, int ply) {
    if (v == VALUE_NONE) return v;
    return v >= VALUE_MATE_IN_MAX_PLY ? v + ply : v <= VALUE_MATED_IN_MAX_PLY ? v - ply : v;
}
Value value_from_tt(Value v, int ply) {
    if (v == VALUE_NONE) return v;
    return v >= VALUE_MATE_IN_MAX_PLY ? v - ply : v <= VALUE_MATED_IN_MAX_PLY ? v + ply : v;
}

std::string score_str(Value v) {
    if (v >= VALUE_MATE_IN_MAX_PLY) return "mate " + std::to_string((VALUE_MATE - v + 1) / 2);
    if (v <= VALUE_MATED_IN_MAX_PLY) return "mate " + std::to_string(-(VALUE_MATE + v) / 2);
    return "cp " + std::to_string(v);
}

int mate_moves(Value v) {
    if (v >= VALUE_MATE_IN_MAX_PLY) return (VALUE_MATE - v + 1) / 2;
    if (v <= VALUE_MATED_IN_MAX_PLY) return -(VALUE_MATE + v) / 2;
    return MAX_PLY;
}

#ifdef JQ_STATS
// Search statistics (diagnostic builds only; single-thread meaningful).
struct Stats {
    uint64_t search, pv, qs, ttcut, rfp, nullTry, nullCut, cext, chance, outcomes, lmr, lmrRe, pvsRe, fh, fh1,
        fhMoves, failLow, failLowMoves, lmp, fut, quietSearched, capSearched, darkSearched;
    uint64_t fhCat[6], fhTT, fh1TT, loopNodes, loopTT;  // cutoff move: tt, good cap, killer/counter, bad cap, quiet, dark
    uint64_t cvCalls, cvProbeN, cvDeepN, mixCalls, mixProbeN, mixDeepN, probeCalls, probeN;  // chance node kinds: calls, nodes
    uint64_t mixK[6][3];  // mix at non-PV nodes by (quiescence probes / searched probes) x (fail low, inside, fail high): calls, probe nodes, deep nodes
    uint64_t probeSubN, evProbe;                    // nodes / evaluations inside ChanceMix probe subtrees (and ChanceFut statics)
    uint64_t chanceSubN, nodesAll, evBig, evSmall;  // nodes under expanded chance nodes; evaluations by net (evSmall: probes only)
    uint64_t updBig, updSmall, refBig, refSmall;    // accumulator updates / refreshes by net
    uint64_t mixPct[2][101][2];  // ChanceMix probes by (other / full-depth identity) x probability % (floor): probes, subtree nodes
    uint64_t trvRootDark, trvRootAtk, trvRootThr, trvTreeDark, trvTreeAtk, trvTreeThr;  // dark moves searched / from-square attacked / triggered
    uint64_t trvCalls;  // threat_reveal evaluations
};
inline Stats g_st;
#define STAT(x) (++g_st.x)
#define STATN(x, n) (g_st.x += (n))
#else
#define STAT(x) ((void)0)
#define STATN(x, n) ((void)0)
#endif

// Least valuable attacker of square s by colour c on board bd (-1 if none), its value in *val. Face-down
// pieces attack with the role of their start square, keeping the palace / own-half limits, and are worth
// darkVal (their side's expected identity). Pins and facing generals are ignored.
#ifdef JQ_BB_VERIFY
int least_attacker_ref(const Piece* bd, Color c, int s, int darkVal, int* val) {
    int best = -1, bv = 1 << 30;
    auto mine = [&](int q) { return bd[q] != NO_PIECE && color_of(bd[q]) == c; };
    auto role = [&](int q) { return type_of(bd[q]) == DARK ? PieceType(T.darkRole[c][q]) : type_of(bd[q]); };
    auto consider = [&](int q) {
        const PieceType t = type_of(bd[q]);
        const int v = t == DARK ? darkVal : t == KING ? 1 << 20 : PieceValue[t];
        if (v < bv) { bv = v; best = q; }
    };
    const int back = c == RED ? -9 : 9;  // c's soldier one step behind s attacks it
    if (s + back >= 0 && s + back < NUM_SQ && mine(s + back) && role(s + back) == SOLDIER) consider(s + back);
    if (file_of(s) > 0 && mine(s - 1) && role(s - 1) == SOLDIER && !own_half(c, s - 1)) consider(s - 1);
    if (file_of(s) < 8 && mine(s + 1) && role(s + 1) == SOLDIER && !own_half(c, s + 1)) consider(s + 1);
    for (int j = 0; j < T.advN[s]; ++j) {
        const int q = T.advTo[s][j];
        if (mine(q) && role(q) == ADVISOR && (type_of(bd[q]) != DARK || in_palace(c, s))) consider(q);
    }
    for (int j = 0; j < T.elephN[s]; ++j) {
        const int q = T.elephTo[s][j];
        if (mine(q) && role(q) == ELEPHANT && !bd[T.elephEye[s][j]] && (type_of(bd[q]) != DARK || own_half(c, s))) consider(q);
    }
    for (int j = 0; j < T.horseFromN[s]; ++j) {
        const int q = T.horseFrom[s][j];
        if (mine(q) && role(q) == HORSE && !bd[T.horseFromLeg[s][j]]) consider(q);
    }
    for (int j = 0; j < T.kingN[s]; ++j)
        if (mine(T.kingTo[s][j]) && type_of(bd[T.kingTo[s][j]]) == KING) consider(T.kingTo[s][j]);
    for (int d = 0; d < 4; ++d) {
        const int n = T.rayLen[s][d];
        const uint8_t* r = T.ray[s][d];
        int i = 0;
        while (i < n && !bd[r[i]]) ++i;
        if (i == n) continue;
        if (mine(r[i]) && role(r[i]) == ROOK) consider(r[i]);
        for (++i; i < n; ++i) {
            if (!bd[r[i]]) continue;
            if (mine(r[i]) && role(r[i]) == CANNON) consider(r[i]);
            break;
        }
    }
    *val = bv;
    return best;
}
#endif

// Bitboard least_attacker: the board is p.board restricted to occ (in SEE the only other change is that s is
// re-occupied by the capturer, and s is never its own attacker). Same answer as the mailbox scan, including the
// tie order among equal values: soldiers (behind, west, east), advisors, elephants, horses (table order), king,
// then rays N,S,E,W (rook-role first blocker, cannon-role second blocker).
int least_attacker(const Position& p, Bitboard occ, Color c, int s, int darkVal, int* val) {
    const Piece* bd = p.board;
    const Bitboard C = p.byColorBB[c] & occ;
    int best = -1, bv = 1 << 30;
    auto consider = [&](int q) {
        const PieceType t = type_of(bd[q]);
        const int v = t == DARK ? darkVal : t == KING ? 1 << 20 : PieceValue[t];
        if (v < bv) { bv = v; best = q; }
    };
    if (const Bitboard sol = C & p.byRoleBB[SOLDIER] & BB::SoldierFrom[c][s]) {
        const int b = s + (c == RED ? -9 : 9);
        if (b >= 0 && b < NUM_SQ && test(sol, b)) consider(b);
        if (s > 0 && test(sol, s - 1)) consider(s - 1);
        if (test(sol, s + 1)) consider(s + 1);
    }
    if (Bitboard adv = C & p.byRoleBB[ADVISOR] & BB::AdvTo[s]) {
        if (!in_palace(c, s)) adv &= ~p.darkBB;
        if (adv)
            for (int j = 0; j < T.advN[s]; ++j)
                if (test(adv, T.advTo[s][j])) consider(T.advTo[s][j]);
    }
    if (Bitboard el = C & p.byRoleBB[ELEPHANT] & BB::ElephTo[s]) {
        if (!own_half(c, s)) el &= ~p.darkBB;
        if (el)
            for (int j = 0; j < T.elephN[s]; ++j)
                if (test(el, T.elephTo[s][j]) && !test(occ, T.elephEye[s][j])) consider(T.elephTo[s][j]);
    }
    if (const Bitboard h = C & p.byRoleBB[HORSE] & BB::HorseFrom[s])
        for (int j = 0; j < T.horseFromN[s]; ++j)
            if (test(h, T.horseFrom[s][j]) && !test(occ, T.horseFromLeg[s][j])) consider(T.horseFrom[s][j]);
    if (const Bitboard k = C & p.byRoleBB[KING] & BB::KingTo[s])
        for (int j = 0; j < T.kingN[s]; ++j)
            if (test(k, T.kingTo[s][j])) consider(T.kingTo[s][j]);
    const Bitboard rk = C & p.byRoleBB[ROOK], cn = C & p.byRoleBB[CANNON];
    if ((rk | cn) & (BB::FileBB[file_of(s)] | BB::RankBB[rank_of(s)]))
        for (int d = 0; d < 4; ++d) {
            int f1, f2;
            BB::blockers(s, d, occ, f1, f2);
            if (f1 < 0) continue;
            if (test(rk, f1)) consider(f1);
            if (f2 >= 0 && test(cn, f2)) consider(f2);
        }
    *val = bv;
#ifdef JQ_BB_VERIFY
    {
        Piece vb[NUM_SQ];
        for (int q = 0; q < NUM_SQ; ++q) vb[q] = test(occ, q) ? bd[q] : NO_PIECE;
        if (!vb[s]) vb[s] = make_piece(~c, SOLDIER);
        int rv;
        const int rb = least_attacker_ref(vb, c, s, darkVal, &rv);
        if (rb != best || (best >= 0 && rv != bv)) { std::fprintf(stderr, "least_attacker mismatch %s s=%d\n", p.fen().c_str(), s); std::abort(); }
    }
#endif
    return best;
}

struct Worker {
    Shared& G;
    TranspositionTable& TT;
    std::vector<std::unique_ptr<Worker>>& peers;
    Worker(Shared& g, TranspositionTable& tt, std::vector<std::unique_ptr<Worker>>& p) : G(g), TT(tt), peers(p) {}
    uint64_t total_nodes() const {
        uint64_t n = 0;
        for (auto& w : peers) n += w->nodes.load(std::memory_order_relaxed);
        return n;
    }
    int64_t elapsed() const { return P_NodesTime ? int64_t(total_nodes() / uint64_t(P_NodesTime)) : G.elapsed(); }
    int id = 0;
    Position pos;
    int16_t history[2][NUM_SQ][NUM_SQ];
    int16_t contHist[16][NUM_SQ][16][NUM_SQ];  // [prev piece][prev to][piece][to]
    int16_t capHist[16][NUM_SQ][9];            // [piece][to][captured type]
    // Correction history: running mean of (search value - static eval) in 1/16 cp, indexed by side to move
    // and a hash of the soldier and face-down piece placement (the slow-changing structure).
    static constexpr int CORR_BITS = 14;
    int16_t corrHist[2][1 << CORR_BITS];
    int8_t movedPc[MAX_PLY + 8];               // pidx16 of the piece moved at ply (-1 = null move)
    Value staticEv[MAX_PLY + 8];
    Move killers[MAX_PLY + 8][2];
    Move counters[PIECE_CODES][NUM_SQ];
    Move pv[MAX_PLY + 8][MAX_PLY + 8];
    int pvLen[MAX_PLY + 8];
    Move curMove[MAX_PLY + 8];
    Move excluded[MAX_PLY + 8];                // singular-extension verification: move skipped at this ply
    std::atomic<uint64_t> nodes{0};
    int selDepth = 0;
    int rootDepth = 0;
    int chanceLayers = 0;
    int maxLayers = 1;                         // chance layer cap of this search
    int revealSplits = 0;
    uint32_t calls = 0;
    // Root bookkeeping.
    Move bestMove = MOVE_NONE, iterBest = MOVE_NONE;
    Value bestScore = -VALUE_INFINITE, iterScore = -VALUE_INFINITE;
    int completedDepth = 0;
    std::vector<Move> bestPv;
    const std::function<void(const std::string&)>* info = nullptr;

    double eK = 300.0, eMin = 0.0, eMax = 1.0;

    // NNUE accumulators per search ply (lazily updated along the current line).
    bool useNN = false;
    nnue::Accumulator acc[MAX_PLY + 10];
    nnue::Dirty dirty[MAX_PLY + 10];
    // Validity per net (bit mask of nnue::NET_MAIN / NET_SMALL): each net's accumulator is brought up to date
    // separately from its own last valid ply, so one net's flag never stands for the other's.
    uint8_t accOk[MAX_PLY + 10];
    // Dual nets with ChanceProbeSmall: inside a ChanceMix probe subtree (inSmallProbe) evaluations use the small
    // net and TT keys are salted (ttSalt) so small-net scores never meet big-net scores in the TT. On the main
    // line both accumulators are kept current so that probe entries update incrementally.
    bool probeSmall = false;
    bool inSmallProbe = false;
    uint64_t ttSalt = 0;
    static constexpr uint64_t SMALL_SALT = 0x5bd1e9955bd1e995ULL;
    uint64_t tkey() const { return pos.key ^ ttSalt; }
    struct SmallProbeGuard {
        Worker& w;
        bool prevIn;
        uint64_t prevSalt;
        SmallProbeGuard(Worker& x, bool on) : w(x), prevIn(x.inSmallProbe), prevSalt(x.ttSalt) {
            if (on) {
                w.inSmallProbe = true;
                w.ttSalt = P_SmallTTSalt ? SMALL_SALT : 0;
            }
        }
        ~SmallProbeGuard() { w.inSmallProbe = prevIn; w.ttSalt = prevSalt; }
    };
    void ensure_acc(int ply, uint8_t b) {
        if (accOk[ply] & b) return;
        int k = ply;
        while (k > 0 && !(accOk[k] & b)) --k;
        bool ok = (accOk[k] & b) != 0;
        for (int j = k + 1; ok && j <= ply; ++j) {
            if (dirty[j].refresh >= 0 || ((b & nnue::NET_MAIN) && dirty[j].bigSkip)) { ok = false; break; }  // a king changed bucket (small v2 nets only) / dual: big delta skipped
            nnue::update(acc[j - 1], acc[j], dirty[j], b);
#ifdef JQ_STATS
            if (b == nnue::NET_MAIN) ++g_st.updBig; else ++g_st.updSmall;
#endif
            accOk[j] |= b;
        }
        if (!ok) {
            nnue::refresh(pos, acc[ply], b);
#ifdef JQ_STATS
            if (b == nnue::NET_MAIN) ++g_st.refBig; else ++g_st.refSmall;
#endif
            accOk[ply] |= b;
        }
    }
    bool useHsm = false;
    Value eval_at(int ply) {
        if (!useNN) return classical_eval(pos);
        if (useHsm) {
            bool sm = inSmallProbe;
            if (!sm) switch (P_DualMode) {
                case 1: sm = std::abs(nnue::simple_material(pos)) > P_DualThr; break;
                case 2: sm = std::abs(nnue::simple_material(pos)) <= P_DualThr; break;
                case 3: sm = nnue::hidden_count(pos) <= P_DualHid; break;
                case 4: sm = nnue::hidden_count(pos) > P_DualHid; break;
                case 5: sm = std::abs(nnue::simple_material(pos)) > P_DualThr || nnue::hidden_count(pos) <= P_DualHid; break;
                case 7: sm = std::abs(nnue::simple_material(pos)) > P_DualThr || nnue::hidden_count(pos) > P_DualHid; break;
                case 6: {
                    ensure_acc(ply, nnue::NET_SMALL);
                    const Value v = nnue::evaluate_hsm(acc[ply], pos.stm);
                    if (std::abs(int(v)) > P_DualThr) { ++g_dualEv[1]; return v; }
                    ++g_dualEv[2];
                    break;
                }
                default: break;
            }
#ifdef JQ_VERIFY_DUAL
            {
                static thread_local nnue::Accumulator chk;
                nnue::refresh(pos, chk, nnue::NET_ALL);
                ensure_acc(ply, sm ? nnue::NET_SMALL : nnue::NET_MAIN);
                const Value a = sm ? nnue::evaluate_hsm(acc[ply], pos.stm) : nnue::evaluate(acc[ply], pos.stm);
                const Value b = sm ? nnue::evaluate_hsm(chk, pos.stm) : nnue::evaluate(chk, pos.stm);
                if (a != b) { std::fprintf(stderr, "DUAL mismatch (%s) %d vs %d\n", sm ? "small" : "big", int(a), int(b)); std::abort(); }
            }
#endif
            ++g_dualEv[sm ? 1 : 0];
            if (sm) { ensure_acc(ply, nnue::NET_SMALL); return nnue::evaluate_hsm(acc[ply], pos.stm); }
            ensure_acc(ply, nnue::NET_MAIN);
            return nnue::evaluate(acc[ply], pos.stm);
        }
        if (inSmallProbe) ensure_acc(ply, nnue::NET_SMALL);
        else {
            ensure_acc(ply, nnue::NET_MAIN);
            if (probeSmall) ensure_acc(ply, nnue::NET_SMALL);
        }
#ifdef JQ_VERIFY_NNUE
        {
            static thread_local nnue::Accumulator chk;
            nnue::refresh(pos, chk);
            if (accOk[ply] & nnue::NET_MAIN) {
                const Value a = nnue::evaluate(acc[ply], pos.stm), b = nnue::evaluate(chk, pos.stm);
                if (a != b) { std::fprintf(stderr, "NNUE mismatch (main) %d vs %d\n", int(a), int(b)); std::abort(); }
            }
            if ((accOk[ply] & nnue::NET_SMALL) && nnue::dual()) {
                const Value a = nnue::evaluate_small(acc[ply], pos.stm), b = nnue::evaluate_small(chk, pos.stm);
                if (a != b) { std::fprintf(stderr, "NNUE mismatch (small) %d vs %d\n", int(a), int(b)); std::abort(); }
            }
        }
#endif
#ifdef JQ_VERIFY_THR
        if (accOk[ply] & nnue::NET_MAIN) {
            static thread_local hnn::Acc chk;
            hnn::refresh_nocache(pos, chk);
            const int W = hnn::width();
            bool bad = false;
            for (int p = 0; p < 2; ++p)
                bad |= std::memcmp(chk.v[p], acc[ply].hnn.v[p], sizeof(int16_t) * std::size_t(W)) != 0 ||
                       std::memcmp(chk.ps[p], acc[ply].hnn.ps[p], sizeof chk.ps[p]) != 0;
            g_vthr.checks.fetch_add(1, std::memory_order_relaxed);
            if (bad) g_vthr.bad.fetch_add(1, std::memory_order_relaxed);
        }
#endif
#ifdef JQ_STATS
        if (inSmallProbe) ++g_st.evSmall; else { ++g_st.evBig; }
        if (mixProbeDepth > 0) ++g_st.evProbe;
#endif
        return inSmallProbe ? nnue::evaluate_small(acc[ply], pos.stm) : nnue::evaluate(acc[ply], pos.stm);
    }

    void clear() {
        std::memset(history, 0, sizeof history);
        std::memset(contHist, 0, sizeof contHist);
        std::memset(capHist, 0, sizeof capHist);
        std::memset(corrHist, 0, sizeof corrHist);
        std::memset(killers, 0, sizeof killers);
        std::memset(counters, 0, sizeof counters);
    }
    double v2e(Value v) const {
        v = std::clamp(v, -SAT, SAT);
        return __builtin_expect(v2eTab.empty(), 0) ? v2e_ref(v) : v2eTab[v + SAT];
    }
    std::vector<double> v2eTab;  // v2eTab[v + SAT] = 1 / (1 + exp(-v / eK)), the same expression evaluated once
    double v2eK = -1.0;
    void set_eK(double k) {
        eK = k;
        if (v2eK == k && !v2eTab.empty()) return;
        v2eTab.resize(2 * SAT + 1);
        for (int v = -SAT; v <= SAT; ++v) v2eTab[v + SAT] = v2e_ref(Value(v));
        v2eK = k;
    }
    __attribute__((noinline)) double v2e_ref(Value v) const { return 1.0 / (1.0 + jqm::exp(-double(v) / eK)); }
    double e2v(double e) const { return eK * jqm::log(e / (1.0 - e)); }  // jqm: glibc-FMA-exact on every tier/OS
    // lround(e2v(e)) (isa): every e2v use is rounded to an integer Value. Tiers without hardware FMA first try the
    // platform log (error < 1e-12 absolute here) and use it unless f lies within 1e-6 of a rounding boundary k + 1/2;
    // there (and on FMA tiers) the exact glibc-equivalent jqm::log decides, so the result is always identical.
    long e2v_r(double e) const {
#if !JQM_HW_FMA
        const double f = eK * std::log(e / (1.0 - e));
        if (std::fabs(f - std::floor(f) - 0.5) > 1e-6 && std::fabs(f) < 1e9) return std::lround(f);
#endif
        return std::lround(e2v(e));
    }
    // floor / ceil of e2v(e): same scheme, boundaries at the integers
    double e2v_floor(double e) const {
#if !JQM_HW_FMA
        const double f = eK * std::log(e / (1.0 - e));
        const double d = f - std::floor(f);
        if (d > 1e-6 && d < 1.0 - 1e-6 && std::fabs(f) < 1e9) return std::floor(f);
#endif
        return std::floor(e2v(e));
    }
    double e2v_ceil(double e) const {
#if !JQM_HW_FMA
        const double f = eK * std::log(e / (1.0 - e));
        const double d = f - std::floor(f);
        if (d > 1e-6 && d < 1.0 - 1e-6 && std::fabs(f) < 1e9) return std::ceil(f);
#endif
        return std::ceil(e2v(e));
    }

#ifdef JQ_STATS
    int mixProbeDepth = 0;  // > 0 inside a ChanceMix probe subtree (any net)
    struct ProbeDepthGuard {
        int& d;
        bool on;
        ProbeDepthGuard(int& x, bool o) : d(x), on(o) { d += on; }
        ~ProbeDepthGuard() { d -= on; }
    };
#define PROBE_DEPTH_GUARD(on) ProbeDepthGuard pdg_(mixProbeDepth, on)
#else
#define PROBE_DEPTH_GUARD(on) ((void)0)
#endif
    void bump() {
        nodes.store(nodes.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
#ifdef JQ_STATS
        ++g_st.nodesAll;
        if (chanceLayers > 0) ++g_st.chanceSubN;
        if (mixProbeDepth > 0) ++g_st.probeSubN;
#endif
    }
    void check_time();
    void iterate();
    Value search(Value alpha, Value beta, int depth, int ply, bool pvNode, bool cutNode);
    Value qsearch(Value alpha, Value beta, int ply, int qply);
    Value child(Move m, Value alpha, Value beta, int depth, int ply, bool pvNode, bool cutNode, int qply);
    int piece_value(int s) const {
        const Piece p = pos.board[s];
        return type_of(p) == DARK ? urn_mean_value(pos, color_of(p)) : PieceValue[type_of(p)];
    }
    bool see_ge(Move m, int threshold) const { return search::see_ge(pos, m, threshold); }
    int structure_index() const {
        return int(pos.skey >> (64 - CORR_BITS));
    }
    Value corrected(Value raw) const {
        if (!P_CorrHist || raw == VALUE_NONE) return raw;
        const int e = corrHist[pos.stm][structure_index()];
        return std::clamp(raw + e * P_CorrHist / 1600, -VALUE_KNOWN_WIN + 1, VALUE_KNOWN_WIN - 1);
    }
    void update_corr(Value raw, Value best, int depth) {
        int16_t& e = corrHist[pos.stm][structure_index()];
        const int target = std::clamp((best - raw) * 16, -8192, 8192);
        const int w = std::min(depth + 1, 16);
        e = int16_t((e * (256 - w) + target * w) / 256);
    }
    static void gravity(int16_t& h, int bonus) { h = int16_t(h + bonus - h * std::abs(bonus) / 16384); }
    void update_history(Color c, Move m, int bonus) { gravity(history[c][from_sq(m)][to_sq(m)], bonus); }
    // Quiet statistics: butterfly + continuation (1 and 2 plies back). pc = pidx16 of the mover.
    int quiet_stat(Color c, Move m, int pc, int ply) const {
        int s = history[c][from_sq(m)][to_sq(m)];
        if (P_ContHist) {
            const int to = to_sq(m);
            if (ply >= 1 && movedPc[ply - 1] >= 0) s += contHist[movedPc[ply - 1]][to_sq(curMove[ply - 1])][pc][to];
            if (ply >= 2 && movedPc[ply - 2] >= 0) s += contHist[movedPc[ply - 2]][to_sq(curMove[ply - 2])][pc][to];
        }
        return s;
    }
    void update_quiet(Color c, Move m, int pc, int ply, int bonus) {
        update_history(c, m, bonus);
        if (!P_ContHist) return;
        const int to = to_sq(m);
        if (ply >= 1 && movedPc[ply - 1] >= 0) gravity(contHist[movedPc[ply - 1]][to_sq(curMove[ply - 1])][pc][to], bonus);
        if (ply >= 2 && movedPc[ply - 2] >= 0) gravity(contHist[movedPc[ply - 2]][to_sq(curMove[ply - 2])][pc][to], bonus);
    }
    void report(int depth, Value v, Bound bound);
};

void Worker::check_time() {
    if (id != 0 || (++calls & 1023) != 0) return;
    if (G.timed && !G.pondering && elapsed() >= G.maximumMs) G.stop = true;
    if (G.nodeLimit && total_nodes() >= uint64_t(G.nodeLimit)) G.stop = true;
}

// Evaluates a move from the side to move's perspective. A face-down mover turns the move into a
// chance node over the urn; outcomes are averaged in expected-score space with Star1 windows.
// Chance results are fail-hard: alpha (fail low), beta (fail high) or an exact value in between.
Value Worker::child(Move m, Value alpha, Value beta, int depth, int ply, bool pvNode, bool cutNode, int qply) {
    Position& p = pos;
    curMove[ply] = base_move(m);
    movedPc[ply] = int8_t(pidx16(p.board[from_sq(m)]));
    auto sub = [&](Move mm, Value a, Value b, int d) {
        StateInfo st;
        TT.prefetch(p.key_after(mm) ^ ttSalt);
        if (useNN) {
            nnue::make_dirty(p, mm, dirty[ply + 1]);
            accOk[ply + 1] = 0;
        }
        p.do_move(mm, st);
        bump();
        const Value v = qply >= 0 ? -qsearch(-b, -a, ply + 1, qply + 1)
                                  : -search(-b, -a, d, ply + 1, pvNode, !cutNode);
        p.undo_move(mm, st);
        return v;
    };
    if (!p.is_dark_move(m)) return sub(m, alpha, beta, depth);
    // An aggregated chance value lies in [-SAT, SAT]. A fail-hard bound outside that range (a window wider than the
    // saturated scale, e.g. the infinite root window) is the saturated value itself, not a bound.
    auto sat = [](Value v) { return std::clamp(v, Value(-SAT), SAT); };
    auto stat = [&](Move mm) {  // static eval after mm, from the mover's side
        StateInfo st;
        if (useNN) {
            nnue::make_dirty(p, mm, dirty[ply + 1]);
            accOk[ply + 1] = 0;
        }
        p.do_move(mm, st);
        const Value v = -corrected(eval_at(ply + 1));
        p.undo_move(mm, st);
        return v;
    };

    const Color us = p.stm;
    const int from = from_sq(m), to = to_sq(m);
    int slots[NUM_SLOTS], cnt[NUM_SLOTS], n = 0;
    for (int sl = 0; sl < NUM_SLOTS; ++sl)
        if (p.urn[us][sl] > 0) {
            int j = n++;
            while (j > 0 && cnt[j - 1] < p.urn[us][sl]) { slots[j] = slots[j - 1]; cnt[j] = cnt[j - 1]; --j; }
            slots[j] = sl;
            cnt[j] = p.urn[us][sl];
        }
    // Bounded chance expansion: deep in quiescence or beyond MaxChanceLayers+1 chance layers on this
    // line, play one representative reveal (the identity whose value is closest to the urn mean).
    int repI = 0;  // slots are sorted by count, so slot 0 is the most numerous identity
    if (n > 1 && P_RepMode != 1) {
        const int mean = urn_mean_value(p, us);
        int bestDiff = 1 << 30;
        for (int i = 0; i < n; ++i) {
            const int diff = std::abs(PieceValue[slot_type(slots[i])] - mean);
            if (diff < bestDiff) { bestDiff = diff; repI = i; }
        }
    }
    auto subPv = [&](Move mm, int d) {  // open-window value of an outcome, searched as a PV node
        StateInfo st;
        TT.prefetch(p.key_after(mm) ^ ttSalt);
        if (useNN) {
            nnue::make_dirty(p, mm, dirty[ply + 1]);
            accOk[ply + 1] = 0;
        }
        p.do_move(mm, st);
        bump();
        Value v = VALUE_NONE;
        if (P_ChanceMixTT) {
            bool hit;
            TTEntry e{};
            TT.probe(tkey(), hit, e);
            if (hit && e.depth >= d && Bound(e.genBound & 3) == BOUND_EXACT) {
                const Value tv = value_from_tt(e.value, ply + 1);
                if (tv != VALUE_NONE && std::abs(tv) < VALUE_KNOWN_WIN) v = -tv;
            }
        }
        if (v == VALUE_NONE) v = -search(-VALUE_INFINITE, VALUE_INFINITE, d, ply + 1, true, false);
        p.undo_move(mm, st);
        return v;
    };
    // Control variate: E[v] ~ sum_i p_i v_s(i) + c (v_deep(rep) - v_s(rep)) in expected-score space, with probe values
    // v_s (static, depth 0 or depth pd) and c = p_rep + (1 - p_rep) rho; rho = 1 gives v_deep(rep) + sum_i p_i
    // (v_s(i) - v_s(rep)). The representative is searched with the window mapped through the correction.
    auto cvValue = [&](int pd, bool pvProbe, bool staticProbe, int cvI, bool repByProbe, int rhoPct, int top,
                       bool mixCall) -> Value {  // mixCall: a ChanceMix node (its probes may use the small net)
        double dl = 0.0, e0[NUM_SLOTS] = {};
#ifdef JQ_STATS
        const uint64_t sn0 = nodes.load(std::memory_order_relaxed);
#endif
        const bool skipRep = P_ChanceMixSkip && pvProbe && rhoPct == 0 && !repByProbe && top <= 1;
        {
            SmallProbeGuard sg(*this, mixCall && probeSmall);
            PROBE_DEPTH_GUARD(mixCall);
            for (int i = 0; i < n; ++i) {
                if (skipRep && i == cvI) continue;  // e0[cvI] = 0 cancels below
                const Move pm = make_move(from, to, slot_type(slots[i]));
                // ChanceMixLow: low-probability identities (never the full-depth ones) get shallower probes.
                const int pdi = mixCall && P_ChanceMixLow && i != cvI && i >= top &&
                                        cnt[i] * 100 < P_ChanceMixLow * p.urnTotal[us]
                                    ? std::max(0, pd - P_ChanceMixLowR)
                                    : pd;
#ifdef JQ_STATS
                const uint64_t pn0 = nodes.load(std::memory_order_relaxed);
#endif
                e0[i] = v2e(staticProbe ? stat(pm) : pvProbe && pdi > 0 ? subPv(pm, pdi) : sub(pm, -VALUE_INFINITE, VALUE_INFINITE, pdi));
#ifdef JQ_STATS
                if (mixCall && !staticProbe) {
                    auto& h = g_st.mixPct[i == cvI || i < top][std::clamp(cnt[i] * 100 / p.urnTotal[us], 0, 100)];
                    ++h[0];
                    h[1] += nodes.load(std::memory_order_relaxed) - pn0;
                }
#endif
                if (G.stop) return VALUE_ZERO;
            }
        }
        if (repByProbe) {
            double mean0 = 0.0, best = 1e9;
            for (int i = 0; i < n; ++i) mean0 = jqm::fma(double(cnt[i]) / p.urnTotal[us], e0[i], mean0);
            for (int i = 0; i < n; ++i)
                if (std::abs(e0[i] - mean0) < best) { best = std::abs(e0[i] - mean0); cvI = i; }
        }
        // Further full-depth identities (least numerous of them first): each is searched with the window that the
        // current values of the others leave for it, and its value (or bound) replaces its probe value.
        double deepMass = 0.0;
        for (int j = top > 1 ? std::min(top, n) - 1 : -1; j >= 0; --j) {
            if (j == cvI) continue;
            const double pj = double(cnt[j]) / p.urnTotal[us];
            double offj = 0.0;
            for (int i = 0; i < n; ++i)
                if (i != j) offj = jqm::fma(double(cnt[i]) / p.urnTotal[us], e0[i], offj);
            const double aj = (v2e(alpha) - offj) / pj, bj = (v2e(beta) - offj) / pj;
            if (aj >= eMax || bj <= eMin) continue;  // this identity alone cannot move the node across the window
            const Value caj = aj <= eMin ? -VALUE_INFINITE : std::max(Value(e2v_floor(aj)), -SAT);
            Value cbj = bj >= eMax ? VALUE_INFINITE : std::min(Value(e2v_ceil(bj)), SAT);
            if (cbj <= caj) cbj = caj + 1;
            const Value vj = sub(make_move(from, to, slot_type(slots[j])), caj, cbj, depth);
            if (G.stop) return VALUE_ZERO;
            e0[j] = std::clamp(v2e(vj), eMin, eMax);
            deepMass += pj;
        }
        for (int i = 0; i < n; ++i) dl = jqm::fma(double(cnt[i]) / p.urnTotal[us], e0[i] - e0[cvI], dl);
        const double pr = double(cnt[cvI]) / p.urnTotal[us];
        const double c = rhoPct >= 100 ? 1.0 : pr + (1.0 - pr - deepMass) * rhoPct / 100.0;
        const double off = jqm::fma(1.0 - c, e0[cvI], dl);  // E = c e(v_deep) + off
        const double a = (v2e(alpha) - off) / c, b = (v2e(beta) - off) / c;
        if (a >= eMax) return alpha;
        if (b <= eMin) return beta;
        Value ca = a <= eMin ? -VALUE_INFINITE : std::max(Value(e2v_floor(a)), -SAT);
        Value cb = b >= eMax ? VALUE_INFINITE : std::min(Value(e2v_ceil(b)), SAT);
        if (cb <= ca) cb = ca + 1;
        auto mapped = [&](Value x) {
            return std::clamp(Value(e2v_r(std::clamp(jqm::fma(c, v2e(x), off), eMin, eMax))), -SAT, SAT);
        };
        if (P_ChanceNW && !pvNode && beta - alpha == 1) {
            // Only the side of beta matters here. A window wider than one point would be searched by a non-PV node
            // without re-searches, and a value inside it is only a bound; so the identity gets the null window at the
            // smallest value whose mapped result reaches beta.
            Value lo = std::max(ca, Value(-SAT)), hi = std::min(cb, SAT);
            while (hi < SAT && mapped(hi) < beta) hi = std::min(SAT, Value(hi + 8));
            while (lo > -SAT && mapped(lo) >= beta) lo = std::max(Value(-SAT), Value(lo - 8));
            if (mapped(hi) < beta) return alpha;
            if (mapped(lo) >= beta) return beta;
            while (hi - lo > 1) {
                const Value mid = lo + (hi - lo) / 2;
                if (mapped(mid) >= beta) hi = mid; else lo = mid;
            }
            ca = hi - 1;
            cb = hi;
        }
#ifdef JQ_STATS
        const uint64_t sn1 = nodes.load(std::memory_order_relaxed);
#endif
        const Value v = sub(make_move(from, to, slot_type(slots[cvI])), ca, cb, depth);
        if (G.stop) return VALUE_ZERO;
#ifdef JQ_STATS
        {
            const uint64_t sn2 = nodes.load(std::memory_order_relaxed);
            if (pvProbe) { ++g_st.mixCalls; g_st.mixProbeN += sn1 - sn0; g_st.mixDeepN += sn2 - sn1; }
            if (pvProbe && !pvNode) {
                const int k = (pd > 0 ? 3 : 0) + (v <= ca ? 0 : v >= cb ? 2 : 1);
                ++g_st.mixK[k][0]; g_st.mixK[k][1] += sn1 - sn0; g_st.mixK[k][2] += sn2 - sn1;
            }
            else { ++g_st.cvCalls; g_st.cvProbeN += sn1 - sn0; g_st.cvDeepN += sn2 - sn1; }
        }
#endif
        const Value r = mapped(v);
        if (v <= ca) return std::min(r, alpha);
        if (v >= cb) return std::max(r, beta);
        return r;  // exact up to rounding
    };
    struct LayerGuard {
        int& c;
        explicit LayerGuard(int& x) : c(x) { ++c; }
        ~LayerGuard() { --c; }
    };
    // Outcomes to search: identity slot, urn count (probability mass) and depth.
    int oSlot[NUM_SLOTS], oCnt[NUM_SLOTS], oDepth[NUM_SLOTS], k = 0;
    int* counter = &chanceLayers;
    const bool xDeep = P_ChanceXDepth && qply < 0 && depth >= P_ChanceXDepth;
    if (n == 1 || qply >= 1 ||
        (!xDeep && (chanceLayers > maxLayers || (P_ChanceTail == 1 && chanceLayers >= maxLayers)))) {
        const Move rm = make_move(from, to, slot_type(slots[repI]));
        // RevealCheck: identities whose reveal checks differently from the representative's are their own
        // outcomes; all the others share the representative's search and probability mass.
        const bool cvDeep = P_ChanceCV && depth >= P_ChanceCV;
        const bool cvLow = !cvDeep && P_ChanceCVLow && depth >= P_ChanceCVLow;
        if ((cvDeep || cvLow) && n > 1 && qply < 0 && alpha < SAT && beta > -SAT) {
            STAT(chance);
            const int pd = P_ChanceCVGap ? std::max(0, depth - P_ChanceCVGap) : std::min(P_ChanceCVProbe, depth - 1);
            return sat(cvValue(pd, false, !(cvDeep && !P_ChanceCVStatic), repI, P_RepMode == 2, P_ChanceCVRho, 1, false));
        }
        if (n == 1 || (qply >= 0 && !P_RevealCheckQs) || revealSplits >= P_RevealCheck || !p.reveal_may_check(to))
            return sub(rm, alpha, beta, depth);
        const bool repCheck = p.reveal_gives_check(from, to, slot_type(slots[repI]));
        oSlot[0] = slots[repI];
        oCnt[0] = cnt[repI];
        oDepth[0] = depth;
        k = 1;
        for (int i = 0; i < n; ++i) {
            if (i == repI) continue;
            if (p.reveal_gives_check(from, to, slot_type(slots[i])) != repCheck) {
                oSlot[k] = slots[i];
                oCnt[k] = cnt[i];
                oDepth[k++] = depth;
            } else
                oCnt[0] += cnt[i];
        }
        if (k == 1) return sub(rm, alpha, beta, depth);
        counter = &revealSplits;
    } else {
        if (P_ChanceMix && qply < 0 && depth >= P_ChanceMixMin && alpha < SAT && beta > -SAT &&
            (!P_ChanceMixPv || (pvNode && beta - alpha > 1))) {
            STAT(chance);
            LayerGuard guard(chanceLayers);
            const int mixR = P_ChanceMixPvR && pvNode && beta - alpha > 1 && (!P_ChanceMixPvPly || ply < P_ChanceMixPvPly)
                                 ? P_ChanceMixPvR : P_ChanceMix;
            const int pd = depth - mixR - (P_ChanceMixDiv ? depth / P_ChanceMixDiv : 0);
            if (P_ChanceFut && !pvNode && beta - alpha == 1 && pd <= 0 && depth <= P_RfpDepth &&
                (P_ChanceMixRho0 >= 0 ? P_ChanceMixRho0 : P_ChanceMixRho) == 0 && P_ChanceMixTop == 1 &&
                std::abs(alpha) < VALUE_KNOWN_WIN &&
                staticEv[ply] != VALUE_NONE && staticEv[ply] + P_ChanceFutPre * depth <= alpha) {
                // After a reveal that gives no check the opponent can stand pat, so a quiescence probe is at most the
                // static value. If the static values of the other identities already leave the full-depth identity
                // more than the reverse-futility margin short, its search would be cut at once and the node fails low.
                const double pr = double(cnt[0]) / p.urnTotal[us];
                double offS = 0.0;
                Value sRep = VALUE_NONE;
                bool ok = true;
                SmallProbeGuard sg(*this, probeSmall);  // static probe values: small net (no TT access here)
                PROBE_DEPTH_GUARD(true);
                for (int i = 0; i < n && ok; ++i) {
                    const Move pm = make_move(from, to, slot_type(slots[i]));
                    StateInfo st;
                    if (useNN) {
                        nnue::make_dirty(p, pm, dirty[ply + 1]);
                        accOk[ply + 1] = 0;
                    }
                    p.do_move(pm, st);
                    bump();
                    if (p.checkHist.back() != 0) ok = false;
                    const Value sv = ok ? Value(-corrected(eval_at(ply + 1))) : VALUE_ZERO;
                    p.undo_move(pm, st);
                    if (i == 0) sRep = sv; else offS = jqm::fma(double(cnt[i]) / p.urnTotal[us], v2e(sv), offS);
                    // The remaining identities at their best cannot rescue the node either: stop early.
                    if (ok && i == 0 && sRep >= VALUE_KNOWN_WIN) ok = false;
                }
                if (ok && jqm::fma(pr, v2e(sRep + P_RfpMargin * depth), offS) <= v2e(alpha)) {
                    STAT(rfp);
                    const Value r = Value(e2v_r(std::clamp(jqm::fma(pr, v2e(sRep), offS), eMin, eMax)));
                    return sat(std::min(alpha, r));
                }
            }
            return sat(cvValue(std::max(0, pd), P_ChanceMixProbe != 0, false, 0, false,
                               pd <= 0 && P_ChanceMixRho0 >= 0 ? P_ChanceMixRho0 : P_ChanceMixRho, P_ChanceMixTop,
                               true));
        }
        // At the outermost permitted layer, outcomes are resolved by quiescence only.
        const int d = chanceLayers < maxLayers || xDeep ? depth : P_ChanceTail == 2 ? depth / 2 : 0;
        const int dRed = std::max(0, d - P_ChanceRed);  // non-representative outcomes
        for (int i = 0; i < n; ++i) {
            oSlot[k] = slots[i];
            oCnt[k] = cnt[i];
            int di = i == repI ? d : dRed;
            if (i != repI && P_ChanceRedLow && cnt[i] * 100 < P_ChanceRedLow * p.urnTotal[us]) di = std::max(0, di - 1);
            oDepth[k++] = di;
        }
    }
    STAT(chance);
    STATN(outcomes, k);
    if (alpha >= SAT) return alpha;
    if (beta <= -SAT) return beta;
    LayerGuard guard(*counter);

    const double total = p.urnTotal[us];
    const double a = v2e(alpha), b = v2e(beta);
    if (P_ChanceMode == 0) {
        // Exact Star1: child windows guarantee that any child failure makes this node fail.
        double S = 0.0, R = 1.0;
        for (int i = 0; i < k; ++i) {
            const double pi = oCnt[i] / total;
            R = i == k - 1 ? 0.0 : R - pi;
            const double lo = jqm::fma(-R, eMax, a - S) / pi;
            const double hi = jqm::fma(-R, eMin, b - S) / pi;
            if (lo >= eMax) return sat(alpha);
            if (hi <= eMin) return sat(beta);
            const Value ca = lo <= eMin ? -VALUE_INFINITE : std::max(Value(e2v_floor(lo)), -SAT);
            Value cb = hi >= eMax ? VALUE_INFINITE : std::min(Value(e2v_ceil(hi)), SAT);
            if (cb <= ca) cb = ca + 1;
            const Value v = sub(make_move(from, to, slot_type(oSlot[i])), ca, cb, oDepth[i]);
            if (G.stop) return VALUE_ZERO;
            if (v <= ca) return sat(alpha);
            if (v >= cb) return sat(beta);
            S = jqm::fma(pi, v2e(v), S);
        }
        const double e = std::clamp(S, eMin, eMax);
        const Value r = std::clamp(Value(e2v_r(e)), -SAT, SAT);
        return std::clamp(r, alpha, beta);
    }
    // Probe mode: every outcome is searched with the parent's window and the fail-soft results are
    // averaged in expected-score space. Sound early exits use one-sided bounds of finished outcomes.
    Value wa = alpha, wb = beta;
    if (P_ChancePvW && pvNode && qply < 0 && beta - alpha > 1) {
        wa = std::max(alpha - P_ChancePvW, -VALUE_INFINITE);
        wb = std::min(beta + P_ChancePvW, VALUE_INFINITE);
    }
    const bool tailFix = qply < 0 && k > 1 && (P_ChanceTailFix == 2 || (P_ChanceTailFix == 1 && pvNode && beta - alpha > 1));
    double S = 0.0, sLo = 0.0, sHi = 0.0, R = 1.0;
#ifdef JQ_STATS
    struct ProbeStat {
        Worker& w; uint64_t n0;
        explicit ProbeStat(Worker& x) : w(x), n0(x.nodes.load(std::memory_order_relaxed)) {}
        ~ProbeStat() { ++g_st.probeCalls; g_st.probeN += w.nodes.load(std::memory_order_relaxed) - n0; }
    } probeStat(*this);
#endif
    for (int i = 0; i < k; ++i) {
        const double pi = oCnt[i] / total;
        R = i == k - 1 ? 0.0 : R - pi;
        const Move om = make_move(from, to, slot_type(oSlot[i]));
        const Value v = sub(om, wa, wb, oDepth[i]);
        if (G.stop) return VALUE_ZERO;
        double e = v2e(v);
        bool tailed = false;
        if (tailFix && (v <= wa || v >= wb) && std::abs(v) < SAT) {
            // The probe gives the side; the open-window search at reduced depth gives the size.
            const Value t = subPv(om, std::max(0, oDepth[i] - P_ChanceTailR));
            if (G.stop) return VALUE_ZERO;
            e = v2e(v <= wa ? std::min(v, t) : std::max(v, t));
            tailed = true;
        }
        // (ISA build, -ffp-contract=off): S/sLo/sHi += pi * x exactly as the reference build (g++ -O3
        // -mfma, contraction on) rounds them: fused on some paths only (decoded from its machine code).
        const bool lowF = v <= wa, highF = v >= wb;
        if (tailFix && lowF) {
            S = jqm::fma(pi, e, S); sLo += pi * eMin; sHi += pi * e;
        } else if (tailFix && highF && !tailed) {
            sLo = jqm::fma(pi, e, sLo); S = jqm::fma(pi, e, S); sHi += pi * eMax;
        } else {
            const double t = pi * e;
            S += t; sLo += lowF ? pi * eMin : t; sHi += highF ? pi * eMax : t;
        }
        if (jqm::fma(R, eMin, sLo) >= b) return sat(std::max(beta, Value(e2v_r(std::clamp(jqm::fma(R, eMin, sLo), eMin, eMax)))));
        if (jqm::fma(R, eMax, sHi) <= a) return sat(std::min(alpha, Value(e2v_r(std::clamp(jqm::fma(R, eMax, sHi), eMin, eMax)))));
    }
    return std::clamp(Value(e2v_r(std::clamp(S, eMin, eMax))), -SAT, SAT);
}

Value Worker::qsearch(Value alpha, Value beta, int ply, int qply) {
    STAT(qs);
    check_time();
    if (G.stop) return VALUE_ZERO;
    pvLen[ply] = 0;
    Position& p = pos;
    if (ply > selDepth) selDepth = ply;
    if (ply >= MAX_PLY) return eval_at(ply);
    const bool inCheck = p.checkHist.back() != 0;
    // The only quiet moves of quiescence are check evasions, so a cycle inside it is one of mutual checks (a cannon
    // and its screen trading discovered checks with a cannon on the next file, say). Without this test such a cycle
    // ran on to MAX_PLY.
    if (inCheck && qply > 0) {
        const int rs = p.repetition_status();
        if (rs == 1) return VALUE_DRAW;
        if (rs == 2) return VALUE_KNOWN_WIN;
        if (rs == 3) return -VALUE_KNOWN_WIN;
    }
    const Color us = p.stm;
    bool ttHit = false;
    TTEntry* tte = nullptr;
    TTEntry ttd{};
    if (P_QsTT) {
        tte = TT.probe(tkey(), ttHit, ttd);
        if (ttHit) {
            const Value tv = value_from_tt(ttd.value, ply);
            const Bound tb = Bound(ttd.genBound & 3);
            if (tv != VALUE_NONE && (tb == BOUND_EXACT || (tb == BOUND_LOWER && tv >= beta) || (tb == BOUND_UPPER && tv <= alpha)))
                return tv;
        }
    }
    const Value oldAlpha = alpha;
    Value bestValue = -VALUE_INFINITE, standPat = VALUE_NONE, rawStand = VALUE_NONE;
    if (!inCheck) {
        rawStand = ttHit && ttd.eval != VALUE_NONE ? Value(ttd.eval) : eval_at(ply);
        standPat = corrected(rawStand);
        if (standPat >= beta) {
            if (P_QsTT && !ttHit) TT.store(tte, tkey(), value_to_tt(standPat, ply), BOUND_LOWER, 0, MOVE_NONE, rawStand);
            return standPat;
        }
        if (standPat > alpha) alpha = standPat;
        bestValue = standPat;
    }
    Move moves[MAX_MOVES];
    int scores[MAX_MOVES];
    const int n = inCheck ? p.gen_moves(moves) : p.gen_captures(moves);
    const Move qsTT = P_QsTTMove && ttHit ? Move(ttd.move) : MOVE_NONE;
    for (int i = 0; i < n; ++i) {
        const int to = to_sq(moves[i]);
        if (moves[i] == qsTT) scores[i] = 1 << 30;
        else if (p.board[to]) scores[i] = (1 << 20) + piece_value(to) * 16 - piece_value(from_sq(moves[i])) / 16;
        else scores[i] = history[us][from_sq(moves[i])][to];
    }
    Move bestMove = MOVE_NONE;
    int legalCount = 0;
    Position::LegalCtx lc;
    p.legal_ctx(lc);
    for (int i = 0; i < n; ++i) {
        int bi = i;
        for (int j = i + 1; j < n; ++j)
            if (scores[j] > scores[bi]) bi = j;
        std::swap(moves[i], moves[bi]);
        std::swap(scores[i], scores[bi]);
        const Move m = moves[i];
#ifdef JQ_MG_OLD
        if (!p.legal(m)) continue;
        if (g_chaseRule == CHASE_RULE_SKYRULE && p.skyrule_forbids_legal_move(m)) continue;
        ++legalCount;
        if (!inCheck) {
            const int vv = piece_value(to_sq(m));
            if (standPat + vv + P_DeltaMargin <= alpha) continue;
#else
        // out of check legalCount is unused, so the delta test may come before the legality test (and
        // before the SkyRule filter, which needs a legal move and only drops moves: same set of searched moves)
        if (!inCheck && standPat + piece_value(to_sq(m)) + P_DeltaMargin <= alpha) continue;
        if (!p.legal(m, lc)) continue;
        if (g_chaseRule == CHASE_RULE_SKYRULE && p.skyrule_forbids_legal_move(m)) continue;
        ++legalCount;
        if (!inCheck) {
            const int vv = piece_value(to_sq(m));
#endif
            if (!p.is_dark_move(m)) TT.prefetch(p.key_after(m) ^ ttSalt);  // ahead of the exchange evaluation
            const int av = piece_value(from_sq(m));
            if (P_QsSee ? !see_ge(m, -P_QsSeeMargin) : (av > vv + P_BadCapMargin && p.attacked(~us, to_sq(m)))) continue;
            if (p.is_dark_move(m) && qply >= P_QsDarkPly) continue;
        }
        const Value v = child(m, alpha, beta, 0, ply, false, false, qply);
        if (G.stop) return VALUE_ZERO;
        if (v > bestValue) {
            bestValue = v;
            if (v > alpha) {
                bestMove = m;
                if (v >= beta) break;
                alpha = v;
            }
        }
    }
    if (inCheck && legalCount == 0) return mated_in(ply);
    if (P_QsTT) {
        const Bound b = bestValue >= beta ? BOUND_LOWER : bestValue > oldAlpha ? BOUND_EXACT : BOUND_UPPER;
        TT.store(tte, tkey(), value_to_tt(bestValue, ply), b, 0, P_QsTTMove ? bestMove : MOVE_NONE, rawStand);
    }
    return bestValue;
}

Value Worker::search(Value alpha, Value beta, int depth, int ply, bool pvNode, bool cutNode) {
    Position& p = pos;
    const bool inCheck = p.checkHist.back() != 0;
    if (inCheck && P_CheckExt && ply < (P_CheckExt == 1 ? 2 * rootDepth + 4 : rootDepth)) { ++depth; STAT(cext); }
    if (depth <= 0) return qsearch(alpha, beta, ply, 0);
    STAT(search);
    if (pvNode) STAT(pv);
    check_time();
    if (G.stop) return VALUE_ZERO;
    pvLen[ply] = 0;
    const bool root = ply == 0;
    if (ply > selDepth) selDepth = ply;
    if (!root) {
        if (ply >= MAX_PLY) return inCheck ? VALUE_DRAW : eval_at(ply);
        const int rs = p.repetition_status();
        if (rs == 1) return VALUE_DRAW;
        if (rs == 2) return VALUE_KNOWN_WIN;
        if (rs == 3) return -VALUE_KNOWN_WIN;
        if (p.noCapture >= NO_CAPTURE_LIMIT) return p.has_legal_move() ? VALUE_DRAW : mated_in(ply);  // mate precedes the clock
        alpha = std::max(mated_in(ply), alpha);
        beta = std::min(mate_in(ply + 1), beta);
        if (alpha >= beta) return alpha;
    }
    const Color us = p.stm;

    bool ttHit;
    TTEntry ttd{};
    TTEntry* tte = TT.probe(tkey(), ttHit, ttd);
    const Value ttValue = ttHit ? value_from_tt(ttd.value, ply) : VALUE_NONE;
    const Bound ttBound = ttHit ? Bound(ttd.genBound & 3) : BOUND_NONE;
    Move ttMove = ttHit ? Move(ttd.move) : MOVE_NONE;
    if (root && bestMove != MOVE_NONE) ttMove = bestMove;
    const Move excl = excluded[ply];
    if (!pvNode && !excl && ttHit && ttd.depth >= depth && ttValue != VALUE_NONE &&
        (ttBound == BOUND_EXACT || (ttBound == BOUND_LOWER && ttValue >= beta) ||
         (ttBound == BOUND_UPPER && ttValue <= alpha))) {
        STAT(ttcut);
        return ttValue;
    }

    Value eval = VALUE_NONE, rawEval = VALUE_NONE;
    staticEv[ply] = VALUE_NONE;
    if (!inCheck) {
        rawEval = ttHit && ttd.eval != VALUE_NONE ? Value(ttd.eval) : eval_at(ply);
        eval = corrected(rawEval);
        staticEv[ply] = eval;
        if (ttValue != VALUE_NONE && std::abs(ttValue) < VALUE_KNOWN_WIN &&
            (ttBound & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttValue;
    }

    const bool improving = P_Improving && !inCheck && ply >= 2 && staticEv[ply - 2] != VALUE_NONE &&
                           staticEv[ply] > staticEv[ply - 2];
    if (!pvNode && !inCheck) {
        if (depth <= P_RazorDepth && !excl && eval + P_RazorMargin * depth < alpha && alpha > -VALUE_KNOWN_WIN) {
            const Value v = qsearch(alpha, alpha + 1, ply, 0);
            if (G.stop) return VALUE_ZERO;
            if (v <= alpha) return v;
        }
        if (depth <= P_RfpDepth && eval - P_RfpMargin * (depth - (improving ? 1 : 0)) >= beta && eval < VALUE_KNOWN_WIN) {
            STAT(rfp);
            return eval;
        }
        if (!excl && depth >= 2 && eval >= beta && ply > 0 && curMove[ply - 1] != MOVE_NULL && p.has_major(us) &&
            beta > -VALUE_KNOWN_WIN) {
            const int R = P_NullBase + depth / P_NullDepthDiv + std::min((eval - beta) / P_NullDiv, 3);
            STAT(nullTry);
            StateInfo st;
            if (useNN) {
                dirty[ply + 1] = nnue::Dirty{};
                accOk[ply + 1] = 0;
            }
            p.do_null(st);
            curMove[ply] = MOVE_NULL;
            movedPc[ply] = -1;
            bump();
            const Value v = -search(-beta, -beta + 1, depth - R, ply + 1, false, !cutNode);
            p.undo_null(st);
            if (G.stop) return VALUE_ZERO;
            if (v >= beta) { STAT(nullCut); return v >= VALUE_KNOWN_WIN ? beta : v; }
        }
    }
    // ProbCut: a face-up capture that beats beta + margin in quiescence and at depth - 4 cuts the node.
    if (P_ProbCut && !pvNode && !inCheck && !excl && depth >= P_ProbCutDepth && std::abs(beta) < VALUE_KNOWN_WIN / 2 &&
        !(ttHit && ttd.depth >= depth - 3 && ttValue != VALUE_NONE && ttValue < beta + P_ProbCut)) {
        const Value pcBeta = beta + P_ProbCut;
        Move pcm[MAX_MOVES];
        int pcs[MAX_MOVES];
        const int npc = p.gen_captures(pcm);
        Position::LegalCtx lc;
        p.legal_ctx(lc);
        for (int i = 0; i < npc; ++i) pcs[i] = piece_value(to_sq(pcm[i])) * 16 - piece_value(from_sq(pcm[i])) / 16;
        for (int i = 0; i < npc; ++i) {
            int bi = i;
            for (int j = i + 1; j < npc; ++j)
                if (pcs[j] > pcs[bi]) bi = j;
            std::swap(pcm[i], pcm[bi]);
            std::swap(pcs[i], pcs[bi]);
            const Move m = pcm[i];
            if (p.is_dark_move(m) || !p.legal(m, lc)) continue;
            const int to = to_sq(m), vv = piece_value(to), av = piece_value(from_sq(m));
            const int gain = p.attacked(~us, to) ? vv - av : vv;
            if (eval == VALUE_NONE || eval + gain < pcBeta) continue;
            Value v = child(m, pcBeta - 1, pcBeta, 0, ply, false, false, 0);
            if (v >= pcBeta && !G.stop) v = child(m, pcBeta - 1, pcBeta, depth - 4, ply, false, !cutNode, -1);
            if (G.stop) return VALUE_ZERO;
            if (v >= pcBeta) {
                TT.store(tte, tkey(), value_to_tt(v, ply), BOUND_LOWER, depth - 3, m, rawEval);
                return v;
            }
        }
    }
    if (!root && ttMove == MOVE_NONE && depth >= 4) --depth;  // internal iterative reduction

    // Singular extension: if every alternative fails well below the TT value, extend the TT move.
    int singularExt = 0;
    if (P_SeDepth && !root && !excl && depth >= P_SeDepth && ttMove != MOVE_NONE && (ttBound & BOUND_LOWER) &&
        ttd.depth >= depth - 3 && ttValue != VALUE_NONE && std::abs(ttValue) < VALUE_KNOWN_WIN && !p.is_dark_move(ttMove)) {
        const Value sBeta = ttValue - P_SeMargin * depth / 8;
        excluded[ply] = ttMove;
        const Value sv = search(sBeta - 1, sBeta, (depth - 1) / 2, ply, false, cutNode);
        excluded[ply] = MOVE_NONE;
        if (G.stop) return VALUE_ZERO;
        if (sv < sBeta) singularExt = 1;
        else if (sBeta >= beta) return sBeta;  // multi-cut: another move also beats beta
    }

    // Threatened face-down pieces of the side to move: evaluated lazily per from-square, only when a ThreatRev
    // option needs it (and in JQ_STATS builds, for the stats7 counters), only for face-down movers.
    const bool trvRootOn = root && (P_ThreatRevRoot || P_ThreatRevScale);
    int8_t trvSq[16], trvSt[16];
    int16_t trvLoss[16];
    int trvN = 0, trvDV[2] = {-1, -1};
    auto trv_get = [&](int s) -> int {  // index into trvSq/trvSt/trvLoss
        for (int j = 0; j < trvN; ++j)
            if (trvSq[j] == s) return j;
        if (trvDV[0] < 0) trvDV[0] = urn_mean_value(p, RED), trvDV[1] = urn_mean_value(p, BLACK);
        int l;
        STAT(trvCalls);
        const int st = search::threat_reveal(p, s, &l, trvDV);
        trvSq[trvN] = int8_t(s); trvSt[trvN] = int8_t(st); trvLoss[trvN] = int16_t(std::min(l, 30000));
        return trvN < 15 ? trvN++ : trvN;
    };
    auto trv_hit = [&](Move mm) { return trvSt[trv_get(from_sq(mm))] == 2; };

    Move moves[MAX_MOVES];
    int scores[MAX_MOVES];
    const int n = p.gen_moves(moves);
    const Move prev = ply > 0 ? curMove[ply - 1] : MOVE_NONE;
    const Move counter = prev && prev != MOVE_NULL ? counters[p.board[to_sq(prev)]][to_sq(prev)] : MOVE_NONE;
    for (int i = 0; i < n; ++i) {
        const Move m = moves[i];
        const int to = to_sq(m);
        if (m == ttMove) scores[i] = 1 << 30;
        else if (p.board[to]) {
            const int vv = piece_value(to), av = piece_value(from_sq(m));
            const bool bad = P_SeeOrder ? !see_ge(m, -P_BadCapMargin) : (av > vv + P_BadCapMargin && p.attacked(~us, to));
            scores[i] = (bad ? -(1 << 20) : (1 << 28)) + vv * 16 - av / 16 +
                        (P_CapHist ? capHist[pidx16(p.board[from_sq(m)])][to][type_of(p.board[to])] / 8 : 0);
        } else if (m == killers[ply][0]) scores[i] = 1 << 27;
        else if (m == killers[ply][1]) scores[i] = (1 << 27) - 1;
        else if (m == counter) scores[i] = 1 << 26;
        else {
            scores[i] = quiet_stat(us, m, pidx16(p.board[from_sq(m)]), ply);
            if (P_ThreatRevOrder && p.is_dark_move(m) && trv_hit(m)) scores[i] += P_ThreatRevOrder;
            if (P_SeeQuietOrder && !see_ge(m, -P_SeeQuietOrder)) scores[i] -= 1 << 18;
        }
    }

    Value bestValue = -VALUE_INFINITE;
    Move best = MOVE_NONE;
    int legalCount = 0, nq = 0, nc = 0;
    STAT(loopNodes);
    if (ttMove != MOVE_NONE) STAT(loopTT);
    Move quiets[64], caps[32];
    Position::LegalCtx lc;
    p.legal_ctx(lc);
    for (int i = 0; i < n; ++i) {
        int bi = i;
        for (int j = i + 1; j < n; ++j)
            if (scores[j] > scores[bi]) bi = j;
        std::swap(moves[i], moves[bi]);
        std::swap(scores[i], scores[bi]);
        const Move m = moves[i];
        if (m == excl || !p.legal(m, lc)) continue;
        if (g_chaseRule == CHASE_RULE_SKYRULE && p.skyrule_forbids_legal_move(m)) continue;
        if (root && !G.rootMoves.empty() && std::find(G.rootMoves.begin(), G.rootMoves.end(), m) == G.rootMoves.end())
            continue;
        const bool deferOn = P_SmpDefer && G.nThreads > 1 && depth >= P_SmpDefer;
        const uint64_t bk = deferOn ? busy_key(tkey(), m) : 0;
        if (deferOn && legalCount > 0 && scores[i] != DEFER_SCORE &&
            busy_slot(bk).load(std::memory_order_relaxed) == bk) {
            scores[i] = DEFER_SCORE;  // another thread is on it: try the remaining moves first
            --i;
            continue;
        }
        if (deferOn) busy_slot(bk).store(bk, std::memory_order_relaxed);
        ++legalCount;
        const bool capture = p.board[to_sq(m)] != NO_PIECE;
        const bool dark = p.is_dark_move(m);
        const bool givesCheck = !dark && p.gives_check(m);
#if !defined(JQ_MG_OLD) && !defined(JQ_NO_HINT)
        if (!dark) p.check_hint(m, givesCheck);  // do_move reuses it (skips check_after)
#endif
        const bool quiet = !capture;
        const int hist = quiet ? quiet_stat(us, m, pidx16(p.board[from_sq(m)]), ply) : 0;

        if (!root && bestValue > VALUE_MATED_IN_MAX_PLY && !inCheck && quiet && !givesCheck && depth <= P_PruneDepth) {
            const int lmp = P_LmpBase + depth * depth;
            if (nq >= (P_Improving && !improving ? lmp / 2 + 1 : lmp)) { STAT(lmp); continue; }
            if (eval + P_FutBase + P_FutMul * depth <= alpha && hist < 4000) { STAT(fut); continue; }
            if (P_HistPrune && depth <= 3 && hist < -P_HistPrune * depth) continue;
        }
        if (!root && bestValue > VALUE_MATED_IN_MAX_PLY && !inCheck && !givesCheck && depth <= P_SeePruneDepth &&
            (capture ? P_SeeCap && !see_ge(m, -P_SeeCap * depth) : P_SeeQuiet && !see_ge(m, -P_SeeQuiet * depth * depth)))
            continue;
        const int newDepth = depth - 1 + (m == ttMove ? singularExt : 0);
#ifdef JQ_STATS
        if (dark) {
            const int st = trvSt[trv_get(from_sq(m))];
            if (root) { STAT(trvRootDark); if (st) STAT(trvRootAtk); if (st == 2) STAT(trvRootThr); }
            else { STAT(trvTreeDark); if (st) STAT(trvTreeAtk); if (st == 2) STAT(trvTreeThr); }
        }
#endif
        Value pen = root && dark && (!P_RootDarkHid || p.darkCount[us] <= P_RootDarkHid) ? Value(P_RootDarkPen) : Value(0);
        if (trvRootOn && dark && trv_hit(m))  // root reveal bonus for a threatened face-down piece: shift opposite to RootDarkPen
            pen -= P_ThreatRevScale ? Value(std::min(300, trvLoss[trv_get(from_sq(m))] * P_ThreatRevScale / 100)) : Value(P_ThreatRevRoot);
        auto child = [&](Move mm, Value a, Value b, int d, int pl, bool pvn, bool cut, int qp) {
            if (!pen) return this->child(mm, a, b, d, pl, pvn, cut, qp);
            const Value sa = std::abs(a) < VALUE_KNOWN_WIN ? Value(a + pen) : a;
            const Value sb = std::abs(b) < VALUE_KNOWN_WIN ? Value(b + pen) : b;
            if (sa >= sb) return this->child(mm, a, b, d, pl, pvn, cut, qp);  // window straddles +-KNOWN_WIN: unshifted
            const Value r = this->child(mm, sa, sb, d, pl, pvn, cut, qp);
            return std::abs(r) < VALUE_KNOWN_WIN ? Value(r - pen) : r;
        };
        Value v;
        if (legalCount == 1) {
            v = child(m, alpha, beta, newDepth, ply, pvNode, false, -1);
        } else {
            int r = 0;
            if (depth >= 3 && quiet && !inCheck && !givesCheck) {
                r = lmrTable[std::min(depth, 63)][std::min(legalCount, 63)];
                if (pvNode) r -= 1;
                if (cutNode) r += 1;
                if (P_Improving && !improving) r += 1;
                if (m == killers[ply][0] || m == killers[ply][1] || m == counter) r -= 1;
                r -= hist / P_HistDiv;
                if (dark && P_ThreatRevLmr != 100 && trv_hit(m)) r = r * P_ThreatRevLmr / 100;  // replaces DarkLmr for threatened reveals
                else if (dark && P_DarkLmr != 100) r = r * P_DarkLmr / 100;
                r = std::clamp(r, 0, newDepth - 1);
            } else if (P_CapLmr && depth >= 3 && capture && !pvNode && !inCheck && !givesCheck && !see_ge(m, 0)) {
                r = std::clamp(P_CapLmr + (cutNode ? 1 : 0), 0, newDepth - 1);
            }
            if (r > 0) STAT(lmr);
            v = child(m, alpha, alpha + 1, newDepth - r, ply, false, true, -1);
            if (v > alpha && r > 0 && !G.stop) { STAT(lmrRe); v = child(m, alpha, alpha + 1, newDepth, ply, false, !cutNode, -1); }
            if (pvNode && v > alpha && v < beta && !G.stop) { STAT(pvsRe); v = child(m, alpha, beta, newDepth, ply, true, false, -1); }
        }
        if (deferOn) {
            uint64_t expect = bk;
            busy_slot(bk).compare_exchange_strong(expect, 0, std::memory_order_relaxed);
        }
        if (G.stop) return VALUE_ZERO;
        if (dark) STAT(darkSearched); else if (capture) STAT(capSearched); else STAT(quietSearched);
        if (quiet && nq < 64) quiets[nq++] = m;
        if (capture && nc < 32) caps[nc++] = m;
        if (v > bestValue) {
            bestValue = v;
            if (v > alpha) {
                best = m;
                if (pvNode) {
                    pv[ply][0] = m;
                    int len = 1;
                    if (!dark)
                        for (int k = 0; k < pvLen[ply + 1]; ++k) pv[ply][len++] = pv[ply + 1][k];
                    pvLen[ply] = len;
                }
                if (root) { iterBest = m; iterScore = v; }
                if (v >= beta) {
                    STAT(fh);
                    if (legalCount == 1) STAT(fh1);
                    STATN(fhMoves, legalCount);
#ifdef JQ_STATS
                    {
                        const int sc = scores[i];
                        const int cat = m == ttMove ? 0 : sc >= (1 << 27) + 2 ? 1 : sc >= (1 << 26) ? 2 : sc < -(1 << 19) ? 3 : dark ? 5 : 4;
                        ++g_st.fhCat[cat];
                        if (ttMove != MOVE_NONE) { ++g_st.fhTT; if (legalCount == 1) ++g_st.fh1TT; }
                    }
#endif
                    break;
                }
                alpha = v;
            }
        }
    }
    if (legalCount == 0) return excl ? alpha : mated_in(ply);  // no legal move loses, including stalemate
#ifdef JQ_STATS
    if (best == MOVE_NONE) { STAT(failLow); STATN(failLowMoves, legalCount); }
#endif

    if (best != MOVE_NONE && bestValue >= beta) {
        const int bonus = std::min(depth * depth * 8, 1600);
        if (p.board[to_sq(best)] == NO_PIECE) {
            update_quiet(us, best, pidx16(p.board[from_sq(best)]), ply, bonus);
            for (int k = 0; k < nq; ++k)
                if (quiets[k] != best) update_quiet(us, quiets[k], pidx16(p.board[from_sq(quiets[k])]), ply, -bonus);
            if (killers[ply][0] != best) { killers[ply][1] = killers[ply][0]; killers[ply][0] = best; }
            if (prev && prev != MOVE_NULL) counters[p.board[to_sq(prev)]][to_sq(prev)] = best;
        } else if (P_CapHist) {
            gravity(capHist[pidx16(p.board[from_sq(best)])][to_sq(best)][type_of(p.board[to_sq(best)])], bonus);
        }
        if (P_CapHist)
            for (int k = 0; k < nc; ++k)
                if (caps[k] != best) gravity(capHist[pidx16(p.board[from_sq(caps[k])])][to_sq(caps[k])][type_of(p.board[to_sq(caps[k])])], -bonus);
    }
    const Bound b = bestValue >= beta ? BOUND_LOWER : (pvNode && best != MOVE_NONE ? BOUND_EXACT : BOUND_UPPER);
    // Correction history learns from non-capture results whose bound points the right way.
    if (P_CorrHist && !inCheck && !excl && rawEval != VALUE_NONE && std::abs(bestValue) < VALUE_KNOWN_WIN &&
        !(best != MOVE_NONE && p.board[to_sq(best)] != NO_PIECE) &&
        ((bestValue < staticEv[ply] && bestValue < beta) || (bestValue > staticEv[ply] && best != MOVE_NONE)))
        update_corr(rawEval, bestValue, depth);
    if (!excl) TT.store(tte, tkey(), value_to_tt(bestValue, ply), b, depth, best, rawEval);
    return bestValue;
}

void Worker::report(int depth, Value v, Bound bound) {
    if (!info || !*info) return;
    const int64_t ms = std::max<int64_t>(1, elapsed());
    const uint64_t nn = total_nodes();
    std::string s = "info depth " + std::to_string(depth) + " seldepth " + std::to_string(selDepth) +
                    " multipv 1 score " + score_str(v);
    if (bound == BOUND_LOWER) s += " lowerbound";
    if (bound == BOUND_UPPER) s += " upperbound";
    s += " nodes " + std::to_string(nn) + " nps " + std::to_string(nn * 1000 / uint64_t(ms)) +
         " hashfull " + std::to_string(TT.hashfull()) + " time " + std::to_string(ms) + " pv";
    for (Move m : bestPv) s += " " + move_str(m);
    (*info)(s);
}

void Worker::iterate() {
    useNN = nnue::loaded();
    useHsm = useNN && nnue::hsm_loaded() && !nnue::is_abj();
    nnue::set_dual_skip(useHsm ? P_DualMode : 0, P_DualThr, P_DualHid, P_DualSkipMargin);
    probeSmall = useNN && P_ChanceProbeSmall && (nnue::dual() || useHsm);
    inSmallProbe = false;
    ttSalt = 0;
    if (useNN) {
        nnue::set_flush_denormals();
        hnn::bind_acc_stack(&acc[0].hnn, sizeof(acc[0]), int(sizeof acc / sizeof acc[0]));  // big-net side table (isa)
        nnue::refresh(pos, acc[0]);
        accOk[0] = nnue::NET_ALL;
    }
    Value prevScore = 0;
    Move lastBest = MOVE_NONE;  // time management (main thread)
    double instab = 0;
    std::fill(std::begin(excluded), std::end(excluded), MOVE_NONE);
    bestMove = MOVE_NONE;
    bestScore = -VALUE_INFINITE;
    completedDepth = 0;
    bestPv.clear();
    for (int d = 1; d < MAX_PLY - 8; ++d) {
        if (G.depthLimit && d > G.depthLimit) break;
        if (id > 0 && d > 2) {  // helper depth diversity
            static constexpr int SkipSize[20] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
            static constexpr int SkipPhase[20] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};
            const int i = (id - 1) % 20;
            if (P_SmpSkip == 1 && (d + id) % 3 == 0) continue;
            if (P_SmpSkip == 2 && ((d + SkipPhase[i]) / SkipSize[i]) % 2) continue;
        }
        rootDepth = d;
        selDepth = 0;
        Value delta = P_AspDelta;
        Value alpha = -VALUE_INFINITE, beta = VALUE_INFINITE;
        if (d >= 4 && std::abs(prevScore) < VALUE_KNOWN_WIN) {
            alpha = std::max(prevScore - delta, -VALUE_INFINITE);
            beta = std::min(prevScore + delta, VALUE_INFINITE);
        }
        Value v = 0;
        iterBest = MOVE_NONE;
        iterScore = -VALUE_INFINITE;
        while (true) {
            v = search(alpha, beta, d, 0, true, false);
            if (G.stop) break;
            // A result at an infinite bound cannot be improved by widening: take it as the value.
            if ((v <= alpha && alpha <= -VALUE_INFINITE) || (v >= beta && beta >= VALUE_INFINITE)) break;
            if (v <= alpha) {
                beta = (alpha + beta) / 2;
                alpha = std::max(v - delta, -VALUE_INFINITE);
            } else if (v >= beta) {
                beta = std::min(v + delta, VALUE_INFINITE);
                if (pvLen[0] > 0) { bestMove = pv[0][0]; bestPv.assign(pv[0], pv[0] + pvLen[0]); }
            } else
                break;
            delta += delta / 2 + 5;
            if (id == 0 && elapsed() > 1000) report(d, v, v <= alpha ? BOUND_UPPER : BOUND_LOWER);
        }
        if (G.stop) {
            if (iterBest != MOVE_NONE && iterScore >= bestScore && pvLen[0] > 0 && pv[0][0] == iterBest) {
                bestMove = iterBest;
                bestPv.assign(pv[0], pv[0] + pvLen[0]);
            }
            break;
        }
        const Value lastIterScore = prevScore;
        prevScore = v;
        completedDepth = d;
        bestScore = v;
        if (pvLen[0] > 0) {
            bestMove = pv[0][0];
            bestPv.assign(pv[0], pv[0] + pvLen[0]);
        }
        if (id == 0) {
            report(d, v, BOUND_EXACT);
            instab = instab * 0.5 + (d > 1 && bestMove != lastBest ? 1.0 : 0.0);
            lastBest = bestMove;
            if (G.mateLimit && std::abs(v) >= VALUE_MATE_IN_MAX_PLY
                && mate_moves(v) <= G.mateLimit)
                break;
            if (G.timed && !G.pondering) {
                double soft = double(G.optimumMs) * (G.clockMode ? P_TmSoft : 55) / 100;
                if (G.clockMode && P_TmStab)
                    soft *= std::clamp(jqm::fma(P_TmStab / 100.0, instab - 0.5, 1.0), 0.5, 2.5);
                if (G.clockMode && P_TmFall && d >= 5)
                    soft *= jqm::fma(P_TmFall / 100.0, std::clamp((lastIterScore - v) / 100.0, -0.5, 1.0), 1.0);
                if (elapsed() >= int64_t(soft)) break;
            }
            if (std::abs(v) >= VALUE_MATE_IN_MAX_PLY && d >= 2 * (VALUE_MATE - std::abs(v)) + 4) {
                if (G.timed) break;
            }
        }
    }
}

uint64_t perft_rec(Position& pos, int depth) {
    Move moves[MAX_MOVES];
    const int n = pos.gen_legal(moves);
    if (depth <= 1) return uint64_t(n);
    uint64_t total = 0;
    for (int i = 0; i < n; ++i) {
        const Move m = moves[i];
        StateInfo st;
        Move mm = m;
        if (pos.is_dark_move(m)) {
            // One representative reveal (first available slot) keeps the tree finite and deterministic.
            for (int sl = 0; sl < NUM_SLOTS; ++sl)
                if (pos.urn[pos.stm][sl] > 0) { mm = make_move(from_sq(m), to_sq(m), slot_type(sl)); break; }
        }
        pos.do_move(mm, st);
        total += perft_rec(pos, depth - 1);
        pos.undo_move(mm, st);
    }
    return total;
}

double perft_chance_rec(Position& pos, int depth, uint64_t& leaves) {
    if (depth == 0) { ++leaves; return 1.0; }
    Move moves[MAX_MOVES];
    const int n = pos.gen_legal(moves);
    double mass = 0.0;
    for (int i = 0; i < n; ++i) {
        const Move m = moves[i];
        if (!pos.is_dark_move(m)) {
            StateInfo st;
            pos.do_move(m, st);
            mass += perft_chance_rec(pos, depth - 1, leaves);
            pos.undo_move(m, st);
            continue;
        }
        const Color us = pos.stm;
        double sub = 0.0;
        for (int sl = 0; sl < NUM_SLOTS; ++sl) {
            if (pos.urn[us][sl] <= 0) continue;
            const double pr = double(pos.urn[us][sl]) / pos.urnTotal[us];
            const Move mm = make_move(from_sq(m), to_sq(m), slot_type(sl));
            StateInfo st;
            pos.do_move(mm, st);
            sub = jqm::fma(pr, perft_chance_rec(pos, depth - 1, leaves), sub);
            pos.undo_move(mm, st);
        }
        mass += sub;
    }
    return mass;
}

}  // namespace

std::vector<Param>& params() {
    static std::vector<Param> p = {
        {"ChanceK", &P_ChanceK, 100, 1000},  {"RfpMargin", &P_RfpMargin, 30, 300},
        {"RfpDepth", &P_RfpDepth, 1, 12},    {"NullBase", &P_NullBase, 1, 6},
        {"NullDiv", &P_NullDiv, 50, 800},    {"FutBase", &P_FutBase, 0, 400},
        {"FutMul", &P_FutMul, 30, 400},      {"LmpBase", &P_LmpBase, 1, 12},
        {"LmrBase", &P_LmrBase, 0, 200},     {"LmrDiv", &P_LmrDiv, 100, 500},
        {"AspDelta", &P_AspDelta, 5, 200},   {"QsDarkPly", &P_QsDarkPly, 0, 16},
        {"HistDiv", &P_HistDiv, 1000, 32000}, {"DeltaMargin", &P_DeltaMargin, 0, 1000},
        {"BadCapMargin", &P_BadCapMargin, -200, 400}, {"ChanceMode", &P_ChanceMode, 0, 1},
        {"MaxChanceLayers", &P_MaxChanceLayers, 0, 12}, {"ContHist", &P_ContHist, 0, 1}, {"ChanceTail", &P_ChanceTail, 0, 2},
        {"ChanceRed", &P_ChanceRed, 0, 8}, {"RevealCheck", &P_RevealCheck, 0, 8}, {"ChanceCV", &P_ChanceCV, 0, 30}, {"DarkLmr", &P_DarkLmr, 0, 200}, {"ChanceCVGap", &P_ChanceCVGap, 0, 30}, {"RepMode", &P_RepMode, 0, 2}, {"ChanceRedLow", &P_ChanceRedLow, 0, 100}, {"ChanceXDepth", &P_ChanceXDepth, 0, 40}, {"TmMtg", &P_TmMtg, 5, 80}, {"TmInc", &P_TmInc, 0, 100}, {"TmSoft", &P_TmSoft, 10, 300}, {"TmStab", &P_TmStab, 0, 300}, {"TmFall", &P_TmFall, 0, 300}, {"TmMtgMin", &P_TmMtgMin, 0, 80}, {"TmMtgDec", &P_TmMtgDec, 0, 100}, {"NodesTime", &P_NodesTime, 0, 100000}, {"ChancePvW", &P_ChancePvW, 0, 30000}, {"ChanceTailFix", &P_ChanceTailFix, 0, 2}, {"ChanceTailR", &P_ChanceTailR, 0, 30}, {"ChanceCVRho", &P_ChanceCVRho, 0, 100}, {"ChanceMix", &P_ChanceMix, 0, 30}, {"ChanceMixRho", &P_ChanceMixRho, 0, 100}, {"ChanceMixMin", &P_ChanceMixMin, 1, 30}, {"ChanceMixPv", &P_ChanceMixPv, 0, 1}, {"ChanceMixProbe", &P_ChanceMixProbe, 0, 1}, {"ChanceMixSkip", &P_ChanceMixSkip, 0, 1}, {"ChanceNW", &P_ChanceNW, 0, 1}, {"ChanceFut", &P_ChanceFut, 0, 1}, {"ChanceFutPre", &P_ChanceFutPre, -1000, 300}, {"ChanceMixRho0", &P_ChanceMixRho0, -1, 100}, {"ChanceMixTT", &P_ChanceMixTT, 0, 1}, {"ChanceMixTop", &P_ChanceMixTop, 1, 6}, {"ChanceMixDiv", &P_ChanceMixDiv, 0, 30}, {"ChanceMixPvR", &P_ChanceMixPvR, 0, 30}, {"ChanceMixPvPly", &P_ChanceMixPvPly, 0, 100}, {"ChanceMixLow", &P_ChanceMixLow, 0, 100}, {"ChanceMixLowR", &P_ChanceMixLowR, 0, 99}, {"ChanceProbeSmall", &P_ChanceProbeSmall, 0, 1}, {"DualMode", &P_DualMode, 0, 7}, {"DualThreshold", &P_DualThr, 0, 100000}, {"DualHidden", &P_DualHid, 0, 32}, {"DualSkipMargin", &P_DualSkipMargin, -1, 100000}, {"SmallTTSalt", &P_SmallTTSalt, 0, 1},{"ChanceLateDark", &P_ChanceLateDark, 0, 30}, {"ChanceLateAdd", &P_ChanceLateAdd, 0, 12}, {"ChanceCVStatic", &P_ChanceCVStatic, 0, 1}, {"ChanceCVLow", &P_ChanceCVLow, 0, 30}, {"RootDarkPen", &P_RootDarkPen, -300, 300}, {"RootDarkHid", &P_RootDarkHid, 0, 15}, {"ThreatRevRoot", &P_ThreatRevRoot, 0, 300}, {"ThreatRevScale", &P_ThreatRevScale, 0, 100}, {"ThreatRevLmr", &P_ThreatRevLmr, 0, 200}, {"ThreatRevOrder", &P_ThreatRevOrder, 0, 30000}, {"ThreatRevSee", &P_ThreatRevSee, -200, 400}, {"ThreatRevMode", &P_ThreatRevMode, 0, 1}, {"ThreatRevProb", &P_ThreatRevProb, 0, 100}, {"ChanceCVProbe", &P_ChanceCVProbe, 0, 10}, {"RevealCheckQs", &P_RevealCheckQs, 0, 1},
        {"SeDepth", &P_SeDepth, 0, 20}, {"SeMargin", &P_SeMargin, 0, 200}, {"HistPrune", &P_HistPrune, 0, 20000},
        {"NullDepthDiv", &P_NullDepthDiv, 2, 8}, {"SmpVote", &P_SmpVote, 0, 1}, {"SmpDefer", &P_SmpDefer, 0, 30}, {"SmpSkip", &P_SmpSkip, 0, 2},
        {"PruneDepth", &P_PruneDepth, 0, 12}, {"RazorDepth", &P_RazorDepth, 0, 6}, {"RazorMargin", &P_RazorMargin, 50, 1000},
        {"CapHist", &P_CapHist, 0, 1}, {"Improving", &P_Improving, 0, 1}, {"QsTT", &P_QsTT, 0, 1},
        {"CheckExt", &P_CheckExt, 0, 2}, {"QsTTMove", &P_QsTTMove, 0, 1},
        {"ProbCut", &P_ProbCut, 0, 1000}, {"ProbCutDepth", &P_ProbCutDepth, 3, 12},
        {"QsSee", &P_QsSee, 0, 1}, {"QsSeeMargin", &P_QsSeeMargin, -500, 500}, {"SeeOrder", &P_SeeOrder, 0, 1},
        {"SeeCap", &P_SeeCap, 0, 1000}, {"SeeQuiet", &P_SeeQuiet, 0, 500}, {"SeePruneDepth", &P_SeePruneDepth, 1, 12},
        {"SeeQuietOrder", &P_SeeQuietOrder, 0, 1000}, {"CorrHist", &P_CorrHist, 0, 400}, {"TTWays", &g_ttWays, 1, 4}, {"TTWaysMT", &g_ttWaysMT, 1, 4}, {"CapLmr", &P_CapLmr, 0, 3},
    };
    return p;
}

struct Searcher::Impl {
    Shared G;
    TranspositionTable TT;
    std::vector<std::unique_ptr<Worker>> workers;
};

Searcher::Searcher(int threads, std::size_t hashMb) : impl_(std::make_unique<Impl>()) {
    search::init();
    impl_->TT.resize(hashMb);
    set_threads(threads);
}

Searcher::~Searcher() = default;

void Searcher::set_threads(int n) {
    n = std::clamp(n, 1, 256);
    impl_->workers.clear();
    for (int i = 0; i < n; ++i) {
        impl_->workers.push_back(std::make_unique<Worker>(impl_->G, impl_->TT, impl_->workers));
        impl_->workers.back()->id = i;
        impl_->workers.back()->clear();
    }
}

int Searcher::threads() const { return int(impl_->workers.size()); }
void Searcher::resize_hash(std::size_t mb) { impl_->TT.resize(mb); }

void Searcher::clear() {
    for (auto& w : impl_->workers) w->clear();
    impl_->TT.clear();
}

void Searcher::stop() { impl_->G.stop = true; }
void Searcher::prepare(bool pondering) {
    Shared& G = impl_->G;
    G.stop = false;
    G.pondering = false;
    G.ponderArmed = pondering;
    G.ponderHitPending = false;
    G.ponderOffsetMs = 0;
}
void Searcher::ponderhit() {
    Shared& G = impl_->G;
    if (G.pondering.load(std::memory_order_relaxed)) {
        G.ponderOffsetMs.store(G.raw_elapsed(), std::memory_order_relaxed);
        G.pondering.store(false, std::memory_order_relaxed);
    } else if (G.ponderArmed.exchange(false, std::memory_order_relaxed))
        G.ponderHitPending.store(true, std::memory_order_relaxed);
}

SearchResult Searcher::go(const Position& root, const Limits& limits, const std::function<void(const std::string&)>& info,
                          bool resetStop) {
    Shared& G = impl_->G;
    auto& workers = impl_->workers;
    G.start = Clock::now();
    G.ponderOffsetMs = 0;
    G.pondering = limits.ponder;
    G.ponderArmed = false;
    if (G.ponderHitPending.exchange(false, std::memory_order_relaxed))
        G.pondering = false;
    if (resetStop) G.stop = false;
    G.depthLimit = limits.depth;
    if (!G.depthLimit && limits.mate > 0)
        G.depthLimit = 2 * std::min(limits.mate, (MAX_PLY - 8) / 2);
    G.mateLimit = limits.mate;
    G.nodeLimit = limits.nodes;
    G.timed = false;
    G.clockMode = false;
    const Color us = root.stm;
    const int overhead = 30;
    if (limits.movetime > 0) {
        G.timed = true;
        G.optimumMs = G.maximumMs = std::max(1, limits.movetime - std::min(10, limits.movetime / 10));
        G.optimumMs = G.maximumMs * 2;  // use the whole budget; iterations stop at maximum
    } else if (limits.time[us] > 0 && !limits.infinite) {
        G.timed = true;
        G.clockMode = true;
        const int64_t t = limits.time[us], inc = limits.inc[us];
        int mtg = limits.movestogo > 0 ? std::min(limits.movestogo, 40) : P_TmMtg;
        if (limits.movestogo <= 0 && P_TmMtgMin)
            mtg = std::max(P_TmMtgMin, P_TmMtg - root.gamePly / 2 * P_TmMtgDec / 10);
        G.optimumMs = t / mtg + inc * P_TmInc / 100;
        G.maximumMs = std::min<int64_t>(t * 2 / 5, G.optimumMs * 4);
        G.optimumMs = std::min(G.optimumMs, t - overhead);
        G.maximumMs = std::max<int64_t>(1, std::min(G.maximumMs, t - overhead));
        G.optimumMs = std::max<int64_t>(1, std::min(G.optimumMs, G.maximumMs));
    }
    impl_->TT.new_search();
    impl_->TT.set_mt_ways(workers.size() > 1 ? g_ttWaysMT : 0);
    G.nThreads = int(workers.size());
    G.rootMoves = limits.searchMoves;
    for (auto& w : workers) {
        w->pos = root;
        w->nodes = 0;
        w->chanceLayers = 0;
        w->maxLayers = P_MaxChanceLayers +
                       (P_ChanceLateDark && root.darkCount[RED] + root.darkCount[BLACK] <= P_ChanceLateDark ? P_ChanceLateAdd : 0);
        w->revealSplits = 0;
        w->info = nullptr;
        const double k = P_ChanceK;
        w->set_eK(k);
        w->eMin = 1.0 / (1.0 + jqm::exp(double(SAT) / k));
        w->eMax = 1.0 / (1.0 + jqm::exp(-double(SAT) / k));
    }
    workers[0]->info = &info;

    Move legal[MAX_MOVES];
    const int nLegal = root.gen_legal(legal);
    SearchResult res;
    if (nLegal == 0) return res;
    Move fallback = legal[0];
    if (g_chaseRule == CHASE_RULE_SKYRULE) {
        for (int i = 0; i < nLegal; ++i)
            if (!workers[0]->pos.skyrule_forbids_legal_move(legal[i])) {
                fallback = legal[i];
                break;
            }
    }

    std::vector<Thread> helpers;
    for (std::size_t i = 1; i < workers.size(); ++i) helpers.emplace_back([&workers, i] { workers[i]->iterate(); });
    workers[0]->iterate();
    G.stop = true;
    for (auto& t : helpers) t.join();

    const Worker* chosen = workers[0].get();
    if (P_SmpVote && workers.size() > 1) {
        // Each finished thread votes for its move with weight (score - worst + 14) * depth.
        Value minScore = VALUE_INFINITE;
        for (auto& w : workers)
            if (w->completedDepth > 0 && w->bestMove != MOVE_NONE) minScore = std::min(minScore, w->bestScore);
        auto votes = [&](Move mv) {
            int64_t v = 0;
            for (auto& w : workers)
                if (w->completedDepth > 0 && w->bestMove == mv) v += int64_t(w->bestScore - minScore + 14) * w->completedDepth;
            return v;
        };
        if (chosen->completedDepth > 0 && chosen->bestMove != MOVE_NONE) {
            int64_t bestVotes = votes(chosen->bestMove);
            for (auto& w : workers) {
                if (w->completedDepth == 0 || w->bestMove == MOVE_NONE || w->bestMove == chosen->bestMove) continue;
                const int64_t v = votes(w->bestMove);
                if (v > bestVotes) { bestVotes = v; chosen = w.get(); }
            }
        }
    }
    const Worker& m = *chosen;
    res.best = m.bestMove != MOVE_NONE ? m.bestMove : fallback;
    res.score = m.bestScore;
    res.depth = workers[0]->completedDepth;
    res.nodes = m.total_nodes();
    res.timeMs = int(workers[0]->elapsed());
    if (P_NodesTime) info("info string vtime " + std::to_string(res.timeMs) + " nodes " + std::to_string(res.nodes));
    res.pv = m.bestPv;
    if (res.pv.empty() || res.pv[0] != res.best) res.pv.assign(1, res.best);
#ifdef JQ_STATS
    {
        const Stats& t = g_st;
        char buf[1024];
        std::snprintf(buf, sizeof buf,
                      "info string stats search %llu pv %llu qs %llu ttcut %llu rfp %llu null %llu/%llu cext %llu chance %llu "
                      "outcomes %llu lmr %llu re %llu pvsRe %llu fh %llu fh1 %llu fhMoves %llu failLow %llu flMoves %llu "
                      "lmp %llu fut %llu searched q/c/d %llu/%llu/%llu",
                      (unsigned long long)t.search, (unsigned long long)t.pv, (unsigned long long)t.qs, (unsigned long long)t.ttcut,
                      (unsigned long long)t.rfp, (unsigned long long)t.nullCut, (unsigned long long)t.nullTry,
                      (unsigned long long)t.cext, (unsigned long long)t.chance, (unsigned long long)t.outcomes,
                      (unsigned long long)t.lmr, (unsigned long long)t.lmrRe, (unsigned long long)t.pvsRe, (unsigned long long)t.fh,
                      (unsigned long long)t.fh1, (unsigned long long)t.fhMoves, (unsigned long long)t.failLow,
                      (unsigned long long)t.failLowMoves, (unsigned long long)t.lmp, (unsigned long long)t.fut,
                      (unsigned long long)t.quietSearched, (unsigned long long)t.capSearched, (unsigned long long)t.darkSearched);
        if (info) info(buf);
        std::snprintf(buf, sizeof buf, "info string stats2 loop %llu withTT %llu fhTT %llu fh1TT %llu fhCat tt/cap/kc/bad/quiet/dark %llu/%llu/%llu/%llu/%llu/%llu",
                      (unsigned long long)t.loopNodes, (unsigned long long)t.loopTT, (unsigned long long)t.fhTT, (unsigned long long)t.fh1TT,
                      (unsigned long long)t.fhCat[0], (unsigned long long)t.fhCat[1], (unsigned long long)t.fhCat[2],
                      (unsigned long long)t.fhCat[3], (unsigned long long)t.fhCat[4], (unsigned long long)t.fhCat[5]);
        if (info) info(buf);
        std::snprintf(buf, sizeof buf, "info string stats3 chance calls:probeNodes+deepNodes cv %llu:%llu+%llu mix %llu:%llu+%llu probeMode %llu:%llu",
                      (unsigned long long)t.cvCalls, (unsigned long long)t.cvProbeN, (unsigned long long)t.cvDeepN,
                      (unsigned long long)t.mixCalls, (unsigned long long)t.mixProbeN, (unsigned long long)t.mixDeepN,
                      (unsigned long long)t.probeCalls, (unsigned long long)t.probeN);
        if (info) info(buf);
        {
            std::string o = "info string stats4 mix non-PV calls:probe+deep  q-probes low/in/high";
            for (int k = 0; k < 6; ++k) {
                if (k == 3) o += "  searched probes low/in/high";
                o += " " + std::to_string(t.mixK[k][0]) + ":" + std::to_string(t.mixK[k][1]) + "+" + std::to_string(t.mixK[k][2]);
            }
            if (info) info(o);
        }
        std::snprintf(buf, sizeof buf, "info string stats5 nodes %llu chanceSub %llu probeSub %llu evProbe %llu evBig %llu evSmall %llu upd big/small %llu/%llu refresh big/small %llu/%llu updSmall/evSmall %.3f",
                      (unsigned long long)t.nodesAll, (unsigned long long)t.chanceSubN,
                      (unsigned long long)t.probeSubN, (unsigned long long)t.evProbe, (unsigned long long)t.evBig,
                      (unsigned long long)t.evSmall, (unsigned long long)t.updBig, (unsigned long long)t.updSmall,
                      (unsigned long long)t.refBig, (unsigned long long)t.refSmall, t.evSmall ? double(t.updSmall) / double(t.evSmall) : 0.0);
        if (info) info(buf);
        for (int r = 0; r < 2; ++r) {
            std::string o = r ? "info string stats6 mixPct rep" : "info string stats6 mixPct other";
            for (int q = 0; q <= 100; ++q)
                if (t.mixPct[r][q][0])
                    o += " " + std::to_string(q) + ":" + std::to_string(t.mixPct[r][q][0]) + ":" + std::to_string(t.mixPct[r][q][1]);
            if (info) info(o);
        }
        std::snprintf(buf, sizeof buf, "info string stats7 trv dark moves root %llu attacked %llu triggered %llu  tree %llu attacked %llu triggered %llu  calls %llu",
                      (unsigned long long)t.trvRootDark, (unsigned long long)t.trvRootAtk, (unsigned long long)t.trvRootThr,
                      (unsigned long long)t.trvTreeDark, (unsigned long long)t.trvTreeAtk, (unsigned long long)t.trvTreeThr,
                      (unsigned long long)t.trvCalls);
        if (info) info(buf);
        g_st = Stats{};
    }
#endif
    return res;
}

int Searcher::hashfull() const { return impl_->TT.hashfull(); }

namespace search {

void init() {
    init_tables();
    for (int d = 0; d < 64; ++d)
        for (int m = 0; m < 64; ++m)
            lmrTable[d][m] = d && m ? int(P_LmrBase / 100.0 + jqm::log(d) * jqm::log(m) / (P_LmrDiv / 100.0)) : 0;
}

uint64_t perft(Position& pos, int depth) { return depth <= 0 ? 1 : perft_rec(pos, depth); }

// Static exchange evaluation on to_sq(m): does the side to move net at least `threshold` from the capture
// sequence there? A face-down mover or victim counts at its side's expected identity value.
#ifdef JQ_BB_VERIFY
bool see_ge_ref(const Position& p, Move m, int threshold) {
    const int from = from_sq(m), to = to_sq(m);
    const int darkV[2] = {urn_mean_value(p, RED), urn_mean_value(p, BLACK)};
    auto val = [&](Piece q) { return !q ? 0 : type_of(q) == DARK ? darkV[color_of(q)] : PieceValue[type_of(q)]; };
    int swap = val(p.board[to]) - threshold;
    if (swap < 0) return false;
    swap = val(p.board[from]) - swap;
    if (swap <= 0) return true;
    Piece bd[NUM_SQ];
    std::memcpy(bd, p.board, sizeof bd);
    bd[to] = bd[from];
    bd[from] = NO_PIECE;
    Color side = p.stm;
    bool res = true;
    while (true) {
        side = ~side;
        int av;
        const int a = least_attacker_ref(bd, side, to, darkV[side], &av);
        if (a < 0) break;
        res = !res;
        if (type_of(bd[a]) == KING) {  // the king may only take an undefended piece
            int dv;
            return least_attacker_ref(bd, ~side, to, darkV[~side], &dv) >= 0 ? !res : res;
        }
        swap = av - swap;
        if (swap < int(res)) break;
        bd[to] = bd[a];
        bd[a] = NO_PIECE;
    }
    return res;
}
int threat_reveal_ref(const Position& p, int s, int* loss, const int* dv) {
    *loss = 0;
    const Color us = p.stm, them = ~us;
    const int darkV[2] = {dv ? dv[0] : urn_mean_value(p, RED), dv ? dv[1] : urn_mean_value(p, BLACK)};
    int av;
    int a = least_attacker_ref(p.board, them, s, darkV[them], &av);
    if (a < 0 || p.urnTotal[us] <= 0) return 0;
    Piece bd[NUM_SQ];
    std::memcpy(bd, p.board, sizeof bd);
    int vals[34], n = 0;
    Color side = them;
    while (true) {
        const bool king = type_of(bd[a]) == KING;
        bd[s] = bd[a];
        bd[a] = NO_PIECE;
        if (king) {  // the king may only take an undefended piece
            int dv;
            if (least_attacker_ref(bd, ~side, s, darkV[~side], &dv) < 0) vals[n++] = av;
            break;
        }
        vals[n++] = av;
        if (n >= 32) break;
        side = ~side;
        a = least_attacker_ref(bd, side, s, darkV[side], &av);
        if (a < 0) break;
    }
    if (n == 0) return 1;  // only a king attacks and the square is defended
    int sum = 0, cntWin = 0;
    for (int sl = 0; sl < NUM_SLOTS; ++sl) {
        const int c = p.urn[us][sl];
        if (!c) continue;
        int g[34];
        g[0] = PieceValue[slot_type(sl)];
        for (int k = 1; k < n; ++k) g[k] = vals[k - 1] - g[k - 1];
        for (int k = n - 1; k > 0; --k) g[k - 1] = -std::max(-g[k - 1], g[k]);
        const int gain = std::max(0, g[0]);
        sum += c * gain;
        if (gain > 0) cntWin += c;
    }
    *loss = sum / p.urnTotal[us];
    if (n == 1) return 2;  // undefended
    const bool trig = P_ThreatRevMode == 0 ? sum > P_ThreatRevSee * p.urnTotal[us] : 100 * cntWin >= P_ThreatRevProb * p.urnTotal[us];
    return trig ? 2 : 1;
}
#define JQ_SIMPL(f) f##_impl
#else
#define JQ_SIMPL(f) f
#endif
bool JQ_SIMPL(see_ge)(const Position& p, Move m, int threshold) {
    const int from = from_sq(m), to = to_sq(m);
    const int darkV[2] = {urn_mean_value(p, RED), urn_mean_value(p, BLACK)};
    auto val = [&](Piece q) { return !q ? 0 : type_of(q) == DARK ? darkV[color_of(q)] : PieceValue[type_of(q)]; };
    int swap = val(p.board[to]) - threshold;
    if (swap < 0) return false;
    swap = val(p.board[from]) - swap;
    if (swap <= 0) return true;
    // Occupancy after the capture: from emptied, to occupied (pieces only ever leave squares afterwards).
    Bitboard occ = (p.occBB & ~sq_bb(from)) | sq_bb(to);
    Color side = p.stm;
    bool res = true;
    while (true) {
        side = ~side;
        int av;
        const int a = least_attacker(p, occ, side, to, darkV[side], &av);
        if (a < 0) break;
        res = !res;
        if (type_of(p.board[a]) == KING) {  // the king may only take an undefended piece
            int dv;
            return least_attacker(p, occ, ~side, to, darkV[~side], &dv) >= 0 ? !res : res;  // king not moved yet
        }
        swap = av - swap;
        if (swap < int(res)) break;
        occ &= ~sq_bb(a);
    }
    return res;
}

// Threatened face-down piece of the side to move on s: the opponent's capture sequence on s (least valuable
// attacker first, alternating; other face-down pieces at their urn mean, pins ignored, a king captures only
// as the last piece) does not depend on the victim's identity, so it is built once and the swap-list minimax
// is evaluated per urn identity of the mover (<= 6 slot types). *loss = probability-weighted opponent gain
// (each identity's gain floored at 0: the opponent may decline). Returns 0 = s not attacked, 1 = attacked but
// not triggered, 2 = triggered (undefended always triggers: loss = full expected value).
int JQ_SIMPL(threat_reveal)(const Position& p, int s, int* loss, const int* dv) {
    *loss = 0;
    const Color us = p.stm, them = ~us;
    const int darkV[2] = {dv ? dv[0] : urn_mean_value(p, RED), dv ? dv[1] : urn_mean_value(p, BLACK)};
    int av;
    Bitboard occ = p.occBB;
    int a = least_attacker(p, occ, them, s, darkV[them], &av);
    if (a < 0 || p.urnTotal[us] <= 0) return 0;
    int vals[34], n = 0;
    Color side = them;
    while (true) {
        const bool king = type_of(p.board[a]) == KING;
        occ &= ~sq_bb(a);  // the capturer moves onto s
        if (king) {  // the king may only take an undefended piece
            int dv;
            if (least_attacker(p, occ, ~side, s, darkV[~side], &dv) < 0) vals[n++] = av;
            break;
        }
        vals[n++] = av;
        if (n >= 32) break;
        side = ~side;
        a = least_attacker(p, occ, side, s, darkV[side], &av);
        if (a < 0) break;
    }
    if (n == 0) return 1;  // only a king attacks and the square is defended
    int sum = 0, cntWin = 0;
    for (int sl = 0; sl < NUM_SLOTS; ++sl) {
        const int c = p.urn[us][sl];
        if (!c) continue;
        int g[34];
        g[0] = PieceValue[slot_type(sl)];
        for (int k = 1; k < n; ++k) g[k] = vals[k - 1] - g[k - 1];
        for (int k = n - 1; k > 0; --k) g[k - 1] = -std::max(-g[k - 1], g[k]);
        const int gain = std::max(0, g[0]);
        sum += c * gain;
        if (gain > 0) cntWin += c;
    }
    *loss = sum / p.urnTotal[us];
    if (n == 1) return 2;  // undefended
    const bool trig = P_ThreatRevMode == 0 ? sum > P_ThreatRevSee * p.urnTotal[us] : 100 * cntWin >= P_ThreatRevProb * p.urnTotal[us];
    return trig ? 2 : 1;
}

#ifdef JQ_BB_VERIFY
bool see_ge(const Position& p, Move m, int threshold) {
    const bool r = see_ge_impl(p, m, threshold);
    if (r != see_ge_ref(p, m, threshold)) { std::fprintf(stderr, "see_ge mismatch %s %s %d\n", p.fen().c_str(), move_str(m).c_str(), threshold); std::abort(); }
    return r;
}
int threat_reveal(const Position& p, int s, int* loss, const int* dv) {
    int l1, l2;
    const int r = threat_reveal_impl(p, s, &l1, dv), r2 = threat_reveal_ref(p, s, &l2, dv);
    if (r != r2 || l1 != l2) { std::fprintf(stderr, "threat_reveal mismatch %s %d\n", p.fen().c_str(), s); std::abort(); }
    *loss = l1;
    return r;
}
#endif

int see_value(const Position& p, Move m) {
    int lo = -4000, hi = 4000;  // largest threshold that still holds
    while (lo < hi) {
        const int mid = lo + (hi - lo + 1) / 2;
        if (see_ge(p, m, mid)) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

double perft_chance(Position& pos, int depth, uint64_t& leaves) {
    leaves = 0;
    return perft_chance_rec(pos, depth, leaves);
}

}  // namespace search
}  // namespace jq
