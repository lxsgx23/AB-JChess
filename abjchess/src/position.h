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
// Board state, move generation and make/unmake for Jieqi.
//
// Information model (from the point of view of the player the engine plays for):
//   urn[c][slot]  candidate identities still unaccounted for among c's face-down pieces.
//   A face-down piece reveals as slot t with probability urn[c][t] / urnTotal[c].
// Captured face-down pieces whose identity the observer did not see stay in the urn;
// by exchangeability the live pieces are then still a uniform draw from the urn.
// Inside search, capturing a face-down piece never branches on its identity for the
// same reason (the urn is left unchanged), while moving one is a chance node.
#pragma once
#include "bitboard.h"
#include "tables.h"
#include "types.h"

#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace jq {

struct StateInfo {
    uint64_t key;        // key before the move
    Piece captured;
    int8_t urnSlot;      // slot drawn by the mover's reveal, -1 if none
    int noCapture;       // counter before the move
    int gamePly;         // move number state before the move
    int16_t lastIrrev;   // irreversible boundary before the move
    uint64_t skey;       // structure key before the move
    uint16_t movedId = 0;
    uint16_t capturedId = 0;
    int16_t urnMean;     // mover's urnMean before the move (restored by undo_move for a reveal)
};

// SkyRule per-position record (one per keyHist entry). A quiet face-up move that does not give
// check is recorded as SKY_PENDING with a snapshot of the board after the move; whether it is a
// chase (newly threatened targets) is only computed when a consumer needs it (lazy, memoised).
constexpr uint8_t SKY_PENDING = 3;
// SkyRule limits: 6 consecutive checks per checking piece (max 3), 18 consecutive chases. Verification
// builds may lower them (-DSKY_CHECK_LIM=.. -DSKY_CHASE_LIM=..) to exercise the forbidden-move paths.
#ifndef SKY_CHECK_LIM
#define SKY_CHECK_LIM 6
#endif
#ifndef SKY_CHASE_LIM
#define SKY_CHASE_LIM 18
#endif
struct SkyEntry {
    Bitboard unionCheck;  // CHECK: union of checking-piece identities over the current check chain
    uint16_t count;       // CHECK/CHASE: chain length (valid when countKnown or action == CHECK)
    uint16_t run;         // upper bound on the chase chain length (pending/chase chain)
    uint8_t action;       // SKYRULE_NONE / SKYRULE_CHECK / SKYRULE_CHASE / SKY_PENDING
    uint8_t countKnown;
    uint8_t from, to;     // the quiet move into this position (pending entries)
    uint8_t actor;
    uint8_t hasBoard;
    uint32_t board;       // index into skyBoards (pending entries)
};
struct SkyBoard { Piece b[96]; };

constexpr int NO_CAPTURE_LIMIT = 80;
// Values are ChaseRule: none=0, default=1, skyrule=2. The integer backing
// keeps the parameter table and legacy numeric UCI commands source-compatible.
extern int g_chaseRule;
constexpr int MAX_MOVES = 160;

class Position {
public:
    // Bitboards (kept in sync with board[] by add/remove/move and do/undo):
    //   occBB = all pieces; byColorBB[c]; byRoleBB[t] = pieces moving as t (a face-down piece counts under the
    //   role of its start square, T.darkRole); darkBB = face-down pieces (both colours).
    Bitboard occBB = 0;
    Bitboard byColorBB[2] = {0, 0};
    Bitboard byRoleBB[9] = {};
    Bitboard darkBB = 0;
    // (all members have defined initial values: a failed set_fen must never leave indeterminate state behind)
    Piece board[NUM_SQ] = {};
    Color stm = RED;
    int8_t urn[2][NUM_SLOTS] = {};
    int8_t urnTotal[2] = {0, 0};
    int16_t urnMean[2] = {0, 0};  // cached urn_mean_value (exact: recomputed whenever urn changes)
    int8_t darkCount[2] = {0, 0};
    int8_t kingSq[2] = {-1, -1};
    int noCapture = 0;
    int gamePly = 0;
    uint8_t plist[2][16] = {};
    int8_t pcount[2] = {0, 0};
    int8_t pidx[NUM_SQ] = {};
    uint64_t key = 0;
    bool chkInv = false;  // the side not to move is known not to be in check (set_fen), kept by legal moves
    uint64_t skey = 0;  // xor of zPiece over soldiers and face-down pieces (correction-history index)
    // Game + search history: key and "side to move is in check" per position.
    std::vector<uint64_t> keyHist;
    std::vector<uint8_t> checkHist;
    std::vector<Move> moveHist;  // moveHist[j]: base move played from position j (MOVE_NULL for a null move)
    mutable std::vector<SkyEntry> skyHist;  // per position, aligned with keyHist (lazily resolved)
    std::vector<SkyBoard> skyBoards;        // board snapshots of pending entries (LIFO)
#ifdef SKY_VERIFY
    std::vector<Bitboard> skyRawHist;       // reference (original eager) implementation
    std::vector<Bitboard> skyChaseIdsHist;
    std::vector<Bitboard> skyStableChaseHist;
    std::vector<Bitboard> skyCheckIdsHist;
    std::vector<Bitboard> skyStableCheckHist;
    std::vector<Bitboard> skyUnionCheckHist;
    std::vector<uint8_t> skyActionHist;
    std::vector<uint16_t> skyCountHist;
    bool ref_forbids(Move m);
    int ref_sky_cycle_verdict(int i, int p) const;
#endif
    int sky_cycle_verdict(int i, int p) const;
    uint16_t pieceIds[NUM_SQ] = {};
    int lastIrrev = 0;  // index into keyHist of the latest position after an irreversible move

    Position() {
        keyHist.reserve(1024);
        checkHist.reserve(1024);
        moveHist.reserve(1024);
        skyHist.reserve(1024);
        skyBoards.reserve(256);
    }

    // Pikafish-dialect FEN: board (ranks 9..0, X/x = face down), side, pools, half-move clock, move number.
    bool set_fen(const std::string& fen, std::string* error = nullptr);
    bool set_fen_impl(const std::string& fen, std::string* error);
    void refresh_caches();
    // History token from-to + optional reveal letter (mover colour) + optional captured identity (victim colour) or 'x'.
    bool apply_token(const std::string& token, std::string* error = nullptr);
    // Same as apply_token with parsed parts (reveal / captured identity NO_TYPE when not observed).
    bool apply_observed(Move base, PieceType reveal, PieceType capturedId, std::string* error = nullptr);
    std::string fen() const;
    std::string pretty() const;

    PieceType role(int s) const {
        const Piece p = board[s];
        return type_of(p) == DARK ? PieceType(T.darkRole[color_of(p)][s]) : type_of(p);
    }
    bool is_dark_move(Move m) const { return type_of(board[from_sq(m)]) == DARK; }
    bool is_capture(Move m) const { return board[to_sq(m)] != NO_PIECE; }
    Bitboard pieces() const { return occBB; }
    Bitboard pieces(Color c) const { return byColorBB[c]; }
    Bitboard pieces(Color c, PieceType role) const { return byColorBB[c] & byRoleBB[role]; }
    // Attack test on a hypothetical board: is square k attacked by colour e when the occupancy is occ, e's pieces
    // are E (subset of occ) and role / dark sets are this position's, except that square x (if >= 0) holds a
    // face-up e piece of type xt? Mirrors king_attacked() exactly.
    bool attacked_bb(Color e, int k, Bitboard occ, Bitboard E, int x = -1, PieceType xt = NO_TYPE) const;
    bool ray_attacked_bb(Color e, int k, int d, Bitboard occ, Bitboard E, int x = -1, PieceType xt = NO_TYPE) const;
    void compute_bitboards();
#ifdef JQ_BB_VERIFY
    bool ray_attacked_impl(Color c, int k, int d) const;
    bool king_attacked_impl(Color c, int k) const;
    template <bool CAP> int gen_impl(Move* list) const;
    bool legal_impl(Move m) const;
    bool reveal_gives_check_impl(int from, int to, PieceType t) const;
    bool check_after_impl(int from, int to, PieceType t) const;
    bool gives_check_impl(Move m) const;
#endif
    bool bitboards_ok() const;

    void update_mean(Color c);
    bool king_attacked(Color c, int k) const;
    bool ray_attacked(Color c, int k, int d) const;  // rook / cannon / facing-king attack along one ray
    bool in_check() const { return king_attacked(stm, kingSq[stm]); }
    // Pseudo-legal generation of base moves (no reveal). Captures-only variant for quiescence.
    int gen_moves(Move* list) const;
    template <bool CAP> int gen(Move* list) const;
    int gen_captures(Move* list) const;
    // Own-king safety for a pseudo-legal base move; identity of a dark mover is irrelevant.
    bool legal(Move m) const;
    // per-node legality context: own king square, in-check flag and (lazily) the first three
    // occupied squares of each king ray. legal(m, c) == legal(m) for every pseudo-legal move m of this position (c
    // must come from legal_ctx() on the same position). Measured slower (-1.5%): off unless -DJQ_LCTX.
    struct LegalCtx { int8_t k; bool chk; bool init; int8_t b[4][3]; };
    void legal_ctx(LegalCtx& c) const {
#if defined(JQ_LCTX) && !defined(JQ_MG_OLD)
        c.k = kingSq[stm]; c.chk = checkHist.back() != 0; c.init = false;
#else
        (void)c;
#endif
    }
    bool legal_slow(Move m, LegalCtx& c) const;
    bool legal(Move m, LegalCtx& c) const {
#if defined(JQ_MG_OLD) || !defined(JQ_LCTX)
        (void)c;
        return legal(m);
#else
        const int from = from_sq(m), to = to_sq(m), k = c.k;
        bool r;
        if (from == k || c.chk) r = legal(m);
        else if (!(((BB::REL[from][k] | BB::REL[to][k]) & 1) | (BB::REL[from][k] & 2))) r = true;
        else r = legal_slow(m, c);
#ifdef JQ_BB_VERIFY
        if (r != legal(m)) { std::fprintf(stderr, "BB VERIFY MISMATCH legal ctx %d %d\n", from, to); std::abort(); }
#endif
        return r;
#endif
    }
    // Whether a legal move by a face-up piece checks the opponent (false for face-down movers).
    bool gives_check(Move m) const;
    bool check_after(int from, int to, PieceType t) const;  // stm in check after the opponent's legal move from->to
    // A face-down mover from -> to revealed as t: does the revealed piece itself attack the enemy king?
    // reveal_may_check is a cheap necessary condition for any identity to differ.
    bool reveal_gives_check(int from, int to, PieceType t) const;
    bool reveal_may_check(int to) const;
    // Square attacked by `by` (king-centric test; adequate for capture safety heuristics).
    bool attacked(Color by, int s) const { return king_attacked(~by, s); }
    int gen_legal(Move* list) const;
    bool has_legal_move() const;

    // the search may hand do_move the gives_check(m) it already computed for face-up m in this position
    // (do_move then skips its own check_after; same value, both exact). Matched by move and key / history length.
    void check_hint(Move m, bool gc) { hintMove = m; hintKey = key; hintLen = uint32_t(keyHist.size()); hintVal = gc; }
    Move hintMove = MOVE_NONE;
    uint64_t hintKey = 0;
    uint32_t hintLen = 0;
    bool hintVal = false;
    // m must carry a reveal type when the mover is face down.
    void do_move(Move m, StateInfo& st);
    uint64_t key_after(Move m) const;  // the key do_move(m) will produce (m carries the reveal for a dark mover)
    void undo_move(Move m, const StateInfo& st);
    void do_null(StateInfo& st);
    void undo_null(const StateInfo& st);

    // 0: none, 1: draw, 2: side to move wins (opponent perpetually checked or chased), 3: side to move loses.
    int repetition_status() const;
    std::bitset<NUM_SQ> chase_threats(Color actor) const;  // enemy squares actor threatens under the chase rule
    // m must already have passed legal(). Inline filter: a move can only cross a limit when the same
    // actor's prior action is a check chain >= 6 or a (possible) chase chain >= 18.
    bool skyrule_forbids_legal_move(Move m) {
#ifndef SKY_VERIFY
        if (g_chaseRule != CHASE_RULE_SKYRULE || is_dark_move(m) || is_capture(m)) return false;
        const std::size_t n = skyHist.size();
        if (n < 3) return false;
        const SkyEntry& e = skyHist[n - 2];
        if (e.action == SKYRULE_CHECK ? e.count < SKY_CHECK_LIM : (e.action == SKYRULE_NONE || e.run < SKY_CHASE_LIM)) return false;
#endif
        return skyrule_forbids_slow(m);
    }
    bool skyrule_forbids_slow(Move m);
    uint8_t sky_action(std::size_t k) const;  // resolves a pending entry
    uint16_t sky_count(std::size_t k) const;  // chain length of a CHASE entry
    bool has_major(Color c) const;
    uint64_t compute_key() const;
    bool valid(std::string* error = nullptr) const;

private:
    int chase_verdict(int first) const;
    void bb_put(int s, Piece p) {
        const Color c = color_of(p);
        const PieceType t = type_of(p);
        const Bitboard b = sq_bb(s);
        occBB |= b;
        byColorBB[c] |= b;
        if (t == DARK) { darkBB |= b; byRoleBB[T.darkRole[c][s]] |= b; }
        else byRoleBB[t] |= b;
    }
    void bb_clear(int s, Piece p) {
        const Color c = color_of(p);
        const PieceType t = type_of(p);
        const Bitboard b = ~sq_bb(s);
        occBB &= b;
        byColorBB[c] &= b;
        if (t == DARK) { darkBB &= b; byRoleBB[T.darkRole[c][s]] &= b; }
        else byRoleBB[t] &= b;
    }
    // Bitboard delta of a move from -> to by pc (revealed as rev if face down) capturing cap; applied by
    // do_move and again (XOR is its own inverse) by undo_move.
    void bb_move(Color us, int from, int to, Piece pc, PieceType rev, Piece cap) {
        const Bitboard bf = sq_bb(from), bt = sq_bb(to);
        if (cap) {
            const Color th = ~us;
            byColorBB[th] ^= bt;
            if (type_of(cap) == DARK) { darkBB ^= bt; byRoleBB[T.darkRole[th][to]] ^= bt; }
            else byRoleBB[type_of(cap)] ^= bt;
            occBB ^= bf;
        } else occBB ^= bf | bt;
        byColorBB[us] ^= bf | bt;
        if (type_of(pc) == DARK) { darkBB ^= bf; byRoleBB[T.darkRole[us][from]] ^= bf; byRoleBB[rev] ^= bt; }
        else byRoleBB[type_of(pc)] ^= bf | bt;
    }
    void add_piece(int s, Piece p);
    void remove_piece(int s);
    void move_piece(int from, int to);
    void push_history() {
        keyHist.push_back(key);
        checkHist.push_back(uint8_t(in_check()));
    }
};

std::string move_str(Move m);
Move parse_move_coords(const std::string& s);
char piece_char(Piece p);

extern const char* START_FEN;

}  // namespace jq
