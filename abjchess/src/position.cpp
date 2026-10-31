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
#include <algorithm>
#include <vector>
#include "position.h"
#include "eval.h"
#include "position_ref.h"
#ifdef JQ_BB_VERIFY
#define JQ_IMPL(f) f##_impl
#else
#define JQ_IMPL(f) f
#endif

#include <cctype>
#include <atomic>
#include <charconv>
#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <bitset>
#include <cstring>
#include <limits>
#include <sstream>

namespace jq {

const char* START_FEN =
    "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1";

namespace {
PieceType type_from_char(char ch) {
    switch (std::toupper(static_cast<unsigned char>(ch))) {
    case 'K': return KING;
    case 'R': return ROOK;
    case 'N': return HORSE;
    case 'B': return ELEPHANT;
    case 'A': return ADVISOR;
    case 'C': return CANNON;
    case 'P': return SOLDIER;
    case 'X': return DARK;
    default: return NO_TYPE;
    }
}
bool fail(std::string* error, const char* text) {
    if (error) *error = text;
    return false;
}

// SkyRule helpers are defined with the chase classifier below. They are
// forward-declared here because make/unmake records the action immediately
// after applying a move.
Bitboard skyrule_checkers(const Position& pos, Color actor);
#ifdef SKY_VERIFY
Bitboard skyrule_chase_targets(const Position& pos, Color actor);
Bitboard skyrule_target_ids(const Position& pos, Bitboard targets);
Bitboard ref_skyrule_checkers(const Position& pos, Color actor);
#endif
const SkyEntry SKY_NONE_ENTRY = {0, 0, 0, SKYRULE_NONE, 1, 0, 0, 0, 0, 0};
uint8_t sky_classify(const SkyEntry& e, const Piece* raw);
#ifdef SKY_VERIFY
std::atomic<uint64_t> V_ct{0}, V_ct_bad{0}, V_act{0}, V_act_bad{0}, V_cnt{0}, V_cnt_bad{0}, V_chk{0}, V_chk_bad{0},
    V_forb{0}, V_forb_bad{0}, V_forb_true{0}, V_forb_slow{0}, V_rep{0}, V_rep_bad{0}, V_rep_sky{0}, V_entry{0}, V_entry_bad{0};
struct VReport { ~VReport() {
    std::fprintf(stderr, "SKY_VERIFY chase_targets %llu bad %llu | lazy actions %llu bad %llu | chase counts %llu bad %llu | "
                 "checkers %llu bad %llu | entry(CHECK/count/union/run-bound) %llu bad %llu | forbids %llu (slow-path %llu, forbidden %llu) bad %llu | "
                 "repetition_status %llu (skyrule cycle verdicts %llu) bad %llu\n",
        (unsigned long long)V_ct, (unsigned long long)V_ct_bad, (unsigned long long)V_act, (unsigned long long)V_act_bad,
        (unsigned long long)V_cnt, (unsigned long long)V_cnt_bad, (unsigned long long)V_chk, (unsigned long long)V_chk_bad,
        (unsigned long long)V_entry, (unsigned long long)V_entry_bad,
        (unsigned long long)V_forb, (unsigned long long)V_forb_slow, (unsigned long long)V_forb_true, (unsigned long long)V_forb_bad,
        (unsigned long long)V_rep, (unsigned long long)V_rep_sky, (unsigned long long)V_rep_bad);
} } g_vreport;
#endif

}  // namespace

char piece_char(Piece p) {
    static const char* chars = ".KRNBACPX";
    if (p == NO_PIECE) return '.';
    const char c = chars[type_of(p)];
    return color_of(p) == RED ? c : char(std::tolower(static_cast<unsigned char>(c)));
}

std::string move_str(Move m) {
    if (m == MOVE_NONE) return "0000";
    std::string s;
    s += char('a' + file_of(from_sq(m)));
    s += char('0' + rank_of(from_sq(m)));
    s += char('a' + file_of(to_sq(m)));
    s += char('0' + rank_of(to_sq(m)));
    return s;
}

Move parse_move_coords(const std::string& s) {
    if (s.size() < 4) return MOVE_NONE;
    const int f1 = s[0] - 'a', r1 = s[1] - '0', f2 = s[2] - 'a', r2 = s[3] - '0';
    if (f1 < 0 || f1 > 8 || f2 < 0 || f2 > 8 || r1 < 0 || r1 > 9 || r2 < 0 || r2 > 9) return MOVE_NONE;
    return make_move(make_sq(f1, r1), make_sq(f2, r2));
}

void Position::update_mean(Color c) {
    if (urnTotal[c] <= 0) { urnMean[c] = 0; return; }
    int sum = 0;
    for (int sl = 0; sl < NUM_SLOTS; ++sl) sum += urn[c][sl] * PieceValue[slot_type(sl)];
    urnMean[c] = int16_t(sum / urnTotal[c]);
}

void Position::compute_bitboards() {
    BB::init();
    occBB = darkBB = 0;
    byColorBB[0] = byColorBB[1] = 0;
    for (auto& b : byRoleBB) b = 0;
    for (int s = 0; s < NUM_SQ; ++s)
        if (board[s]) bb_put(s, board[s]);
}

bool Position::bitboards_ok() const {
    Position q;
    std::memcpy(q.board, board, sizeof board);
    q.compute_bitboards();
    if (q.occBB != occBB || q.darkBB != darkBB || q.byColorBB[0] != byColorBB[0] || q.byColorBB[1] != byColorBB[1]) return false;
    for (int t = 0; t < 9; ++t)
        if (q.byRoleBB[t] != byRoleBB[t]) return false;
    return true;
}

void Position::refresh_caches() {
    compute_bitboards();
    chkInv = (stm == RED || stm == BLACK) && kingSq[0] >= 0 && kingSq[1] >= 0 && kingSq[0] < NUM_SQ && kingSq[1] < NUM_SQ && !king_attacked(~stm, kingSq[~stm]);
    update_mean(RED);
    update_mean(BLACK);
    skey = 0;
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < pcount[c]; ++i) {
            const int sq = plist[c][i];
            if (type_of(board[sq]) == SOLDIER || type_of(board[sq]) == DARK) skey ^= T.zPiece[board[sq]][sq];
        }
}

void Position::add_piece(int s, Piece p) {
    board[s] = p;
    const Color c = color_of(p);
    pidx[s] = pcount[c];
    plist[c][pcount[c]++] = uint8_t(s);
}

void Position::remove_piece(int s) {
    const Color c = color_of(board[s]);
    const int i = pidx[s];
    const int last = plist[c][--pcount[c]];
    plist[c][i] = uint8_t(last);
    pidx[last] = int8_t(i);
    pidx[s] = -1;
    board[s] = NO_PIECE;
}

void Position::move_piece(int from, int to) {
    const Color c = color_of(board[from]);
    const int i = pidx[from];
    plist[c][i] = uint8_t(to);
    pidx[to] = int8_t(i);
    pidx[from] = -1;
    board[to] = board[from];
    board[from] = NO_PIECE;
}

uint64_t Position::compute_key() const {
    uint64_t k = 0;
    for (int s = 0; s < NUM_SQ; ++s)
        if (board[s]) k ^= T.zPiece[board[s]][s];
    if (stm == BLACK) k ^= T.zSide;
    for (int c = 0; c < 2; ++c)
        for (int sl = 0; sl < NUM_SLOTS; ++sl) k ^= T.zUrn[c][sl][urn[c][sl]];
    return k;
}

bool Position::set_fen(const std::string& fen, std::string* error) {
    const bool ok = set_fen_impl(fen, error);
    refresh_caches();
    return ok;
}

bool Position::set_fen_impl(const std::string& fen, std::string* error) {
    init_tables();
    auto parse_nonnegative = [&](const std::string& text, const char* field, int& value) {
        if (text.empty() || !std::all_of(text.begin(), text.end(), [](char ch) {
                return ch >= '0' && ch <= '9';
            }))
            return fail(error, (std::string("invalid ") + field + " in FEN").c_str());
        const char* first = text.data();
        const char* last = first + text.size();
        const auto result = std::from_chars(first, last, value);
        if (result.ec != std::errc{} || result.ptr != last || value < 0)
            return fail(error, (std::string("invalid ") + field + " in FEN").c_str());
        return true;
    };

    std::vector<std::string> fields;
    std::istringstream in(fen);
    for (std::string field; in >> field;) fields.push_back(field);
    if (fields.size() == 4)
        fields.insert(fields.begin() + 2, "-");
    if (fields.size() != 5 && fields.size() != 6)
        return fail(error, "FEN requires five or six fields");

    const std::string& boardText = fields[0];
    const std::string& side = fields[1];
    const std::string& pools = fields[2];
    const std::string& clock = fields[fields.size() == 6 ? 4 : 3];
    const std::string& number = fields[fields.size() == 6 ? 5 : 4];
    int halfmove = 0, fullmove = 0;
    if (!parse_nonnegative(clock, "halfmove clock", halfmove)
        || !parse_nonnegative(number, "fullmove number", fullmove))
        return fail(error, "invalid move counters in FEN");
    const int64_t ply = (fullmove == 0 ? 0LL : 2LL * (fullmove - 1)) + (side == "b");
    if (ply > std::numeric_limits<int>::max())
        return fail(error, "fullmove number is too large");
    std::memset(board, 0, sizeof board);
    std::memset(pieceIds, 0, sizeof pieceIds);
    std::memset(pidx, -1, sizeof pidx);
    pcount[0] = pcount[1] = 0;
    darkCount[0] = darkCount[1] = 0;
    kingSq[0] = kingSq[1] = -1;
    // Everything else is reset before parsing too, so a FEN rejected half-way leaves a defined (empty) state and
    // never the previous position's or indeterminate pool / counters / history.
    std::memset(urn, 0, sizeof urn);
    urnTotal[0] = urnTotal[1] = 0;
    stm = RED;
    noCapture = 0;
    gamePly = 0;
    lastIrrev = 0;
    key = 0;
    keyHist.clear();
    checkHist.clear();
    moveHist.clear();
    skyHist.clear();
    skyBoards.clear();
    int kingCount[2] = {};
    int rank = 9, file = 0;
    for (char ch : boardText) {
        if (ch == '/') {
            if (file != 9) return fail(error, "rank does not have 9 files");
            --rank;
            file = 0;
            continue;
        }
        if (ch >= '1' && ch <= '9') {
            file += ch - '0';
            if (file > 9) return fail(error, "rank contains more than nine files");
            continue;
        }
        const PieceType t = type_from_char(ch);
        if (t == NO_TYPE || rank < 0 || file > 8) return fail(error, "bad board character");
        const Color c = std::isupper(static_cast<unsigned char>(ch)) ? RED : BLACK;
        const int s = make_sq(file, rank);
        if (t == DARK && T.darkRole[c][s] == NO_TYPE) return fail(error, "face-down piece off its start point");
        if (pcount[c] >= 16) return fail(error, "too many pieces");
        add_piece(s, make_piece(c, t));
        pieceIds[s] = uint16_t(s + 1);
        if (t == KING) { kingSq[c] = int8_t(s); ++kingCount[c]; }
        if (t == DARK) ++darkCount[c];
        ++file;
    }
    if (rank != 0 || file != 9) return fail(error, "board must have 10 ranks");
    if (kingCount[0] != 1 || kingCount[1] != 1) return fail(error, "board must contain exactly one king per side");
    if (side == "w") stm = RED;
    else if (side == "b") stm = BLACK;
    else return fail(error, "bad side to move");
    std::memset(urn, 0, sizeof urn);
    if (pools != "-") {
        // Pool letters are R A C P N B in Pikafish order; accept any order.
        bool seen[2][NUM_SLOTS] = {};
        for (std::size_t i = 0; i < pools.size();) {
            const char ch = pools[i++];
            const PieceType t = type_from_char(ch);
            if (t < ROOK || t > SOLDIER) return fail(error, "bad pool letter");
            int n = 1;
            if (i < pools.size() && std::isdigit(static_cast<unsigned char>(pools[i]))) {
                const std::size_t start = i;
                while (i < pools.size() && std::isdigit(static_cast<unsigned char>(pools[i]))) ++i;
                const std::string digits = pools.substr(start, i - start);
                if (!parse_nonnegative(digits, "rest-piece count", n))
                    return fail(error, "invalid rest-piece count");
            }
            const Color c = std::isupper(static_cast<unsigned char>(ch)) ? RED : BLACK;
            const int slot = type_slot(t);
            if (seen[c][slot]) return fail(error, "duplicate rest-piece identity");
            seen[c][slot] = true;
            if (n > URN_INIT[type_slot(t)]) return fail(error, "pool count too large");
            urn[c][slot] = int8_t(n);
        }
    }
    for (int c = 0; c < 2; ++c) {
        urnTotal[c] = 0;
        for (int sl = 0; sl < NUM_SLOTS; ++sl) urnTotal[c] = int8_t(urnTotal[c] + urn[c][sl]);
        if (urnTotal[c] < darkCount[c]) return fail(error, "pool smaller than face-down count");
    }
    noCapture = halfmove;
    gamePly = int(ply);
    compute_bitboards();
    key = compute_key();
    keyHist.clear();
    checkHist.clear();
    moveHist.clear();
    skyHist.clear();
    skyBoards.clear();
    skyHist.push_back(SKY_NONE_ENTRY);
#ifdef SKY_VERIFY
    skyRawHist.clear();
    skyChaseIdsHist.clear();
    skyStableChaseHist.clear();
    skyCheckIdsHist.clear();
    skyStableCheckHist.clear();
    skyUnionCheckHist.clear();
    skyActionHist.clear();
    skyCountHist.clear();
#endif
    lastIrrev = 0;
    push_history();
#ifdef SKY_VERIFY
    skyRawHist.push_back(0);
    skyChaseIdsHist.push_back(0);
    skyStableChaseHist.push_back(0);
    skyCheckIdsHist.push_back(0);
    skyStableCheckHist.push_back(0);
    skyUnionCheckHist.push_back(0);
    skyActionHist.push_back(SKYRULE_NONE);
    skyCountHist.push_back(0);
#endif
    return true;
}

std::string Position::fen() const {
    std::string out;
    for (int r = 9; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 9; ++f) {
            const Piece p = board[make_sq(f, r)];
            if (!p) { ++empty; continue; }
            if (empty) out += char('0' + empty), empty = 0;
            out += piece_char(p);
        }
        if (empty) out += char('0' + empty);
        if (r) out += '/';
    }
    out += stm == RED ? " w " : " b ";
    static const PieceType order[6] = {ROOK, ADVISOR, CANNON, SOLDIER, HORSE, ELEPHANT};
    std::string pools;
    for (int c = 0; c < 2; ++c)
        for (PieceType t : order) {
            const int n = urn[c][type_slot(t)];
            if (!n) continue;
            pools += piece_char(make_piece(Color(c), t));
            pools += std::to_string(n);
        }
    out += pools.empty() ? "-" : pools;
    out += " " + std::to_string(noCapture) + " " + std::to_string(gamePly / 2 + 1);
    return out;
}

std::string Position::pretty() const {
    std::string out;
    for (int r = 9; r >= 0; --r) {
        out += char('0' + r);
        out += ' ';
        for (int f = 0; f < 9; ++f) {
            out += piece_char(board[make_sq(f, r)]);
            out += ' ';
        }
        out += '\n';
    }
    out += "  a b c d e f g h i\n";
    out += "fen " + fen() + "\n";
    return out;
}

// ---- attack tests on bitboards --------------------------------------------------------------------------------
// Same answers as the original mailbox scans (kept in position_ref.cpp for the JQ_BB_VERIFY build).
bool Position::ray_attacked_bb(Color e, int k, int d, Bitboard occ, Bitboard E, int x, PieceType xt) const {
    // Only an e rook / cannon / king on this ray can attack along it.
    const Bitboard rck = (byRoleBB[ROOK] | byRoleBB[CANNON] | byRoleBB[KING]) & E;
    if (!(BB::Ray[k][d] & rck) && (x < 0 || (xt != ROOK && xt != CANNON && xt != KING) || !test(BB::Ray[k][d], x))) return false;
    int f1, f2;
    BB::blockers(k, d, occ, f1, f2);
    if (f1 < 0) return false;
    if (test(E, f1)) {
        if (f1 == x ? xt == ROOK : test(byRoleBB[ROOK], f1)) return true;
        if (d < 2 && (f1 == x ? xt == KING : test(byRoleBB[KING], f1))) return true;  // facing generals
    }
    return f2 >= 0 && test(E, f2) && (f2 == x ? xt == CANNON : test(byRoleBB[CANNON], f2));
}

bool Position::attacked_bb(Color e, int k, Bitboard occ, Bitboard E, int x, PieceType xt) const {
    const Bitboard xm = x >= 0 ? ~sq_bb(x) : ~Bitboard(0);
    const Bitboard xb = x >= 0 ? sq_bb(x) : 0;
    auto rs = [&](PieceType t) { return (byRoleBB[t] & E & xm) | (xt == t ? xb : 0); };
    // rook / cannon / facing king along k's file and rank
    const Bitboard lines = BB::FileBB[file_of(k)] | BB::RankBB[rank_of(k)];
#ifdef JQ_MG_OLD
    if ((rs(ROOK) | rs(CANNON) | (rs(KING) & BB::FileBB[file_of(k)])) & lines)
        for (int d = 0; d < 4; ++d)
            if (ray_attacked_bb(e, k, d, occ, E, x, xt)) return true;
#else  // all four rays from one scan; same per-ray logic as ray_attacked_bb
    if ((rs(ROOK) | rs(CANNON) | (rs(KING) & BB::FileBB[file_of(k)])) & lines) {
        const Bitboard R = rs(ROOK), C = rs(CANNON), K = rs(KING);
        int f1[4], f2[4];
        BB::scan4(k, occ, f1, f2);
        for (int d = 0; d < 4; ++d) {
            const int a = f1[d];
            if (a < 0) continue;
            if (test(R, a) || (d < 2 && test(K, a))) return true;
            const int b = f2[d];
            if (b >= 0 && test(C, b)) return true;
        }
    }
#endif
    if (Bitboard h = BB::HorseFrom[k] & rs(HORSE)) {
        do {
            const int p = pop_lsb(h);
            if (!test(occ, BB::Leg[p][k])) return true;
        } while (h);
    }
    if (BB::SoldierFrom[e][k] & rs(SOLDIER)) return true;
    const Bitboard dk = darkBB & E & xm;
    if (Bitboard a = BB::AdvTo[k] & rs(ADVISOR)) {
        if (!in_palace(e, k)) a &= ~dk;  // a face-down advisor stays in its palace
        if (a) return true;
    }
    if (Bitboard el = BB::ElephTo[k] & rs(ELEPHANT)) {
        if (!own_half(e, k)) el &= ~dk;  // a face-down elephant stays on its own half
        while (el) {
            const int p = pop_lsb(el);
            if (!test(occ, BB::Eye[k][p])) return true;
        }
    }
    return false;
}

bool Position::JQ_IMPL(ray_attacked)(Color c, int k, int d) const { return ray_attacked_bb(~c, k, d, occBB, byColorBB[~c]); }

bool Position::JQ_IMPL(king_attacked)(Color c, int k) const { return attacked_bb(~c, k, occBB, byColorBB[~c]); }

namespace {
inline void add(Move*& out, int from, int to) { *out++ = make_move(from, to); }
}  // namespace

// Move order is exactly the mailbox order (piece-list order; per piece: rays N,S,E,W outwards / table order),
// because the search's move ordering breaks ties by generation order.
template <bool CAP>
int Position::JQ_IMPL(gen)(Move* list) const {
    Move* out = list;
    const Color us = stm, them = ~us;
    const Bitboard occ = occBB;
    const Bitboard capt = byColorBB[them] & ~byRoleBB[KING];  // capturable squares
    const Bitboard tgt = CAP ? capt : capt | (BB_ALL & ~occ);   // legal destinations
    for (int i = 0; i < pcount[us]; ++i) {
        const int s = plist[us][i];
        const Piece pc = board[s];
        const bool dark = type_of(pc) == DARK;
        const PieceType t = role(s);
        switch (t) {
        case KING:
            if (CAP && !(BB::KingTo[s] & tgt)) break;
            for (int j = 0; j < T.kingN[s]; ++j)
                if (test(tgt, T.kingTo[s][j])) add(out, s, T.kingTo[s][j]);
            break;
        case ROOK:
            if (CAP && !(capt & (BB::FileBB[file_of(s)] | BB::RankBB[rank_of(s)]))) break;
#ifndef JQ_MG_OLD
            {   // the four first blockers from one scan
                int f1[4], f2[4];
                BB::scan4(s, occ, f1, f2);
                for (int d = 0; d < 4; ++d) {
                    const int b = f1[d];
                    if (!CAP) {
                        const uint8_t* r = T.ray[s][d];
                        for (int j = 0, n = T.rayLen[s][d]; j < n && r[j] != b; ++j) add(out, s, r[j]);
                    }
                    if (b >= 0 && test(capt, b)) add(out, s, b);
                }
                break;
            }
#endif
            for (int d = 0; d < 4; ++d) {
                const int b = BB::first_blocker(s, d, occ);
                if (!CAP) {
                    const uint8_t* r = T.ray[s][d];
                    for (int j = 0, n = T.rayLen[s][d]; j < n && r[j] != b; ++j) add(out, s, r[j]);
                }
                if (b >= 0 && test(capt, b)) add(out, s, b);
            }
            break;
        case CANNON:
            if (CAP && !(capt & (BB::FileBB[file_of(s)] | BB::RankBB[rank_of(s)]))) break;
#ifndef JQ_MG_OLD
            {   // first / second blockers of the four rays from one scan
                int f1[4], f2[4];
                BB::scan4(s, occ, f1, f2);
                for (int d = 0; d < 4; ++d) {
                    const int b1 = f1[d], b2 = f2[d];
                    if (!CAP) {
                        const uint8_t* r = T.ray[s][d];
                        for (int j = 0, n = T.rayLen[s][d]; j < n && r[j] != b1; ++j) add(out, s, r[j]);
                    }
                    if (b2 >= 0 && test(capt, b2)) add(out, s, b2);
                }
                break;
            }
#endif
            for (int d = 0; d < 4; ++d) {
                int b1, b2;
                BB::blockers(s, d, occ, b1, b2);
                if (!CAP) {
                    const uint8_t* r = T.ray[s][d];
                    for (int j = 0, n = T.rayLen[s][d]; j < n && r[j] != b1; ++j) add(out, s, r[j]);
                }
                if (b2 >= 0 && test(capt, b2)) add(out, s, b2);
            }
            break;
        case HORSE:
            if (!(BB::HorseTo[s] & tgt)) break;
            for (int j = 0; j < T.horseN[s]; ++j)
                if (!board[T.horseLeg[s][j]] && test(tgt, T.horseTo[s][j])) add(out, s, T.horseTo[s][j]);
            break;
        case ELEPHANT: {
            Bitboard m = BB::ElephTo[s] & tgt;
            if (dark) m &= BB::Half[us];
            if (!m) break;
            for (int j = 0; j < T.elephN[s]; ++j)
                if (test(m, T.elephTo[s][j]) && !board[T.elephEye[s][j]]) add(out, s, T.elephTo[s][j]);
            break;
        }
        case ADVISOR: {
            Bitboard m = BB::AdvTo[s] & tgt;
            if (dark) m &= BB::Palace[us];
            if (!m) break;
            for (int j = 0; j < T.advN[s]; ++j)
                if (test(m, T.advTo[s][j])) add(out, s, T.advTo[s][j]);
            break;
        }
        case SOLDIER: {
            const Bitboard m = BB::SoldierTo[us][s] & tgt;
            if (!m) break;
            const int to = s + (us == RED ? 9 : -9);  // order: forward, west, east (as the mailbox loop)
            if (to >= 0 && to < NUM_SQ && test(m, to)) add(out, s, to);
            if (s > 0 && test(m, s - 1)) add(out, s, s - 1);
            if (test(m, s + 1)) add(out, s, s + 1);  // s + 1 <= 90 < 128
            break;
        }
        default: break;
        }
    }
    return int(out - list);
}

int Position::gen_moves(Move* list) const { return gen<false>(list); }

// Captures only, in the same order as gen_moves() would list them.
int Position::gen_captures(Move* list) const { return gen<true>(list); }

namespace {
#ifdef JQ_MG_OLD
inline bool same_line(int a, int b) { return file_of(a) == file_of(b) || rank_of(a) == rank_of(b); }
inline bool diag_adjacent(int a, int b) {
    return std::abs(file_of(a) - file_of(b)) == 1 && std::abs(rank_of(a) - rank_of(b)) == 1;
}
inline int cheb(int a, int b) { return std::max(std::abs(file_of(a) - file_of(b)), std::abs(rank_of(a) - rank_of(b))); }
inline bool cheb_le2(int a, int b) { return cheb(a, b) <= 2; }
#else  // one 8 KB relation table instead of div / mod by 9
inline bool same_line(int a, int b) { return BB::REL[a][b] & 1; }
inline bool diag_adjacent(int a, int b) { return BB::REL[a][b] & 2; }
inline bool cheb_le2(int a, int b) { return BB::REL[a][b] & 4; }
#endif
// A non-king move can change attacks on king k only through k's file/rank (sliders, cannon screens,
// facing generals) or k's diagonal neighbours (horse legs and elephant eyes).
inline bool touches_king_lines(int from, int to, int k) {
    return same_line(from, k) || same_line(to, k) || diag_adjacent(from, k);
}
}  // namespace

bool Position::JQ_IMPL(legal)(Move m) const {
    const int from = from_sq(m), to = to_sq(m);
    const bool kingMove = type_of(board[from]) == KING;
    if (!kingMove && !checkHist.back() && !touches_king_lines(from, to, kingSq[stm])) return true;
    // Board after the move: the mover (identity irrelevant for own safety) on to, a captured enemy removed.
    const Color e = ~stm;
    const Bitboard occ = (occBB & ~sq_bb(from)) | sq_bb(to);
    const Bitboard E = byColorBB[e] & ~sq_bb(to);
    bool ok;
    const int k0 = kingSq[stm];
    if (kingMove || checkHist.back() || diag_adjacent(from, k0))
        ok = !attacked_bb(e, kingMove ? to : k0, occ, E);
    else {
        // Not in check and the king stays: only rays through from (discovery) or to (new cannon
        // screen) can carry a new attack.
        const int df = T.dirTo[k0][from], dt = T.dirTo[k0][to];
        ok = !((df >= 0 && ray_attacked_bb(e, k0, df, occ, E)) || (dt >= 0 && dt != df && ray_attacked_bb(e, k0, dt, occ, E)));
    }
    return ok;
}

#if defined(JQ_LCTX) && !defined(JQ_MG_OLD)
// Legal-context slow path (see position.h; off by default): not in check, not a king move, from or to on
// a king line / from on a diagonal neighbour. Same answer as legal(m).
bool Position::legal_slow(Move m, LegalCtx& c) const {
    const int from = from_sq(m), to = to_sq(m), k = c.k;
    if (diag_adjacent(from, k)) return legal(m);
    const int df = T.dirTo[k][from], dt = T.dirTo[k][to];
    if (df == dt) return legal(m);
    if (!c.init) {
        int f1[4], f2[4];
        BB::scan4(k, occBB, f1, f2);
        for (int d = 0; d < 4; ++d) {
            c.b[d][0] = int8_t(f1[d]); c.b[d][1] = int8_t(f2[d]);
            c.b[d][2] = int8_t(f2[d] >= 0 ? BB::first_blocker(f2[d], d, occBB) : -1);
        }
        c.init = true;
    }
    const Bitboard E = byColorBB[~stm] & ~sq_bb(to);
    auto hit = [&](int d, int a, int b) {
        if (a < 0) return false;
        if (test(E, a) && (test(byRoleBB[ROOK], a) || (d < 2 && test(byRoleBB[KING], a)))) return true;
        return b >= 0 && test(E, b) && test(byRoleBB[CANNON], b);
    };
    if (df >= 0) {
        const int8_t* L = c.b[df];
        if (from == L[0] ? hit(df, L[1], L[2]) : from == L[1] ? hit(df, L[0], L[2]) : false) return false;
    }
    if (dt >= 0) {
        const int8_t* L = c.b[dt];
        const bool asc = BB::ascending(dt);
        auto closer = [&](int x, int y) { return y < 0 || (asc ? x < y : x > y); };
        int a, b;
        if (to == L[0]) a = to, b = L[1];
        else if (closer(to, L[0])) a = to, b = L[0];
        else if (to == L[1] || closer(to, L[1])) a = L[0], b = to;
        else a = L[0], b = L[1];
        if (hit(dt, a, b)) return false;
    }
    return true;
}
#else
bool Position::legal_slow(Move m, LegalCtx&) const { return legal(m); }
#endif

bool Position::reveal_may_check(int to) const {
    const int k = kingSq[~stm];
    return same_line(to, k) || cheb_le2(to, k);
}

bool Position::JQ_IMPL(reveal_gives_check)(int from, int to, PieceType t) const {
    const Bitboard occ = (occBB & ~sq_bb(from)) | sq_bb(to);
    const Bitboard U = (byColorBB[stm] & ~sq_bb(from)) | sq_bb(to);
    return attacked_bb(stm, kingSq[~stm], occ, U, to, t);
}

bool Position::JQ_IMPL(check_after)(int from, int to, PieceType t) const {
    const int k = kingSq[stm];
    if (!touches_king_lines(from, to, k) && (t == ROOK || t == CANNON || t == KING || !cheb_le2(to, k))) return false;
    if (diag_adjacent(from, k) || (t != ROOK && t != CANNON && t != KING && cheb_le2(to, k))) return king_attacked(stm, k);
    const int df = T.dirTo[k][from], dt = T.dirTo[k][to];
    return (df >= 0 && ray_attacked(stm, k, df)) || (dt >= 0 && dt != df && ray_attacked(stm, k, dt));
}

bool Position::JQ_IMPL(gives_check)(Move m) const {
    const int from = from_sq(m), to = to_sq(m);
    const Piece moving = board[from];
    if (type_of(moving) == DARK) return false;
    const int k = kingSq[~stm];
    const PieceType t = type_of(moving);
    // Off the enemy king's lines a rook/cannon/king cannot attack it, and the others need to be close.
    if (!touches_king_lines(from, to, k) && (t == ROOK || t == CANNON || t == KING || !cheb_le2(to, k))) return false;
    const Bitboard occ = (occBB & ~sq_bb(from)) | sq_bb(to);
    const Bitboard U = (byColorBB[stm] & ~sq_bb(from)) | sq_bb(to);
    if (diag_adjacent(from, k) || (t != ROOK && t != CANNON && t != KING && cheb_le2(to, k)))
        return attacked_bb(stm, k, occ, U, to, t);
    // The mover can only check along a ray through to; discoveries come through from's ray.
    const int df = T.dirTo[k][from], dt = T.dirTo[k][to];
    return (df >= 0 && ray_attacked_bb(stm, k, df, occ, U, to, t)) || (dt >= 0 && dt != df && ray_attacked_bb(stm, k, dt, occ, U, to, t));
}

int Position::gen_legal(Move* list) const {
    Move tmp[MAX_MOVES];
    const int n = gen_moves(tmp);
    int k = 0;
    for (int i = 0; i < n; ++i)
        if (legal(tmp[i])) list[k++] = tmp[i];
    return k;
}

bool Position::has_legal_move() const {
    Move tmp[MAX_MOVES];
    const int n = gen_moves(tmp);
    for (int i = 0; i < n; ++i)
        if (legal(tmp[i])) return true;
    return false;
}

void Position::do_move(Move m, StateInfo& st) {
    const int from = from_sq(m), to = to_sq(m);
    const Color us = stm;
    const Piece pc = board[from];
    const Piece cap = board[to];
    const bool sky = g_chaseRule == CHASE_RULE_SKYRULE;
#ifdef SKY_VERIFY
    const Bitboard beforeTargets = sky && !cap && type_of(pc) != DARK ? skyrule_chase_targets(*this, us) : 0;
    const Bitboard beforeTargetIds = sky && !cap && type_of(pc) != DARK ? skyrule_target_ids(*this, beforeTargets) : 0;
#endif
    st.key = key;
    st.captured = cap;
    st.urnSlot = -1;
    st.noCapture = noCapture;
    st.gamePly = gamePly;
    st.lastIrrev = int16_t(lastIrrev);
    st.skey = skey;
    st.movedId = pieceIds[from];
    st.capturedId = cap ? pieceIds[to] : 0;
    bb_move(us, from, to, pc, reveal_of(m), cap);
    if (cap) {
        key ^= T.zPiece[cap][to];
        if (type_of(cap) == SOLDIER || type_of(cap) == DARK) skey ^= T.zPiece[cap][to];
        if (type_of(cap) == DARK) --darkCount[~us];
        remove_piece(to);
    }
    key ^= T.zPiece[pc][from];
    if (type_of(pc) == SOLDIER || type_of(pc) == DARK) skey ^= T.zPiece[pc][from];
    move_piece(from, to);
    pieceIds[to] = st.movedId;
    pieceIds[from] = 0;
    Piece np = pc;
    if (type_of(pc) == DARK) {
        const int sl = type_slot(reveal_of(m));
        np = make_piece(us, reveal_of(m));
        board[to] = np;
        st.urnMean = urnMean[us];
        key ^= T.zUrn[us][sl][urn[us][sl]];
        --urn[us][sl];
        key ^= T.zUrn[us][sl][urn[us][sl]];
        --urnTotal[us];
        update_mean(us);
        --darkCount[us];
        st.urnSlot = int8_t(sl);
    } else if (type_of(pc) == KING) {
        kingSq[us] = int8_t(to);
    }
    key ^= T.zPiece[np][to];
    if (type_of(np) == SOLDIER) skey ^= T.zPiece[np][to];
    key ^= T.zSide;
    noCapture = cap ? 0 : noCapture + 1;
    ++gamePly;
    stm = ~us;
    keyHist.push_back(key);
#if !defined(JQ_MG_OLD) && !defined(JQ_NO_HINT)
    if (m == hintMove && st.key == hintKey && keyHist.size() == hintLen + 1) {  // keyHist already holds the new key
        hintMove = MOVE_NONE;
#ifdef JQ_BB_VERIFY
        if (hintVal != (chkInv ? check_after(from, to, type_of(np)) : in_check())) { std::fprintf(stderr, "BB VERIFY MISMATCH check hint\n"); std::abort(); }
#endif
        checkHist.push_back(uint8_t(hintVal));
    } else
#endif
    checkHist.push_back(uint8_t(chkInv ? check_after(from, to, type_of(np)) : in_check()));
    moveHist.push_back(base_move(m));
    {
        SkyEntry e = SKY_NONE_ENTRY;
        if (sky && !cap && type_of(pc) != DARK) {
            const std::size_t cur = skyHist.size();
            const Bitboard checkIds = checkHist.back() ? skyrule_checkers(*this, us) : Bitboard(0);
            if (checkIds) {
                e.action = SKYRULE_CHECK;
                e.unionCheck = checkIds;
                e.count = 1;
                if (cur >= 3) {
                    const SkyEntry& p = skyHist[cur - 2];
                    if (p.action == SKYRULE_CHECK) {
                        e.count = uint16_t(p.count + 1);
                        e.unionCheck = p.unionCheck | checkIds;
                    }
                }
            } else {
                e.action = SKY_PENDING;
                e.countKnown = 0;
                e.run = 1;
                if (cur >= 3) {
                    const SkyEntry& p = skyHist[cur - 2];
                    if (p.action == SKY_PENDING || p.action == SKYRULE_CHASE) e.run = uint16_t(p.run + 1);
                }
                e.from = uint8_t(from);
                e.to = uint8_t(to);
                e.actor = uint8_t(us);
                e.hasBoard = 1;
                e.board = uint32_t(skyBoards.size());
                skyBoards.emplace_back();
                std::memcpy(skyBoards.back().b, board, NUM_SQ);
            }
        }
        skyHist.push_back(e);
#ifdef SKY_ABL_EAGER  // ablation build only: classify every record at once (no laziness)
        if (e.action == SKY_PENDING && sky_action(skyHist.size() - 1) == SKYRULE_CHASE) sky_count(skyHist.size() - 1);
#endif
    }
#ifdef SKY_VERIFY
    skyRawHist.push_back(0);
    skyChaseIdsHist.push_back(0);
    skyStableChaseHist.push_back(0);
    skyCheckIdsHist.push_back(0);
    skyStableCheckHist.push_back(0);
    skyUnionCheckHist.push_back(0);
    skyActionHist.push_back(SKYRULE_NONE);
    skyCountHist.push_back(0);
    if (sky && !cap && type_of(pc) != DARK) {
        const Bitboard checkIds = checkHist.back() ? ref_skyrule_checkers(*this, us) : Bitboard(0);
        const Bitboard afterTargets = checkIds ? Bitboard(0) : skyrule_chase_targets(*this, us);
        const Bitboard afterTargetIds = checkIds ? Bitboard(0) : skyrule_target_ids(*this, afterTargets);
        const Bitboard newTargetIds = afterTargetIds & ~beforeTargetIds;
        const std::size_t current = skyActionHist.size() - 1;
        if (checkIds) {
            skyActionHist[current] = SKYRULE_CHECK;
            skyCheckIdsHist[current] = checkIds;
            skyStableCheckHist[current] = checkIds;
            skyUnionCheckHist[current] = checkIds;
        } else if (newTargetIds) {
            Bitboard raw = 0;
            for_each_sq(afterTargets, [&](int s) {
                const uint16_t id = pieceIds[s];
                if (id && (newTargetIds & sq_bb(id - 1))) raw |= sq_bb(s);
            });
            skyActionHist[current] = SKYRULE_CHASE;
            skyRawHist[current] = raw;
            skyChaseIdsHist[current] = newTargetIds;
            skyStableChaseHist[current] = newTargetIds;
        }
        if (current >= 3) {
            const std::size_t previous = current - 2;
            if (skyActionHist[current] == skyActionHist[previous]
                && skyActionHist[current] != SKYRULE_NONE) {
                skyCountHist[current] = uint16_t(skyCountHist[previous] + 1);
                if (skyActionHist[current] == SKYRULE_CHECK) {
                    skyStableCheckHist[current] = skyStableCheckHist[previous] & skyCheckIdsHist[current];
                    skyUnionCheckHist[current] = skyUnionCheckHist[previous] | skyCheckIdsHist[current];
                } else {
                    skyStableChaseHist[current] = skyStableChaseHist[previous] & skyChaseIdsHist[current];
                }
            } else if (skyActionHist[current] != SKYRULE_NONE)
                skyCountHist[current] = 1;
        } else if (skyActionHist[current] != SKYRULE_NONE)
            skyCountHist[current] = 1;
    }
#endif
#ifdef SKY_VERIFY
    {
        const std::size_t k = skyHist.size() - 1;
        const SkyEntry& e = skyHist[k];
        bool bad = skyHist.size() != skyActionHist.size() || skyHist.size() != keyHist.size();
        const uint8_t ra = skyActionHist[k];
        if (ra == SKYRULE_CHECK) bad = bad || e.action != SKYRULE_CHECK || e.count != skyCountHist[k] || e.unionCheck != skyUnionCheckHist[k];
        else if (e.action == SKYRULE_CHECK) bad = true;
        else if (e.action == SKY_PENDING) {
            const uint8_t a = sky_classify(e, skyBoards[e.board].b);
            bad = bad || a != ra || (ra == SKYRULE_CHASE && e.run < skyCountHist[k]);
        } else bad = bad || ra != SKYRULE_NONE || skyCountHist[k] != 0;
        for (int s2 = 0; s2 < NUM_SQ; ++s2) if (board[s2] && !pieceIds[s2]) bad = true;
        ++V_entry;
        if (bad) ++V_entry_bad;
    }
#endif
    if (cap || type_of(pc) == DARK) lastIrrev = int(keyHist.size()) - 1;
}

uint64_t Position::key_after(Move m) const {
    const int from = from_sq(m), to = to_sq(m);
    const Piece pc = board[from], cap = board[to];
    uint64_t k = key ^ T.zPiece[pc][from] ^ T.zSide;
    if (cap) k ^= T.zPiece[cap][to];
    Piece np = pc;
    if (type_of(pc) == DARK) {
        const int sl = type_slot(reveal_of(m));
        np = make_piece(stm, reveal_of(m));
        k ^= T.zUrn[stm][sl][urn[stm][sl]] ^ T.zUrn[stm][sl][urn[stm][sl] - 1];
    }
    return k ^ T.zPiece[np][to];
}

void Position::undo_move(Move m, const StateInfo& st) {
    const int from = from_sq(m), to = to_sq(m);
    stm = ~stm;
    const Color us = stm;
    keyHist.pop_back();
    checkHist.pop_back();
    moveHist.pop_back();
    if (skyHist.back().hasBoard) skyBoards.pop_back();
    skyHist.pop_back();
#ifdef SKY_VERIFY
    skyRawHist.pop_back();
    skyChaseIdsHist.pop_back();
    skyStableChaseHist.pop_back();
    skyCheckIdsHist.pop_back();
    skyStableCheckHist.pop_back();
    skyUnionCheckHist.pop_back();
    skyActionHist.pop_back();
    skyCountHist.pop_back();
#endif
    if (st.urnSlot >= 0) {
        bb_move(us, from, to, make_piece(us, DARK), type_of(board[to]), st.captured);
        ++urn[us][st.urnSlot];
        ++urnTotal[us];
#ifdef JQ_MG_OLD
        update_mean(us);
#else
        urnMean[us] = st.urnMean;  // restore instead of recompute (same value: urn restored exactly)
#endif
        ++darkCount[us];
        board[to] = make_piece(us, DARK);
    } else {
        bb_move(us, from, to, board[to], NO_TYPE, st.captured);
        if (type_of(board[to]) == KING) kingSq[us] = int8_t(from);
    }
    move_piece(to, from);
    pieceIds[from] = st.movedId;
    pieceIds[to] = 0;
    if (st.captured) {
        add_piece(to, st.captured);
        pieceIds[to] = st.capturedId;
        if (type_of(st.captured) == DARK) ++darkCount[~us];
    }
    key = st.key;
    skey = st.skey;
    noCapture = st.noCapture;
    gamePly = st.gamePly;
    lastIrrev = st.lastIrrev;
}

void Position::do_null(StateInfo& st) {
    st.key = key;
    st.noCapture = noCapture;
    st.gamePly = gamePly;
    st.lastIrrev = int16_t(lastIrrev);
    key ^= T.zSide;
    stm = ~stm;
    ++noCapture;
    ++gamePly;
    keyHist.push_back(key);
    checkHist.push_back(0);
    moveHist.push_back(MOVE_NULL);
    skyHist.push_back(SKY_NONE_ENTRY);
#ifdef SKY_VERIFY
    skyRawHist.push_back(0);
    skyChaseIdsHist.push_back(0);
    skyStableChaseHist.push_back(0);
    skyCheckIdsHist.push_back(0);
    skyStableCheckHist.push_back(0);
    skyUnionCheckHist.push_back(0);
    skyActionHist.push_back(SKYRULE_NONE);
    skyCountHist.push_back(0);
#endif
    lastIrrev = int(keyHist.size()) - 1;  // a null move breaks repetition cycles
}

void Position::undo_null(const StateInfo& st) {
    stm = ~stm;
    key = st.key;
    noCapture = st.noCapture;
    gamePly = st.gamePly;
    lastIrrev = st.lastIrrev;
    keyHist.pop_back();
    checkHist.pop_back();
    moveHist.pop_back();
    skyHist.pop_back();
#ifdef SKY_VERIFY
    skyRawHist.pop_back();
    skyChaseIdsHist.pop_back();
    skyStableChaseHist.pop_back();
    skyCheckIdsHist.pop_back();
    skyStableCheckHist.pop_back();
    skyUnionCheckHist.pop_back();
    skyActionHist.pop_back();
    skyCountHist.pop_back();
#endif
}

int g_chaseRule = CHASE_RULE_SKYRULE;

namespace {
// ---- Perpetual-chase classification -------------------------------------------------------------
// Original port of the project's referee (native/chase.cpp, specified in ADJUDICATION.md). It works on
// a board of public positional types: a face-down piece acts as a face-up piece of its start-square
// role (so a dark advisor or elephant roams freely here, unlike in ordinary move generation).
using Mask = std::bitset<NUM_SQ>;
int sgn(int v) { return (v > 0) - (v < 0); }

int intervening(const Piece* b, int from, int to) {
    if (from == to || (file_of(from) != file_of(to) && rank_of(from) != rank_of(to))) return -1;
    const int step = sgn(file_of(to) - file_of(from)) + 9 * sgn(rank_of(to) - rank_of(from));
    int n = 0;
    for (int s = from + step; s != to; s += step) n += b[s] != NO_PIECE;
    return n;
}

bool geometry(const Piece* b, int from, int to) {
    const Piece pc = b[from];
    if (!pc || from == to) return false;
    const int dx = file_of(to) - file_of(from), dy = rank_of(to) - rank_of(from);
    const int ax = std::abs(dx), ay = std::abs(dy);
    const Color c = color_of(pc);
    switch (type_of(pc)) {
    case KING: return ax + ay == 1 && in_palace(c, to);
    case ROOK: return intervening(b, from, to) == 0;
    case CANNON: return intervening(b, from, to) == (b[to] ? 1 : 0);
    case HORSE:
        if (!((ax == 2 && ay == 1) || (ax == 1 && ay == 2))) return false;
        return !b[from + (ax == 2 ? sgn(dx) : 9 * sgn(dy))];
    case ELEPHANT: return ax == 2 && ay == 2 && !b[from + dx / 2 + 9 * (dy / 2)];
    case ADVISOR: return ax == 1 && ay == 1;
    case SOLDIER: return (dx == 0 && dy == (c == RED ? 1 : -1)) || (dy == 0 && ax == 1 && !own_half(c, from));
    default: return false;
    }
}

int king_square(const Piece* b, Color c) {
    for (int s = 0; s < NUM_SQ; ++s)
        if (b[s] == make_piece(c, KING)) return s;
    return -1;
}

// King safety of the hypothetical test: the enemy king counts as a rook ray (facing generals).
bool king_exposed(const Piece* b, Color c) {
    const int k = king_square(b, c);
    if (k < 0) return true;
    for (int s = 0; s < NUM_SQ; ++s) {
        const Piece pc = b[s];
        if (!pc || color_of(pc) == c) continue;
        if (type_of(pc) == KING ? intervening(b, s, k) == 0 : geometry(b, s, k)) return true;
    }
    return false;
}

bool safe_capture(Piece* b, int from, int to) {
    const Piece m = b[from], v = b[to];
    if (from == to || !m || !v || type_of(v) == KING || color_of(m) == color_of(v) || !geometry(b, from, to)) return false;
    b[to] = m;
    b[from] = NO_PIECE;
    const bool exposed = king_exposed(b, color_of(m));
    b[from] = m;
    b[to] = v;
    return !exposed;
}

bool major(PieceType t) { return t == ROOK || t == HORSE || t == CANNON; }

// Pieces screening an enemy line onto `actor`'s king ("blocked") and the enemy pieces pinning one of
// actor's pieces ("pinning"). For non-cannon lines, other non-cannon candidates are not screens.
void pins(const Piece* b, Color actor, Mask& blocked, Mask& pinning) {
    const int king = king_square(b, actor);
    if (king < 0) return;
    Mask candidates, omitted;
    int path[NUM_SQ][9];
    int plen[NUM_SQ] = {};
    for (int s = 0; s < NUM_SQ; ++s) {
        const Piece pc = b[s];
        if (!pc || color_of(pc) == actor) continue;
        const int dx = file_of(king) - file_of(s), dy = rank_of(king) - rank_of(s);
        const int ax = std::abs(dx), ay = std::abs(dy);
        const PieceType t = type_of(pc);
        bool cand = false;
        if ((t == ROOK || t == CANNON || t == KING) && (dx == 0 || dy == 0)) {
            cand = true;
            const int step = sgn(dx) + 9 * sgn(dy);
            for (int q = s + step; q != king; q += step) path[s][plen[s]++] = q;
        } else if (t == HORSE && ((ax == 2 && ay == 1) || (ax == 1 && ay == 2))) {
            cand = true;
            path[s][plen[s]++] = s + (ax == 2 ? sgn(dx) : 9 * sgn(dy));
        } else if (t == ELEPHANT && ax == 2 && ay == 2) {
            cand = true;
            path[s][plen[s]++] = s + dx / 2 + 9 * (dy / 2);
        }
        if (cand) {
            candidates.set(s);
            if (t != CANNON) omitted.set(s);
        }
    }
    for (int s = 0; s < NUM_SQ; ++s) {
        if (!candidates[s]) continue;
        const bool cannon = type_of(b[s]) == CANNON;
        Mask screens;
        bool friendly = false;
        for (int k = 0; k < plen[s]; ++k) {
            const int q = path[s][k];
            if (b[q] && (cannon || !omitted[q])) {
                screens.set(q);
                friendly = friendly || color_of(b[q]) == actor;
            }
        }
        if (screens.count() == (cannon ? 2u : 1u)) {
            blocked |= screens;
            if (friendly) pinning.set(s);
        }
    }
}

// Enemy pieces (squares) that `actor` threatens in the sense of the chase rule.
Mask chase_targets(Piece* b, Color actor) {
    Mask blocked, pinning, out;
    pins(b, actor, blocked, pinning);
    for (int target = 0; target < NUM_SQ; ++target) {
        const Piece victim = b[target];
        if (!victim || type_of(victim) == KING || color_of(victim) == actor) continue;
        const PieceType vt = type_of(victim);
        for (int a = 0; a < NUM_SQ; ++a) {
            const Piece pc = b[a];
            if (!pc || color_of(pc) != actor || type_of(pc) == KING || type_of(pc) == SOLDIER) continue;
            if (blocked[a]) {
                if (!pinning[target]) continue;
            } else if (vt == SOLDIER && own_half(color_of(victim), target)) continue;
            if (!safe_capture(b, a, target)) continue;
            const PieceType at = type_of(pc);
            const bool always = ((at == HORSE || at == CANNON) && vt == ROOK) || ((at == ADVISOR || at == ELEPHANT) && major(vt));
            if (!always) {
                b[target] = pc;  // hypothetical capture; any legal recapture protects the target
                b[a] = NO_PIECE;
                bool defended = false;
                for (int d = 0; d < NUM_SQ && !defended; ++d)
                    if (b[d] && color_of(b[d]) == color_of(victim)) defended = safe_capture(b, d, target);
                b[a] = pc;
                b[target] = victim;
                if (defended) continue;
                if (at == vt && safe_capture(b, target, a)) continue;  // mutual attack of equal pieces
            }
            out.set(target);
            break;
        }
    }
    return out;
}

// ---- Fast chase classifier (same result as chase_targets above) ----------------------------------
// Piece lists replace the 90-square scans; every hypothetical board the original builds only moves a
// piece onto a square of the opposite colour's list or removes one, so scanning the lists of the
// original board (skipping squares whose occupant no longer matches) visits exactly the same pieces.
struct ChaseCtx {
    uint8_t sq[2][NUM_SQ];
    int n[2];
    int king[2];
};

inline bool geometry_pc(const Piece* b, Piece pc, int from, int to) {
    const int dx = file_of(to) - file_of(from), dy = rank_of(to) - rank_of(from);
    const int ax = std::abs(dx), ay = std::abs(dy);
    const Color c = color_of(pc);
    switch (type_of(pc)) {
    case KING: return ax + ay == 1 && in_palace(c, to);
    case ROOK: return intervening(b, from, to) == 0;
    case CANNON: return intervening(b, from, to) == (b[to] ? 1 : 0);
    case HORSE:
        if (!((ax == 2 && ay == 1) || (ax == 1 && ay == 2))) return false;
        return !b[from + (ax == 2 ? sgn(dx) : 9 * sgn(dy))];
    case ELEPHANT: return ax == 2 && ay == 2 && !b[from + dx / 2 + 9 * (dy / 2)];
    case ADVISOR: return ax == 1 && ay == 1;
    case SOLDIER: return (dx == 0 && dy == (c == RED ? 1 : -1)) || (dy == 0 && ax == 1 && !own_half(c, from));
    default: return false;
    }
}

inline void ctx_build(const Piece* b, ChaseCtx& x) {
    x.n[0] = x.n[1] = 0;
    x.king[0] = x.king[1] = -1;
    for (int s = 0; s < NUM_SQ; ++s) {
        const Piece pc = b[s];
        if (!pc) continue;
        const Color c = color_of(pc);
        x.sq[c][x.n[c]++] = uint8_t(s);
        if (type_of(pc) == KING && x.king[c] < 0) x.king[c] = s;
    }
}

// king_exposed(b, c) with c's king on square k.
inline bool king_exposed_f(const Piece* b, Color c, int k, const ChaseCtx& x) {
    if (k < 0) return true;
    const Color e = ~c;
    for (int i = 0; i < x.n[e]; ++i) {
        const int s = x.sq[e][i];
        const Piece pc = b[s];
        if (!pc || color_of(pc) == c || s == k) continue;
        if (type_of(pc) == KING ? intervening(b, s, k) == 0 : geometry_pc(b, pc, s, k)) return true;
    }
    return false;
}

inline bool safe_capture_f(Piece* b, int from, int to, const ChaseCtx& x) {
    const Piece m = b[from], v = b[to];
    if (from == to || !m || !v || type_of(v) == KING || color_of(m) == color_of(v) || !geometry_pc(b, m, from, to)) return false;
    b[to] = m;
    b[from] = NO_PIECE;
    const Color c = color_of(m);
    const bool exposed = king_exposed_f(b, c, type_of(m) == KING ? to : x.king[c], x);
    b[from] = m;
    b[to] = v;
    return !exposed;
}

inline void pins_f(const Piece* b, Color actor, const ChaseCtx& x, Bitboard& blocked, Bitboard& pinning) {
    const int king = x.king[actor];
    if (king < 0) return;
    const Color e = ~actor;
    Bitboard omitted = 0;
    int cand[NUM_SQ], plen[NUM_SQ], nc = 0;
    int path[NUM_SQ][9];
    for (int i = 0; i < x.n[e]; ++i) {
        const int s = x.sq[e][i];
        const Piece pc = b[s];
        const int dx = file_of(king) - file_of(s), dy = rank_of(king) - rank_of(s);
        const int ax = std::abs(dx), ay = std::abs(dy);
        const PieceType t = type_of(pc);
        int len = 0;
        if ((t == ROOK || t == CANNON || t == KING) && (dx == 0 || dy == 0)) {
            const int step = sgn(dx) + 9 * sgn(dy);
            for (int q = s + step; q != king; q += step) path[nc][len++] = q;
        } else if (t == HORSE && ((ax == 2 && ay == 1) || (ax == 1 && ay == 2))) {
            path[nc][len++] = s + (ax == 2 ? sgn(dx) : 9 * sgn(dy));
        } else if (t == ELEPHANT && ax == 2 && ay == 2) {
            path[nc][len++] = s + dx / 2 + 9 * (dy / 2);
        } else continue;
        cand[nc] = s;
        plen[nc++] = len;
        if (t != CANNON) omitted |= sq_bb(s);
    }
    for (int k = 0; k < nc; ++k) {
        const int s = cand[k];
        const bool cannon = type_of(b[s]) == CANNON;
        Bitboard screens = 0;
        int ns = 0;
        bool friendly = false;
        for (int j = 0; j < plen[k]; ++j) {
            const int q = path[k][j];
            if (b[q] && (cannon || !(omitted & sq_bb(q)))) {
                screens |= sq_bb(q);
                ++ns;
                friendly = friendly || color_of(b[q]) == actor;
            }
        }
        if (ns == (cannon ? 2 : 1)) {
            blocked |= screens;
            if (friendly) pinning |= sq_bb(s);
        }
    }
}


// Enemy pieces (squares) that `actor` threatens in the sense of the chase rule: chase_targets() as a bitboard.
Bitboard chase_targets_bb(Piece* b, Color actor) {
#ifdef SKY_ABL_SLOWCT  // ablation build only: original 90x90-scan classifier
    { const Mask r = chase_targets(b, actor); Bitboard o = 0; for (int s = 0; s < NUM_SQ; ++s) if (r[s]) o |= sq_bb(s); return o; }
#endif
    ChaseCtx x;
    ctx_build(b, x);
    Bitboard blocked = 0, pinning = 0, out = 0;
    pins_f(b, actor, x, blocked, pinning);
    const Color e = ~actor;
    for (int ti = 0; ti < x.n[e]; ++ti) {
        const int target = x.sq[e][ti];
        const Piece victim = b[target];
        if (type_of(victim) == KING) continue;
        const PieceType vt = type_of(victim);
        const bool victimOwnHalfSoldier = vt == SOLDIER && own_half(color_of(victim), target);
        const bool targetPinning = (pinning & sq_bb(target)) != 0;
        for (int ai = 0; ai < x.n[actor]; ++ai) {
            const int a = x.sq[actor][ai];
            const Piece pc = b[a];
            const PieceType at = type_of(pc);
            if (at == KING || at == SOLDIER) continue;
            if (blocked & sq_bb(a)) {
                if (!targetPinning) continue;
            } else if (victimOwnHalfSoldier) continue;
            if (!safe_capture_f(b, a, target, x)) continue;
            const bool always = ((at == HORSE || at == CANNON) && vt == ROOK) || ((at == ADVISOR || at == ELEPHANT) && major(vt));
            if (!always) {
                b[target] = pc;  // hypothetical capture; any legal recapture protects the target
                b[a] = NO_PIECE;
                bool defended = false;
                for (int di = 0; di < x.n[e] && !defended; ++di) {
                    const int d = x.sq[e][di];
                    if (b[d] && color_of(b[d]) == e) defended = safe_capture_f(b, d, target, x);
                }
                b[a] = pc;
                b[target] = victim;
                if (defended) continue;
                if (at == vt && safe_capture_f(b, target, a, x)) continue;  // mutual attack of equal pieces
            }
            out |= sq_bb(target);
            break;
        }
    }
#ifdef SKY_VERIFY
    {
        Piece c[NUM_SQ];
        std::memcpy(c, b, NUM_SQ);
        const Mask ref = chase_targets(c, actor);
        Bitboard r = 0;
        for (int s = 0; s < NUM_SQ; ++s) if (ref[s]) r |= sq_bb(s);
        ++V_ct;
        if (r != out || std::memcmp(c, b, NUM_SQ)) ++V_ct_bad;
    }
#endif
    return out;
}

Mask chase_targets_mask(Piece* b, Color actor) {
    Mask m;
    for_each_sq(chase_targets_bb(b, actor), [&](int s) { m.set(s); });
    return m;
}

// Exact memo of chase_targets_bb by (public board, actor); the full board is compared on lookup.
struct ChaseCacheEntry {
    alignas(32) Piece b[96];
    Bitboard targets;
    uint8_t actor;
    uint8_t valid;
};
constexpr std::size_t CHASE_CACHE = 512;
thread_local ChaseCacheEntry t_chaseCache[CHASE_CACHE];

inline uint64_t board_hash(const Piece* b) {
    uint64_t h = 0x9E3779B97F4A7C15ull, w;
    for (int i = 0; i < 96; i += 8) {
        std::memcpy(&w, b + i, 8);
        h = (h ^ w) * 0xff51afd7ed558ccdull;
        h ^= h >> 29;
    }
    return h;
}

Bitboard chase_targets_cached(Piece* b /*96 bytes, padded with 0*/, Color actor) {
    const uint64_t h = board_hash(b) ^ uint64_t(actor);
    ChaseCacheEntry& e = t_chaseCache[(h ^ (h >> 32)) & (CHASE_CACHE - 1)];
    if (e.valid && e.actor == actor && std::memcmp(e.b, b, 96) == 0) return e.targets;
    const Bitboard t = chase_targets_bb(b, actor);
    std::memcpy(e.b, b, 96);
    e.targets = t;
    e.actor = uint8_t(actor);
    e.valid = 1;
    return t;
}

#ifdef SKY_VERIFY
// ---- Reference (original) SkyRule helpers, verification builds only ------------------------------
Bitboard skyrule_chase_targets(const Position& pos, Color actor) {
    struct CacheEntry {
        uint64_t key = 0;
        Bitboard targets = 0;
        Color actor = RED;
        bool valid = false;
    };
    constexpr std::size_t CACHE_SIZE = 1024;
    static thread_local CacheEntry cache[CACHE_SIZE];
    CacheEntry& entry = cache[(pos.key ^ (pos.key >> 32)) & (CACHE_SIZE - 1)];
    if (entry.valid && entry.key == pos.key && entry.actor == actor) return entry.targets;

    Piece b[NUM_SQ];
    for (int s = 0; s < NUM_SQ; ++s)
        b[s] = pos.board[s] && type_of(pos.board[s]) == DARK
             ? make_piece(color_of(pos.board[s]), pos.role(s)) : pos.board[s];
    Bitboard out = 0;
    const Mask targets = chase_targets(b, actor);
    for (int s = 0; s < NUM_SQ; ++s)
        if (targets[s]) out |= sq_bb(s);
    entry.key = pos.key;
    entry.targets = out;
    entry.actor = actor;
    entry.valid = true;
    return out;
}

Bitboard skyrule_target_ids(const Position& pos, Bitboard targets) {
    Bitboard ids = 0;
    for_each_sq(targets, [&](int s) {
        const uint16_t id = pos.pieceIds[s];
        if (id) ids |= sq_bb(id - 1);
    });
    return ids;
}

Bitboard ref_skyrule_checkers(const Position& pos, Color actor) {
    Piece b[NUM_SQ];
    for (int s = 0; s < NUM_SQ; ++s)
        b[s] = pos.board[s] && type_of(pos.board[s]) == DARK
             ? make_piece(color_of(pos.board[s]), pos.role(s)) : pos.board[s];
    const int king = pos.kingSq[~actor];
    if (king < 0) return 0;
    Bitboard out = 0;
    for (int s = 0; s < NUM_SQ; ++s) {
        const Piece pc = b[s];
        if (!pc || color_of(pc) != actor) continue;
        const bool attacks = type_of(pc) == KING ? intervening(b, s, king) == 0
                                                 : geometry(b, s, king);
        if (attacks) out |= sq_bb(pos.pieceIds[s] ? pos.pieceIds[s] - 1 : s);
    }
    return out;
}
#endif

// Identities of `actor`'s pieces that attack the enemy king on the public board (dark pieces act as
// their start-square role). Emptiness is all geometry needs from other squares, so the raw board is used.
Bitboard skyrule_checkers(const Position& pos, Color actor) {
    const int king = pos.kingSq[~actor];
    if (king < 0) return 0;
    Bitboard out = 0;
    for_each_sq(pos.byColorBB[actor], [&](int s) {
        const PieceType t = pos.role(s);
        const bool attacks = t == KING ? intervening(pos.board, s, king) == 0
                                       : (s != king && geometry_pc(pos.board, make_piece(actor, t), s, king));
        if (attacks) out |= sq_bb(pos.pieceIds[s] ? pos.pieceIds[s] - 1 : s);
    });
#ifdef SKY_VERIFY
    ++V_chk;
    if (out != ref_skyrule_checkers(pos, actor)) ++V_chk_bad;
#endif
    return out;
}
}  // namespace

namespace {
// Chase classification of a pending record: does the quiet move from e.from to e.to newly threaten a target?
uint8_t sky_classify(const SkyEntry& e, const Piece* raw) {
    alignas(32) Piece A[96], B[96];
    for (int s = 0; s < NUM_SQ; ++s) {
        const Piece p = raw[s];
        A[s] = p && type_of(p) == DARK ? make_piece(color_of(p), PieceType(T.darkRole[color_of(p)][s])) : p;
    }
    for (int s = NUM_SQ; s < 96; ++s) A[s] = NO_PIECE;
    std::memcpy(B, A, 96);
    B[e.from] = A[e.to];
    B[e.to] = NO_PIECE;
    const Color actor = Color(e.actor);
    const Bitboard before = chase_targets_cached(B, actor);
    const Bitboard after = chase_targets_cached(A, actor);
    // Targets are enemy pieces, which a quiet move leaves in place, so new target identities exist iff new
    // target squares do (identities are unique and non-zero on occupied squares).
    return (after & ~before) ? SKYRULE_CHASE : SKYRULE_NONE;
}
}  // namespace

uint8_t Position::sky_action(std::size_t k) const {
    SkyEntry& e = skyHist[k];
    if (e.action != SKY_PENDING) return e.action;
    e.action = sky_classify(e, skyBoards[e.board].b);
    if (e.action == SKYRULE_NONE) { e.count = 0; e.countKnown = 1; }
#ifdef SKY_VERIFY
    ++V_act;
    if (k < skyActionHist.size() && skyActionHist[k] != e.action) ++V_act_bad;
#endif
    return e.action;
}

uint16_t Position::sky_count(std::size_t k) const {
    std::size_t j = k;
    while (!skyHist[j].countKnown && j >= 3 && sky_action(j - 2) == SKYRULE_CHASE) j -= 2;
    if (!skyHist[j].countKnown) { skyHist[j].count = 1; skyHist[j].countKnown = 1; }
    for (std::size_t q = j + 2; q <= k; q += 2) {
        skyHist[q].count = uint16_t(skyHist[q - 2].count + 1);
        skyHist[q].countKnown = 1;
    }
#ifdef SKY_VERIFY
    for (std::size_t q = j; q <= k; q += 2) {
        ++V_cnt;
        if (q < skyCountHist.size() && skyCountHist[q] != skyHist[q].count) ++V_cnt_bad;
    }
#endif
    return skyHist[k].count;
}

// Cycle from position `first` to the current one (no captures, reveals or checks inside it): a side
// that newly threatens one fixed physical piece with every one of its moves is a perpetual chaser.
// Returns 0 (draw: both or neither chase), 2 (side to move wins) or 3 (side to move loses).
int Position::chase_verdict(int first) const {
    const int p = int(keyHist.size()) - 1;
    const int len = p - first;
    if (len > 2 * MAX_PLY) return 0;
    Piece boards[2 * MAX_PLY + 1][NUM_SQ];
    for (int s = 0; s < NUM_SQ; ++s) {
        const Piece pc = board[s];
        boards[len][s] = pc && type_of(pc) == DARK ? make_piece(color_of(pc), role(s)) : pc;
    }
    for (int j = p - 1; j >= first; --j) {  // undo the cycle's pure relocations
        const Move m = moveHist[j];
        if (m == MOVE_NULL) return 0;
        Piece* b = boards[j - first];
        std::memcpy(b, boards[j - first + 1], NUM_SQ);
        b[from_sq(m)] = b[to_sq(m)];
        b[to_sq(m)] = NO_PIECE;
    }
    int id[NUM_SQ];
    for (int s = 0; s < NUM_SQ; ++s) id[s] = boards[0][s] ? s : -1;
    Mask common[2];
    common[RED].set();
    common[BLACK].set();
    const Color stmFirst = (len & 1) ? ~stm : stm;
    for (int j = 0; j < len; ++j) {
        const Color actor = (j & 1) ? ~stmFirst : stmFirst;
        const Move m = moveHist[first + j];
        Mask before, after;
        if (common[actor].any()) {
            const Mask bsq = chase_targets_mask(boards[j], actor);
            for (int s = 0; s < NUM_SQ; ++s)
                if (bsq[s]) before.set(id[s]);
        }
        id[to_sq(m)] = id[from_sq(m)];
        id[from_sq(m)] = -1;
        if (common[actor].none()) continue;
        const Mask asq = chase_targets_mask(boards[j + 1], actor);
        for (int s = 0; s < NUM_SQ; ++s)
            if (asq[s]) after.set(id[s]);
        common[actor] &= after & ~before;
        if (common[RED].none() && common[BLACK].none()) return 0;
    }
    const bool usChase = common[stm].any(), themChase = common[~stm].any();
    if (usChase == themChase) return 0;
    return usChase ? 3 : 2;
}

std::bitset<NUM_SQ> Position::chase_threats(Color actor) const {
    Piece b[NUM_SQ];
    for (int s = 0; s < NUM_SQ; ++s) b[s] = board[s] && type_of(board[s]) == DARK ? make_piece(color_of(board[s]), role(s)) : board[s];
    return chase_targets_mask(b, actor);
}

namespace {
inline bool sky_forbids_new(Position& P, Move m);
}

bool Position::skyrule_forbids_slow(Move m) {
#ifdef SKY_VERIFY
    const bool nv = sky_forbids_new(*this, m);
    const bool rv = ref_forbids(m);
    ++V_forb;
    if (nv) ++V_forb_true;
    if (nv != rv) ++V_forb_bad;
    return nv;
#else
    return sky_forbids_new(*this, m);
#endif
}

namespace {
inline bool sky_forbids_new(Position& P, Move m) {
    if (g_chaseRule != CHASE_RULE_SKYRULE || P.is_dark_move(m) || P.is_capture(m)) return false;
    const std::size_t n = P.skyHist.size();
    if (n < 3) return false;
    const std::size_t previous = n - 2;
    {
        const SkyEntry& e = P.skyHist[previous];
        if (e.action == SKYRULE_CHECK ? e.count < SKY_CHECK_LIM : (e.action == SKYRULE_NONE || e.run < SKY_CHASE_LIM)) return false;
    }
#ifdef SKY_VERIFY
    ++V_forb_slow;
#endif
    const uint8_t previousAction = P.sky_action(previous);
    const bool couldRepeatCheck = previousAction == SKYRULE_CHECK && P.skyHist[previous].count >= SKY_CHECK_LIM;
    const bool couldRepeatChase = previousAction == SKYRULE_CHASE && P.sky_count(previous) >= SKY_CHASE_LIM;
    if (!couldRepeatCheck && !couldRepeatChase) return false;

    StateInfo st;
    P.do_move(m, st);
    const std::size_t current = n;
    bool forbidden = false;
    const SkyEntry& c = P.skyHist[current];
    if (c.action == SKYRULE_CHECK) {
        const int checkers = std::min(3, std::max(1, popcount(c.unionCheck)));
        forbidden = c.count > SKY_CHECK_LIM * checkers;
    } else if (couldRepeatChase && P.sky_action(current) == SKYRULE_CHASE)
        forbidden = P.sky_count(current) > SKY_CHASE_LIM;  // (after a check chain a chase restarts at 1)
    P.undo_move(m, st);
    return forbidden;
}
}  // namespace

#ifdef SKY_VERIFY
bool Position::ref_forbids(Move m) {
    if (g_chaseRule != CHASE_RULE_SKYRULE || is_dark_move(m) || is_capture(m)) return false;
    if (skyActionHist.size() < 3) return false;
    const std::size_t previous = skyActionHist.size() - 2;
    const uint8_t previousAction = skyActionHist[previous];
    const uint16_t previousCount = skyCountHist[previous];
    const bool couldRepeatCheck = previousAction == SKYRULE_CHECK && previousCount >= SKY_CHECK_LIM;
    const bool couldRepeatChase = previousAction == SKYRULE_CHASE && previousCount >= SKY_CHASE_LIM;
    if (!couldRepeatCheck && !couldRepeatChase) return false;
    StateInfo st;
    do_move(m, st);
    const std::size_t current = skyActionHist.size() - 1;
    bool forbidden = false;
    if (skyActionHist[current] == SKYRULE_CHECK) {
        const Bitboard all = skyUnionCheckHist[current];
        const int checkers = std::min(3, std::max(1, popcount(all)));
        forbidden = skyCountHist[current] > SKY_CHECK_LIM * checkers;
    } else if (skyActionHist[current] == SKYRULE_CHASE)
        forbidden = skyCountHist[current] > SKY_CHASE_LIM;
    undo_move(m, st);
    return forbidden;
}

int Position::ref_sky_cycle_verdict(int i, int p) const {
    bool anyCheck = false;
    for (int j = i; j < p; ++j) anyCheck = anyCheck || checkHist[j + 1] != 0;
    bool chase[2] = {false, false};
    for (int j = i; j < p; ++j) {
        const std::size_t state = std::size_t(j + 1);
        if (state >= skyActionHist.size()) continue;
        const Color actor = ((p - j) & 1) == 0 ? stm : ~stm;
        if (skyActionHist[state] == SKYRULE_CHASE) chase[actor] = true;
    }
    if (anyCheck) return 1;
    if (chase[stm] != chase[~stm]) return chase[stm] ? 3 : 2;
    return 1;
}
#endif

// SkyRule keeps the same responsibility convention as the reference engine: a unilateral forcing
// suffix loses for the side to move at the repeated position, while mutual forcing is a draw. Chase
// records are resolved lazily, and only while they can still change the verdict.
int Position::sky_cycle_verdict(int i, int p) const {
    for (int j = i; j < p; ++j)
        if (checkHist[j + 1]) return 1;
    bool chase[2] = {false, false};
    for (int j = i; j < p; ++j) {
        const std::size_t state = std::size_t(j + 1);
        if (state >= skyHist.size()) continue;
        const Color actor = ((p - j) & 1) == 0 ? stm : ~stm;
        if (!chase[actor] && sky_action(state) == SKYRULE_CHASE) chase[actor] = true;
    }
    if (chase[stm] != chase[~stm]) return chase[stm] ? 3 : 2;
    return 1;
}

int Position::repetition_status() const {
    const int p = int(keyHist.size()) - 1;
    for (int i = p - 4; i >= lastIrrev; i -= 2) {
        if (keyHist[i] != key) continue;
        // Moves in the cycle are made from positions i..p-1; the mover at position j gave check
        // iff the side to move at j+1 is in check. The mover at j is stm when (p-j) is even.
        bool perpUs = true, perpThem = true, anyCheck = false;
        for (int j = i; j < p; ++j) {
            const bool gaveCheck = checkHist[j + 1] != 0;
            anyCheck = anyCheck || gaveCheck;
            if (((p - j) & 1) == 0) perpUs = perpUs && gaveCheck;
            else perpThem = perpThem && gaveCheck;
        }
        if (perpUs && !perpThem) return 3;
        if (perpThem && !perpUs) return 2;
        if (g_chaseRule == CHASE_RULE_NONE) return 1;
        if (g_chaseRule == CHASE_RULE_DEFAULT) {
            if (anyCheck) return 1;  // mutual perpetual or intermittent check: draw
            const int v = chase_verdict(i);
            return v ? v : 1;
        }
#ifdef SKY_VERIFY
        const int nv = sky_cycle_verdict(i, p), rv = ref_sky_cycle_verdict(i, p);
        ++V_rep; ++V_rep_sky;
        if (nv != rv) ++V_rep_bad;
        return nv;
#else
        return sky_cycle_verdict(i, p);
#endif
    }
    return 0;
}

bool Position::has_major(Color c) const {
    for (int i = 0; i < pcount[c]; ++i) {
        const PieceType t = type_of(board[plist[c][i]]);
        if (t == ROOK || t == HORSE || t == CANNON || t == DARK) return true;
    }
    return false;
}

bool Position::apply_token(const std::string& token, std::string* error) {
    if (token.size() < 4 || token.size() > 6)
        return fail(error, "move token must contain four coordinates and valid identities");
    const Move base = parse_move_coords(token);
    if (base == MOVE_NONE) return fail(error, "bad move token");
    const Piece mover = board[from_sq(base)];
    const Piece victim = board[to_sq(base)];
    const bool movedDark = mover && type_of(mover) == DARK;
    const bool capturedDark = victim && type_of(victim) == DARK;
    const std::size_t minimumSize = 4 + std::size_t(movedDark);
    const std::size_t maximumSize = minimumSize + std::size_t(capturedDark);
    if (token.size() < minimumSize || token.size() > maximumSize)
        return fail(error, "identity suffix does not match the hidden move/capture");

    auto parse_identity = [&](char ch, Color expected, PieceType& parsed, const char* role) {
        const PieceType t = type_from_char(ch);
        if (t < ROOK || t > SOLDIER) {
            if (error) *error = std::string("invalid ") + role + " identity suffix";
            return false;
        }
        const Color color = std::isupper(static_cast<unsigned char>(ch)) ? RED : BLACK;
        if (color != expected) {
            if (error) *error = std::string("wrong color for ") + role + " identity suffix";
            return false;
        }
        parsed = t;
        return true;
    };

    PieceType reveal = NO_TYPE, capturedId = NO_TYPE;
    std::size_t suffix = 4;
    if (movedDark && !parse_identity(token[suffix++], stm, reveal, "moved-dark")) return false;
    if (capturedDark && suffix < token.size()
        && !parse_identity(token[suffix++], ~stm, capturedId, "captured-dark")) return false;
    return apply_observed(base, reveal, capturedId, error);
}

bool Position::apply_observed(Move base, PieceType reveal, PieceType capturedId, std::string* error) {
    base = base_move(base);
    const int from = from_sq(base), to = to_sq(base);
    const Piece mover = board[from];
    if (!mover || color_of(mover) != stm) return fail(error, "no piece of side to move on origin");
    const Piece victim = board[to];
    Move legalMoves[MAX_MOVES];
    const int n = gen_legal(legalMoves);
    bool found = false;
    for (int i = 0; i < n; ++i) found = found || legalMoves[i] == base;
    if (!found) return fail(error, "illegal move in history");
    if (type_of(mover) == DARK) {
        if (reveal == NO_TYPE) return fail(error, "face-down mover without reveal");
        if (urn[stm][type_slot(reveal)] <= 0) return fail(error, "reveal not in pool");
    }
    if (capturedId != NO_TYPE
        && (!victim || type_of(victim) != DARK || urn[~stm][type_slot(capturedId)] <= 0))
        return fail(error, "captured identity is absent from the pool");
    StateInfo st;
    do_move(type_of(mover) == DARK ? make_move(from, to, reveal) : base, st);
    if (victim && type_of(victim) == DARK && capturedId != NO_TYPE) {
        // The observer saw the captured identity: remove it from that side's urn.
        const Color vc = color_of(victim);
        const int sl = type_slot(capturedId);
        if (urn[vc][sl] <= 0) return fail(error, "captured identity not in pool");
        key ^= T.zUrn[vc][sl][urn[vc][sl]];
        --urn[vc][sl];
        key ^= T.zUrn[vc][sl][urn[vc][sl]];
        --urnTotal[vc];
        update_mean(vc);
        keyHist.back() = key;
    }
    return true;
}

bool Position::valid(std::string* error) const {
    if (key != compute_key()) return fail(error, "key mismatch");
    int dark[2] = {0, 0}, cnt[2] = {0, 0};
    for (int s = 0; s < NUM_SQ; ++s) {
        const Piece p = board[s];
        if (!p) { if (pidx[s] != -1) return fail(error, "pidx on empty"); continue; }
        const Color c = color_of(p);
        ++cnt[c];
        if (type_of(p) == DARK) ++dark[c];
        if (pidx[s] < 0 || plist[c][pidx[s]] != s) return fail(error, "piece list mismatch");
        if (type_of(p) == KING && kingSq[c] != s) return fail(error, "king square mismatch");
    }
    for (int c = 0; c < 2; ++c) {
        if (cnt[c] != pcount[c] || dark[c] != darkCount[c]) return fail(error, "count mismatch");
        int tot = 0;
        for (int sl = 0; sl < NUM_SLOTS; ++sl) { if (urn[c][sl] < 0) return fail(error, "negative urn"); tot += urn[c][sl]; }
        if (tot != urnTotal[c] || tot < dark[c]) return fail(error, "urn total mismatch");
    }
    return true;
}

#ifdef JQ_BB_VERIFY
namespace {
[[noreturn]] void bb_fail(const Position& p, const char* what, int a, int b) {
    std::fprintf(stderr, "BB VERIFY MISMATCH %s (%d,%d) fen %s\n", what, a, b, p.fen().c_str());
    std::abort();
}
}  // namespace
bool Position::ray_attacked(Color c, int k, int d) const {
    const bool r = ray_attacked_impl(c, k, d);
    if (r != RefPos(*this).ray_attacked(c, k, d)) bb_fail(*this, "ray_attacked", k, d);
    return r;
}
bool Position::king_attacked(Color c, int k) const {
    const bool r = king_attacked_impl(c, k);
    if (r != RefPos(*this).king_attacked(c, k)) bb_fail(*this, "king_attacked", c, k);
    return r;
}
template <bool CAP>
int Position::gen(Move* list) const {
    if (!bitboards_ok()) bb_fail(*this, "bitboards", 0, 0);
    const int n = gen_impl<CAP>(list);
    Move ref[MAX_MOVES];
    const int rn = RefPos(*this).gen<CAP>(ref);
    if (n != rn) bb_fail(*this, "gen count", n, rn);
    for (int i = 0; i < n; ++i)
        if (ref[i] != list[i]) bb_fail(*this, "gen order", i, CAP);
    return n;
}
bool Position::legal(Move m) const {
    const bool r = legal_impl(m);
    if (r != RefPos(*this).legal(m)) bb_fail(*this, "legal", from_sq(m), to_sq(m));
    return r;
}
bool Position::reveal_gives_check(int from, int to, PieceType t) const {
    const bool r = reveal_gives_check_impl(from, to, t);
    if (r != RefPos(*this).reveal_gives_check(from, to, t)) bb_fail(*this, "reveal_gives_check", from, to);
    return r;
}
bool Position::check_after(int from, int to, PieceType t) const {
    const bool r = check_after_impl(from, to, t);
    if (r != RefPos(*this).check_after(from, to, t)) bb_fail(*this, "check_after", from, to);
    return r;
}
bool Position::gives_check(Move m) const {
    const bool r = gives_check_impl(m);
    if (r != RefPos(*this).gives_check(m)) bb_fail(*this, "gives_check", from_sq(m), to_sq(m));
    return r;
}
template int Position::gen<true>(Move*) const;
template int Position::gen<false>(Move*) const;
#endif

}  // namespace jq
