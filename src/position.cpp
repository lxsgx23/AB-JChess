/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "position.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

#include "bitboard.h"
#include "misc.h"
#include "movegen.h"
#include "tt.h"
#include "uci.h"

using std::string;

namespace Stockfish {

namespace Zobrist {

Key psq[PIECE_NB + 1][SQUARE_NB];
Key side, noPawns;
}

namespace {

constexpr std::string_view PieceToChar(" RACPNBK racpnbkXx");

constexpr std::string_view DarkPieces("RNBAKABNR"
                                      "........."
                                      ".C.....C."
                                      "P.P.P.P.P"
                                      "........."
                                      "........."
                                      "p.p.p.p.p"
                                      ".c.....c."
                                      "........."
                                      "rnbakabnr");

static constexpr Piece Pieces[] = {W_ROOK, W_ADVISOR, W_CANNON, W_PAWN, W_KNIGHT, W_BISHOP, W_KING,
                                   B_ROOK, B_ADVISOR, B_CANNON, B_PAWN, B_KNIGHT, B_BISHOP, B_KING};

constexpr bool is_identity_piece(Piece pc) {
    return pc > NO_PIECE && pc < PIECE_NB && type_of(pc) >= ROOK && type_of(pc) <= BISHOP;
}

constexpr int identity_inventory(Piece pc) { return type_of(pc) == PAWN ? 5 : 2; }

Piece parse_piece_character(char token, const char* fieldName) {
    const std::size_t idx = PieceToChar.find(token);
    if (idx == std::string_view::npos || idx == NO_PIECE || idx >= PIECE_NB)
        throw std::invalid_argument(std::string("invalid piece character in ") + fieldName);
    return Piece(idx);
}

int parse_nonnegative_integer(std::string_view text, const char* fieldName) {
    int value = 0;
    if (text.empty())
        throw std::invalid_argument(std::string("missing ") + fieldName + " in FEN");
    if (!std::all_of(text.begin(), text.end(),
                     [](char token) { return token >= '0' && token <= '9'; }))
        throw std::invalid_argument(std::string("invalid ") + fieldName + " in FEN");

    const char* first  = text.data();
    const char* last   = first + text.size();
    const auto  result = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || result.ptr != last || value < 0)
        throw std::invalid_argument(std::string("invalid ") + fieldName + " in FEN");
    return value;
}

void parse_rest_field(const std::string& field, std::array<int, PIECE_NB>& restPieces) {
    if (field.find('|') != std::string::npos)
        throw std::invalid_argument("AB-JChess FEN does not support legacy unknown-loss fields");

    if (field == "-")
        return;
    if (field.empty())
        throw std::invalid_argument("empty rest-piece field in FEN");

    std::array<bool, PIECE_NB> seen{};
    for (std::size_t i = 0; i < field.size();)
    {
        const Piece pc = parse_piece_character(field[i++], "rest-piece field");
        if (!is_identity_piece(pc))
            throw std::invalid_argument("invalid rest-piece field in FEN");
        if (seen[pc])
            throw std::invalid_argument("duplicate identity in rest-piece field");
        seen[pc] = true;

        const std::size_t digitsBegin = i;
        while (i < field.size() && field[i] >= '0' && field[i] <= '9')
            ++i;

        const int count =
          digitsBegin == i
            ? 1
            : parse_nonnegative_integer(
                std::string_view(field).substr(digitsBegin, i - digitsBegin), "rest-piece count");
        if (count > identity_inventory(pc))
            throw std::invalid_argument("invalid rest-piece count in FEN");
        restPieces[pc] = count;
    }
}

struct ParsedFen {
    std::array<Piece, SQUARE_NB> board{};
    std::array<bool, SQUARE_NB>  dark{};
    std::array<int, PIECE_NB>    restPieces{};
    Color                        sideToMove = WHITE;
    int                          rule40     = 0;
    int                          gamePly    = 0;
};

ParsedFen parse_fen(const std::string& fenStr) {
    std::istringstream         stream(fenStr);
    std::array<std::string, 5> fields;
    for (auto& field : fields)
        if (!(stream >> field))
            throw std::invalid_argument("AB-JChess FEN requires exactly five fields");

    std::string extra;
    if (stream >> extra)
        throw std::invalid_argument("AB-JChess FEN requires exactly five fields");

    ParsedFen                 parsed;
    std::array<int, COLOR_NB> colorCounts{};
    std::array<int, COLOR_NB> darkCounts{};
    std::array<int, COLOR_NB> kingCounts{};
    int                       rank = 0;
    int                       file = 0;

    for (char token : fields[0])
    {
        if (token == '/')
        {
            if (file != FILE_NB || rank >= RANK_9)
                throw std::invalid_argument("FEN board must contain ten ranks of nine squares");
            ++rank;
            file = 0;
            continue;
        }

        if (token >= '1' && token <= '9')
        {
            file += token - '0';
            if (file > FILE_NB)
                throw std::invalid_argument("FEN rank contains more than nine squares");
            continue;
        }

        if (file >= FILE_NB)
            throw std::invalid_argument("FEN rank contains more than nine squares");

        const Square square = make_square(File(file), Rank(int(RANK_9) - rank));
        Piece        pc;
        if (token == 'X' || token == 'x')
        {
            const char underlying = DarkPieces[square];
            if (underlying == '.' || (token == 'X') != (underlying >= 'A' && underlying <= 'Z'))
                throw std::invalid_argument("dark piece is not on a valid unrevealed square");
            pc = parse_piece_character(underlying, "piece placement");
            if (!is_identity_piece(pc))
                throw std::invalid_argument("king cannot be an unrevealed piece");
            parsed.dark[square] = true;
            ++darkCounts[color_of(pc)];
        }
        else
            pc = parse_piece_character(token, "piece placement");

        parsed.board[square] = pc;
        ++colorCounts[color_of(pc)];
        if (type_of(pc) == KING)
            ++kingCounts[color_of(pc)];
        ++file;
    }

    if (rank != RANK_9 || file != FILE_NB)
        throw std::invalid_argument("FEN board must contain ten ranks of nine squares");
    if (kingCounts[WHITE] != 1 || kingCounts[BLACK] != 1)
        throw std::invalid_argument("FEN board must contain exactly one king per side");
    if (colorCounts[WHITE] > 16 || colorCounts[BLACK] > 16)
        throw std::invalid_argument("FEN board contains too many pieces for one side");

    if (fields[1] == "w")
        parsed.sideToMove = WHITE;
    else if (fields[1] == "b")
        parsed.sideToMove = BLACK;
    else
        throw std::invalid_argument("FEN active color must be 'w' or 'b'");

    parse_rest_field(fields[2], parsed.restPieces);
    for (Color color : {WHITE, BLACK})
    {
        int identityCandidates = 0;
        for (PieceType pt = ROOK; pt < KING; ++pt)
            identityCandidates += parsed.restPieces[make_piece(color, pt)];

        if (identityCandidates < darkCounts[color])
            throw std::invalid_argument("not enough rest-piece identities for dark pieces in FEN");
    }

    parsed.rule40      = parse_nonnegative_integer(fields[3], "halfmove clock");
    const int fullmove = parse_nonnegative_integer(fields[4], "fullmove number");

    const std::int64_t basePly = fullmove == 0 ? 0 : 2LL * (fullmove - 1);
    const std::int64_t gamePly = basePly + (parsed.sideToMove == BLACK);
    if (gamePly > std::numeric_limits<int>::max())
        throw std::invalid_argument("fullmove number is too large");
    parsed.gamePly = int(gamePly);

    return parsed;
}

Bitboard possible_chase_targets_after_quiet(const Position& pos, Move m) {

    assert(m.is_ok());
    assert(!pos.move_dark(m));
    assert(!pos.capture(m));

    Color     us       = pos.side_to_move();
    Color     them     = ~us;
    Square    from     = m.from_sq();
    Square    to       = m.to_sq();
    PieceType pt       = type_of(pos.piece_on(from));
    Bitboard  occupied = (pos.pieces() ^ from) | to;

    if (pt == PAWN)
        return attacks_bb<PAWN>(to, us) & pos.pieces(them, KING);

    if (pt == KING)
        return (attacks_bb<KING>(to) | attacks_bb<ROOK>(to, occupied)) & pos.pieces(them, KING);

    return attacks_bb(pt, to, occupied) & pos.pieces(them);
}

// Return attacks on a king after a hypothetical capture.  Position::checkers_to()
// reads the live piece bitboards, so it cannot remove the captured victim from
// those masks.  SkyRule needs the post-capture view when deciding whether an
// attack is a legal capture (and therefore a real chase).
Bitboard skyrule_checkers_after_capture(const Position& pos, Color attackerColor,
                                        Square king, Bitboard occupiedAfter,
                                        Square capturedInPosition = SQ_NONE) {

    const Color enemy = ~attackerColor;
    const Bitboard ignored = is_ok(capturedInPosition) ? square_bb(capturedInPosition) : 0;
    // Hidden Jieqi pieces have an underlying type for identity bookkeeping,
    // but they do not attack until revealed. Keep them out of hypothetical
    // king-safety captures as well as ordinary chase snapshots.
    const Bitboard enemyPieces = pos.pieces(enemy) & occupiedAfter & ~ignored
                               & ~pos.pieces(DARK);

    return ((attacks_bb<PAWN_TO>(king, enemy) & enemyPieces & pos.pieces(PAWN))
            | (attacks_bb<KNIGHT_TO>(king, occupiedAfter) & enemyPieces
               & pos.pieces(KNIGHT))
            | (attacks_bb<ROOK>(king, occupiedAfter) & enemyPieces
               & pos.pieces(KING, ROOK))
            | (attacks_bb<CANNON>(king, occupiedAfter) & enemyPieces
               & pos.pieces(CANNON))
            | (attacks_bb<BISHOP>(king, occupiedAfter) & enemyPieces
               & pos.pieces(BISHOP))
            | (attacks_bb<ADVISOR>(king) & enemyPieces & pos.pieces(ADVISOR)));
}

// SkyRule is deliberately kept on top of the V8 attack tables.  In
// particular, advisors and elephants use the actual V8 tables here instead
// of imposing a second palace/river model (which would be wrong for Jieqi).
bool skyrule_can_capture_from(const Position& pos, Color color, PieceType pt, Square from,
                              Square to, Bitboard occupied,
                              Square capturedInPosition = SQ_NONE) {

    if (pos.is_dark(from) || pos.is_dark(to) || !is_ok(from) || !is_ok(to))
        return false;

    Bitboard attacks = pt == PAWN ? attacks_bb<PAWN>(from, color)
                                  : attacks_bb(pt, from, occupied);
    if (!(attacks & to))
        return false;

    const Square king = pt == KING ? to : pos.king_square(color);
    const Bitboard occupiedAfter = (occupied ^ from) | to;
    return !skyrule_checkers_after_capture(pos, color, king, occupiedAfter, capturedInPosition);
}

bool skyrule_has_true_root(const Position& pos, Color victim, Square target, Square attacker,
                           Bitboard occupied, Square capturedAttackerInPosition = SQ_NONE,
                           Bitboard excluded = 0,
                           Bitboard* trueRoots = nullptr) {

    const Bitboard occupiedAfterCapture = occupied ^ attacker;
    Bitboard roots = pos.attackers_to(target, occupiedAfterCapture) & pos.pieces(victim)
                   & ~square_bb(target) & ~excluded;
    bool found = false;

    while (roots)
    {
        const Square root = pop_lsb(roots);
        const Piece pc = pos.piece_on(root);
        if (pc == NO_PIECE
            || !skyrule_can_capture_from(pos, victim, type_of(pc), root, target,
                                         occupiedAfterCapture, capturedAttackerInPosition))
            continue;

        found = true;
        if (trueRoots)
            *trueRoots |= root;
        else
            return true;
    }
    return found;
}

struct SkyruleRelations {
    Bitboard eligibleAttackers = 0;
    Bitboard trueRooters = 0;
    Bitboard sameTypeNeutralizers = 0;
    Bitboard visibleTrueRooters = 0;
    Bitboard chasers = 0;
    Bitboard rooters = 0;
};

struct SkyruleSnapshot {
    Bitboard chased = 0;
    Bitboard rooted = 0;
    std::array<SkyruleRelations, SQUARE_NB> relation{};
};

void skyrule_add_chased(const Position& pos, Color attackerColor, Square attacker,
                        PieceType attackerType, Bitboard attacks, Bitboard occupied,
                        Square capturedAttackerInPosition,
                        Bitboard& chased, SkyruleRelations* relation,
                        Bitboard excludedRoots = 0,
                        Bitboard visibleRootCandidates = ~Bitboard(0)) {

    if (attackerType == KING || attackerType == PAWN || pos.is_dark(attacker))
        return;

    const Color victim = ~attackerColor;
    attacks &= pos.pieces(victim);

    while (attacks)
    {
        const Square target = pop_lsb(attacks);
        const Piece targetPiece = pos.piece_on(target);
        if (targetPiece == NO_PIECE || pos.is_dark(target))
            continue;

        const PieceType targetType = type_of(targetPiece);
        if (targetType == KING || (targetType == PAWN && !(square_bb(target) & HalfBB[attackerColor])))
            continue;

        if (relation)
            relation->eligibleAttackers |= square_bb(attacker);

        if (!skyrule_can_capture_from(pos, attackerColor, attackerType, attacker, target,
                                       occupied, target))
            continue;

        // A same-type victim which can legally recapture the attacker is not
        // being chased. A pinned or horse-leg-blocked recapture remains a
        // chase, as required by Duffish.
        if (targetType == attackerType
            && skyrule_can_capture_from(pos, victim, targetType, target, attacker, occupied,
                                        capturedAttackerInPosition))
        {
            if (relation)
                relation->sameTypeNeutralizers |= square_bb(attacker);
            continue;
        }

        if (targetType == ROOK && attackerType != ROOK && attackerType != PAWN)
        {
            chased |= square_bb(target);
            if (relation)
                relation->chasers |= square_bb(attacker);
            continue;
        }

        Bitboard roots = 0;
        if (!skyrule_has_true_root(pos, victim, target, attacker, occupied,
                                   capturedAttackerInPosition, excludedRoots, &roots))
        {
            chased |= square_bb(target);
            if (relation)
                relation->chasers |= square_bb(attacker);
        }
        else
        {
            if (relation)
            {
                relation->trueRooters |= square_bb(attacker);
                if (roots & visibleRootCandidates)
                    relation->visibleTrueRooters |= square_bb(attacker);
                relation->rooters |= square_bb(attacker);
            }
        }
    }
}

SkyruleSnapshot skyrule_snapshot(const Position& pos, Color attacker, Color victim) {

    SkyruleSnapshot snapshot;
    const Bitboard occupied = pos.pieces();
    Bitboard targets = pos.pieces(victim);

    while (targets)
    {
        const Square target = pop_lsb(targets);
        const Piece targetPiece = pos.piece_on(target);
        if (targetPiece == NO_PIECE || pos.is_dark(target))
            continue;

        const PieceType targetType = type_of(targetPiece);
        if (targetType == KING || (targetType == PAWN && !(square_bb(target) & HalfBB[attacker])))
            continue;

        const Bitboard visibleRootCandidates =
          pos.attackers_to(target, occupied) & pos.pieces(victim);
        Bitboard attackers = pos.attackers_to(target, occupied) & pos.pieces(attacker);
        attackers &= ~pos.pieces(attacker, KING, PAWN);
        while (attackers)
        {
            const Square from = pop_lsb(attackers);
            const Piece pc = pos.piece_on(from);
            if (pc == NO_PIECE)
                continue;
            skyrule_add_chased(pos, attacker, from, type_of(pc), square_bb(target), occupied,
                               from,
                               snapshot.chased, &snapshot.relation[target], Bitboard(0),
                               visibleRootCandidates);
        }
        // Duffish excludes advisors (FERS) from the rooted-target aggregate;
        // V8 elephants remain ordinary rooted targets even when revealed
        // outside orthodox river/palace regions.
        if (snapshot.relation[target].rooters && targetType != ADVISOR)
            snapshot.rooted |= square_bb(target);
    }
    return snapshot;
}

Bitboard undo_skyrule_move_board(Bitboard board, Move move) {

    if (!move.is_ok())
        return board;
    const Bitboard from = square_bb(move.from_sq());
    const Bitboard to = square_bb(move.to_sq());
    if (board & to)
        board = (board ^ to) | from;
    return board;
}

struct SkyruleDelta {
    Bitboard chased = 0;
    Bitboard rooted = 0;
    Bitboard pinnedRoot = 0;
    Bitboard withdrawnScreen = 0;
};

Bitboard skyrule_chased_by_targets(const Position& pos, Color chaser, Bitboard targets,
                                   const SkyruleSnapshot& snapshot);

SkyruleDelta skyrule_delta(const Position& pos, const Position& previous, Color attacker,
                           Color victim, Move move, PieceType movedType,
                           const SkyruleSnapshot& after, const SkyruleSnapshot& before,
                           bool includeWithdrawnScreen, const int* afterIds,
                           const int* beforeIds) {

    SkyruleDelta delta;
    const Square from = move.from_sq();
    const Square to = move.to_sq();
    Bitboard targets = after.chased | after.rooted;
    while (targets)
    {
        const Square target = pop_lsb(targets);
        const SkyruleRelations& a = after.relation[target];
        Square beforeTarget = target;
        if (afterIds && beforeIds && afterIds[target] > 0)
        {
            const int targetId = afterIds[target];
            for (Bitboard pieces = previous.pieces(victim); pieces;)
            {
                const Square candidate = pop_lsb(pieces);
                if (beforeIds[candidate] == targetId
                    && type_of(previous.piece_on(candidate)) == type_of(pos.piece_on(target)))
                {
                    beforeTarget = candidate;
                    break;
                }
            }
        }
        else if (pos.state()->previous && pos.state()->previous->move.is_ok()
                 && !pos.state()->previous->capturedPiece && !pos.state()->previous->movedDark
                 && pos.state()->previous->move.to_sq() == target)
            beforeTarget = pos.state()->previous->move.from_sq();

        const SkyruleRelations& b = before.relation[beforeTarget];
        const Bitboard afterChasers = undo_skyrule_move_board(a.chasers, move);
        const Bitboard afterRooters = undo_skyrule_move_board(a.rooters, move);
        const Bitboard newChasers = afterChasers & ~b.chasers;
        const Bitboard caused = newChasers
                              & (~b.eligibleAttackers
                                 | b.sameTypeNeutralizers
                                 | b.visibleTrueRooters);
#ifdef SKY_DEBUG
        if (target == SQ_E4)
            std::fprintf(stderr, "delta e4 aC=%llx bC=%llx elig=%llx roots=%llx vis=%llx new=%llx caused=%llx\\n",
                         (unsigned long long)a.chasers, (unsigned long long)b.chasers,
                         (unsigned long long)b.eligibleAttackers,
                         (unsigned long long)b.trueRooters,
                         (unsigned long long)b.visibleTrueRooters,
                         (unsigned long long)newChasers,
                         (unsigned long long)caused);
#endif

        Bitboard causedChasers = caused;
        if (movedType == CANNON
            && skyrule_chased_by_targets(pos, attacker, square_bb(target), after))
        {
            // A cannon which becomes the sole screen of a friendly cannon can
            // be a mutual pursuit. The screen itself did not create a new
            // unilateral chase, so remove it from the delta.
            Bitboard screenCannons = a.chasers & pos.pieces(attacker, CANNON) & ~square_bb(to);
            while (screenCannons)
            {
                const Square chaser = pop_lsb(screenCannons);
                const Bitboard screens = (between_bb(chaser, target) ^ target) & pos.pieces();
                if (screens == square_bb(to))
                    causedChasers &= ~undo_skyrule_move_board(square_bb(chaser), move);
            }
        }

        Bitboard withdrawnScreenChasers = 0;
        if (includeWithdrawnScreen && from != SQ_NONE && beforeTarget != SQ_NONE)
        {
            auto rootDependsOnWithdrawnScreen = [&](Square attackerSq) {
                const Bitboard occupiedAfterCapture = previous.pieces() ^ square_bb(attackerSq);
                Bitboard roots = previous.attackers_to(beforeTarget, occupiedAfterCapture)
                               & previous.pieces(victim) & ~square_bb(beforeTarget);
                bool foundWithdrawnScreenRoot = false;

                while (roots)
                {
                    const Square root = pop_lsb(roots);
                    const Piece rootPiece = previous.piece_on(root);
                    if (rootPiece == NO_PIECE
                        || !skyrule_can_capture_from(previous, victim, type_of(rootPiece), root,
                                                     beforeTarget, occupiedAfterCapture,
                                                     attackerSq))
                        continue;

                    const Bitboard screens = type_of(rootPiece) == CANNON
                                           ? (between_bb(root, beforeTarget) ^ beforeTarget)
                                               & occupiedAfterCapture
                                           : Bitboard(0);
                    if (screens == square_bb(from))
                        foundWithdrawnScreenRoot = true;
                    else
                        return false;
                }
                return foundWithdrawnScreenRoot;
            };

            Bitboard candidates = causedChasers & b.trueRooters;
            while (candidates)
            {
                const Square candidate = pop_lsb(candidates);
                if (rootDependsOnWithdrawnScreen(candidate))
                    withdrawnScreenChasers |= square_bb(candidate);
            }
        }

        if (causedChasers && (after.chased & target))
        {
            delta.chased |= square_bb(target);
            if (causedChasers & b.trueRooters)
                delta.pinnedRoot |= square_bb(target);
            if (includeWithdrawnScreen && !(causedChasers & ~withdrawnScreenChasers))
                delta.withdrawnScreen |= square_bb(target);
        }
        if ((afterRooters & ~b.rooters) && (after.rooted & target))
            delta.rooted |= square_bb(target);
    }
    return delta;
}

Bitboard skyrule_direct_chased(const Position& pos, bool includeCheckingMove = false) {

    const StateInfo* state = pos.state();
    if (state->move == Move::none() || state->capturedPiece
        || (state->checkersBB && !includeCheckingMove))
        return 0;

    const Color attackerColor = ~pos.side_to_move();
    const Color victim = pos.side_to_move();
    const Square from = state->move.from_sq();
    const Square to = state->move.to_sq();
    const Piece moved = pos.piece_on(to);
    if (moved == NO_PIECE || color_of(moved) != attackerColor || pos.is_dark(to))
        return 0;

    const PieceType movedType = type_of(moved);
    if (movedType == KING || movedType == PAWN)
        return 0;

    Bitboard directAttacks = attacks_bb(movedType, to, pos.pieces()) & pos.pieces(victim);
    if (movedType == ROOK || movedType == CANNON)
        directAttacks &= ~line_bb(from, to);

    Bitboard chased = 0;
    skyrule_add_chased(pos, attackerColor, to, movedType, directAttacks, pos.pieces(), to,
                       chased, nullptr);
    return chased;
}

Bitboard skyrule_pinned_same_type_chased(const Position& pos) {

    const StateInfo* state = pos.state();
    if (!state->previous || state->movedDark || state->capturedPiece)
        return 0;

    const Color attackerColor = ~pos.side_to_move();
    const Color victim = pos.side_to_move();
    Bitboard newPins = state->blockersForKing[victim]
                     & ~state->previous->blockersForKing[victim]
                     & pos.pieces(victim);
    Bitboard chased = 0;

    while (newPins)
    {
        const Square pinned = pop_lsb(newPins);
        const Piece pinnedPiece = pos.piece_on(pinned);
        if (pinnedPiece == NO_PIECE || color_of(pinnedPiece) != victim)
            continue;

        const PieceType pinnedType = type_of(pinnedPiece);
        Bitboard sameTypeAttackers = pos.attackers_to(pinned, pos.pieces())
                                   & pos.pieces(attackerColor, pinnedType);
        while (sameTypeAttackers)
        {
            const Square attacker = pop_lsb(sameTypeAttackers);
            skyrule_add_chased(pos, attackerColor, attacker, pinnedType, square_bb(pinned),
                               pos.pieces(), attacker, chased, nullptr);
        }
    }

    return chased;
}

Bitboard skyrule_chased_by_targets(const Position& pos, Color chaser, Bitboard targets,
                                   const SkyruleSnapshot& snapshot) {

    const Color counterChaser = ~chaser;
    const Bitboard occupied = pos.pieces();
    Bitboard chased = 0;
    targets &= pos.pieces(counterChaser);

    while (targets)
    {
        const Square attacker = pop_lsb(targets);
        const Piece piece = pos.piece_on(attacker);
        if (piece == NO_PIECE || color_of(piece) != counterChaser)
            continue;

        const PieceType type = type_of(piece);
        if (type == KING || type == PAWN)
            continue;
        const Bitboard attacks = attacks_bb(type, attacker, occupied) & pos.pieces(chaser);
        const Bitboard originalChasers = snapshot.relation[attacker].chasers;
        skyrule_add_chased(pos, counterChaser, attacker, type, attacks, occupied, attacker,
                           chased, nullptr, originalChasers);
    }

    return chased;
}

uint32_t skyrule_chased_ids(Bitboard chased, const int* pieceIds) {

    uint32_t ids = 0;
    while (chased)
    {
        const Square target = pop_lsb(chased);
        const int id = pieceIds[target];
        if (id > 0 && id <= 32)
            ids |= uint32_t(1) << (id - 1);
    }
    return ids;
}

uint32_t skyrule_chaser_ids(Bitboard targets, const int* pieceIds,
                            const SkyruleSnapshot& snapshot) {

    uint32_t ids = 0;
    while (targets)
    {
        const Square target = pop_lsb(targets);
        Bitboard chasers = snapshot.relation[target].chasers;
        while (chasers)
        {
            const int id = pieceIds[pop_lsb(chasers)];
            if (id > 0 && id <= 32)
                ids |= uint32_t(1) << (id - 1);
        }
    }
    return ids;
}

void skyrule_undo_id_board(Move move, int* pieceIds) {

    if (!move.is_ok())
        return;
    const Square from = move.from_sq();
    const Square to = move.to_sq();
    pieceIds[from] = pieceIds[to];
    // The temporary checker map is zero-based and uses -1 for an empty
    // square.  Writing zero here would alias the first piece identity after
    // the next undo pair and could falsely keep a checker persistent.
    pieceIds[to] = -1;
}

int skyrule_checking_piece_count(const Position& root, int maxCheckStates,
                                  bool* hasPersistentChecker = nullptr,
                                  int* persistentCheckStates = nullptr) {

    Position rollback;
    rollback.copy_for_chasing_from(root);

    int pieceIds[SQUARE_NB];
    std::fill_n(pieceIds, SQUARE_NB, -1);

    int nextId[COLOR_NB] = {0, 0};
    for (int sq = 0; sq < SQUARE_NB; ++sq)
    {
        const Square s = Square(sq);
        const Piece  pc = rollback.piece_on(s);
        if (pc != NO_PIECE)
            pieceIds[s] = nextId[color_of(pc)]++;
    }

    uint32_t checkerIds = 0;
    uint32_t persistentCheckerIds = 0;
    int      persistentStates = 0;
    bool     suffixActive = true;
    int      states = 0;

    while (states < maxCheckStates && rollback.state()->skyruleAction == SKYRULE_CHECK)
    {
        if (rollback.state()->capturedPiece != NO_PIECE
            || rollback.state()->captureDark || rollback.state()->movedDark)
            break;
        Bitboard checkers = rollback.state()->checkersBB;
        uint32_t stateCheckerIds = 0;
        while (checkers)
        {
            const Square checker = pop_lsb(checkers);
            const int    id = pieceIds[checker];
            if (id >= 0 && id < 32)
            {
                checkerIds |= uint32_t(1) << id;
                stateCheckerIds |= uint32_t(1) << id;
            }
        }

        if (suffixActive && states == 0)
        {
            persistentCheckerIds = stateCheckerIds;
            persistentStates = persistentCheckerIds ? 1 : 0;
        }
        else if (suffixActive)
        {
            const uint32_t continuingIds = persistentCheckerIds & stateCheckerIds;
            if (!continuingIds)
                suffixActive = false;
            else
            {
                persistentCheckerIds = continuingIds;
                persistentStates = states + 1;
            }
        }

        ++states;

        const Move checkingMove = rollback.state()->move;
        if (!checkingMove.is_ok())
            break;
        skyrule_undo_id_board(checkingMove, pieceIds);
        rollback.undo_move(checkingMove);

        const Move reply = rollback.state()->move;
        if (!reply.is_ok())
            break;
        skyrule_undo_id_board(reply, pieceIds);
        rollback.undo_move(reply);
    }

    int count = 0;
    while (checkerIds)
    {
        checkerIds &= checkerIds - 1;
        ++count;
    }

    if (hasPersistentChecker)
        *hasPersistentChecker = persistentStates > 0;
    if (persistentCheckStates)
        *persistentCheckStates = persistentStates;

    return std::max(count, 1);
}

bool skyrule_has_overlong_persistent_check(const Position& pos) {

    if (pos.state()->skyruleAction != SKYRULE_CHECK || pos.state()->skyruleCount <= 6)
        return false;

    bool hasPersistentChecker = false;
    int  persistentCheckStates = 0;
    skyrule_checking_piece_count(pos, pos.state()->skyruleCount,
                                 &hasPersistentChecker, &persistentCheckStates);
    return hasPersistentChecker && persistentCheckStates > 6;
}
}  // namespace

// Returns an ASCII representation of the position
std::ostream& operator<<(std::ostream& os, const Position& pos) {

    os << "\n +---+---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_9; r >= RANK_0; --r)
    {
        for (File f = FILE_A; f <= FILE_I; ++f)
        {
            Square sq = make_square(f, r);
            Piece  pc = pos.piece_on(sq);
            os << " | " << (pos.is_dark(sq) ? "Xx"[color_of(pc)] : PieceToChar[pc]);
        }

        os << " | " << int(r) << "\n +---+---+---+---+---+---+---+---+---+\n";
    }

    os << "   a   b   c   d   e   f   g   h   i\n"
       << "\nFen: " << pos.fen() << "\nKey: " << std::hex << std::uppercase << std::setfill('0')
       << std::setw(16) << pos.key() << std::setfill(' ') << std::dec << "\nCheckers: ";

    for (Bitboard b = pos.checkers(); b;)
        os << UCIEngine::square(pop_lsb(b)) << " ";

    return os;
}


// Initializes at startup the various arrays used to compute hash keys
void Position::init() {

    PRNG rng(1070372);

    for (Piece pc : Pieces)
        for (Square s = SQ_A0; s <= SQ_I9; ++s)
            Zobrist::psq[pc][s] = rng.rand<Key>();

    for (Square s = SQ_A0; s <= SQ_I9; ++s)
        Zobrist::psq[DARK_PIECE][s] = rng.rand<Key>();

    Zobrist::side    = rng.rand<Key>();
    Zobrist::noPawns = rng.rand<Key>();
}


// Initializes the position object with the given FEN string. The complete FEN
// is parsed into bounded temporary storage before the active position is touched.
Position& Position::set(const string& fenStr, StateInfo* si) {
    /*
   A FEN string defines a particular position using only the ASCII character set.

   An AB-JChess FEN string contains five fields separated by spaces:

   1) Piece placement (from white's perspective). Each rank is described, starting
      with rank 9 and ending with rank 0. Within each rank, the contents of each
      square are described from file A through file I. Following the Standard
      Algebraic Notation (SAN), each piece is identified by a single letter taken
      from the standard English names. White pieces are designated using upper-case
      letters ("RACPNBK") whilst Black uses lowercase ("racpnbk"). Blank squares are
      noted using digits 1 through 9 (the number of blank squares), and "/"
      separates ranks.

   2) Active color. "w" means white moves next, "b" means black.

   3) Remaining hidden-piece identities for both sides, or "-" when empty.

   4) Halfmove clock. This is the number of halfmoves since the last pawn advance
      or capture. This is used to determine if a draw can be claimed under the
      fifty-move rule.

   5) Fullmove number. The number of the full move. It starts at 1, and is
      incremented after Black's move.
*/

    if (!si)
        throw std::invalid_argument("Position::set requires a StateInfo object");

    const ParsedFen parsed = parse_fen(fenStr);

    std::memset(this, 0, sizeof(Position));
    std::memset(si, 0, sizeof(StateInfo));
    st = si;

    sideToMove = parsed.sideToMove;
    gamePly    = parsed.gamePly;
    st->rule40 = parsed.rule40;
    std::copy(parsed.restPieces.begin(), parsed.restPieces.end(), restPieces);

    for (Square square = SQ_A0; square <= SQ_I9; ++square)
        if (const Piece pc = parsed.board[square]; pc != NO_PIECE)
        {
            put_piece(pc, square);
            if (parsed.dark[square])
                byTypeBB[DARK] |= square;
            if (type_of(pc) == KING)
                kingSquare[color_of(pc)] = square;
        }

    set_state();
    skyrule_assign_piece_ids();

    assert(pos_is_ok());

    return *this;
}


void Position::swap(Position& other) noexcept {
    using std::swap;
    swap(board, other.board);
    swap(byTypeBB, other.byTypeBB);
    swap(byColorBB, other.byColorBB);
    swap(kingSquare, other.kingSquare);
    swap(pieceCount, other.pieceCount);
    swap(st, other.st);
    swap(gamePly, other.gamePly);
    swap(sideToMove, other.sideToMove);
    swap(restPieces, other.restPieces);
    swap(filter, other.filter);
    swap(idBoard, other.idBoard);
}


// Sets king attacks to detect if a move gives check
void Position::set_check_info() const {

    update_blockers<WHITE>();
    update_blockers<BLACK>();

    Square ksq = king_square(~sideToMove);

    // We have to take special cares about the cannon and checks
    st->needSlowCheck =
      checkers() || (attacks_bb<ROOK>(king_square(sideToMove)) & pieces(~sideToMove, CANNON));

    st->checkSquares[PAWN]    = attacks_bb<PAWN_TO>(ksq, sideToMove);
    st->checkSquares[KNIGHT]  = attacks_bb<KNIGHT_TO>(ksq, pieces());
    st->checkSquares[CANNON]  = attacks_bb<CANNON>(ksq, pieces());
    st->checkSquares[ROOK]    = attacks_bb<ROOK>(ksq, pieces());
    st->checkSquares[ADVISOR] = attacks_bb<ADVISOR>(ksq);
    st->checkSquares[BISHOP]  = attacks_bb<BISHOP>(ksq, pieces());
    st->checkSquares[KING]    = 0;

    Bitboard hollowCannons = st->checkSquares[ROOK] & pieces(sideToMove, CANNON);
    if (hollowCannons)
    {
        Bitboard hollowCannonDiscover = Bitboard(0);
        while (hollowCannons)
            hollowCannonDiscover |= between_bb(ksq, pop_lsb(hollowCannons));
        for (PieceType pt = ROOK; pt < KING; ++pt)
            st->checkSquares[pt] |= hollowCannonDiscover;
    }
}


// Computes the hash keys of the position, and other
// data that once computed is updated incrementally as moves are made.
// The function is only used when a new position is set up
void Position::set_state() const {

    st->key               = 0;
    st->minorPieceKey     = 0;
    st->nonPawnKey[WHITE] = st->nonPawnKey[BLACK] = 0;
    st->pawnKey                                   = Zobrist::noPawns;
    st->majorMaterial[WHITE] = st->majorMaterial[BLACK] = VALUE_ZERO;
    st->checkersBB = checkers_to(~sideToMove, king_square(sideToMove));
    st->move       = Move::none();
    st->skyruleRawChased = 0;
    st->skyruleChased = 0;
    st->skyruleAction = SKYRULE_NONE;
    st->skyruleCount = 0;

    set_check_info();

    for (Bitboard b = pieces(); b;)
    {
        Square    s  = pop_lsb(b);
        Piece     pc = piece_on(s);
        PieceType pt = type_of(pc);

        st->key ^= Zobrist::psq[is_dark(s) ? DARK_PIECE : pc][s];

        if (pt == PAWN)
            st->pawnKey ^= Zobrist::psq[pc][s];

        else
        {
            st->nonPawnKey[color_of(pc)] ^= Zobrist::psq[pc][s];

            if (pt != KING && (pt & 1))
            {
                st->majorMaterial[color_of(pc)] += PieceValue[pc];
                if (pt != ROOK)
                    st->minorPieceKey ^= Zobrist::psq[pc][s];
            }
        }
    }

    for (const auto& color : {WHITE, BLACK})
        for (const auto& [piece, count] : rest_pieces(color))
            for (int i = 0; i < count; ++i)
                st->key ^= Zobrist::psq[piece][i];

    if (sideToMove == BLACK)
        st->key ^= Zobrist::side;
}


// Returns a FEN representation of the position.
string Position::fen() const {

    int                emptyCnt;
    std::ostringstream ss;

    for (Rank r = RANK_9; r >= RANK_0; --r)
    {
        for (File f = FILE_A; f <= FILE_I; ++f)
        {
            for (emptyCnt = 0; f <= FILE_I && empty(make_square(f, r)); ++f)
                ++emptyCnt;

            if (emptyCnt)
                ss << emptyCnt;

            if (f <= FILE_I)
            {
                Piece pc = piece_on(make_square(f, r));
                ss << (is_dark(make_square(f, r)) ? "Xx"[color_of(pc)] : PieceToChar[pc]);
            }
        }

        if (r > RANK_0)
            ss << '/';
    }

    ss << (sideToMove == WHITE ? " w " : " b ");

    bool wroteRest = false;
    for (Color c : {WHITE, BLACK})
        for (PieceType pt = ROOK; pt < KING; ++pt)
        {
            Piece pc = make_piece(c, pt);
            if (restPieces[pc] > 0)
            {
                ss << PieceToChar[pc] << restPieces[pc];
                wroteRest = true;
            }
        }
    if (!wroteRest)
        ss << "-";

    ss << " " << st->rule40 << " " << 1 + (gamePly - (sideToMove == BLACK)) / 2;

    return ss.str();
}


// Calculates st->blockersForKing[c] and st->pinners[~c],
// which store respectively the pieces preventing king of color c from being in check
// and the slider pieces of color ~c pinning pieces of color c to the king.
template<Color c>
void Position::update_blockers() const {

    Square ksq             = king_square(c);
    st->blockersForKing[c] = 0;
    st->pinners[~c]        = 0;

    // Snipers are pieces that attack 's' when a piece and other pieces are removed
    Bitboard snipers =
      ((attacks_bb<ROOK>(ksq) & (pieces(ROOK) | pieces(CANNON) | pieces(KING)))
       | (attacks_bb<KNIGHT>(ksq) & pieces(KNIGHT)) | (attacks_bb<BISHOP>(ksq) & pieces(BISHOP)))
      & pieces(~c);
    Bitboard occupancy = pieces() ^ (snipers & ~pieces(CANNON));

    while (snipers)
    {
        Square   sniperSq = pop_lsb(snipers);
        bool     isCannon = type_of(piece_on(sniperSq)) == CANNON;
        Bitboard b = between_bb(ksq, sniperSq) & (isCannon ? pieces() ^ sniperSq : occupancy);

        if (b && ((!isCannon && !more_than_one(b)) || (isCannon && popcount(b) == 2)))
        {
            st->blockersForKing[c] |= b;
            if (b & pieces(c))
                st->pinners[~c] |= sniperSq;
        }
    }
}


// Computes a bitboard of all pieces which attack a given square.
// Slider attacks use the occupied bitboard to indicate occupancy.
Bitboard Position::attackers_to(Square s, Bitboard occupied) const {

    return (attacks_bb<PAWN_TO>(s, WHITE) & pieces(WHITE, PAWN))
         | (attacks_bb<PAWN_TO>(s, BLACK) & pieces(BLACK, PAWN))
         | (attacks_bb<KNIGHT_TO>(s, occupied) & pieces(KNIGHT))
         | (attacks_bb<ROOK>(s, occupied) & pieces(ROOK))
         | (attacks_bb<CANNON>(s, occupied) & pieces(CANNON))
         | (attacks_bb<BISHOP>(s, occupied) & pieces(BISHOP))
         | (attacks_bb<ADVISOR>(s) & pieces(ADVISOR)) | (attacks_bb<KING>(s) & pieces(KING));
}


// Computes a bitboard of all pieces of a given color
// which gives check to a given square. Slider attacks use the occupied bitboard
// to indicate occupancy.
Bitboard Position::checkers_to(Color c, Square s, Bitboard occupied) const {

    return ((attacks_bb<PAWN_TO>(s, c) & pieces(PAWN))
            | (attacks_bb<KNIGHT_TO>(s, occupied) & pieces(KNIGHT))
            | (attacks_bb<ROOK>(s, occupied) & pieces(KING, ROOK))
            | (attacks_bb<CANNON>(s, occupied) & pieces(CANNON))
            | (attacks_bb<BISHOP>(s, occupied) & pieces(BISHOP))
            | (attacks_bb<ADVISOR>(s, occupied) & pieces(ADVISOR)))
         & pieces(c);
}


// Tests whether a pseudo-legal move is legal
bool Position::legal(Move m) const {

    assert(m.is_ok());

    Color     us       = sideToMove;
    Square    from     = m.from_sq();
    Square    to       = m.to_sq();
    Piece     pc       = piece_on(from);
    PieceType pt       = type_of(pc);
    Bitboard  occupied = (pieces() ^ from) | to;
    Square    ksq      = king_square(us);

    assert(color_of(pc) == us);
    assert(piece_on(ksq) == make_piece(us, KING));

    // If the moving piece is a dark advisor, make sure the destination square
    // is within the palace
    if (pt == ADVISOR && is_dark(from) && !(Palace & to))
        return false;

    // If the moving piece is a king, check whether the destination square is
    // attacked by the opponent.
    if (pt == KING)
        return !(checkers_to(~us, to, occupied));

    // If we don't need slow check. A non-king move is always legal when either:
    // 1. Not moving a pinned piece.
    // 2. Moving a pinned non-cannon piece and aligned with king.
    // 3. Moving a pinned cannon and aligned with king but it's not a capture move.
    if (!st->needSlowCheck
        && (!(blockers_for_king(us) & from)
            || ((pt != CANNON || empty(to)) && aligned(from, to, ksq))))
        return true;

    // A non-king move is legal if the king is not under attack after the move.
    return !(checkers_to(~us, ksq, occupied) & ~square_bb(to));
}


// Takes a random move and tests whether the move is
// pseudo-legal. It is used to validate moves from TT that can be corrupted
// due to SMP concurrent access or hash position key aliasing.
bool Position::pseudo_legal(const Move m) const {

    Color  us   = sideToMove;
    Square from = m.from_sq();
    Square to   = m.to_sq();
    Piece  pc   = moved_piece(m);

    // If the 'from' square is not occupied by a piece belonging to the side to
    // move, the move is obviously not legal.
    if (pc == NO_PIECE || color_of(pc) != us)
        return false;

    // The destination square cannot be occupied by a friendly piece
    if (pieces(us) & to)
        return false;

    // Handle the special cases
    if (type_of(pc) == PAWN)
        return bool(attacks_bb<PAWN>(from, us) & to);
    else if (type_of(pc) == CANNON && !capture(m))
        return bool(attacks_bb<ROOK>(from, pieces()) & to);
    else
        return bool(attacks_bb(type_of(pc), from, pieces()) & to);
}


// Tests whether a pseudo-legal move gives a check
bool Position::gives_check(Move m) const {

    assert(m.is_ok());
    Color us = sideToMove;
    Piece pc = moved_piece(m);
    assert(color_of(pc) == us);

    Square from = m.from_sq();
    Square to   = m.to_sq();
    Square ksq  = king_square(~us);

    PieceType pt       = type_of(pc);
    Bitboard  occupied = (pieces() ^ from) | to;

    // Is there a direct check?
    if (pt == CANNON && aligned(from, to, ksq))
    {
        if (attacks_bb<CANNON>(to, occupied) & ksq)
            return true;
    }
    else if (check_squares(pt) & to)
        return true;

    // Is there a discovered check?
    if (attacks_bb<ROOK>(ksq) & pieces(us, CANNON))
        return bool(checkers_to(us, ksq, occupied) & ~square_bb(from));
    else if ((blockers_for_king(~us) & from) && !aligned(from, to, ksq))
        return true;

    return false;
}


Piece Position::do_flip(Square s, Piece pc, DirtyPiece* dp, const TranspositionTable* tt) {

    assert(is_ok(s) && is_dark(s));

    Color us     = ~sideToMove;
    Color them   = ~us;
    Piece fromPc = piece_on(s);

    if (dp)
    {
        dp->add_pc = pc;
        dp->add_sq = s;
    }

    assert(color_of(pc) == us);

    // Flip the piece
    remove_piece(s);
    put_piece(pc, s);
    byTypeBB[DARK] ^= s;
    restPieces[pc]--;

    // Recalculate checkersBB
    st->checkersBB = checkers_to(us, king_square(them));

    // Update hash key
    st->key ^= Zobrist::psq[pc][s] ^ Zobrist::psq[pc][restPieces[pc]];
    // If the moving piece is a pawn, update pawn hash key.
    if (type_of(pc) == PAWN)
        st->pawnKey ^= Zobrist::psq[pc][s];
    else
    {
        st->nonPawnKey[us] ^= Zobrist::psq[pc][s];

        if (type_of(pc) == KNIGHT || type_of(pc) == CANNON)
            st->minorPieceKey ^= Zobrist::psq[pc][s];

        if (type_of(pc) != KING && (type_of(pc) & 1))
            st->majorMaterial[us] += PieceValue[pc];
    }

    if (tt)
        prefetch(tt->first_entry(key()));

    // Update king attacks used for fast check detection
    set_check_info();

    assert(pos_is_ok());

    return fromPc;
}


void Position::undo_flip(Square s, Piece fromPc) {

    assert(is_ok(s) && !is_dark(s));

    Color us = ~sideToMove;
    Piece pc = piece_on(s);

    restPieces[pc]++;
    remove_piece(s);
    put_piece(fromPc, s);
    byTypeBB[DARK] ^= s;
    st->rule40 = st->capturedPiece ? 0 : st->previous->rule40 + 1;

    // Update hash key
    st->key ^= Zobrist::psq[pc][s] ^ Zobrist::psq[pc][restPieces[pc]];
    // If the moving piece is a pawn, update pawn hash key.
    if (type_of(pc) == PAWN)
        st->pawnKey ^= Zobrist::psq[pc][s];
    else
    {
        st->nonPawnKey[us] ^= Zobrist::psq[pc][s];

        if (type_of(pc) == KNIGHT || type_of(pc) == CANNON)
            st->minorPieceKey ^= Zobrist::psq[pc][s];

        if (type_of(pc) != KING && (type_of(pc) & 1))
            st->majorMaterial[us] -= PieceValue[pc];
    }

    // The hidden state produced by a dark move never exposes a checker. Its
    // auxiliary check information is not consumed before the next flip or undo.
    st->checkersBB = 0;

    assert(pos_is_ok());
}


void Position::remove_rest_piece(Piece pc) {

    if (!is_identity_piece(pc) || restPieces[pc] <= 0)
        throw std::invalid_argument("identity is not available in the rest-piece pool");

    --restPieces[pc];
    st->key ^= Zobrist::psq[pc][restPieces[pc]];
}


void Position::restore_rest_piece(Piece pc) {

    if (!is_identity_piece(pc) || restPieces[pc] >= identity_inventory(pc))
        throw std::logic_error("identity cannot be restored to the rest-piece pool");

    st->key ^= Zobrist::psq[pc][restPieces[pc]];
    ++restPieces[pc];
}


// Makes a move, and saves all information necessary
// to a StateInfo object. The move is assumed to be legal. Pseudo-legal
// moves should be filtered out before this function is called.
// If a pointer to the TT table is passed, the entry for the new position
// will be prefetched
DirtyPiece
Position::do_move(Move m, StateInfo& newSt, bool givesCheck, const TranspositionTable* tt) {

    assert(m.is_ok());
    assert(&newSt != st);
    // Update the bloom filter
    ++filter[st->key];

    Key k = st->key ^ Zobrist::side;

    // Copy some fields of the old state to our new StateInfo object except the
    // ones which are going to be recalculated from scratch anyway and then switch
    // our state pointer to point to the new (ready to be updated) state.
    std::memcpy(&newSt, st, offsetof(StateInfo, key));
    newSt.previous = st;
    st             = &newSt;
    st->move       = m;

    Color  us       = sideToMove;
    Color  them     = ~us;
    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  pc       = piece_on(from);
    Piece  captured = piece_on(to);
    bool   moveDark = move_dark(m);

    st->movedId    = idBoard[from];
    st->capturedId = captured ? idBoard[to] : 0;
    st->movedDark  = moveDark;

    DirtyPiece dp;
    dp.pc     = moveDark ? DARK_PIECE : pc;
    dp.from   = from;
    dp.to     = moveDark ? SQ_NONE : to;
    dp.add_sq = SQ_NONE;

    assert(color_of(pc) == us);
    assert(captured == NO_PIECE || color_of(captured) == them);
    assert(type_of(captured) != KING);

    // Increment ply counters.
    // In particular, rule40 will be reset to zero later on in case of a capture.
    ++gamePly;
    ++st->rule40;
    ++st->pliesFromNull;

    if (captured)
    {
        Square capsq = to;

        st->captureDark = is_dark(to);
        // If the captured piece is a pawn, update pawn hash key, otherwise
        // update major material.
        if (type_of(captured) == PAWN)
            st->pawnKey ^= Zobrist::psq[captured][capsq];

        else
        {
            st->nonPawnKey[them] ^= Zobrist::psq[captured][capsq];

            if (type_of(captured) & 1)
            {
                st->majorMaterial[them] -= PieceValue[captured];
                if (type_of(captured) != ROOK)
                    st->minorPieceKey ^= Zobrist::psq[captured][capsq];
            }
        }

        dp.remove_pc = st->captureDark ? DARK_PIECE : captured;
        dp.remove_sq = capsq;

        if (st->captureDark)
            byTypeBB[DARK] ^= to;

        remove_piece(capsq);

        // Update hash key
        k ^= Zobrist::psq[st->captureDark ? DARK_PIECE : captured][capsq];
        // Reset rule 40 counter
        st->rule40 = 0;
    }
    else
    {
        st->captureDark = false;
        dp.remove_sq    = SQ_NONE;
    }

    // Update hash key
    k ^= moveDark ? Zobrist::psq[DARK_PIECE][from] : Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    // If the moving piece is a pawn, update pawn hash key.
    if (type_of(pc) == PAWN)
        st->pawnKey ^= Zobrist::psq[pc][from] ^ (moveDark ? 0 : Zobrist::psq[pc][to]);
    else
    {
        st->nonPawnKey[us] ^= Zobrist::psq[pc][from] ^ (moveDark ? 0 : Zobrist::psq[pc][to]);

        if (type_of(pc) == KNIGHT || type_of(pc) == CANNON)
            st->minorPieceKey ^= Zobrist::psq[pc][from] ^ (moveDark ? 0 : Zobrist::psq[pc][to]);

        if (moveDark && type_of(pc) != KING && (type_of(pc) & 1))
            st->majorMaterial[us] -= PieceValue[pc];
    }

    // Move the piece.
    move_piece(from, to);
    idBoard[to]   = idBoard[from];
    idBoard[from] = 0;

    // Update the key with the final value
    st->key = k;
    if (!moveDark && tt)
        prefetch(tt->first_entry(key()));

    // Set capture piece
    st->capturedPiece = captured;

    // Calculate checkers bitboard (if move gives check)
    st->checkersBB = !moveDark && givesCheck ? checkers_to(us, king_square(them)) : Bitboard(0);
    assert(moveDark || givesCheck == bool(st->checkersBB));

    sideToMove = ~sideToMove;

    // Update king attacks used for fast check detection
    if (!moveDark)
        set_check_info();
    else
        byTypeBB[DARK] ^= from | to;

    // SkyRule state is computed only after the side-to-move switch, so the
    // snapshot sees the moved piece as the attacker and the opponent as the
    // victim. Captures and dark moves are explicitly reset by the helper.
    set_skyrule_info(st);

    assert(pos_is_ok());

    assert(dp.pc != NO_PIECE);
    assert(!bool(captured) ^ (dp.remove_sq != SQ_NONE));
    assert(dp.from != SQ_NONE && (moveDark || dp.to != SQ_NONE));
    assert(dp.add_sq == SQ_NONE);
    return dp;
}


// Unmakes a move. When it returns, the position should
// be restored to exactly the same state as before the move was made.
void Position::undo_move(Move m) {

    assert(m.is_ok());

    sideToMove = ~sideToMove;

    Square from = m.from_sq();
    Square to   = m.to_sq();

    assert(empty(from));
    assert(type_of(st->capturedPiece) != KING);

    // Restore the visibility of the moved piece based on the move metadata.
    // The destination may have been a hidden capture square; after do_move()
    // that square is no longer marked DARK, so inspecting is_dark(to) here
    // cannot distinguish a dark mover from a visible mover.
    if (st->movedDark)
        byTypeBB[DARK] ^= from | to;

    move_piece(to, from);  // Put the piece back at the source square

    if (st->capturedPiece)
    {
        Square capsq = to;

        put_piece(st->capturedPiece, capsq);  // Restore the captured piece
        idBoard[capsq] = st->capturedId;

        if (st->captureDark)
            byTypeBB[DARK] ^= capsq;
    }
    else
        idBoard[to] = 0;

    idBoard[from] = st->movedId;

    // Finally point our state pointer back to the previous state
    st = st->previous;
    --gamePly;

    // Update the bloom filter
    --filter[st->key];

    assert(pos_is_ok());
}

// Used to do a "null move": it flips
// the side to move without executing any move on the board.
void Position::do_null_move(StateInfo& newSt, const TranspositionTable& tt) {

    assert(!checkers());
    assert(&newSt != st);

    // Update the bloom filter
    ++filter[st->key];

    std::memcpy(&newSt, st, sizeof(StateInfo));

    newSt.previous = st;
    st             = &newSt;

    st->move          = Move::null();
    st->capturedPiece = NO_PIECE;
    st->captureDark   = false;
    st->movedDark     = false;
    st->movedId       = 0;
    st->capturedId    = 0;
    st->skyruleRawChased = 0;
    st->skyruleChased = 0;
    st->skyruleAction = SKYRULE_NONE;
    st->skyruleCount = 0;

    st->key ^= Zobrist::side;
    prefetch(tt.first_entry(key()));

    st->pliesFromNull = 0;

    sideToMove = ~sideToMove;

    set_check_info();

    assert(pos_is_ok());
}


// Must be used to undo a "null move"
void Position::undo_null_move() {

    assert(!checkers());

    st         = st->previous;
    sideToMove = ~sideToMove;

    // Update the bloom filter
    --filter[st->key];
}


// Tests if the SEE (Static Exchange Evaluation)
// value of move is greater or equal to the given threshold. We'll use an
// algorithm similar to alpha-beta pruning with a null window.
bool Position::see_ge(Move m, int threshold) const {

    assert(m.is_ok());

    Square from = m.from_sq(), to = m.to_sq();

    int swap = PieceValue[piece_on(to)] - threshold;
    if (swap < 0)
        return false;

    swap = PieceValue[piece_on(from)] - swap;
    if (swap <= 0)
        return true;

    assert(color_of(piece_on(from)) == sideToMove);
    Bitboard occupied  = pieces() ^ from ^ to;  // xoring to is important for pinned piece logic
    Color    stm       = sideToMove;
    Bitboard attackers = attackers_to(to, occupied);

    // Flying general
    if (attackers & pieces(stm, KING))
        attackers |= attacks_bb<ROOK>(to, occupied & ~pieces(ROOK)) & pieces(~stm, KING);
    if (attackers & pieces(~stm, KING))
        attackers |= attacks_bb<ROOK>(to, occupied & ~pieces(ROOK)) & pieces(stm, KING);

    Bitboard nonCannons = attackers & ~pieces(CANNON);
    Bitboard cannons    = attackers & pieces(CANNON);
    Bitboard stmAttackers, bb;
    int      res = 1;

    while (true)
    {
        stm = ~stm;
        attackers &= occupied;

        // If stm has no more attackers then give up: stm loses
        if (!(stmAttackers = attackers & pieces(stm)))
            break;

        // Don't allow pinned pieces to attack as long as there are
        // pinners on their original square.
        if (pinners(~stm) & occupied)
        {
            stmAttackers &= ~blockers_for_king(stm);

            if (!stmAttackers)
                break;
        }

        res ^= 1;

        // Locate and remove the next least valuable attacker, and add to the
        // bitboard 'attackers' any protential attackers when it is removed.
        if ((bb = stmAttackers & pieces(PAWN)))
        {
            if ((swap = PawnValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            nonCannons |= attacks_bb<ROOK>(to, occupied) & pieces(ROOK);
            cannons   = attacks_bb<CANNON>(to, occupied) & pieces(CANNON);
            attackers = nonCannons | cannons;
        }

        else if ((bb = stmAttackers & pieces(BISHOP)))
        {
            if ((swap = BishopValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
        }

        else if ((bb = stmAttackers & pieces(ADVISOR)))
        {
            if ((swap = AdvisorValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            nonCannons |= attacks_bb<KNIGHT_TO>(to, occupied) & pieces(KNIGHT);
            attackers = nonCannons | cannons;
        }

        else if ((bb = stmAttackers & pieces(CANNON)))
        {
            if ((swap = CannonValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            cannons   = attacks_bb<CANNON>(to, occupied) & pieces(CANNON);
            attackers = nonCannons | cannons;
        }

        else if ((bb = stmAttackers & pieces(KNIGHT)))
        {
            if ((swap = KnightValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
        }

        else if ((bb = stmAttackers & pieces(ROOK)))
        {
            swap = RookValue - swap;
            occupied ^= least_significant_square_bb(bb);

            nonCannons |= attacks_bb<ROOK>(to, occupied) & pieces(ROOK);
            cannons   = attacks_bb<CANNON>(to, occupied) & pieces(CANNON);
            attackers = nonCannons | cannons;
        }

        else  // KING
              // If we "capture" with the king but the opponent still has attackers,
              // reverse the result.
            return (attackers & ~pieces(stm)) ? res ^ 1 : res;
    }

    return bool(res);
}


// A lighter version of do_move(), used in chasing detection
std::pair<Piece, int> Position::do_move(Move m) {

    assert(capture(m));

    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  captured = piece_on(to);
    int    id       = idBoard[to];

    // Update id board
    idBoard[to]   = idBoard[from];
    idBoard[from] = 0;

    // Update board and piece lists
    remove_piece(to);
    // Chasing uses a lightweight temporary board mutation and historically
    // leaves the DARK mask unchanged while testing recaptures.
    const Bitboard darkPieces = byTypeBB[DARK];
    move_piece(from, to);
    byTypeBB[DARK] = darkPieces;

    sideToMove = ~sideToMove;

    return {captured, id};
}


// A lighter version of undo_move(), used in chasing detection
void Position::undo_move(Move m, Piece captured, int id) {

    sideToMove = ~sideToMove;

    Square from = m.from_sq();
    Square to   = m.to_sq();

    // Put back id board
    idBoard[from] = idBoard[to];
    idBoard[to]   = id;

    // Match the lightweight do_move() semantics: move_piece() owns DARK-mask
    // updates for real moves, but chasing must preserve its temporary mask.
    const Bitboard darkPieces = byTypeBB[DARK];
    move_piece(to, from);  // Put the piece back at the source square
    byTypeBB[DARK] = darkPieces;

    if (captured)
        put_piece(captured, to);  // Restore the captured piece
}


// Tests whether a pseudo-legal move is chase legal
bool Position::chase_legal(Move m) const {

    assert(m.is_ok());

    Color    us       = sideToMove;
    Square   from     = m.from_sq();
    Square   to       = m.to_sq();
    Bitboard occupied = (pieces() ^ from) | to;

    assert(color_of(moved_piece(m)) == us);
    assert(piece_on(king_square(us)) == make_piece(us, KING));

    // If the moving piece is a king, check whether the destination
    // square is not under new attack after the move.
    if (type_of(piece_on(from)) == KING)
        return !(checkers_to(~us, to, occupied));

    // A non-king move is chase legal if the king is not under new attack after the move.
    return !(checkers_to(~us, king_square(us), occupied) & ~square_bb(to));
}

bool Position::skyrule_move_chases(Square attacker, Square target) const {

    assert(is_ok(attacker) && is_ok(target));

    Piece pc = piece_on(attacker);
    if (pc == NO_PIECE || is_dark(attacker) || piece_on(target) == NO_PIECE
        || color_of(pc) == color_of(piece_on(target)))
        return false;

    struct SideToMoveGuard {
        Position* pos;
        Color     oldSide;
        ~SideToMoveGuard() { pos->sideToMove = oldSide; }
    };

    Position*       self = const_cast<Position*>(this);
    SideToMoveGuard guard{self, sideToMove};
    self->sideToMove = color_of(pc);

    if (type_of(piece_on(target)) == KING)
        return bool(checkers_to(color_of(pc), target) & attacker);

    PieceType attackerType = type_of(pc);
    if (attackerType == PAWN || attackerType == KING)
        return false;

    Bitboard attacks = attacks_bb(attackerType, attacker, pieces()) & target;
    if (!attacks)
        return false;

    Move m(attacker, target);
    if (!chase_legal(m))
        return false;

    if ((attackerType == KNIGHT || attackerType == CANNON) && type_of(piece_on(target)) == ROOK)
        return true;

    if ((attackerType == ADVISOR || attackerType == BISHOP) && (type_of(piece_on(target)) & 1))
        return true;

    bool trueChase             = true;
    const auto& [captured, id] = const_cast<Position*>(this)->do_move(m);
    Bitboard recaptures        = attackers_to(target) & pieces(sideToMove);
    while (recaptures)
    {
        Square s = pop_lsb(recaptures);
        if (chase_legal(Move(s, target)))
        {
            trueChase = false;
            break;
        }
    }
    const_cast<Position*>(this)->undo_move(m, captured, id);

    if (!trueChase)
        return false;

    if (attackerType == type_of(piece_on(target)))
    {
        self->sideToMove = ~color_of(pc);
        const bool asymmetric =
          (attackerType == KNIGHT && ((between_bb(attacker, target) ^ target) & pieces()))
          || !chase_legal(Move(target, attacker));
        return asymmetric;
    }

    return true;
}


void Position::skyrule_assign_piece_ids() {

    std::memset(idBoard, 0, sizeof(idBoard));

    int whiteId = 1;
    int blackId = 1;
    for (Square s = SQ_A0; s <= SQ_I9; ++s)
        if (piece_on(s) != NO_PIECE)
            idBoard[s] = color_of(piece_on(s)) == WHITE ? whiteId++ : blackId++;
}


void Position::copy_for_chasing_from(const Position& src) {
    // SkyRule rollback needs an independent Position and StateInfo chain.
    // A byte-for-byte copy of Position is unsafe because it copies src.st and
    // every StateInfo::previous pointer verbatim; subsequent undo/do_move calls
    // can then alias caller or stack storage and trigger the do_move assertion.
    if (this == &src)
        return;

    std::memcpy(board, src.board, sizeof(board));
    std::memcpy(byTypeBB, src.byTypeBB, sizeof(byTypeBB));
    std::memcpy(byColorBB, src.byColorBB, sizeof(byColorBB));
    std::memcpy(kingSquare, src.kingSquare, sizeof(kingSquare));
    std::memcpy(pieceCount, src.pieceCount, sizeof(pieceCount));
    std::memcpy(restPieces, src.restPieces, sizeof(restPieces));
    std::memcpy(&filter, &src.filter, sizeof(filter));
    std::memcpy(idBoard, src.idBoard, sizeof(idBoard));
    gamePly = src.gamePly;
    sideToMove = src.sideToMove;

    std::array<const StateInfo*, 128> chain{};
    size_t count = 0;
    for (const StateInfo* p = src.st; p && count < chain.size(); p = p->previous)
        chain[count++] = p;

    chasingStateCount = int(count);
    if (!count)
    {
        st = nullptr;
        return;
    }

    // chain[0] is the current state; copy oldest-to-newest and relink locally.
    for (size_t i = count; i-- > 0;)
    {
        chasingStateStorage[i] = *chain[i];
        chasingStateStorage[i].previous = (i + 1 < count) ? &chasingStateStorage[i + 1] : nullptr;
    }
    st = &chasingStateStorage[0];
}


void Position::do_quiet_move_for_chasing(Move m, StateInfo& newSt) {

    assert(m.is_ok());
    assert(!move_dark(m));
    assert(!capture(m));

    Square from = m.from_sq();
    Square to   = m.to_sq();

    newSt.previous      = st;
    newSt.move          = m;
    newSt.capturedPiece = NO_PIECE;
    newSt.captureDark   = false;
    newSt.movedDark     = false;
    newSt.movedId       = idBoard[from];
    newSt.capturedId    = 0;

    st = &newSt;

    idBoard[to]   = idBoard[from];
    idBoard[from] = 0;
    move_piece(from, to);
    sideToMove = ~sideToMove;
}

Bitboard Position::skyrule_chased(Bitboard* rooted, Bitboard* pinnedRoot,
                                   Bitboard* withdrawnScreen) const {

    if (rooted)
        *rooted = 0;
    if (pinnedRoot)
        *pinnedRoot = 0;
    if (withdrawnScreen)
        *withdrawnScreen = 0;

    if (st->move == Move::none() || st->capturedPiece || st->movedDark || checkers())
        return 0;

    const Square to = st->move.to_sq();
    const Piece moved = piece_on(to);
    if (moved == NO_PIECE || is_dark(to) || color_of(moved) == sideToMove)
        return 0;

    Position previous;
    previous.copy_for_chasing_from(*this);
    previous.undo_move(st->move);

    const Color attacker = ~sideToMove;
    const Color victim = sideToMove;
    const SkyruleSnapshot after = skyrule_snapshot(*this, attacker, victim);
    const SkyruleSnapshot before = skyrule_snapshot(previous, attacker, victim);
    const SkyruleDelta delta = skyrule_delta(*this, previous, attacker, victim, st->move,
                                              type_of(moved), after, before,
                                              withdrawnScreen != nullptr, idBoard,
                                              previous.idBoard);

    if (rooted)
        *rooted = delta.rooted;
    if (pinnedRoot)
        *pinnedRoot = delta.pinnedRoot;
    if (withdrawnScreen)
        *withdrawnScreen = delta.withdrawnScreen;
    return delta.chased;
}

void Position::set_skyrule_info(StateInfo* si) const {

    si->skyruleAction = SKYRULE_NONE;
    si->skyruleRawChased = 0;
    si->skyruleChased = 0;
    si->skyruleCount = 0;

    // Hidden movement, flipping and captures are identity-changing events and
    // therefore terminate a SkyRule suffix.
    if (si->move == Move::none() || si->capturedPiece || si->movedDark)
        return;

    if (si->checkersBB)
        si->skyruleAction = SKYRULE_CHECK;
    else
    {
        si->skyruleRawChased = skyrule_chased();
        si->skyruleChased = si->skyruleRawChased;
        if (si->skyruleChased)
            si->skyruleAction = SKYRULE_CHASE;
    }

    if (si->skyruleAction == SKYRULE_NONE)
        return;

    si->skyruleCount = 1;
    StateInfo* previousSameSide = si->previous && si->previous->previous
                                    ? si->previous->previous
                                    : nullptr;
    // Captures, hidden moves, null moves and flips terminate the reversible
    // identity suffix. Do not relay a same-side action across the intervening
    // boundary even when the board geometry happens to look unchanged.
    if (!previousSameSide || !si->previous || !si->previous->move.is_ok()
        || si->previous->capturedPiece != NO_PIECE || si->previous->captureDark
        || si->previous->movedDark || previousSameSide->skyruleAction != si->skyruleAction)
        return;

    if (si->skyruleAction == SKYRULE_CHECK)
    {
        si->skyruleCount = previousSameSide->skyruleCount + 1;
        return;
    }

    Bitboard continuing = si->skyruleChased & previousSameSide->skyruleChased;
    const Move reply = si->previous->move;
    if (reply.is_ok())
    {
        const Square from = reply.from_sq();
        const Square to = reply.to_sq();
        // A target can continue through the intervening reply only when the
        // piece at the destination is the same stable identity that occupied
        // the old target square. A square-only test incorrectly relays a
        // chase when another piece arrives there after a capture/reveal.
        const bool sameIdentity = si->previous->capturedPiece == NO_PIECE
                               && !si->previous->movedDark
                               && si->previous->movedId > 0
                               && idBoard[to] == si->previous->movedId;
        if (sameIdentity && (previousSameSide->skyruleChased & from)
            && (si->skyruleChased & to))
            continuing |= square_bb(to);
    }
    if (continuing)
    {
        si->skyruleChased = continuing;
        si->skyruleCount = previousSameSide->skyruleCount + 1;
    }
    else
    {
        // A newly-created chase starts a fresh suffix even when it targets a
        // different piece from the previous same-side action.
        si->skyruleCount = 1;
    }
}


int Position::skyrule_continuous_chase_count_in_place(Square attacker, Square target, int limit) {

    int count = 0;

    assert(limit > 0);

    const Piece attackerPiece = piece_on(attacker);
    const Piece targetPiece   = piece_on(target);
    const Color attackerColor = color_of(attackerPiece);
    const Color targetColor   = color_of(targetPiece);
    const int   attackerId    = idBoard[attacker];
    const int   targetId      = idBoard[target];

    auto undo_quiet_move = [&]() {
        StateInfo* previous = st->previous;
        if (!previous || !st->move.is_ok() || st->movedDark || st->capturedPiece || st->captureDark)
            return false;

        Move   lastMove   = st->move;
        Square from       = lastMove.from_sq();
        Square to         = lastMove.to_sq();
        int    movedId    = st->movedId;
        Color  movedColor = color_of(piece_on(to));

        if (is_dark(to))
            return false;

        sideToMove = ~sideToMove;
        move_piece(to, from);
        idBoard[from] = movedId;
        idBoard[to]   = 0;
        st            = previous;
        --gamePly;

        if (movedColor == attackerColor && movedId == attackerId)
            attacker = from;

        if (movedColor == targetColor && movedId == targetId)
            target = from;

        return true;
    };

    while (is_ok(attacker) && is_ok(target))
    {
        if (!st->move.is_ok() || st->movedDark || st->capturedPiece || st->captureDark)
            break;

        Move   chaseMove   = st->move;
        Square chaseTo     = chaseMove.to_sq();
        Piece  chasePiece  = piece_on(chaseTo);
        Color  chaseColor  = color_of(chasePiece);
        int    chaseMoveId = st->movedId;

        if (is_dark(chaseTo) || chaseColor != attackerColor || chaseMoveId != attackerId
            || !skyrule_move_chases(attacker, target))
            break;

        ++count;
        if (count >= limit)
            return count;

        if (!undo_quiet_move() || !st->move.is_ok())
            break;

        if (st->movedDark || st->capturedPiece || st->captureDark)
            break;

        Move   responseMove  = st->move;
        Square responseTo    = responseMove.to_sq();
        Piece  responsePiece = piece_on(responseTo);
        Color  responseColor = color_of(responsePiece);
        bool   targetMoved   = responseColor == targetColor && st->movedId == targetId;

        if (is_dark(responseTo) || (!targetMoved && !skyrule_move_chases(attacker, target))
            || !undo_quiet_move())
            break;
    }

    return count;
}


bool Position::skyrule_move_forbidden(Move m) const {

    assert(m.is_ok());
    if (move_dark(m) || capture(m))
        return false;

    StateInfo* previousSameSide = st->previous;
    // The current state's move is the intervening reply. A dark move, capture,
    // or null/flip boundary must prevent the next action from inheriting the
    // prior same-side limit.
    if (!previousSameSide || !st->move.is_ok() || st->capturedPiece != NO_PIECE
        || st->captureDark || st->movedDark)
        return false;

    // The next move is only a candidate for the hard limit when the previous
    // same-side action has already reached that action's threshold. This keeps
    // ordinary quiet replies and action-class changes out of the expensive
    // temporary-position path.
    const int limit = previousSameSide->skyruleAction == SKYRULE_CHECK ? 6 : 18;
    if (previousSameSide->skyruleCount < limit)
        return false;

    const bool givesCheck = this->gives_check(m);
    if (previousSameSide->skyruleAction == SKYRULE_CHECK && !givesCheck)
        return false;  // a quiet reply breaks a check suffix

    if (previousSameSide->skyruleAction != SKYRULE_CHECK
        && previousSameSide->skyruleAction != SKYRULE_CHASE)
        return false;

    Position* position = const_cast<Position*>(this);
    StateInfo newSt;
    position->do_move(m, newSt, givesCheck, nullptr);

    auto checker_suffix = [](Position& current, int maxStates) {
        Position rollback;
        rollback.copy_for_chasing_from(current);
        uint32_t checkerIds = 0;
        int states = 0;

        while (states < maxStates && rollback.st->skyruleAction == SKYRULE_CHECK)
        {
            Bitboard checkers = rollback.st->checkersBB;
            uint32_t currentIds = 0;
            while (checkers)
            {
                const Square checker = pop_lsb(checkers);
                const int id = rollback.idBoard[checker];
                if (id > 0 && id <= 32)
                    currentIds |= uint32_t(1) << (id - 1);
            }
            checkerIds |= currentIds;
            ++states;

            const Move checkingMove = rollback.st->move;
            if (!checkingMove.is_ok())
                break;
            rollback.undo_move(checkingMove);
            if (!rollback.st || !rollback.st->move.is_ok())
                break;
            const Move reply = rollback.st->move;
            rollback.undo_move(reply);
        }

        int count = 0;
        while (checkerIds)
        {
            checkerIds &= checkerIds - 1;
            ++count;
        }
        return std::max(count, 1);
    };

    bool forbidden = false;
    if (previousSameSide->skyruleAction == SKYRULE_CHECK)
    {
        if (newSt.skyruleAction == SKYRULE_CHECK)
        {
            const int checkingPieces = std::min(checker_suffix(*position, newSt.skyruleCount), 3);
            forbidden = newSt.skyruleCount > 6 * checkingPieces;
        }
    }
    else if (newSt.skyruleAction == SKYRULE_CHASE)
        forbidden = newSt.skyruleCount > 18;

    position->undo_move(m);
    return forbidden;
}

bool Position::forbidden_by_skyrule_jieqi(Move m) const {

    // Compatibility entry point retained for the V8 search and root move
    // filters. The implementation now uses Duffish's action/count state.
    return skyrule_move_forbidden(m);
}


// Calculates the chase information for a given color.
uint16_t Position::chased(Color c) {

    uint16_t chase = 0;

    std::swap(c, sideToMove);

    // King and pawn can legally perpetual chase
    Bitboard attackers = pieces(sideToMove) ^ pieces(sideToMove, KING, PAWN);
    while (attackers)
    {
        Square    from         = pop_lsb(attackers);
        PieceType attackerType = type_of(piece_on(from));
        Bitboard  attacks      = attacks_bb(attackerType, from, pieces());

        // Restrict to pinners if pinned, otherwise exclude attacks on unpromoted pawns and checks
        if (blockers_for_king(sideToMove) & from)
            attacks &= pinners(~sideToMove) & ~pieces(KING);
        else
            attacks &= (pieces(~sideToMove) ^ pieces(~sideToMove, KING, PAWN))
                     | (pieces(~sideToMove, PAWN) & HalfBB[sideToMove]);

        while (attacks)
        {
            Square to = pop_lsb(attacks);
            Move   m  = Move(from, to);

            if (chase_legal(m))
            {
                // Attacks against stronger pieces
                if ((attackerType == KNIGHT || attackerType == CANNON)
                    && type_of(piece_on(to)) == ROOK)
                    chase |= (1 << idBoard[to]);
                if ((attackerType == ADVISOR || attackerType == BISHOP)
                    && type_of(piece_on(to)) & 1)
                    chase |= (1 << idBoard[to]);
                // Attacks against potentially unprotected pieces
                else
                {
                    bool trueChase             = true;
                    const auto& [captured, id] = do_move(m);
                    Bitboard recaptures        = attackers_to(to) & pieces(sideToMove);
                    while (recaptures)
                    {
                        Square s = pop_lsb(recaptures);
                        if (chase_legal(Move(s, to)))
                        {
                            trueChase = false;
                            break;
                        }
                    }
                    undo_move(m, captured, id);

                    if (trueChase)
                    {
                        // Exclude mutual/symmetric attacks except pins
                        if (attackerType == type_of(piece_on(to)))
                        {
                            sideToMove = ~sideToMove;
                            if ((attackerType == KNIGHT && ((between_bb(from, to) ^ to) & pieces()))
                                || !chase_legal(Move(to, from)))
                                chase |= (1 << idBoard[to]);
                            sideToMove = ~sideToMove;
                        }
                        else
                            chase |= (1 << idBoard[to]);
                    }
                }
            }
        }
    }

    std::swap(c, sideToMove);

    return chase;
}


// Detects checks and chases from state st - d to state st.  The detector keeps
// identities in per-colour bitsets while walking the cycle backwards.  This
// is intentionally separate from StateInfo's square bitboards: a target may
// move between the two repeated positions and a stationary attacker may keep
// the same relationship alive without creating a new chase.
Value Position::detect_chases(int d, int ply) {

    if (d < 4 || !st)
        return VALUE_NONE;

    constexpr int MAX_CYCLE_STATES = 256;
    struct ActionRecord {
        Move move = Move::none();
        SkyruleAction action = SKYRULE_NONE;
        PieceType type = NO_PIECE_TYPE;
        uint32_t pieceId = 0;
        uint32_t targets = 0;
        uint32_t chasers = 0;
        uint32_t counterChasers = 0;
        uint32_t newTargets = 0;
        uint32_t previouslyChasedTargets = 0;
        uint32_t pinnedRootTargets = 0;
        uint32_t withdrawnScreenTargets = 0;
        bool direct = false;
        bool escapedCheck = false;
        // Duffish keeps these relations per action.  They are deliberately
        // identity based so a checker/target may move between repeated
        // positions without losing the relation.
        bool checkMoveCanBeCaptured = false;
        uint32_t checkMoveChasedTargets = 0;
        bool chaseMoveUsesPinnedSameType = false;
        bool chaseMovePinsRoot = false;
        bool chaseMoveTargetsRook = false;
        bool quietRetreatEscapedCheck = false;
    };

    struct ForcingSide {
        bool allForcing = true;
        bool hasCheck = false;
        bool hasChase = false;
        bool hasCheckChase = false;
        bool hasQuiet = false;
        bool checkRetreatCycle = false;
        bool chaseRetreatCycle = false;
        bool checkChaseRetreatCycle = false;
        bool pairedRetreatCycle = false;
        bool splitMixedCycle = false;
        bool allChasesEscapedCheck = true;
        bool allChasesSingleTarget = true;
        int actions = 0;
        int checkMoves = 0;
        int chaseMoves = 0;
        int quietMoves = 0;
        int persistentChaseStates = 0;
        uint32_t chase = UINT32_MAX;
        uint32_t persistentTargets = UINT32_MAX;
        uint32_t anyTargets = 0;
        uint32_t anyChasers = 0;
        uint32_t checkers = 0;
        uint32_t checkingPieceIntersection = UINT32_MAX;
        uint32_t checkChased = 0;
        uint32_t counterChasers = 0;
        uint32_t persistentCounterChasers = UINT32_MAX;
        uint32_t newChaseTargets = 0;
        uint32_t pinnedRootTargets = 0;
        uint32_t withdrawnScreenTargets = 0;
        uint32_t movedIds = 0;
        uint32_t movedTargetIds = 0;
        uint32_t quietSelfEnteredChaseTargets = 0;
        // Rooted attacks are not ordinary chases, but their piece/type and
        // screen metadata affect the Duffish repetition matrix. Keep these
        // identities per side so a later return can distinguish a real root
        // from an unrooted advisor/horse/cannon attack.
        bool rootedAttackHasNonPawn = false;
        bool rootedAttackHasShieldedHorse = false;
        bool rootedAttackHasAdvisor = false;
        bool rootedAdvisorAttackUsesOpponentScreen = false;
        bool rootedAdvisorHasStableRoot = false;
        bool rootedAdvisorHasOpponentScreenCannonRoot = false;
        Move rootedAttackMoves[256]{};
        PieceType rootedAttackMoveTypes[256]{};
        int rootedAttackMoveCount = 0;
        uint32_t quietChasers[32]{};
        uint32_t checkSuppressedChasers[32]{};
        bool checkMoveCanBeCaptured[256]{};
        uint32_t checkMoveChasedTargets[256]{};
        bool chaseMoveUsesPinnedSameType[256]{};
        bool chaseMovePinsRoot[256]{};
        bool chaseMoveTargetsRook[256]{};
        bool quietRetreatEscapedCheck[256]{};
        uint32_t persistentChasers[32]{};
        ActionRecord records[MAX_CYCLE_STATES]{};
    } forcing[COLOR_NB];

    for (Color c : {WHITE, BLACK})
        std::fill_n(forcing[c].persistentChasers, 32, UINT32_MAX);

    auto id_bit = [](int id) -> uint32_t {
        return id > 0 && id <= 32 ? uint32_t(1) << (id - 1) : uint32_t(0);
    };

    Position rollback;
    rollback.copy_for_chasing_from(*this);

    for (int i = 0; i < d && i < MAX_CYCLE_STATES && rollback.st
                 && rollback.st->move.is_ok(); ++i)
    {
        const StateInfo* state = rollback.st;
        // Captures, dark moves and reveal/capture transitions terminate the
        // reversible SkyRule suffix. Never classify them as quiet retreats or
        // relay stable identities across the boundary.
        if (state->capturedPiece != NO_PIECE || state->captureDark || state->movedDark)
            break;
        const Color mover = ~rollback.sideToMove;
        ForcingSide& side = forcing[mover];
        ActionRecord record;
        record.move = state->move;
        record.action = state->skyruleAction;
        record.escapedCheck = state->previous && bool(state->previous->checkersBB);

        const Square to = state->move.to_sq();
        if (to != SQ_NONE)
        {
            record.pieceId = id_bit(rollback.idBoard[to]);
            const Piece moved = rollback.piece_on(to);
            if (moved != NO_PIECE)
                record.type = type_of(moved);
        }

        ++side.actions;
        side.movedIds |= record.pieceId;
        forcing[~mover].movedTargetIds |= record.pieceId;

        // Persistence is a property of the complete current attack relation,
        // not of this move's new-chase delta.  Intersect it at every state in
        // the cycle so a stationary attacker and a fleeing target retain their
        // stable identities across quiet replies.
        const SkyruleSnapshot snapshot = skyrule_snapshot(rollback, mover, ~mover);
        const uint32_t snapshotTargets =
          skyrule_chased_ids(snapshot.chased, rollback.idBoard);
        side.persistentTargets &= snapshotTargets;
        ++side.persistentChaseStates;

        // Reuse one before/after delta for both chase and rooted-attack
        // classification.  A quiet move can create a true-root relation even
        // when it does not create a chased target; Duffish keeps that metadata
        // for the final repetition matrix.
        SkyruleDelta chaseDelta;
        uint32_t beforeTargetIds = 0;
        Bitboard rooted = 0;
        bool rootedHasNonPawn = false;
        if (state->move.is_ok() && !state->capturedPiece && !state->captureDark
            && !state->movedDark)
        {
            Position previous;
            previous.copy_for_chasing_from(rollback);
            previous.undo_move(state->move);
            const SkyruleSnapshot before = skyrule_snapshot(previous, mover, ~mover);
            beforeTargetIds = skyrule_chased_ids(before.chased, previous.idBoard);
            chaseDelta = skyrule_delta(rollback, previous, mover, ~mover,
                                       state->move, record.type, snapshot, before,
                                       true, rollback.idBoard, previous.idBoard);
            rooted = chaseDelta.rooted;
        }

        for (Bitboard victimPieces = rollback.pieces(~mover); victimPieces;)
        {
            const Square target = pop_lsb(victimPieces);
            const int targetId = rollback.idBoard[target];
            if (targetId <= 0 || targetId > 32)
                continue;

            uint32_t currentChasers = 0;
            if (snapshot.chased & target)
            {
                Bitboard chasers = snapshot.relation[target].chasers;
                while (chasers)
                    currentChasers |= id_bit(rollback.idBoard[pop_lsb(chasers)]);
            }
            side.persistentChasers[targetId - 1] &= currentChasers;
        }

        if (state->skyruleAction == SKYRULE_CHECK)
        {
            side.hasCheck = true;
            ++side.checkMoves;
            record.direct = to != SQ_NONE && bool(state->checkersBB & to);

            // A checking piece that can simply be captured is not a stable
            // forcing checker.  Use the V8 attack/king-safety tables rather
            // than a geometric attackers-only test.
            if (to != SQ_NONE)
            {
                Bitboard captures = rollback.attackers_to(to, rollback.pieces())
                                  & rollback.pieces(~mover);
                while (captures && !record.checkMoveCanBeCaptured)
                {
                    const Square from = pop_lsb(captures);
                    const Piece capturer = rollback.piece_on(from);
                    record.checkMoveCanBeCaptured =
                      capturer != NO_PIECE
                      && skyrule_can_capture_from(rollback, ~mover,
                                                  type_of(capturer), from, to,
                                                  rollback.pieces(), to);
                }
            }

            Bitboard checkers = state->checkersBB;
            uint32_t checkerIds = 0;
            while (checkers)
                checkerIds |= id_bit(rollback.idBoard[pop_lsb(checkers)]);
            // A synthetic state may mark CHECK without carrying checkersBB;
            // retain the moved identity so checker limits remain meaningful.
            if (!checkerIds)
                checkerIds = record.pieceId;
            side.checkers |= checkerIds;
            side.checkingPieceIntersection &= checkerIds;
            side.anyChasers |= checkerIds;

            // CHECK has priority in StateInfo, but a checking move can also
            // create a chase. Keep that component for mixed-cycle priority.
            const uint32_t checkChased =
              skyrule_chased_ids(skyrule_direct_chased(rollback, true), rollback.idBoard);
            side.checkChased |= checkChased;
            side.hasCheckChase |= checkChased != 0;
            record.checkMoveChasedTargets = checkChased;

            if (side.checkMoves <= 256)
            {
                const int idx = side.checkMoves - 1;
                side.checkMoveCanBeCaptured[idx] = record.checkMoveCanBeCaptured;
                side.checkMoveChasedTargets[idx] = checkChased;
            }

            // Preserve the pin/root suppression relation for mixed check
            // cycles.  An attack suppressed by a newly exposed pin or by a
            // defending true root can become a chase when the checker
            // retreats; ordinary unrelated roots are intentionally ignored.
            const Color victim = ~mover;
            const Square kingSq = rollback.king_square(victim);
            for (Bitboard victims = rollback.pieces(victim); victims;)
            {
                const Square target = pop_lsb(victims);
                const int targetId = rollback.idBoard[target];
                if (targetId <= 0 || targetId > 32)
                    continue;
                const SkyruleRelations& rel = snapshot.relation[target];
                Bitboard candidates = rel.eligibleAttackers & ~rel.chasers
                                    & ~square_bb(to);
                while (candidates)
                {
                    const Square attackerSq = pop_lsb(candidates);
                    const Piece attacker = rollback.piece_on(attackerSq);
                    bool pinned = attacker != NO_PIECE
                               && !skyrule_can_capture_from(
                                    rollback, mover, type_of(attacker), attackerSq,
                                    target, rollback.pieces(), target);
                    bool kingRooted = false;
                    if (!pinned && kingSq != SQ_NONE
                        && (rel.trueRooters & attackerSq))
                    {
                        Bitboard roots = 0;
                        skyrule_has_true_root(rollback, victim, target, attackerSq,
                                              rollback.pieces(), attackerSq,
                                              Bitboard(0), &roots);
                        kingRooted = bool(roots & kingSq);
                    }
                    const int attackerId = rollback.idBoard[attackerSq];
                    if ((pinned || kingRooted) && attackerId > 0 && attackerId <= 32)
                        side.checkSuppressedChasers[targetId - 1] |= id_bit(attackerId);
                }
            }
        }
        else if (state->skyruleAction == SKYRULE_CHASE)
        {
            side.hasChase = true;
            ++side.chaseMoves;

            Bitboard targets = state->skyruleRawChased;
            if (!targets)
                targets = snapshot.chased;
            Bitboard targetSquares = targets;
            uint32_t targetIds = 0;
            uint32_t chaserIds = 0;
            while (targetSquares)
            {
                const Square target = pop_lsb(targetSquares);
                targetIds |= id_bit(rollback.idBoard[target]);

                Bitboard chasers = snapshot.relation[target].chasers;
                while (chasers)
                    chaserIds |= id_bit(rollback.idBoard[pop_lsb(chasers)]);
            }

            // Synthetic/legacy StateInfo fixtures may carry only a target
            // square. Keep the moved identity as a conservative chaser in
            // that case; real positions always populate the snapshot above.
            if (!chaserIds)
                chaserIds = record.pieceId;

            record.targets = targetIds;
            record.chasers = chaserIds;
            record.direct = bool(state->skyruleRawChased
                                 & skyrule_direct_chased(rollback));
            const Bitboard pinnedSameType = skyrule_pinned_same_type_chased(rollback);
            record.chaseMoveUsesPinnedSameType = bool(targets & pinnedSameType);
            record.chaseMoveTargetsRook = bool(targets
                                              & rollback.pieces(rollback.sideToMove, ROOK));

            side.chase &= targetIds;
            side.anyTargets |= targetIds;
            side.anyChasers |= chaserIds;
            side.allChasesEscapedCheck &= record.escapedCheck;
            side.allChasesSingleTarget &= !(targetIds & (targetIds - 1));
            Bitboard counter = skyrule_chased_by_targets(
              rollback, mover, targets, snapshot);
            uint32_t counterIds = 0;
            while (counter)
                counterIds |= id_bit(rollback.idBoard[pop_lsb(counter)]);
            record.counterChasers = counterIds;
            side.counterChasers |= counterIds;
            side.persistentCounterChasers &= counterIds;

            // Compare snapshots before and after the move. This is where
            // discovered attacks, pins, cannon screens and blocked horse/eye
            // roots become new chase events rather than repeated geometry.
            record.newTargets = skyrule_chased_ids(chaseDelta.chased, rollback.idBoard);
            record.pinnedRootTargets = skyrule_chased_ids(chaseDelta.pinnedRoot, rollback.idBoard);
            record.withdrawnScreenTargets =
              skyrule_chased_ids(chaseDelta.withdrawnScreen, rollback.idBoard);
            record.previouslyChasedTargets = targetIds & beforeTargetIds;
            side.newChaseTargets |= record.newTargets;
            side.pinnedRootTargets |= record.pinnedRootTargets;
            side.withdrawnScreenTargets |= record.withdrawnScreenTargets;

            record.chaseMovePinsRoot = bool(record.pinnedRootTargets);
            if (side.chaseMoves <= 256)
            {
                const int idx = side.chaseMoves - 1;
                side.chaseMoveUsesPinnedSameType[idx] = record.chaseMoveUsesPinnedSameType;
                side.chaseMovePinsRoot[idx] = record.chaseMovePinsRoot;
                side.chaseMoveTargetsRook[idx] = record.chaseMoveTargetsRook;
            }

            // CHECK/CHASE action priority means a rooted relation is only
            // retained as standalone metadata for a quiet action.
            rooted = 0;

        }
        else
        {
            side.hasQuiet = true;
            side.allForcing = false;
            ++side.quietMoves;
            record.quietRetreatEscapedCheck = record.escapedCheck;

            // If the opponent voluntarily moved a target into an already
            // stationary attack on the preceding ply, this quiet move must not
            // inherit responsibility for starting a persistent chase.
            if (state->previous && state->previous->move.is_ok()
                && !state->previous->capturedPiece && !state->previous->movedDark)
            {
                const Move targetMove = state->previous->move;
                const Square targetSq = targetMove.to_sq();
                const Piece target = rollback.piece_on(targetSq);
                const int targetId = rollback.idBoard[targetSq];
                if (target != NO_PIECE && color_of(target) == ~mover
                    && targetId > 0 && targetId <= 32
                    && (snapshot.chased & targetSq))
                {
                    Position beforeTargetMove;
                    beforeTargetMove.copy_for_chasing_from(rollback);
                    beforeTargetMove.undo_move(state->move);
                    if (beforeTargetMove.state()
                        && beforeTargetMove.state()->move == targetMove)
                    {
                        beforeTargetMove.undo_move(targetMove);
                        const SkyruleSnapshot beforeTarget =
                          skyrule_snapshot(beforeTargetMove, mover, ~mover);
                        const int stableId = rollback.idBoard[targetSq];
                        bool chasedBefore = false;
                        for (Bitboard oldTargets = beforeTargetMove.pieces(~mover); oldTargets;)
                        {
                            const Square oldTarget = pop_lsb(oldTargets);
                            if (beforeTargetMove.idBoard[oldTarget] == stableId)
                            {
                                chasedBefore = bool(beforeTarget.chased & oldTarget);
                                break;
                            }
                        }
                        if (!chasedBefore)
                            side.quietSelfEnteredChaseTargets |= id_bit(targetId);
                    }
                }
            }

            // Keep the stationary chaser identities for check-suppression and
            // paired retreat classification.
            for (Bitboard targets = snapshot.chased; targets;)
            {
                const Square target = pop_lsb(targets);
                const int targetId = rollback.idBoard[target];
                if (targetId <= 0 || targetId > 32)
                    continue;
                Bitboard chasers = snapshot.relation[target].chasers;
                while (chasers)
                {
                    const int chaserId = rollback.idBoard[pop_lsb(chasers)];
                    if (chaserId > 0 && chaserId <= 32)
                        side.quietChasers[targetId - 1] |= id_bit(chaserId);
                }
            }
            if (side.quietMoves <= 256)
                side.quietRetreatEscapedCheck[side.quietMoves - 1] = record.escapedCheck;
        }

        if (rooted)
        {
            // A newly rooted target is not itself a chase, but its material
            // and shielding shape affect several Duffish return exceptions.
            for (Bitboard targets = rooted; targets; )
            {
                const Square targetSq = pop_lsb(targets);
                const Piece targetPiece = rollback.piece_on(targetSq);
                rootedHasNonPawn |= targetPiece != NO_PIECE
                                 && type_of(targetPiece) != PAWN;

                if (record.type == CANNON && targetPiece != NO_PIECE
                    && type_of(targetPiece) != PAWN)
                {
                    const Color victim = rollback.sideToMove;
                    const Square kingSq = rollback.king_square(victim);
                    const Bitboard occupiedAfterCapture = rollback.pieces()
                                                         ^ square_bb(to);
                    Bitboard shields = kingSq == SQ_NONE
                                     ? Bitboard(0)
                                     : (between_bb(targetSq, kingSq) ^ kingSq)
                                         & rollback.pieces(victim);
                    while (shields)
                    {
                        const Square shieldSq = pop_lsb(shields);
                        const Piece shield = rollback.piece_on(shieldSq);
                        const bool shieldIsRoot =
                          shield != NO_PIECE
                          && skyrule_can_capture_from(rollback, victim,
                                                      type_of(shield), shieldSq,
                                                      targetSq, occupiedAfterCapture,
                                                      to);
                        if (type_of(targetPiece) == KNIGHT && shield != NO_PIECE
                            && !shieldIsRoot)
                            side.rootedAttackHasShieldedHorse = true;
                    }
                }
            }

        }

        // A cannon attack on an advisor with a legal true root is a special
        // rooted relation. It is intentionally checked even when `rooted` is
        // empty because advisors are excluded from the aggregate rooted mask.
        if (record.type == CANNON && state->skyruleAction != SKYRULE_CHASE
            && state->skyruleAction != SKYRULE_CHECK)
        {
                const Color victim = rollback.sideToMove;
                Bitboard advisors = attacks_bb(CANNON, to, rollback.pieces())
                                  & rollback.pieces(victim, ADVISOR);
                while (advisors)
                {
                    const Square advisorSq = pop_lsb(advisors);
                    if (!skyrule_can_capture_from(rollback, mover, CANNON, to,
                                                   advisorSq, rollback.pieces(),
                                                   advisorSq)
                        || !skyrule_has_true_root(rollback, victim, advisorSq, to,
                                                  rollback.pieces(), to))
                        continue;

                    side.rootedAttackHasAdvisor = true;
                    const Bitboard occupiedAfterCapture = rollback.pieces()
                                                         ^ square_bb(to);
                    Bitboard roots = rollback.attackers_to(advisorSq,
                                                           occupiedAfterCapture)
                                  & rollback.pieces(victim) & ~square_bb(advisorSq);
                    while (roots)
                    {
                        const Square rootSq = pop_lsb(roots);
                        const Piece rootPiece = rollback.piece_on(rootSq);
                        if (rootPiece == NO_PIECE
                            || !skyrule_can_capture_from(rollback, victim,
                                                         type_of(rootPiece), rootSq,
                                                         advisorSq, occupiedAfterCapture,
                                                         to))
                            continue;

                        bool opponentScreenCannonRoot = false;
                        if (type_of(rootPiece) == CANNON)
                        {
                            const Bitboard rootScreens =
                              (between_bb(rootSq, advisorSq) ^ advisorSq)
                              & occupiedAfterCapture;
                            if (rootScreens && !more_than_one(rootScreens))
                            {
                                const Piece screen = rollback.piece_on(lsb(rootScreens));
                                opponentScreenCannonRoot = screen != NO_PIECE
                                                         && color_of(screen) != victim;
                            }
                        }
                        side.rootedAdvisorHasOpponentScreenCannonRoot |=
                          opponentScreenCannonRoot;
                        side.rootedAdvisorHasStableRoot |= !opponentScreenCannonRoot;
                    }

                    const Bitboard screens =
                      (between_bb(to, advisorSq) ^ advisorSq) & rollback.pieces();
                    if (screens && !more_than_one(screens))
                    {
                        const Piece screen = rollback.piece_on(lsb(screens));
                        side.rootedAdvisorAttackUsesOpponentScreen |=
                          screen != NO_PIECE && color_of(screen) != mover;
                    }
                }
        }

        if (rooted && side.rootedAttackMoveCount < 256)
        {
                const int idx = side.rootedAttackMoveCount++;
                side.rootedAttackMoves[idx] = record.move;
                side.rootedAttackMoveTypes[idx] = record.type;
                side.rootedAttackHasNonPawn |= rootedHasNonPawn;
        }

        if (side.actions <= MAX_CYCLE_STATES)
            side.records[side.actions - 1] = record;

        rollback.undo_move(state->move);
    }

    const Color us = sideToMove;
    const Color them = ~us;

    auto reverses = [](Move first, Move second) {
        return first.is_ok() && second.is_ok()
            && first.from_sq() == second.to_sq()
            && first.to_sq() == second.from_sq();
    };

    auto quiet_reversal = [&](Color c, PieceType type) {
        const ForcingSide& side = forcing[c];
        if (side.quietMoves != 2)
            return false;

        const ActionRecord* first = nullptr;
        const ActionRecord* second = nullptr;
        for (int i = 0; i < side.actions; ++i)
            if (side.records[i].action == SKYRULE_NONE
                && side.records[i].type == type)
                (first ? second : first) = &side.records[i];

        return first && second && reverses(first->move, second->move);
    };

    auto first_action = [&](Color c, SkyruleAction action, PieceType type) {
        const ForcingSide& side = forcing[c];
        for (int i = 0; i < side.actions; ++i)
            if (side.records[i].action == action
                && (type == NO_PIECE_TYPE || side.records[i].type == type))
                return &side.records[i];
        return static_cast<const ActionRecord*>(nullptr);
    };

    // Classify quiet returns after the full cycle is available.  Duffish
    // deliberately treats a black pure check-retreat differently: it can keep
    // forcing status where the symmetric white retreat does not.
    for (Color c : {WHITE, BLACK})
    {
        ForcingSide& side = forcing[c];
        if (side.quietMoves)
        {
            bool allCheckRetreats = side.hasCheck;
            bool allChaseRetreats = side.hasChase;
            bool hasChaseRetreat = false;

            for (int i = 0; i < side.actions && (allCheckRetreats || allChaseRetreats); ++i)
            {
                const ActionRecord& quiet = side.records[i];
                if (quiet.action != SKYRULE_NONE)
                    continue;

                bool reversesCheck = false;
                bool reversesChase = false;
                for (int j = 0; j < side.actions; ++j)
                {
                    const ActionRecord& action = side.records[j];
                    reversesCheck |= action.action == SKYRULE_CHECK
                                  && reverses(quiet.move, action.move);
                    reversesChase |= action.action == SKYRULE_CHASE
                                  && reverses(quiet.move, action.move);
                }

                hasChaseRetreat |= reversesChase;
                allCheckRetreats &= reversesCheck;
                allChaseRetreats &= reversesChase
                                  || (!side.hasCheck && quiet.type == KING
                                      && quiet.escapedCheck);
            }

            if (allCheckRetreats && !side.hasChase)
            {
                side.checkRetreatCycle = true;
                side.allForcing = c == BLACK;
            }
            else
            {
                side.allForcing = false;
                side.chaseRetreatCycle = !allCheckRetreats
                                       && allChaseRetreats && hasChaseRetreat;
            }
        }

        if (side.hasCheck && side.hasChase)
        {
            bool allPaired = true;
            for (int i = 0; allPaired && i < side.actions; ++i)
            {
                const ActionRecord& action = side.records[i];
                if (action.action != SKYRULE_CHECK && action.action != SKYRULE_CHASE)
                    continue;

                const SkyruleAction opposite = action.action == SKYRULE_CHECK
                                              ? SKYRULE_CHASE : SKYRULE_CHECK;
                bool hasReverse = false;
                for (int j = 0; !hasReverse && j < side.actions; ++j)
                    hasReverse = side.records[j].action == opposite
                              && reverses(action.move, side.records[j].move);
                allPaired &= hasReverse;
            }
            side.checkChaseRetreatCycle = allPaired;

            // A mixed sequence may be paired through a quiet retreat rather
            // than by a literal check<->chase reversal. Duffish treats such a
            // cycle as non-forcing; retain the classification separately so
            // the final matrix can distinguish it from an ordinary mixed
            // violation.
            if (side.quietMoves)
            {
                auto hasReverse = [&](Move m, SkyruleAction kind) {
                    for (int j = 0; j < side.actions; ++j)
                        if (side.records[j].action == kind
                            && reverses(m, side.records[j].move))
                            return true;
                    return false;
                };
                bool paired = true;
                for (int i = 0; paired && i < side.actions; ++i)
                {
                    const ActionRecord& action = side.records[i];
                    if (action.action == SKYRULE_CHECK)
                        paired = hasReverse(action.move, SKYRULE_CHASE)
                               || hasReverse(action.move, SKYRULE_NONE);
                    else if (action.action == SKYRULE_CHASE)
                        paired = hasReverse(action.move, SKYRULE_CHECK)
                               || hasReverse(action.move, SKYRULE_NONE)
                               || hasReverse(action.move, SKYRULE_CHASE);
                    else if (action.action == SKYRULE_NONE)
                        paired = hasReverse(action.move, SKYRULE_CHECK)
                               || hasReverse(action.move, SKYRULE_CHASE);
                }
                side.pairedRetreatCycle = paired;
                if (paired)
                    side.allForcing = false;
            }
        }

        // Split mixed cycles contain both checks and chases but no single
        // persistent target intersection. They are relevant only when paired
        // against a genuine checking sequence on the other side.
        if (side.allForcing && side.hasCheck && side.hasChase && !side.chase)
        {
            side.splitMixedCycle = true;
            side.allForcing = false;
        }
    }

    // Duffish retains one deliberately asymmetric red/white exception: a
    // white rook which checks and immediately returns can remain forcing when
    // black's two king retreats are horizontal and a second white rook forms
    // an advisor battery on the returned file.  Keep the geometry limited to
    // the explicit line/rank conditions; advisor reachability itself comes
    // from V8's actual attack tables, with no orthodox palace assumption.
    const ActionRecord* whiteRookCheck = first_action(WHITE, SKYRULE_CHECK, ROOK);
    const ActionRecord* whiteRookQuiet = first_action(WHITE, SKYRULE_NONE, ROOK);
    const ActionRecord* blackKingQuiet[2] = { nullptr, nullptr };
    int blackKingQuietCount = 0;
    for (int i = 0; i < forcing[BLACK].actions && blackKingQuietCount < 2; ++i)
        if (forcing[BLACK].records[i].action == SKYRULE_NONE
            && forcing[BLACK].records[i].type == KING)
            blackKingQuiet[blackKingQuietCount++] = &forcing[BLACK].records[i];
    const bool blackKingHorizontalReturn =
      blackKingQuietCount == 2
      && reverses(blackKingQuiet[0]->move, blackKingQuiet[1]->move)
      && rank_of(blackKingQuiet[0]->move.from_sq())
         == rank_of(blackKingQuiet[0]->move.to_sq());
    const bool whiteRookHorizontalCheckRetreat =
      forcing[WHITE].checkRetreatCycle
      && !forcing[WHITE].hasChase
      && forcing[WHITE].checkMoves == 1
      && forcing[WHITE].quietMoves == 1
      && whiteRookCheck && whiteRookQuiet
      && reverses(whiteRookCheck->move, whiteRookQuiet->move)
      && !forcing[BLACK].hasCheck
      && !forcing[BLACK].hasChase
      && forcing[BLACK].quietMoves == 2
      && blackKingHorizontalReturn
      && whiteRookQuiet->type == ROOK;

    auto white_rook_advisor_battery = [&] {
        if (!whiteRookCheck || !whiteRookQuiet)
            return false;

        const Square batterySq = whiteRookQuiet->move.to_sq();
        if (!is_ok(batterySq) || !is_ok(whiteRookCheck->move.to_sq()))
            return false;
        const Bitboard visible = ~pieces(DARK);
        Bitboard rooks = pieces(WHITE, ROOK) & visible
                       & ~square_bb(whiteRookCheck->move.to_sq());
        while (rooks)
        {
            const Square rookSq = pop_lsb(rooks);
            if (file_of(rookSq) != file_of(batterySq))
                continue;

            const Bitboard advisors = attacks_bb(ROOK, rookSq, pieces())
                                    & pieces(BLACK, ADVISOR) & visible;
            if (advisors)
                return true;
        }
        return false;
    };

    const Square whiteKing = king_square(WHITE);
    const Square blackKing = king_square(BLACK);
    const bool whiteRookCheckRetreatAgainstKing =
      whiteRookHorizontalCheckRetreat
      && st->previous
      && st->previous->skyruleAction == SKYRULE_CHECK
      && whiteKing != SQ_NONE && blackKing != SQ_NONE
      && file_of(whiteKing) == file_of(blackKing)
      && white_rook_advisor_battery();

    if (whiteRookCheckRetreatAgainstKing)
        forcing[WHITE].allForcing = true;

    auto has_common_persistent_chaser = [&](Color c) {
        uint32_t targets = forcing[c].persistentTargets;
        for (int id = 0; id < 32; ++id)
            if ((targets & (uint32_t(1) << id)) && forcing[c].persistentChasers[id])
                return true;
        return false;
    };

    auto every_chase_pursues_persistent_target = [&](Color c) {
        const ForcingSide& side = forcing[c];
        for (int i = 0; i < side.actions; ++i)
            if (side.records[i].action == SKYRULE_CHASE
                && !(side.records[i].targets & side.persistentTargets))
                return false;
        return side.chaseMoves > 0;
    };

    auto initiates_persistent_chase = [&](Color c) {
        const ForcingSide& side = forcing[c];
        for (int i = 0; i < side.actions; ++i)
        {
            const ActionRecord& chase = side.records[i];
            if (chase.action == SKYRULE_CHASE
                && (side.persistentTargets & chase.targets
                    & ~chase.previouslyChasedTargets
                    & ~chase.withdrawnScreenTargets))
                return true;
        }
        return false;
    };

    auto persistent_targets_are_pawns = [&](Color c) {
        uint32_t pawnIds = 0;
        for (Bitboard pawns = pieces(~c, PAWN); pawns;)
            pawnIds |= id_bit(idBoard[pop_lsb(pawns)]);
        return forcing[c].persistentTargets
            && !(forcing[c].persistentTargets & ~pawnIds);
    };

    auto target_moved_before_chaser_closed = [&](Color c) {
        if (!st->move.is_ok())
            return false;
        const Piece closingPiece = piece_on(st->move.to_sq());
        return closingPiece != NO_PIECE && color_of(closingPiece) == c
            && bool(forcing[c].persistentTargets & forcing[c].movedTargetIds);
    };

    auto redundant_direct_chaser_reversal = [&](Color c) {
        const ForcingSide& side = forcing[c];
        if (d != 4 || side.chaseMoves != 1 || side.quietMoves != 1)
            return false;

        const ActionRecord* chase = nullptr;
        const ActionRecord* quiet = nullptr;
        for (int i = 0; i < side.actions; ++i)
        {
            if (side.records[i].action == SKYRULE_CHASE)
                chase = &side.records[i];
            else if (side.records[i].action == SKYRULE_NONE)
                quiet = &side.records[i];
        }
        if (!chase || !quiet || !chase->direct || chase->pieceId != quiet->pieceId
            || !reverses(chase->move, quiet->move))
            return false;

        uint32_t target = side.persistentTargets & chase->previouslyChasedTargets;
        if (!target || (target & (target - 1)))
            return false;
        int targetId = 0;
        while (!(target & (uint32_t(1) << targetId)))
            ++targetId;
        return bool(chase->chasers & chase->pieceId)
            && bool(side.persistentChasers[targetId] & ~chase->pieceId);
    };

    auto persistent_chase = [&](Color c) {
        const ForcingSide& side = forcing[c];
        const int states = d / 2;
        const bool commonChaser = has_common_persistent_chaser(c);
        return !side.hasCheck && side.hasChase
            && side.chaseMoves + side.quietMoves == states
            && side.persistentChaseStates == states
            && side.persistentTargets
            && side.persistentTargets != UINT32_MAX
            && (side.persistentTargets & side.anyTargets)
            && (every_chase_pursues_persistent_target(c)
                || persistent_targets_are_pawns(c))
            && initiates_persistent_chase(c)
            && !(side.persistentTargets & side.quietSelfEnteredChaseTargets)
            && !(side.persistentCounterChasers != UINT32_MAX
                 && side.persistentCounterChasers)
            && (!side.quietMoves || commonChaser
                || persistent_targets_are_pawns(c)
                || target_moved_before_chaser_closed(c))
            && (!side.quietMoves || commonChaser
                || !rollback.state()->capturedPiece)
            && !redundant_direct_chaser_reversal(c);
    };

    auto ordinary_chase = [&](Color c) {
        const ForcingSide& side = forcing[c];
        return side.allForcing && !side.hasCheck && side.hasChase
            && side.chase && side.chase != UINT32_MAX
            && (!(side.chase & (side.chase - 1))
                || side.persistentCounterChasers == UINT32_MAX
                || !side.persistentCounterChasers);
    };

    auto checking_sequence = [&](Color c) {
        const ForcingSide& side = forcing[c];
        if (!side.hasCheck)
            return false;
        uint32_t checkerIds = 0;
        for (int i = 0; i < side.actions; ++i)
            if (side.records[i].action == SKYRULE_CHECK)
                checkerIds |= side.records[i].pieceId;
        if (!checkerIds)
            return false;
        for (int i = 0; i < side.actions; ++i)
            if (side.records[i].action == SKYRULE_NONE
                && !(side.records[i].pieceId & checkerIds))
                return false;
        return true;
    };

    auto split_chasing_against_check = [&](Color c) {
        const Color defender = ~c;
        const ForcingSide& side = forcing[c];
        const ForcingSide& reply = forcing[defender];
        if (!side.hasChase || side.hasCheck || !checking_sequence(defender)
            || reply.hasChase)
            return false;

        // A split chase is meaningful when each chase move contributes to a
        // distinct target (empty intersection), or when king escapes answer a
        // preceding check. A chase that only appears while escaping check is
        // deliberately suppressed.
        const bool ordinarySplit = side.allForcing && !side.chase
                                 && side.chaseMoves + side.quietMoves == d / 2
                                 && [&] {
                                        for (int i = 0; i < side.actions; ++i)
                                            if (side.records[i].action == SKYRULE_CHASE
                                                && side.records[i].escapedCheck)
                                                return false;
                                        return true;
                                    }();

        // Duffish also accepts a split chase whose quiet plies are forced king
        // escapes from an earlier check.  V8 has no orthodox palace predicate
        // here: the move legality and the actual king attack table already
        // enforce the Jieqi geometry.
        const bool kingEscapeSplit = [&] {
            if (side.chase || !side.quietMoves
                || side.chaseMoves + side.quietMoves != d / 2)
                return false;
            if (d > 4)
            {
                StateInfo* anchor = st;
                for (int i = 0; i < d && anchor; ++i)
                    anchor = anchor->previous;
                if (!anchor || anchor->skyruleAction != SKYRULE_CHECK)
                    return false;
            }
            for (int i = 0; i < side.actions; ++i)
            {
                const ActionRecord& action = side.records[i];
                if (action.action == SKYRULE_CHASE && action.escapedCheck)
                    return false;
                if (action.action == SKYRULE_NONE
                    && (action.type != KING || !action.escapedCheck))
                    return false;
            }
            return true;
        }();

        const bool sustainedRetreatSplit = [&] {
            if (side.chase || side.chaseMoves < 3 || !side.quietMoves
                || side.chaseMoves + side.quietMoves != d / 2)
                return false;
            bool hasChaseRetreat = false;
            for (int i = 0; i < side.actions; ++i)
            {
                const ActionRecord& quiet = side.records[i];
                if (quiet.action != SKYRULE_NONE)
                    continue;
                bool reversesChase = false;
                for (int j = 0; j < side.actions; ++j)
                {
                    const ActionRecord& chase = side.records[j];
                    reversesChase |= chase.action == SKYRULE_CHASE
                                  && reverses(quiet.move, chase.move);
                }
                hasChaseRetreat |= reversesChase;
                if (!quiet.escapedCheck && !reversesChase)
                    return false;
            }
            return hasChaseRetreat;
        }();

        return ordinarySplit || kingEscapeSplit || sustainedRetreatSplit;
    };

    auto persistent_mixed = [&](Color c) {
        const ForcingSide& side = forcing[c];
        return side.allForcing && side.hasCheck
            && (side.hasChase || side.hasCheckChase)
            && !side.pairedRetreatCycle
            && !side.splitMixedCycle
            && side.checkMoves + side.chaseMoves == side.actions;
    };

    // Duffish's mixed-cycle priority is narrower than the general mixed
    // classification above: one stable target must survive every state, and
    // no persistent counter-chase may be present.  Keep this separate so a
    // multi-target or mutual cycle cannot enter the returning-check rule.
    auto persistent_mixed_chasing = [&](Color c) {
        const ForcingSide& side = forcing[c];
        const int states = d / 2;
        return side.allForcing && side.hasCheck && side.hasChase
            && side.checkMoves + side.chaseMoves + side.quietMoves == states
            && side.persistentChaseStates == states
            && side.persistentTargets
            && side.persistentTargets != UINT32_MAX
            && !(side.persistentTargets & (side.persistentTargets - 1))
            && !(side.persistentCounterChasers != UINT32_MAX
                 && side.persistentCounterChasers);
    };

    // A check-retreat exemption only applies when the exact stable checker
    // identity is pursued by the other side.  Using anyTargets here is too
    // broad: an unrelated chase in the same cycle must not make a retreating
    // checker appear to be chased.
    auto checking_move_chases_target = [&](Color c, uint32_t target) {
        if (!target)
            return false;

        const ForcingSide& side = forcing[c];
        for (int i = 0; i < side.actions; ++i)
            if (side.records[i].action == SKYRULE_CHECK
                && (side.records[i].checkMoveChasedTargets & target))
                return true;
        return false;
    };

    auto checking_retreat_piece_chased = [&](Color c) {
        const ForcingSide& side = forcing[c];
        if (!side.checkRetreatCycle || side.checkMoves != 1 || side.quietMoves != 1)
            return false;

        uint32_t checker = 0;
        uint32_t retreat = 0;
        Move retreatMove = Move::none();
        for (int i = 0; i < side.actions; ++i)
        {
            const ActionRecord& action = side.records[i];
            if (action.action == SKYRULE_CHECK)
                checker = action.pieceId;
            else if (action.action == SKYRULE_NONE)
            {
                retreat = action.pieceId;
                retreatMove = action.move;
            }
        }

        // The quiet reply must be the same physical checker returning.  This
        // identity test also rejects synthetic square-only fixtures where the
        // two actions happen to use the same coordinates with different men.
        if (!checker || checker != retreat)
            return false;

        const ForcingSide& opponent = forcing[~c];
        const bool chasedInCycle =
          (opponent.hasChase && opponent.chase != UINT32_MAX
           && bool(opponent.chase & checker))
          || checking_move_chases_target(~c, checker);
        if (!chasedInCycle || !retreatMove.is_ok())
            return false;

        // Duffish checks the actual repeated root as well as the cycle
        // metadata.  Recompute the V8 relation snapshot so pins, true roots,
        // horse legs and cannon screens are handled by the same legality code
        // used for ordinary chase detection.
        const Square pieceSq = retreatMove.to_sq();
        if (pieceSq == SQ_NONE || idBoard[pieceSq] <= 0
            || idBoard[pieceSq] > 32
            || !(id_bit(idBoard[pieceSq]) & checker))
            return false;
        const SkyruleSnapshot rootSnapshot = skyrule_snapshot(*this, ~c, c);
        return bool(rootSnapshot.chased & pieceSq)
            && bool(rootSnapshot.relation[pieceSq].chasers);
    };

    // A check-retreat chase which first appeared on the repetition-anchor
    // capture is treated differently from a chase that was already present
    // before the reversible suffix.  The capture terminates the old identity
    // history, so a black checker created by that capture keeps the ordinary
    // check-retreat exemption.  Recompute the pre-capture snapshot with the
    // same V8 attack/true-root predicates used by the main detector.
    auto checking_retreat_chase_created_by_anchor_capture = [&](Color c) {
        const ForcingSide& side = forcing[c];
        if (side.checkMoves != 1 || side.quietMoves != 1
            || !rollback.state() || rollback.state()->capturedPiece == NO_PIECE
            || !rollback.state()->move.is_ok())
            return false;

        const ActionRecord* quiet = first_action(c, SKYRULE_NONE, NO_PIECE_TYPE);
        if (!quiet || !quiet->move.is_ok())
            return false;

        const Square pieceSq = quiet->move.to_sq();
        Position beforeCapture;
        beforeCapture.copy_for_chasing_from(rollback);
        beforeCapture.undo_move(beforeCapture.state()->move);

        const Piece piece = beforeCapture.piece_on(pieceSq);
        if (piece == NO_PIECE || color_of(piece) != c || beforeCapture.is_dark(pieceSq))
            return false;

        const SkyruleSnapshot before = skyrule_snapshot(beforeCapture, ~c, c);
        return !(before.chased & pieceSq);
    };

    auto opponent_royal_reversal = [&](Color c) {
        const ForcingSide& reply = forcing[~c];
        if (reply.quietMoves != 2)
            return false;
        const ActionRecord* first = nullptr;
        const ActionRecord* second = nullptr;
        for (int i = 0; i < reply.actions; ++i)
            if (reply.records[i].action == SKYRULE_NONE)
                (first ? second : first) = &reply.records[i];
        return first && second && first->type == second->type
            && (first->type == KING || first->type == ADVISOR)
            && reverses(first->move, second->move);
    };

    auto forcing_violation = [&](Color c) {
        const ForcingSide& side = forcing[c];
        if (c == BLACK && side.checkRetreatCycle && !side.hasChase
            && (!checking_retreat_piece_chased(c)
                || checking_retreat_chase_created_by_anchor_capture(c))
            && (side.checkMoves != 1 || [&] {
                    for (int i = 0; i < side.actions; ++i)
                        if (side.records[i].action == SKYRULE_CHECK)
                            return side.records[i].direct
                                || side.records[i].type != PAWN
                                || opponent_royal_reversal(c);
                    return true;
                }())
            && !forcing[~c].hasCheck
            && (!forcing[~c].hasChase
                || (!forcing[~c].allForcing
                    && !forcing[~c].chaseRetreatCycle
                    && !forcing[~c].checkChaseRetreatCycle)))
            return false;

        if (side.allForcing && side.hasCheck && side.hasChase
            && forcing[~c].hasCheck && !forcing[~c].allForcing)
            return true;

        return side.allForcing && side.hasCheck
            && (!side.hasChase || forcing[~c].hasChase
                || (!forcing[~c].hasCheck && !forcing[~c].hasChase
                    && (c == WHITE || !side.checkChaseRetreatCycle
                        || ((us == BLACK || checkers())
                            && side.checkChaseRetreatCycle))));
    };

    auto persistent_at_root = [&](Color c) {
        const SkyruleSnapshot rootSnapshot = skyrule_snapshot(*this, c, ~c);
        uint32_t active = skyrule_chased_ids(rootSnapshot.chased, idBoard);
        uint32_t pinnedChasersAtRoot = 0;

        // Returning a target to the safe anchor of a matched four-ply
        // reversal does not extend a single newly-created chase.  This is an
        // identity check, rather than a square-only check, so a capture or
        // reveal cannot accidentally relay the old relation.
        bool matchedReversalEscape = false;
        if (d == 4 && forcing[c].chaseMoves == 1 && forcing[c].quietMoves == 1
            && forcing[~c].quietMoves == 2)
        {
            const ActionRecord* chase = first_action(c, SKYRULE_CHASE, NO_PIECE_TYPE);
            const ActionRecord* quiet = first_action(c, SKYRULE_NONE, NO_PIECE_TYPE);
            const ActionRecord* replies[2] = { nullptr, nullptr };
            int replyCount = 0;
            for (int i = 0; i < forcing[~c].actions && replyCount < 2; ++i)
                if (forcing[~c].records[i].action == SKYRULE_NONE)
                    replies[replyCount++] = &forcing[~c].records[i];

            matchedReversalEscape = chase && quiet && replyCount == 2
                                  && chase->pieceId == quiet->pieceId
                                  && chase->pieceId != 0
                                  && reverses(chase->move, quiet->move)
                                  && replies[0]->pieceId == replies[1]->pieceId
                                  && replies[0]->pieceId != 0
                                  && reverses(replies[0]->move, replies[1]->move);
        }

        // A defender may close the cycle by pinning a persistent chaser. The
        // geometric attack then ceases to be a legal capture at the repeated
        // root, but it still completes the preceding long chase. Preserve
        // that liability unless a same-type recapture or a true root defuses
        // it.
        for (int targetId = 0; targetId < 32; ++targetId)
        {
            const uint32_t targetBit = uint32_t(1) << targetId;
            if (!(forcing[c].persistentTargets & targetBit))
                continue;

            Square targetSq = SQ_NONE;
            for (Bitboard targets = pieces(~c); targets; )
            {
                const Square sq = pop_lsb(targets);
                if (idBoard[sq] == targetId + 1)
                {
                    targetSq = sq;
                    break;
                }
            }
            if (targetSq == SQ_NONE)
                continue;

            const SkyruleRelations& relation = rootSnapshot.relation[targetSq];
            for (Bitboard chasers = pieces(c); chasers; )
            {
                const Square chaserSq = pop_lsb(chasers);
                const int chaserId = idBoard[chaserSq];
                const Piece chaserPiece = piece_on(chaserSq);
                if (chaserPiece == NO_PIECE || chaserId <= 0 || chaserId > 32
                    || !(forcing[c].persistentChasers[targetId] & id_bit(chaserId))
                    // A pinned attacker is deliberately absent from
                    // relation.chasers.  Duffish tests the eligible attack
                    // set here so a root pin can preserve the preceding
                    // chase liability.
                    || !(relation.eligibleAttackers & chaserSq))
                    continue;

                const Piece targetPiece = piece_on(targetSq);
                const PieceType chaserType = type_of(chaserPiece);
                const PieceType targetType = type_of(targetPiece);
                const Bitboard occupied = pieces();
                const Square kingSq = king_square(c);
                const bool kingSafeBefore =
                  kingSq != SQ_NONE
                  && !skyrule_checkers_after_capture(*this, c, kingSq, occupied);
                const Bitboard attack =
                  chaserType == PAWN ? attacks_bb<PAWN>(chaserSq, c)
                                     : attacks_bb(chaserType, chaserSq, occupied);
                const bool pinned = kingSafeBefore
                                 && (attack & targetSq)
                                 && !skyrule_can_capture_from(
                                      *this, c, chaserType, chaserSq, targetSq,
                                      occupied, targetSq);
                if (!pinned)
                    continue;

                const bool sameTypeCounterCapture =
                  targetType == chaserType
                  && skyrule_can_capture_from(*this, ~c, targetType, targetSq,
                                              chaserSq, occupied, chaserSq);
                const bool rootedRookException =
                  targetType == ROOK && chaserType != ROOK && chaserType != PAWN;
                const bool trueRooted =
                  !rootedRookException
                  && skyrule_has_true_root(*this, ~c, targetSq, chaserSq,
                                           occupied, chaserSq);

                if (!sameTypeCounterCapture && !trueRooted)
                {
                    pinnedChasersAtRoot |= targetBit;
                    break;
                }
            }
        }

        if (st->move.is_ok())
        {
            const Square to = st->move.to_sq();
            const Piece moved = piece_on(to);
            const int movedId = idBoard[to];
            if (moved != NO_PIECE && movedId > 0 && movedId <= 32
                && color_of(moved) == ~c
                && (!forcing[c].quietMoves
                    || forcing[c].persistentChasers[movedId - 1])
                && !matchedReversalEscape)
                active |= id_bit(movedId);
        }
        return bool(forcing[c].persistentTargets & (active | pinnedChasersAtRoot));
    };

    const bool checkUs = forcing_violation(us);
    const bool checkThem = forcing_violation(them);
    const bool ordinaryChaseUs = ordinary_chase(us)
                              || split_chasing_against_check(us);
    const bool ordinaryChaseThem = ordinary_chase(them)
                                || split_chasing_against_check(them);
    const bool usePersistent = !checkUs && !checkThem
                            && !ordinaryChaseUs && !ordinaryChaseThem;
    const bool persistentUs = persistent_chase(us);
    const bool persistentThem = persistent_chase(them);
    const bool activePersistentUs = usePersistent && persistentUs
                                 && persistent_at_root(us);
    const bool activePersistentThem = usePersistent && persistentThem
                                   && persistent_at_root(them);
    bool chaseUs = ordinaryChaseUs
                || activePersistentUs;
    bool chaseThem = ordinaryChaseThem
                  || activePersistentThem;
    const bool mixedUs = persistent_mixed(us);
    const bool mixedThem = persistent_mixed(them);
    bool forceUs = checkUs || chaseUs;
    bool forceThem = checkThem || chaseThem;

    auto result_for = [&](Color violator) {
        if (ply == 0)
            return Value(VALUE_NONE);
        return us == violator ? mated_in(ply) : mate_in(ply);
    };

    // These root-only deferrals mirror Duffish's search timing.  A repeated
    // quiet or split-mixed root is left available for one breaker move; the
    // internal node will adjudicate it once the search has entered the cycle.
    if (ply == 0 && !forceUs && !forceThem
        && (forcing[WHITE].splitMixedCycle || forcing[BLACK].splitMixedCycle))
        return VALUE_NONE;

    if (ply == 0 && !forceUs && !forceThem
        && !forcing[WHITE].hasCheck && !forcing[WHITE].hasChase
        && !forcing[BLACK].hasCheck && !forcing[BLACK].hasChase
        && forcing[WHITE].quietMoves + forcing[BLACK].quietMoves == d)
        return VALUE_NONE;

    if (ply == 0 && whiteRookHorizontalCheckRetreat
        && !whiteRookCheckRetreatAgainstKing)
        return VALUE_NONE;

    if (ply == 0 && d == 8
        && (forcing[WHITE].pairedRetreatCycle || forcing[BLACK].pairedRetreatCycle))
    {
        const Color paired = forcing[WHITE].pairedRetreatCycle ? WHITE : BLACK;
        const Color other = ~paired;
        if (!forcing[other].hasCheck && !forcing[other].hasChase
            && forcing[other].quietMoves == 4)
            return VALUE_NONE;
    }

    if (!forceUs && !forceThem && d == 4 && us == WHITE
        && forcing[BLACK].allForcing
        && forcing[BLACK].hasCheck && forcing[BLACK].hasChase
        && forcing[BLACK].checkChaseRetreatCycle
        && !forcing[WHITE].hasCheck && !forcing[WHITE].hasChase
        && st->skyruleAction == SKYRULE_CHASE)
    {
        const ActionRecord* blackCannonCheck = first_action(BLACK, SKYRULE_CHECK,
                                                             CANNON);
        const ActionRecord* blackCannonChase = first_action(BLACK, SKYRULE_CHASE,
                                                             CANNON);
        if (!blackCannonCheck || !blackCannonChase
            || !blackCannonChase->chaseMovePinsRoot)
            return VALUE_NONE;
    }

    // A mixed check/chase side remains responsible when the opponent's direct
    // checker returns to a square that was already covered by the stable
    // chase.  The answering check is an escape, not a fresh interruption.
    const ActionRecord* blackReturningCheck = first_action(BLACK, SKYRULE_CHECK,
                                                            NO_PIECE_TYPE);
    const ActionRecord* blackReturningQuiet = first_action(BLACK, SKYRULE_NONE,
                                                            NO_PIECE_TYPE);
    const ActionRecord* whiteMixedCheck = first_action(WHITE, SKYRULE_CHECK,
                                                       NO_PIECE_TYPE);
    const bool mixedCheckAgainstReturningChasedPiece = d == 4
        && forcing[WHITE].allForcing
        && forcing[WHITE].hasCheck && forcing[WHITE].hasChase
        && persistent_mixed_chasing(WHITE)
        && forcing[BLACK].checkRetreatCycle
        && forcing[BLACK].hasCheck && !forcing[BLACK].hasChase
        && blackReturningCheck && blackReturningQuiet && whiteMixedCheck
        && blackReturningCheck->direct
        && !blackReturningQuiet->escapedCheck
        && (st->move == blackReturningQuiet->move
            || st->move == whiteMixedCheck->move);

    if (mixedCheckAgainstReturningChasedPiece)
    {
        if (ply > 0)
            return us == WHITE ? mated_in(ply) : mate_in(ply - 1);
        return VALUE_NONE;
    }

    // A defender moving a pinned true root away does not turn the chaser's
    // quiet return into a fresh continuation. Duffish treats this completed
    // four-ply shape as a draw (or root deferral), preserving the stable
    // target/chaser identities collected above.
    auto pinned_root_chase_ends_on_retreat = [&](Color chaser) {
        const Color defender = ~chaser;
        const bool activePersistent = chaser == us ? activePersistentUs
                                                   : activePersistentThem;
        const ActionRecord* chase = first_action(chaser, SKYRULE_CHASE,
                                                 NO_PIECE_TYPE);
        const ActionRecord* quiet = first_action(chaser, SKYRULE_NONE,
                                                 NO_PIECE_TYPE);
        return d == 4 && activePersistent
            && forcing[chaser].chaseRetreatCycle
            && forcing[chaser].chaseMoves == 1
            && forcing[chaser].quietMoves == 1
            && forcing[chaser].chaseMovePinsRoot[0]
            && chase && quiet && reverses(chase->move, quiet->move)
            && quiet_reversal(defender, KING)
            && st->move == quiet->move;
    };

    if (pinned_root_chase_ends_on_retreat(WHITE)
        || pinned_root_chase_ends_on_retreat(BLACK))
        return ply > 0 ? VALUE_DRAW : VALUE_NONE;

    // A single piece that alternates check and chase and then makes its one
    // quiet return remains a mutual forcing tour against an active persistent
    // chase. This identity check avoids treating unrelated mixed moves as a
    // draw.
    auto single_piece_mixed_tour_against_persistent_chase = [&](Color mixed) {
        const Color persistent = ~mixed;
        const bool activePersistent = persistent == us ? activePersistentUs
                                                       : activePersistentThem;
        const int states = d / 2;
        const ForcingSide& side = forcing[mixed];
        if (!activePersistent || side.allForcing || !side.hasCheck || !side.hasChase
            || side.checkMoves == 0 || side.chaseMoves == 0
            || side.quietMoves != 1
            || side.checkMoves + side.chaseMoves + 1 != states
            || !st->move.is_ok())
            return false;

        const ActionRecord* quiet = first_action(mixed, SKYRULE_NONE,
                                                 NO_PIECE_TYPE);
        if (!quiet || quiet->move != st->move || !quiet->pieceId
            || (quiet->pieceId & (quiet->pieceId - 1)))
            return false;

        for (int i = 0; i < side.actions; ++i)
        {
            const ActionRecord& action = side.records[i];
            if ((action.action == SKYRULE_CHECK || action.action == SKYRULE_CHASE)
                && action.pieceId != quiet->pieceId)
                return false;
        }
        return true;
    };

    if (single_piece_mixed_tour_against_persistent_chase(WHITE)
        || single_piece_mixed_tour_against_persistent_chase(BLACK))
        return ply > 0 ? VALUE_DRAW : VALUE_NONE;

    // A checking move can suppress a geometrical attack through a king root
    // or a pin. If that same attacker is revealed as a legal chase when the
    // checker retreats, the cycle remains forcing even though its target
    // intersection is otherwise empty.
    auto check_dislodges_root_into_chase = [&](Color c) {
        const Color defender = ~c;
        const ForcingSide& side = forcing[c];
        const ForcingSide& reply = forcing[defender];
        const ActionRecord* check = first_action(c, SKYRULE_CHECK,
                                                 NO_PIECE_TYPE);
        const ActionRecord* quiet = first_action(c, SKYRULE_NONE,
                                                 NO_PIECE_TYPE);
        if (d != 4 || c != us || !side.checkRetreatCycle || side.hasChase
            || side.checkMoves != 1 || side.quietMoves != 1
            // The cycle is walked backwards, so records[0] is not
            // necessarily the checking action. Select by action kind instead
            // of relying on traversal order.
            || !check || !quiet || !check->direct
            || check->pieceId == 0
            || check->pieceId != quiet->pieceId
            || !reverses(check->move, quiet->move)
            || reply.hasCheck || reply.hasChase || reply.quietMoves != 2)
            return false;

        const ActionRecord* replies[2] = { nullptr, nullptr };
        int n = 0;
        for (int i = 0; i < reply.actions && n < 2; ++i)
            if (reply.records[i].action == SKYRULE_NONE)
                replies[n++] = &reply.records[i];
        if (n != 2 || replies[0]->type != KING || replies[1]->type != KING
            || !reverses(replies[0]->move, replies[1]->move)
            || replies[0]->escapedCheck == replies[1]->escapedCheck)
            return false;

        bool closesOnKingReturn = false;
        for (const ActionRecord* r : replies)
            closesOnKingReturn |= st->move == r->move && !r->escapedCheck;
        if (!closesOnKingReturn)
            return false;

        for (int id = 0; id < 32; ++id)
            if (side.checkSuppressedChasers[id] & side.quietChasers[id])
                return true;
        return false;
    };

    for (Color c : {WHITE, BLACK})
        if (check_dislodges_root_into_chase(c))
            return result_for(c);

    auto draw_result = [&] { return ply == 0 ? Value(VALUE_NONE) : Value(VALUE_DRAW); };

    auto checks_between_persistent_chase_escapes = [&](Color checker,
                                                       bool chaserEscapes) {
        const Color chaser = ~checker;
        const int states = d / 2;
        const ForcingSide& checkingSide = forcing[checker];
        const ForcingSide& chasingSide = forcing[chaser];

        if (!persistent_chase(chaser)
            || !checkingSide.hasCheck || checkingSide.hasChase
            || chasingSide.hasCheck
            || checkingSide.checkMoves < 2
            || chasingSide.chaseMoves < 2
            || !checkingSide.quietMoves
            || bool(chasingSide.quietMoves) != chaserEscapes
            || checkingSide.checkMoves + checkingSide.quietMoves != states)
            return false;

        // Every quiet move by the checking side must be the stable chased
        // target, and every quiet reply by the chaser must have escaped check.
        for (int i = 0; i < checkingSide.actions; ++i)
            if (checkingSide.records[i].action == SKYRULE_NONE
                && !(checkingSide.records[i].pieceId
                     & chasingSide.persistentTargets))
                return false;

        for (int i = 0; i < chasingSide.actions; ++i)
            if (chasingSide.records[i].action == SKYRULE_NONE
                && !chasingSide.records[i].quietRetreatEscapedCheck)
                return false;

        return true;
    };

    // Checks made while the chased target escapes do not interrupt a pure
    // chase.  If the chaser never had to answer check, its closing chase is
    // the losing move; if it did, both sides are mutually forcing.
    if (ply > 0 && st->skyruleAction == SKYRULE_CHASE
        && checks_between_persistent_chase_escapes(us, false))
        return mate_in(ply - 1);

    if (checks_between_persistent_chase_escapes(WHITE, true)
        || checks_between_persistent_chase_escapes(BLACK, true))
        return ply > 0 ? VALUE_DRAW : VALUE_NONE;

    // Duffish keeps one asymmetric discovered-check exception for a black
    // pawn retreat.  The black pawn gives an indirect check, reverses back
    // to its anchor, and White's own non-king/non-advisor check retreat
    // closes the same four-ply cycle.  This is a responsibility rule rather
    // than a geometric chase rule, so it must run before the generic
    // persistent-root delay below.
    const ActionRecord* blackPawnCheck = first_action(BLACK, SKYRULE_CHECK,
                                                       PAWN);
    const ActionRecord* blackPawnQuiet = first_action(BLACK, SKYRULE_NONE,
                                                       PAWN);
    const ActionRecord* whiteCycleQuiet = first_action(WHITE, SKYRULE_NONE,
                                                        NO_PIECE_TYPE);
    const bool blackPawnClosesDiscoveredCheckCycle = us == WHITE
        && d == 4
        && !(checkUs || chaseUs)
        && checkThem && !chaseThem
        && forcing[BLACK].allForcing
        && forcing[BLACK].checkRetreatCycle
        && !forcing[BLACK].hasChase
        && forcing[BLACK].checkMoves == 1
        && forcing[BLACK].quietMoves == 1
        && blackPawnCheck && blackPawnQuiet
        && !blackPawnCheck->direct
        && blackPawnQuiet->type == PAWN
        && reverses(blackPawnCheck->move, blackPawnQuiet->move)
        && st->move == blackPawnQuiet->move
        && forcing[WHITE].checkRetreatCycle
        && !forcing[WHITE].hasChase
        && forcing[WHITE].quietMoves == 1
        && whiteCycleQuiet
        && whiteCycleQuiet->type != KING
        && whiteCycleQuiet->type != ADVISOR;

    // The closing black pawn retreat is the losing move.  At a fresh root
    // defer the claim so search can still choose a breaker, matching the
    // root/internal timing used by Duffish.
    if (blackPawnClosesDiscoveredCheckCycle)
    {
        if (ply > 0)
            return mate_in(ply - 1);
        return VALUE_NONE;
    }

    // A persistent chase detected on the opponent's side at a new root is
    // deliberately deferred.  Search must get one chance to find a breaker;
    // internal nodes retain the Duffish mate distance.
    // V8 keeps the force flags split between check and chase classifications;
    // Duffish's combined `forceUs` is equivalent to their union here.
    if (!(checkUs || chaseUs) && activePersistentThem)
    {
        if (ply > 0)
            return mate_in(ply - 1);
        return VALUE_NONE;
    }

    auto paired_split_chase_against_bounding_check = [&](Color c) {
        const ForcingSide& side = forcing[c];
        const ForcingSide& reply = forcing[~c];
        if (c != us
            || !side.chaseRetreatCycle
            || !side.hasChase || side.chase
            || side.chaseMoves < 2
            || side.chaseMoves != side.quietMoves
            || !side.allChasesSingleTarget
            || st->skyruleAction != SKYRULE_CHECK
            || reply.checkMoves != 1)
            return false;

        const ActionRecord* boundingCheck = first_action(~c, SKYRULE_CHECK,
                                                         NO_PIECE_TYPE);
        if (!boundingCheck || !boundingCheck->direct
            || boundingCheck->move != st->move)
            return false;

        StateInfo* anchor = st;
        for (int i = 0; i < d && anchor; ++i)
            anchor = anchor->previous;
        if (!anchor || anchor->skyruleAction != SKYRULE_CHECK)
            return false;

        uint32_t chasePieces = 0;
        int chaseCount = 0;
        for (int i = 0; i < side.actions; ++i)
        {
            const ActionRecord& chase = side.records[i];
            if (chase.action != SKYRULE_CHASE)
                continue;

            ++chaseCount;
            if (chase.escapedCheck)
                return false;
            chasePieces |= chase.pieceId;

            bool paired = false;
            for (int j = 0; j < side.actions; ++j)
                if (side.records[j].action == SKYRULE_NONE
                    && reverses(chase.move, side.records[j].move))
                {
                    paired = true;
                    break;
                }
            if (!paired)
                return false;
        }

        return chaseCount == side.chaseMoves && chasePieces
            && (chasePieces & (chasePieces - 1));
    };

    // A check-bounded paired split chase is deferred at root, but remains the
    // losing move once the search is already inside the repeated cycle.
    if (paired_split_chase_against_bounding_check(us))
    {
        if (ply > 0)
            return mated_in(ply);
        return VALUE_NONE;
    }

    auto indirect_king_reversal_chase = [&](Color c) {
        const ForcingSide& side = forcing[c];
        if (!ordinary_chase(c) || side.chaseMoves != 2)
            return false;

        const ActionRecord* chases[2] = {nullptr, nullptr};
        int count = 0;
        for (int i = 0; i < side.actions && count < 2; ++i)
            if (side.records[i].action == SKYRULE_CHASE)
                chases[count++] = &side.records[i];

        return count == 2
            && chases[0]->type == KING && chases[1]->type == KING
            && !chases[0]->direct && !chases[1]->direct
            && chases[0]->escapedCheck != chases[1]->escapedCheck
            && reverses(chases[0]->move, chases[1]->move);
    };

    // An indirect king-only chase does not outrank a four-ply rook
    // check-retreat that closes on its quiet return.
    for (Color chaseSide : {WHITE, BLACK})
    {
        const Color checkSide = ~chaseSide;
        const ActionRecord* check = first_action(checkSide, SKYRULE_CHECK, ROOK);
        const ActionRecord* quiet = first_action(checkSide, SKYRULE_NONE, ROOK);
        if (d == 4
            && indirect_king_reversal_chase(chaseSide)
            && forcing[checkSide].checkRetreatCycle
            && !forcing[checkSide].hasChase
            && forcing[checkSide].checkMoves == 1
            && forcing[checkSide].quietMoves == 1
            && check && quiet
            && st->move == quiet->move)
            return VALUE_DRAW;
    }

    // An indirect chase can be answered by the chased target when the same
    // stationary attacker remains in the relation on both legs of a
    // four-ply cycle. Duffish treats this narrow shape as mutual pursuit,
    // even though the answering side only makes a quiet rook reversal. Keep
    // the test identity- and action-based so unrelated quiet moves cannot
    // manufacture a draw.
    auto indirect_counter_chase_cycle = [&](Color c) {
        const ForcingSide& side  = forcing[c];
        const ForcingSide& reply = forcing[~c];
        if (d != 4 || !side.allForcing || side.hasCheck || !side.hasChase
            || !side.chase || side.chase == UINT32_MAX || side.chaseMoves != 2
            || side.persistentCounterChasers == UINT32_MAX
            || !side.persistentCounterChasers || reply.hasCheck || reply.hasChase
            || !quiet_reversal(~c, ROOK))
            return false;

        const ActionRecord* chases[2] = {nullptr, nullptr};
        int count = 0;
        for (int i = 0; i < side.actions && count < 2; ++i)
            if (side.records[i].action == SKYRULE_CHASE)
                chases[count++] = &side.records[i];

        return count == 2 && !chases[0]->direct && !chases[1]->direct
            && reverses(chases[0]->move, chases[1]->move);
    };

    for (Color c : {WHITE, BLACK})
        if (indirect_counter_chase_cycle(c))
            return draw_result();

    // A target that counter-chased immediately before the repetition anchor
    // can answer an unchanged indirect attacker throughout a longer cycle.
    // All IDs below are stable V8 identities (id_bit() performs the required
    // one-based to bit-zero-based conversion).
    auto anchor_target_counter_chase_cycle = [&](Color c) {
        const Color counter = ~c;
        const int states = d / 2;
        StateInfo* anchor = rollback.state();
        if (d < 4 || !ordinary_chase(c)
            || forcing[c].chaseMoves != states || forcing[c].quietMoves
            || forcing[counter].hasCheck || forcing[counter].hasChase
            || forcing[counter].quietMoves != states || !anchor
            || !anchor->move.is_ok() || anchor->capturedPiece
            || anchor->skyruleAction != SKYRULE_CHASE)
            return false;

        const Square anchorTo = anchor->move.to_sq();
        const int anchorTargetId = is_ok(anchorTo) ? rollback.idBoard[anchorTo] : 0;
        const uint32_t anchorTarget = id_bit(anchorTargetId);
        if (!anchorTarget || !(forcing[c].chase & anchorTarget)
            || (forcing[c].movedTargetIds & anchorTarget))
            return false;

        uint32_t commonChasers = UINT32_MAX;
        int chaseRecords = 0;
        for (int i = 0; i < forcing[c].actions; ++i)
        {
            const ActionRecord& action = forcing[c].records[i];
            if (action.action != SKYRULE_CHASE)
                continue;
            ++chaseRecords;
            if (action.direct || !action.chasers || (action.chasers & action.pieceId))
                return false;
            commonChasers &= action.chasers;
        }
        if (chaseRecords != states || !commonChasers)
            return false;

        const uint32_t anchorDirectChase =
          skyrule_chased_ids(anchor->skyruleRawChased
                               & skyrule_direct_chased(rollback), rollback.idBoard);
        uint32_t counterTargets = forcing[c].persistentCounterChasers
                                & anchorDirectChase
                                & forcing[counter].persistentTargets;
        while (counterTargets)
        {
            int counterTargetId = 0;
            while (!(counterTargets & (uint32_t(1) << counterTargetId)))
                ++counterTargetId;
            if (forcing[counter].persistentChasers[counterTargetId] & anchorTarget)
                return true;
            counterTargets &= counterTargets - 1;
        }
        return false;
    };

    for (Color c : {WHITE, BLACK})
        if (anchor_target_counter_chase_cycle(c))
            return draw_result();

    // A cannon mixed cycle which relies on a newly pinned same-type recapture
    // keeps the checking side's responsibility when Black closes the cycle
    // with a king reversal.  This branch is independent of palace geometry;
    // V8 legality has already established the pin and king escape.
    if (us == BLACK && ply > 0 && d == 4
        && forcing[WHITE].allForcing
        && forcing[WHITE].hasCheck && forcing[WHITE].hasChase
        && forcing[WHITE].checkChaseRetreatCycle
        && forcing[WHITE].checkMoves == 1 && forcing[WHITE].chaseMoves == 1
        && first_action(WHITE, SKYRULE_CHECK, CANNON)
        && first_action(WHITE, SKYRULE_CHASE, CANNON)
        && first_action(WHITE, SKYRULE_CHASE, CANNON)->chaseMoveUsesPinnedSameType
        && !forcing[BLACK].hasCheck && !forcing[BLACK].hasChase
        && quiet_reversal(BLACK, KING))
        return mate_in(ply - 1);

    // If White checks with the same piece on both legs while Black gives one
    // check and retreats, the long checking side remains responsible.  This
    // is the mixed-check analogue of the persistent-check limit.
    if (d == 4 && forcing[WHITE].allForcing
        && forcing[WHITE].hasCheck && !forcing[WHITE].hasChase
        && forcing[WHITE].checkMoves == 2 && forcing[WHITE].quietMoves == 0)
    {
        const ActionRecord* checks[2] = { nullptr, nullptr };
        int count = 0;
        for (int i = 0; i < forcing[WHITE].actions && count < 2; ++i)
            if (forcing[WHITE].records[i].action == SKYRULE_CHECK)
                checks[count++] = &forcing[WHITE].records[i];
        const ActionRecord* blackCheck = first_action(BLACK, SKYRULE_CHECK,
                                                       NO_PIECE_TYPE);
        const ActionRecord* blackQuiet = first_action(BLACK, SKYRULE_NONE,
                                                       NO_PIECE_TYPE);
        if (count == 2 && checks[0]->type == checks[1]->type
            && reverses(checks[0]->move, checks[1]->move)
            && forcing[BLACK].checkRetreatCycle && !forcing[BLACK].hasChase
            && forcing[BLACK].checkMoves == 1 && forcing[BLACK].quietMoves == 1
            && blackCheck && blackQuiet && blackCheck->type == blackQuiet->type)
        {
            if (ply == 0)
                return VALUE_NONE;
            return us == BLACK ? mate_in(ply - 1) : mated_in(ply);
        }
    }

    // A moving screen can alternate a pure chase with a chase-retreat.  When
    // the actual stable attackers target each other, neither side has a
    // unilateral forcing sequence.  V8's BISHOP attack table is used as-is;
    // no orthodox river restriction is applied here.
    auto effective_chase_retreat_cycle = [&](Color c) {
        return forcing[c].chaseRetreatCycle
            && !(forcing[c].hasChase && forcing[c].allChasesEscapedCheck);
    };

    auto escaped_check_chase_reply = [&](Color c) {
        return forcing[c].hasChase && forcing[c].allChasesEscapedCheck
            && !effective_chase_retreat_cycle(c);
    };

    // A mutual chase-retreat cycle is a pending draw at a fresh search root.
    // Let search try a breaker before converting the repeated root; an
    // internal node still reaches the ordinary mutual-draw handling below.
    if (ply == 0 && !forceUs && !forceThem
        && effective_chase_retreat_cycle(WHITE)
        && effective_chase_retreat_cycle(BLACK))
        return VALUE_NONE;

    // The repeated-position anchor may itself contain a capture.  If the
    // captured target then directly counter-chases the moving member of an
    // otherwise alternating chase and retreats, both threats are mutual in
    // Duffish.  Keep the test identity-based so a later quiet move cannot
    // manufacture a draw from an unrelated capture.
    auto repetition_anchor_capturer_id = [&] {
        const StateInfo* anchorState = rollback.state();
        if (!anchorState || anchorState->capturedPiece == NO_PIECE
            || !anchorState->move.is_ok())
            return uint32_t(0);

        const Square to = anchorState->move.to_sq();
        return is_ok(to) ? id_bit(rollback.idBoard[to]) : uint32_t(0);
    };

    auto chased_target_counter_chase_retreat = [&](Color pure) {
        const Color reply = ~pure;
        const ForcingSide& pureSide = forcing[pure];
        const ForcingSide& replySide = forcing[reply];
        const ActionRecord* replyChase = first_action(reply, SKYRULE_CHASE,
                                                        ROOK);
        const ActionRecord* replyQuiet = first_action(reply, SKYRULE_NONE,
                                                        NO_PIECE_TYPE);
        if (!ordinary_chase(pure)
            || !effective_chase_retreat_cycle(reply)
            || replySide.chaseMoves != 1 || replySide.quietMoves != 1
            || !replyChase || !replyQuiet
            || !replyChase->direct || !replyChase->pieceId
            || !(repetition_anchor_capturer_id() & replyChase->pieceId)
            || replyChase->pieceId != replyQuiet->pieceId)
            return false;

        uint32_t pureMovers = 0;
        for (int i = 0; i < pureSide.actions; ++i)
            if (pureSide.records[i].action == SKYRULE_CHASE)
                pureMovers |= pureSide.records[i].pieceId;

        return bool(pureSide.chase & replyChase->pieceId)
            && bool(replySide.chase & pureMovers);
    };

    for (Color pure : {WHITE, BLACK})
        if (chased_target_counter_chase_retreat(pure))
            return draw_result();

    auto reciprocal_actual_chase_retreat = [&](Color pure) {
        const Color reply = ~pure;
        if (!ordinary_chase(pure) || forcing[pure].hasCheck || forcing[reply].hasCheck
            || !effective_chase_retreat_cycle(reply) || forcing[reply].allForcing)
            return false;

        uint32_t pureChasers = 0;
        for (int i = 0; i < forcing[pure].actions; ++i)
        {
            const ActionRecord& action = forcing[pure].records[i];
            if (action.action == SKYRULE_CHASE)
                pureChasers |= action.chasers ? action.chasers : action.pieceId;
        }

        uint32_t replyChasers = 0;
        int replyChaseCount = 0;
        for (int i = 0; i < forcing[reply].actions; ++i)
        {
            const ActionRecord& action = forcing[reply].records[i];
            if (action.action != SKYRULE_CHASE)
                continue;
            ++replyChaseCount;
            if (action.type != BISHOP || !action.chasers
                || (action.chasers & action.pieceId))
                return false;
            replyChasers |= action.chasers;
        }

        return replyChaseCount > 0 && pureChasers && replyChasers
            && (forcing[reply].chase & pureChasers) == pureChasers
            && (forcing[pure].chase & replyChasers) == replyChasers;
    };

    for (Color c : {WHITE, BLACK})
        if (reciprocal_actual_chase_retreat(c))
            return draw_result();

    auto alternating_responder_piece_id = [&](Color c) {
        if (d != 4)
            return uint32_t(0);

        const ForcingSide& side = forcing[c];
        if (side.checkChaseRetreatCycle && side.checkMoves == 1
            && side.chaseMoves == 1)
        {
            const ActionRecord* check = first_action(c, SKYRULE_CHECK, NO_PIECE_TYPE);
            const ActionRecord* chase = first_action(c, SKYRULE_CHASE, NO_PIECE_TYPE);
            if (check && chase && check->pieceId && check->pieceId == chase->pieceId)
                return check->pieceId;
        }

        if (effective_chase_retreat_cycle(c) && side.chaseMoves == 1
            && side.quietMoves == 1)
        {
            const ActionRecord* chase = first_action(c, SKYRULE_CHASE, NO_PIECE_TYPE);
            const ActionRecord* quiet = first_action(c, SKYRULE_NONE, NO_PIECE_TYPE);
            if (chase && quiet && chase->pieceId
                && chase->pieceId == quiet->pieceId
                && chase->type == quiet->type
                && reverses(chase->move, quiet->move))
                return chase->pieceId;
        }

        return uint32_t(0);
    };

    auto pure_chase_targets_alternating_responder = [&](Color pure) {
        const Color responder = ~pure;
        const uint32_t responderId = alternating_responder_piece_id(responder);
        return responderId && forcing[responder].checkChaseRetreatCycle
            && ordinary_chase(pure) && bool(forcing[pure].chase & responderId);
    };

    auto recent_repeated_check = [&](Color c) {
        // V8 StateInfo does not carry Duffish's auxiliary repetition bit. A
        // consecutive check action in the short anchor window is the
        // equivalent signal for the cannon priority cases.
        StateInfo* cursor = st;
        Color mover = ~us;
        for (int i = 0; i < 3 && cursor; ++i, cursor = cursor->previous,
                                      mover = ~mover)
            if (mover == c && cursor->skyruleAction == SKYRULE_CHECK)
                return true;
        return false;
    };

    auto latest_pure_chase_escaped_check = [&](Color pure) {
        for (int i = 0; i < forcing[pure].actions; ++i)
            if (forcing[pure].records[i].action == SKYRULE_CHASE
                && forcing[pure].records[i].move == st->move
                && forcing[pure].records[i].escapedCheck)
                return true;
        return false;
    };

    auto pure_chase_escaped_check = [&](Color pure) {
        for (int i = 0; i < forcing[pure].actions; ++i)
            if (forcing[pure].records[i].action == SKYRULE_CHASE
                && forcing[pure].records[i].escapedCheck)
                return true;
        return false;
    };

    auto persistent_pure_chase_uses_different_chasers = [&](Color pure) {
        const ForcingSide& side = forcing[pure];
        if (side.chaseMoves < 2 || st->skyruleAction != SKYRULE_CHASE
            || st->skyruleCount <= d / 2)
            return false;

        uint32_t commonChasers = UINT32_MAX;
        for (int i = 0; i < side.actions; ++i)
            if (side.records[i].action == SKYRULE_CHASE)
                commonChasers &= side.records[i].chasers
                               ? side.records[i].chasers
                               : side.records[i].pieceId;
        return !commonChasers;
    };

    auto pure_chase_escapes_mixed_check = [&](Color pure) {
        const Color responder = ~pure;
        const uint32_t responderId = alternating_responder_piece_id(responder);
        const ActionRecord* check = first_action(responder, SKYRULE_CHECK,
                                                  NO_PIECE_TYPE);
        const ActionRecord* chase = first_action(responder, SKYRULE_CHASE,
                                                  NO_PIECE_TYPE);
        if (!responderId || !check || !chase
            || !forcing[responder].checkChaseRetreatCycle
            || !(forcing[pure].chase & responderId))
            return false;

        const bool ordinaryCannonCheck = check->type == CANNON
                                       && chase->type == CANNON
                                       && !chase->chaseMoveTargetsRook;
        const bool horseCheck = check->type == KNIGHT && chase->type == KNIGHT
                             && ply == 1;
        return (ordinaryCannonCheck || horseCheck)
            && latest_pure_chase_escaped_check(pure);
    };

    auto repetition_anchor_check_escape_kind = [&] {
        StateInfo* anchor = st;
        for (int i = 0; i < d && anchor; ++i)
            anchor = anchor->previous;

        auto check_escape_kind = [](StateInfo* escape) {
            if (!escape || !escape->previous
                || escape->previous->skyruleAction != SKYRULE_CHECK)
                return 0;
            if (escape->skyruleAction == SKYRULE_NONE)
                return 1;

            StateInfo* reply = escape->previous->previous;
            StateInfo* priorCheck = reply ? reply->previous : nullptr;
            return escape->skyruleAction == SKYRULE_CHASE && reply
                && reply->skyruleAction == SKYRULE_NONE && priorCheck
                && priorCheck->skyruleAction == SKYRULE_CHECK ? 2 : 0;
        };

        StateInfo* previous = anchor ? anchor->previous : nullptr;
        int kind = check_escape_kind(anchor) | check_escape_kind(previous);
        StateInfo* claimed = previous ? previous->previous : nullptr;
        return kind | (check_escape_kind(claimed) & 2);
    };

    auto direct_mixed_outranks_pure_chase = [&](Color pure) {
        const Color responder = ~pure;
        const uint32_t responderId = alternating_responder_piece_id(responder);
        const ActionRecord* check = first_action(responder, SKYRULE_CHECK,
                                                  NO_PIECE_TYPE);
        const ActionRecord* chase = first_action(responder, SKYRULE_CHASE,
                                                  NO_PIECE_TYPE);
        if (!ordinary_chase(pure) || !responderId || !check || !chase
            || !forcing[responder].checkChaseRetreatCycle
            || !check->direct || !chase->direct)
            return false;

        if (pure_chase_escapes_mixed_check(pure))
            return false;

        const bool pureClosesNewCycle = ~us == pure
                                     && (!st->previous
                                         || st->previous->skyruleAction != SKYRULE_CHECK);
        auto actual_pure_chaser = [&](Color sideColor, Color retreatColor) {
            uint32_t pureChasers = 0;
            for (int i = 0; i < forcing[sideColor].actions; ++i)
            {
                const ActionRecord& action = forcing[sideColor].records[i];
                if (action.action == SKYRULE_CHASE)
                    pureChasers |= action.chasers ? action.chasers : action.pieceId;
            }
            return pureChasers && bool(forcing[retreatColor].chase & pureChasers);
        };

        if ((forcing[pure].chase & responderId)
            && (actual_pure_chaser(pure, responder)
                || responder == WHITE
                || (check->type == KNIGHT
                    && !(ply == 1 && latest_pure_chase_escaped_check(pure)))
                || (check->type == CANNON
                    && (!chase->chaseMoveTargetsRook
                        || pureClosesNewCycle
                        || (~us == responder && st->previous
                            && st->previous->skyruleAction == SKYRULE_CHASE
                            && st->previous->skyruleCount <= d / 2
                            && pure_chase_escaped_check(pure))
                        || recent_repeated_check(responder)
                        || (st->previous
                            && st->previous->skyruleAction == SKYRULE_CHECK)))))
            return true;

        for (int i = 0; i < forcing[pure].actions; ++i)
            if (forcing[pure].records[i].action == SKYRULE_CHASE
                && (forcing[pure].records[i].type != KING
                    || forcing[pure].records[i].direct))
                return false;
        return forcing[pure].chaseMoves > 0;
    };

    // Direct mixed cycles take precedence over a pure chase of the same
    // responder.  Preserve Duffish's root delay and mate-distance offset.
    for (Color pure : {WHITE, BLACK})
    {
        const bool targetsResponder = pure_chase_targets_alternating_responder(pure);
        const bool directPriority = direct_mixed_outranks_pure_chase(pure);
        const bool immediatePriority = pure_chase_escapes_mixed_check(pure);
        if (targetsResponder || directPriority)
        {
            const Color responder = ~pure;
            const int anchorEscape = repetition_anchor_check_escape_kind();
            const bool multiPiece = persistent_pure_chase_uses_different_chasers(pure);
            const Color violator = (directPriority && !(anchorEscape & 2)
                                    && !multiPiece)
                                 || (immediatePriority && !anchorEscape)
                                   ? responder
                                   : pure;
            if (ply > 0)
                return us == violator ? mated_in(ply)
                                      : mate_in(ply - int(directPriority
                                                          || immediatePriority));
            return VALUE_NONE;
        }
    }

    auto retreat_chases_actual_pure_chaser = [&](Color pure, Color retreat) {
        uint32_t pureChasers = 0;
        const ForcingSide& pureSide = forcing[pure];
        for (int i = 0; i < pureSide.actions; ++i)
        {
            const ActionRecord& action = pureSide.records[i];
            if (action.action != SKYRULE_CHASE)
                continue;

            const uint32_t actual = action.chasers;
            const bool multiple = actual && (actual & (actual - 1));
            if (action.direct && multiple)
                pureChasers |= actual;
            else if (!action.direct && actual
                     && (action.type == PAWN
                         || !pureSide.persistentCounterChasers))
                pureChasers |= actual;
            else
                pureChasers |= action.pieceId;
        }
        return pureChasers && bool(forcing[retreat].chase & pureChasers);
    };

    // For direct mixed cycles Duffish distinguishes the physical attacker
    // from the piece which moved the screen.  A direct move with multiple
    // actual attackers keeps all attackers responsible; an indirect move
    // keeps the actual attackers except for a counter-chased non-pawn, where
    // the moving identity is the conservative fallback.
    auto retreat_chases_pure_chaser = [&](Color pure, Color retreat) {
        uint32_t pureChasers = 0;
        const ForcingSide& pureSide = forcing[pure];
        for (int i = 0; i < pureSide.actions; ++i)
        {
            const ActionRecord& action = pureSide.records[i];
            if (action.action != SKYRULE_CHASE)
                continue;

            const uint32_t actual = action.chasers;
            const bool multiple = actual && (actual & (actual - 1));
            if (action.direct && multiple)
                pureChasers |= actual;
            else if (!action.direct && actual
                     && (action.type == PAWN || action.type == ADVISOR
                         || pureSide.persistentCounterChasers == 0))
                pureChasers |= actual;
            else
                pureChasers |= action.pieceId;
        }
        return pureChasers && bool(forcing[retreat].chase & pureChasers);
    };

    // An unrelated chase followed by a retreat does not neutralize the
    // opponent's continuous pure chase.  The pawn exception mirrors Duffish:
    // a pawn retreat is treated as a legal escape even when its geometry also
    // touches the pure chaser.
    auto independent_chase_retreat_reply = [&](Color pure) {
        const Color reply = ~pure;
        const ForcingSide& side = forcing[pure];
        const ForcingSide& response = forcing[reply];
        bool allChasesDirect = side.chaseMoves > 0;
        for (int i = 0; allChasesDirect && i < side.actions; ++i)
            if (side.records[i].action == SKYRULE_CHASE)
                allChasesDirect &= side.records[i].direct;

        const ActionRecord* responseChase = nullptr;
        for (int i = 0; i < response.actions; ++i)
            if (response.records[i].action == SKYRULE_CHASE)
                responseChase = &response.records[i];

        return d == 4 && ordinary_chase(pure)
            && (side.allChasesSingleTarget || allChasesDirect || persistent_chase(pure))
            && effective_chase_retreat_cycle(reply) && !response.allForcing
            && response.chaseMoves == 1 && response.quietMoves == 1
            && responseChase
            && (responseChase->type == PAWN
                || !retreat_chases_actual_pure_chaser(pure, reply));
    };

    // A split counter-chase is mutual only when it continuously covers every
    // actual chaser of the pure side.  This is stricter than merely observing
    // one transient attack and avoids turning a one-ply counter-shot into a
    // draw claim.
    auto reciprocal_split_counter_chase = [&](Color pure) {
        const Color counter = ~pure;
        const ForcingSide& side = forcing[pure];
        const ForcingSide& reply = forcing[counter];
        if (d != 4 || !ordinary_chase(pure) || side.hasCheck || reply.hasCheck
            || !reply.allForcing || !reply.hasChase
            || reply.chaseMoves != d / 2 || reply.quietMoves
            || reply.persistentCounterChasers == UINT32_MAX
            || !reply.persistentCounterChasers)
            return false;

        uint32_t pureChasers = 0;
        for (int i = 0; i < side.actions; ++i)
        {
            const ActionRecord& action = side.records[i];
            if (action.action == SKYRULE_CHASE)
                pureChasers |= action.chasers ? action.chasers : action.pieceId;
        }
        return pureChasers && (reply.chase & pureChasers) == pureChasers;
    };

    for (Color c : {WHITE, BLACK})
        if (reciprocal_split_counter_chase(c))
            return draw_result();

    auto chase_retreat_reversal = [&](Color c, PieceType type) {
        const ForcingSide& side = forcing[c];
        if (!side.chaseRetreatCycle || side.chaseMoves != 1 || side.quietMoves != 1)
            return false;
        const ActionRecord* chase = first_action(c, SKYRULE_CHASE, type);
        const ActionRecord* quiet = first_action(c, SKYRULE_NONE, type);
        return chase && quiet && reverses(chase->move, quiet->move);
    };

    // A pawn reversal touching the advisor target is a concrete Duffish
    // exception.  Use V8's actual pawn attack table; no river predicate is
    // imposed on a revealed Jieqi pawn.
    if (!forceUs && !forceThem
        && chase_retreat_reversal(WHITE, ADVISOR)
        && quiet_reversal(BLACK, PAWN))
    {
        const ActionRecord* advisorQuiet = first_action(WHITE, SKYRULE_NONE,
                                                         ADVISOR);
        const ActionRecord* pawnQuiet = first_action(BLACK, SKYRULE_NONE, PAWN);
        if (advisorQuiet && pawnQuiet)
        {
            const Square advisorSq = advisorQuiet->move.to_sq();
            const bool attacksFrom = bool(attacks_bb<PAWN>(pawnQuiet->move.from_sq(), BLACK)
                                          & advisorSq);
            const bool attacksTo = bool(attacks_bb<PAWN>(pawnQuiet->move.to_sq(), BLACK)
                                        & advisorSq);
            if ((attacksFrom || attacksTo) && st->move == pawnQuiet->move)
                return ply > 0 ? (us == WHITE ? mated_in(ply) : mate_in(ply))
                               : VALUE_NONE;
        }
    }

    // A horse returning against an unrooted advisor is a concrete Duffish
    // exception.  Use V8's actual horse/advisor attack tables so this remains
    // valid for revealed Jieqi pieces outside orthodox palace/river regions.
    auto horse_return_attacks_unrooted_advisor = [&] {
        const ActionRecord* horse = first_action(WHITE, SKYRULE_NONE, KNIGHT);
        if (!horse)
            return false;

        const Square horseSq = horse->move.to_sq();
        Bitboard advisors = attacks_bb(KNIGHT, horseSq, pieces())
                          & pieces(BLACK, ADVISOR) & ~pieces(DARK);
        while (advisors)
        {
            const Square advisorSq = pop_lsb(advisors);
            if (!skyrule_has_true_root(*this, BLACK, advisorSq, horseSq,
                                       pieces(), horseSq))
                return true;
        }
        return false;
    };

    // A rooted cannon check/return is kept separate from an ordinary chase.
    // These flags are populated while walking the reversible suffix above;
    // requiring an otherwise non-forcing cycle keeps this exception narrow.
    auto cannon_rooted_retreat = [&](Color c) {
        const ForcingSide& side = forcing[c];
        const ActionRecord* check = first_action(c, SKYRULE_CHECK, CANNON);
        return !side.hasChase && side.checkRetreatCycle && side.checkMoves == 1
            && check && check->direct && side.rootedAttackMoveCount == 1
            && side.rootedAttackMoveTypes[0] == CANNON
            && reverses(side.rootedAttackMoves[0], check->move);
    };

    const ActionRecord* blackCheck = first_action(BLACK, SKYRULE_CHECK, CANNON);
    const ActionRecord* blackQuiet = first_action(BLACK, SKYRULE_NONE, CANNON);
    const bool bareBlackCannonCheckRetreat =
      !checkUs && !checkThem && !chaseUs && !chaseThem
      && d == 4 && forcing[BLACK].allForcing
      && forcing[BLACK].checkRetreatCycle && !forcing[BLACK].hasChase
      && forcing[BLACK].checkMoves == 1 && forcing[BLACK].quietMoves == 1
      && blackCheck && blackQuiet && blackCheck->direct
      && !blackCheck->checkMoveCanBeCaptured
      && forcing[BLACK].rootedAttackMoveCount == 0
      && forcing[BLACK].rootedAdvisorAttackUsesOpponentScreen
      && forcing[BLACK].rootedAdvisorHasOpponentScreenCannonRoot
      && !forcing[BLACK].rootedAdvisorHasStableRoot
      && quiet_reversal(WHITE, KING);

    if (bareBlackCannonCheckRetreat)
    {
        if (ply > 0)
            return us == BLACK ? mated_in(ply) : mate_in(ply - 1);
        return VALUE_NONE;
    }

    // A black mixed check/chase by one piece against a quiet royal/advisor
    // reversal is decisive.  A pinned-root cannon gets the same one-ply
    // distance adjustment as Duffish's rooted-cannon branch.
    if (us == WHITE && d == 4 && forcing[BLACK].allForcing
        && forcing[BLACK].hasCheck && forcing[BLACK].hasChase
        && forcing[BLACK].checkChaseRetreatCycle
        && forcing[BLACK].checkMoves == 1 && forcing[BLACK].chaseMoves == 1
        && !forcing[WHITE].hasCheck && !forcing[WHITE].hasChase
        && (quiet_reversal(WHITE, KING) || quiet_reversal(WHITE, ADVISOR)))
    {
        const ActionRecord* check = first_action(BLACK, SKYRULE_CHECK, NO_PIECE_TYPE);
        const ActionRecord* chase = first_action(BLACK, SKYRULE_CHASE, NO_PIECE_TYPE);
        if (check && chase && check->type == chase->type)
        {
            const int pinnedAdjustment = chase->chaseMovePinsRoot ? 1 : 0;
            if (ply > 0)
                return mate_in(ply - pinnedAdjustment);
            return VALUE_NONE;
        }
    }

    if (!checkUs && !checkThem && !chaseUs && !chaseThem
        && us == BLACK && d == 4
        && quiet_reversal(BLACK, ADVISOR)
        && quiet_reversal(WHITE, KNIGHT)
        && horse_return_attacks_unrooted_advisor())
    {
        if (ply > 0)
            return mated_in(ply);
        return VALUE_NONE;
    }

    if (ply == 0 && d == 4 && cannon_rooted_retreat(WHITE)
        && forcing[WHITE].rootedAttackHasShieldedHorse
        && quiet_reversal(BLACK, KING))
        return VALUE_NONE;

    for (Color c : {WHITE, BLACK})
        if (ply == 0 && d == 4 && cannon_rooted_retreat(c)
            && !forcing[c].rootedAttackHasNonPawn
            && quiet_reversal(~c, KING))
            return VALUE_NONE;

    auto check_move_escaped_check = [&](Color c) {
        const ActionRecord* check = first_action(c, SKYRULE_CHECK, NO_PIECE_TYPE);
        return check && check->escapedCheck;
    };

    // A discovered white pawn check answering a rook check does not transfer
    // responsibility when Black's direct rook retreat closes the cycle.
    if (us == WHITE && d == 4 && checkUs != checkThem
        && forcing[WHITE].checkRetreatCycle && forcing[BLACK].checkRetreatCycle
        && forcing[WHITE].checkMoves == 1 && forcing[BLACK].checkMoves == 1
        && first_action(WHITE, SKYRULE_CHECK, PAWN)
        && first_action(BLACK, SKYRULE_CHECK, ROOK)
        && first_action(BLACK, SKYRULE_NONE, ROOK)
        && !first_action(WHITE, SKYRULE_CHECK, PAWN)->direct
        && check_move_escaped_check(WHITE)
        && first_action(BLACK, SKYRULE_CHECK, ROOK)->direct
        && st->move == first_action(BLACK, SKYRULE_NONE, ROOK)->move
        && !forcing[WHITE].hasChase && !forcing[BLACK].hasChase)
        return ply > 0 ? mated_in(ply) : VALUE_NONE;

    // Reciprocal check-retreats are mutual even when the red retreat
    // exemption makes only Black forcing.  At a new root this remains a
    // pending draw so search can choose a breaker.
    if (d == 4 && checkUs != checkThem
        && forcing[WHITE].checkRetreatCycle && forcing[BLACK].checkRetreatCycle
        && forcing[WHITE].hasCheck && forcing[BLACK].hasCheck
        && forcing[WHITE].checkMoves == 1 && forcing[BLACK].checkMoves == 1
        && (first_action(WHITE, SKYRULE_CHECK, NO_PIECE_TYPE)->direct
            || check_move_escaped_check(WHITE))
        && first_action(BLACK, SKYRULE_CHECK, NO_PIECE_TYPE)->direct
        && !forcing[WHITE].hasChase && !forcing[BLACK].hasChase)
        return ply > 0 ? VALUE_DRAW : VALUE_NONE;

    if (checkUs && checkThem && !forcing[WHITE].hasChase && !forcing[BLACK].hasChase)
        return VALUE_DRAW;

    for (Color c : {WHITE, BLACK})
        if (independent_chase_retreat_reply(c))
            return result_for(c);

    auto check_retreat_responder_piece_id = [&](Color c) {
        const ForcingSide& side = forcing[c];
        if (d != 4 || !side.checkRetreatCycle || side.hasChase
            || side.checkMoves != 1 || side.quietMoves != 1)
            return uint32_t(0);
        const ActionRecord* check = first_action(c, SKYRULE_CHECK, NO_PIECE_TYPE);
        const ActionRecord* quiet = first_action(c, SKYRULE_NONE, NO_PIECE_TYPE);
        if (!check || !quiet || !check->pieceId || check->pieceId != quiet->pieceId
            || !reverses(check->move, quiet->move))
            return uint32_t(0);
        return check->pieceId;
    };

    if (us == WHITE && !forceUs && forceThem && checkThem && !chaseThem
        && forcing[~us].checkRetreatCycle && !forcing[~us].hasChase
        && effective_chase_retreat_cycle(us) && !forcing[us].allForcing)
        return VALUE_DRAW;

    if (!forceUs && forceThem && effective_chase_retreat_cycle(us))
    {
        chaseUs = true;
        forceUs = true;
    }

    if (!forceThem && forceUs && effective_chase_retreat_cycle(~us)
        && !(forcing[us].allForcing && forcing[us].hasCheck && forcing[us].hasChase))
    {
        chaseThem = true;
        forceThem = true;
    }

    if (!forceUs && checkThem && !chaseThem && escaped_check_chase_reply(us)
        && forcing[~us].checkRetreatCycle
        && !checking_move_chases_target(us, check_retreat_responder_piece_id(~us)))
        return VALUE_DRAW;

    if (!forceThem && checkUs && !chaseUs && escaped_check_chase_reply(~us)
        && forcing[us].checkRetreatCycle && ply > 0)
        return VALUE_DRAW;

    // Mutual pure chases are a draw only after the search has entered the
    // repeated node.  At the root Duffish leaves the position searchable so
    // a breaker move can be selected instead of converting the repetition
    // immediately.
    if (ply == 0 && forceUs && forceThem && chaseUs && chaseThem
        && !checkUs && !checkThem)
        return VALUE_NONE;

    const uint32_t blackResponderId = check_retreat_responder_piece_id(BLACK);
    if (d == 4 && forcing[WHITE].allForcing
        && forcing[WHITE].hasCheck && forcing[WHITE].hasChase
        && forcing[WHITE].checkChaseRetreatCycle
        && forcing[WHITE].checkMoves == 1 && forcing[WHITE].chaseMoves == 1
        && forcing[BLACK].checkRetreatCycle && !forcing[BLACK].hasChase
        && forcing[BLACK].checkMoves == 1 && forcing[BLACK].quietMoves == 1)
    {
        const ActionRecord* whiteCheck = first_action(WHITE, SKYRULE_CHECK,
                                                       NO_PIECE_TYPE);
        const ActionRecord* whiteChase = first_action(WHITE, SKYRULE_CHASE,
                                                       NO_PIECE_TYPE);
        const ActionRecord* blackCheck = first_action(BLACK, SKYRULE_CHECK,
                                                       NO_PIECE_TYPE);
        const ActionRecord* blackQuiet = first_action(BLACK, SKYRULE_NONE,
                                                       NO_PIECE_TYPE);
        if (whiteCheck && whiteChase && blackCheck && blackQuiet
            && whiteCheck->type == whiteChase->type
            && blackCheck->type == blackQuiet->type)
        {
            const bool typeException = !whiteCheck->direct
                || ((whiteCheck->type == ROOK || whiteCheck->type == CANNON)
                    && (blackCheck->type == CANNON || blackCheck->type == BISHOP));
            const bool targetException = blackResponderId
                && ((forcing[WHITE].chase & blackResponderId)
                    || checking_move_chases_target(WHITE, blackResponderId));
            if (typeException || targetException)
            {
                if (ply > 0)
                    return us == WHITE ? mated_in(ply) : mate_in(ply - 1);
                return VALUE_NONE;
            }
        }
    }

    // The remaining mixed-force matrix is intentionally asymmetric in
    // Duffish.  Keep these checks ahead of the generic cross-target draw
    // logic so an escape-created chase cannot accidentally equal a genuine
    // persistent forcing sequence.
    if (forceUs && forceThem)
    {
        StateInfo* previousUs = st ? st->previous : nullptr;
        const int usActionCount = previousUs ? previousUs->skyruleCount : 0;

        if (checkUs && !chaseUs && checkThem && !chaseThem
            && forcing[us].checkChaseRetreatCycle
            && forcing[~us].checkChaseRetreatCycle
            && escaped_check_chase_reply(us) != escaped_check_chase_reply(~us))
        {
            const Color violator = escaped_check_chase_reply(us) ? ~us : us;
            return result_for(violator);
        }

        if (us == BLACK && !checkUs && chaseUs && checkThem && !chaseThem
            && previousUs && previousUs->skyruleAction == SKYRULE_CHASE
            && (!forcing[~us].hasChase || (usActionCount & 1))
            && usActionCount < d / 2 + 2)
            return VALUE_NONE;

        if (us == BLACK && checkUs && !chaseUs && !checkThem && chaseThem
            && !forcing[us].hasChase && effective_chase_retreat_cycle(~us)
            && !forcing[~us].allForcing)
            return VALUE_DRAW;

        if (checkUs && !chaseUs && !checkThem && chaseThem
            && forcing[us].hasChase && forcing[us].checkChaseRetreatCycle
            && effective_chase_retreat_cycle(~us)
            && !forcing[~us].allForcing)
            return ply > 0 ? mated_in(ply) : VALUE_NONE;

        if (checkUs && !chaseUs && !checkThem && chaseThem
            && escaped_check_chase_reply(~us) && !forcing[us].hasChase)
            return ply > 0 ? mated_in(ply) : VALUE_NONE;

        if (!checkUs && chaseUs && checkThem && !chaseThem
            && escaped_check_chase_reply(us) && !forcing[~us].hasChase)
            return mate_in(ply);

        if (us == BLACK && checkUs && !chaseUs && !checkThem && chaseThem
            && !forcing[us].hasChase)
            return ply > 0 ? mate_in(ply) : VALUE_NONE;

        if (us == WHITE && !checkUs && chaseUs && checkThem && !chaseThem
            && !forcing[us].allChasesEscapedCheck
            && !forcing[~us].hasChase)
            return ply > 0 ? mated_in(ply) : VALUE_NONE;

        if (!checkUs && chaseUs && !checkThem && chaseThem
            && !effective_chase_retreat_cycle(us)
            && effective_chase_retreat_cycle(~us)
            && retreat_chases_pure_chaser(us, ~us)
            && !forcing[~us].allForcing)
            return ply > 0 ? mated_in(ply) : VALUE_NONE;

        if (!checkUs && chaseUs && !checkThem && chaseThem
            && effective_chase_retreat_cycle(us)
            && !effective_chase_retreat_cycle(~us)
            && retreat_chases_pure_chaser(~us, us)
            && !forcing[us].allForcing)
            return ply > 0 ? mate_in(ply - 1) : VALUE_NONE;

        if (us == WHITE && checkUs && !chaseUs && !checkThem && chaseThem
            && forcing[WHITE].hasChase
            && forcing[WHITE].checkChaseRetreatCycle
            && forcing[BLACK].chaseMoves == 2)
        {
            const ActionRecord* firstBlackChase = first_action(BLACK, SKYRULE_CHASE,
                                                                NO_PIECE_TYPE);
            const ActionRecord* secondBlackChase = nullptr;
            bool seen = false;
            for (int i = 0; i < forcing[BLACK].actions; ++i)
                if (forcing[BLACK].records[i].action == SKYRULE_CHASE)
                {
                    if (seen)
                    {
                        secondBlackChase = &forcing[BLACK].records[i];
                        break;
                    }
                    seen = true;
                }
            if (firstBlackChase && secondBlackChase
                && firstBlackChase->type == ADVISOR
                && secondBlackChase->type == ADVISOR)
                return ply > 0 ? mated_in(ply) : VALUE_NONE;
        }

        if (us == WHITE && checkUs && !chaseUs && !checkThem && chaseThem
            && (!forcing[us].hasChase
                || (previousUs && previousUs->skyruleAction == SKYRULE_CHECK)))
        {
            if (ply > 0 && !forcing[us].hasChase)
                return mated_in(ply);
            return VALUE_NONE;
        }

        return (checkUs || checkThem) ? mate_in(ply) : VALUE_DRAW;
    }

    // A persistent chase which is defused at the repeated root is a draw (or
    // a root deferral), not an immediate win for the nominal chaser.
    if (!forceUs && !forceThem
        && ((persistentUs && !activePersistentUs)
            || (persistentThem && !activePersistentThem)))
        return ply > 0 ? VALUE_DRAW : VALUE_NONE;

    // Mutual perpetual checking is neutral.  Mutual chase requires the stable
    // target/chaser identities to cross in both directions, or both sides to
    // have a target that legally counter-attacks their opponent's chaser.  Two
    // unrelated chases are not silently converted into a draw.
    const bool crossUs = chaseUs && chaseThem
                      && bool(forcing[us].persistentTargets & forcing[them].anyChasers);
    const bool crossThem = chaseUs && chaseThem
                        && bool(forcing[them].persistentTargets & forcing[us].anyChasers);
    const bool reciprocalCounter = chaseUs && chaseThem
                                 && forcing[us].persistentCounterChasers != UINT32_MAX
                                 && forcing[them].persistentCounterChasers != UINT32_MAX
                                 && bool(forcing[us].persistentCounterChasers
                                         & forcing[us].anyChasers)
                                 && bool(forcing[them].persistentCounterChasers
                                         & forcing[them].anyChasers);

    if ((checkUs && checkThem) || (crossUs && crossThem) || reciprocalCounter
        || (mixedUs && mixedThem))
        return draw_result();

    // A mixed check/chase cycle is neutral only when the chase is aimed at the
    // opponent's stable checker. Otherwise the side supplying the chase is the
    // SkyRule violator, even if the other side also gives checks.
    const bool chaseUsChecksThem = chaseUs
                                && bool(forcing[us].persistentTargets & forcing[them].checkers);
    const bool chaseThemChecksUs = chaseThem
                                && bool(forcing[them].persistentTargets & forcing[us].checkers);
    if ((chaseUs && (checkThem || mixedThem)) && chaseUsChecksThem)
        return draw_result();
    if ((chaseThem && (checkUs || mixedUs)) && chaseThemChecksUs)
        return draw_result();

    if (chaseUs && chaseThem)
    {
        // If only one side's target set reaches the opponent's actual chaser,
        // the other side's chase is the effective counter-pursuit and the
        // original persistent chaser remains responsible.
        if (crossUs && !crossThem)
            return result_for(them);
        if (crossThem && !crossUs)
            return result_for(us);
        return draw_result();
    }

    if ((checkUs || mixedUs) && (checkThem || mixedThem))
    {
        if (mixedUs && !mixedThem)
            return result_for(us);
        if (mixedThem && !mixedUs)
            return result_for(them);
        return draw_result();
    }

    if (chaseUs && (checkThem || mixedThem))
        return result_for(us);
    if (chaseThem && (checkUs || mixedUs))
        return result_for(them);

    if (chaseUs || checkUs || mixedUs)
        return result_for(us);
    if (chaseThem || checkThem || mixedThem)
        return result_for(them);

    // A repeated root without a stable forcing suffix remains searchable. This
    // is important at ply zero, where the caller can still select a breaker.
    return ply == 0 ? VALUE_NONE : VALUE_DRAW;
}


// Tests whether the position may end the game by rule 60, insufficient material, draw repetition,
// perpetual check repetition or perpetual chase repetition that allows a player to claim a game result.
int Position::skyrule_repeat_count(int end) const {

    int count = 1;

    for (StateInfo* stp = st; end >= 2 && stp->previous && stp->previous->previous; end -= 2)
    {
        stp = stp->previous->previous;
        if (stp->key == st->key)
            ++count;
    }

    return count;
}


bool Position::skyrule_repeat_draw() const {
    // Compatibility wrapper for callers that used the former five-fold
    // shortcut. SkyRule repetitions are now adjudicated by rule_judge(),
    // including root VALUE_NONE deferral and forcing-side responsibility.
    Value result = VALUE_NONE;
    return const_cast<Position*>(this)->rule_judge(result, 0, CHASING_RULE_SKYRULE_JIEQI)
        && result == VALUE_DRAW;
}


bool Position::rule_judge(Value& result, int ply, ChasingRule rule) {

    if (rule == CHASING_RULE_SKYRULE_JIEQI && ply > 0)
    {
        if (skyrule_has_overlong_persistent_check(*this))
        {
            result = mate_in(ply - 1);
            return true;
        }

        if (st->move.is_ok())
        {
            Position previous;
            previous.copy_for_chasing_from(*this);
            previous.undo_move(previous.state()->move);

            if (skyrule_has_overlong_persistent_check(previous))
            {
                result = mated_in(ply);
                return true;
            }
        }
    }

    int end = std::min(st->rule40, st->pliesFromNull);

    if (end >= 4 && filter[st->key] >= 1)
    {
        int        cnt       = 0;
        StateInfo* stp       = st->previous->previous;
        bool       checkThem = st->checkersBB && stp->checkersBB;
        bool       checkUs   = st->previous->checkersBB && stp->previous->checkersBB;

        for (int i = 4; i <= end; i += 2)
        {
            stp = stp->previous->previous;
            checkThem &= bool(stp->checkersBB);

            // SkyRule uses the first reversible repetition as its anchor. At
            // the root the detector may return VALUE_NONE so search can try a
            // breaker; the legacy Github rule keeps its historical two-fold
            // root handling below.
            if (stp->key == st->key)
            {
                if (rule == CHASING_RULE_SKYRULE_JIEQI)
                {
                    // SkyRule repetitions are not ordinary five-fold draws.
                    // The detector may deliberately return VALUE_NONE at the
                    // root to let search find a move which breaks the cycle.
                    Position rollback;
                    rollback.copy_for_chasing_from(*this);
                    result = rollback.detect_chases(i, ply);
                    if (result == VALUE_NONE)
                        continue;
                    return true;
                }
                ++cnt;
                if (cnt != 2 && ply <= i)
                    continue;

                if (!checkThem && !checkUs)
                {
                    // Copy the current position to a rollback struct, so we don't need to do those moves again
                    Position rollback;
                    rollback.copy_for_chasing_from(*this);

                    // Chasing detection
                    result = rollback.detect_chases(i, ply);
                }
                else
                    // Checking detection
                    result = !checkUs ? mate_in(ply) : !checkThem ? mated_in(ply) : VALUE_DRAW;

                // 3 folds and 2 fold draws can be judged immediately for the
                // legacy Github rule. SkyRule was handled above.
                if (result == VALUE_DRAW || cnt == 2)
                    return true;

                // 2 fold mates need further investigations
                if (filter[st->key] <= 1)
                {
                    // Not exceeding rule 40 and have the same previous step
                    if (st->rule40 < 80 && st->previous->key == stp->previous->key)
                    {
                        // Even if we entering this loop again, it will not lead to a 3 fold repetition
                        StateInfo* prev = st->previous;
                        while ((prev = prev->previous) != stp)
                            if (filter[prev->key] > 1)
                                break;
                        if (prev == stp)
                            return true;
                    }
                    // We know there can't be another fold
                    break;
                }
            }

            if (i + 1 <= end)
                checkUs &= bool(stp->previous->checkersBB);
        }
    }

    // 40 move rule
    if (st->rule40 >= 80)
    {
        result = MoveList<LEGAL>(*this).size() ? VALUE_DRAW : mated_in(ply);
        return true;
    }

    return false;
}


// Flips position with the white and black sides reversed. This
// is only useful for debugging e.g. for finding evaluation symmetry bugs.
void Position::flip() {

    string            f, token;
    std::stringstream ss(fen());

    for (Rank r = RANK_9; r >= RANK_0; --r)  // Piece placement
    {
        std::getline(ss, token, r > RANK_0 ? '/' : ' ');
        f.insert(0, token + (f.empty() ? " " : "/"));
    }

    ss >> token;                        // Active color
    f += (token == "w" ? "B " : "W ");  // Will be lowercased later

    ss >> token;
    f += token + " ";

    std::transform(f.begin(), f.end(), f.begin(),
                   [](char c) { return char(islower(c) ? toupper(c) : tolower(c)); });

    ss >> token;
    f += token;

    std::getline(ss, token);  // Half and full moves
    f += token;

    set(f, st);

    assert(pos_is_ok());
}


// Performs some consistency checks for the position object
// and raise an assert if something wrong is detected.
// This is meant to be helpful when debugging.
bool Position::pos_is_ok() const {

    constexpr bool Fast = true;  // Quick (default) or full check?

    if ((sideToMove != WHITE && sideToMove != BLACK) || piece_on(king_square(WHITE)) != W_KING
        || piece_on(king_square(BLACK)) != B_KING)
        assert(0 && "pos_is_ok: Default");

    if (Fast)
        return true;

    if (pieceCount[W_KING] != 1 || pieceCount[B_KING] != 1
        || checkers_to(sideToMove, king_square(~sideToMove)))
        assert(0 && "pos_is_ok: Kings");

    if ((pieces(WHITE, PAWN) & ~PawnBB[WHITE]) || (pieces(BLACK, PAWN) & ~PawnBB[BLACK])
        || pieceCount[W_PAWN] > 5 || pieceCount[B_PAWN] > 5)
        assert(0 && "pos_is_ok: Pawns");

    if ((pieces(WHITE) & pieces(BLACK)) || (pieces(WHITE) | pieces(BLACK)) != pieces()
        || popcount(pieces(WHITE)) > 16 || popcount(pieces(BLACK)) > 16)
        assert(0 && "pos_is_ok: Bitboards");

    for (PieceType p1 = PAWN; p1 <= KING; ++p1)
        for (PieceType p2 = PAWN; p2 <= KING; ++p2)
            if (p1 != p2 && (pieces(p1) & pieces(p2)))
                assert(0 && "pos_is_ok: Bitboards");

    for (Piece pc : Pieces)
        if (pieceCount[pc] != popcount(pieces(color_of(pc), type_of(pc)))
            || pieceCount[pc] != std::count(board, board + SQUARE_NB, pc))
            assert(0 && "pos_is_ok: Pieces");

    return true;
}

}  // namespace Stockfish
