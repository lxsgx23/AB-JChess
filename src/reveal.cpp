/*
  AB-JChess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "reveal.h"

#include <algorithm>
#include <cassert>
#include <numeric>

#include "bitboard.h"
#include "position.h"

namespace Stockfish::Reveal {

std::int64_t round_divide(std::int64_t numerator, std::int64_t denominator) {
    assert(denominator > 0);

    const std::int64_t quotient  = numerator / denominator;
    const std::int64_t remainder = numerator % denominator;
    const std::int64_t magnitude = remainder < 0 ? -remainder : remainder;

    return quotient + (magnitude * 2 >= denominator ? (numerator < 0 ? -1 : 1) : 0);
}

Features make_features(const FeatureInputs& inputs) {
    const int phase = int(round_divide(std::int64_t(inputs.globalDark) * 1024, 30));

    const std::int64_t poolDenominator = std::int64_t(inputs.poolTotal) * (RookValue - PawnValue);
    const int          pool = poolDenominator > 0
                              ? int(round_divide(inputs.poolValueAbovePawn * 1024, poolDenominator))
                              : 0;

    const int unknownCount = std::clamp(inputs.poolTotal - inputs.ownBoardDark, 0, 15);
    const int unknown      = int(round_divide(std::int64_t(unknownCount) * 1024, 15));

    const int clampedEval = std::clamp(int(inputs.rawParentEval), -800, 800);
    const int comeback    = int(round_divide(std::int64_t(-clampedEval) * 1024, 800));

    return {phase, pool, unknown, comeback};
}

Features make_features(const Position& position, Value rawParentEval) {
    Position::RestPieceList restPieces;
    const int restPieceTypes = position.rest_pieces(position.side_to_move(), restPieces);

    int          poolTotal          = 0;
    std::int64_t poolValueAbovePawn = 0;
    for (int i = 0; i < restPieceTypes; ++i)
    {
        const auto [piece, count] = restPieces[i];
        poolTotal += count;
        poolValueAbovePawn += std::int64_t(count) * (PieceValue[piece] - PawnValue);
    }

    return make_features({popcount(position.pieces(DARK)), poolTotal,
                          popcount(position.pieces(position.side_to_move(), DARK)),
                          poolValueAbovePawn, rawParentEval});
}

QuietFeatures make_quiet_features(const QuietFeatureInputs& inputs) {
    const int phase = int(
      round_divide(std::int64_t(2 * inputs.globalDark - 30) * 1024, 30));

    const std::int64_t poolTotal =
      std::accumulate(inputs.poolCounts.begin(), inputs.poolCounts.end(), 0LL);
    int highValue = 0;
    int diversity = 0;
    if (poolTotal > 0)
    {
        const std::int64_t highCount = std::int64_t(inputs.poolCounts[0])
                                     + inputs.poolCounts[2] + inputs.poolCounts[4];
        const std::int64_t centeredHighValue = 5 * highCount - 2 * poolTotal;
        highValue = int(round_divide(centeredHighValue * 1024,
                                     (centeredHighValue >= 0 ? 3 : 2) * poolTotal));

        std::int64_t squaredCounts = 0;
        for (int count : inputs.poolCounts)
            squaredCounts += std::int64_t(count) * count;
        diversity = std::clamp(int(round_divide(
                                 (poolTotal * poolTotal - 5 * squaredCounts) * 256,
                                 poolTotal * poolTotal)),
                               -1024, 1024);
    }

    const int safety = inputs.safety == QuietSafety::Safe    ? 1024
                     : inputs.safety == QuietSafety::Loose   ? -1024
                                                             : 0;
    const int raw = int(round_divide(
      std::int64_t(-std::clamp(int(inputs.rawParentEval), -800, 800)) * 1024, 800));
    const int gate = inputs.safety == QuietSafety::Safe         ? 1024
                   : inputs.safety == QuietSafety::Contested    ? 512
                                                                 : 0;
    return {phase, safety, highValue, diversity,
            int(round_divide(std::int64_t(raw) * gate, 1024))};
}

QuietFeatures
make_quiet_features(const Position& position, QuietSafety safety, Value rawParentEval) {
    Position::RestPieceList restPieces;
    const int restPieceTypes = position.rest_pieces(position.side_to_move(), restPieces);

    std::array<int, 6> poolCounts{};
    for (int i = 0; i < restPieceTypes; ++i)
    {
        const auto [piece, count] = restPieces[i];
        switch (type_of(piece))
        {
        case ROOK: poolCounts[0] += count; break;
        case ADVISOR: poolCounts[1] += count; break;
        case CANNON: poolCounts[2] += count; break;
        case PAWN: poolCounts[3] += count; break;
        case KNIGHT: poolCounts[4] += count; break;
        case BISHOP: poolCounts[5] += count; break;
        default: assert(false); break;
        }
    }

    return make_quiet_features(
      {popcount(position.pieces(DARK)), poolCounts, safety, rawParentEval});
}

QuietSafety quiet_safety(const Position& pos, Move move) {
    assert(pos.move_dark(move));
    assert(!pos.capture(move));

    const Bitboard fromBB        = square_bb(move.from_sq());
    const Bitboard toBB          = square_bb(move.to_sq());
    const Bitboard occupiedAfter = (pos.pieces() ^ fromBB) | toBB;
    const Bitboard attackers =
      pos.attackers_to(move.to_sq(), occupiedAfter) & ~pos.pieces(DARK);
    const Bitboard enemy = attackers & pos.pieces(~pos.side_to_move());
    const Bitboard friendly =
      (attackers & pos.pieces(pos.side_to_move())) & ~fromBB;

    return !enemy ? QuietSafety::Safe
                  : friendly ? QuietSafety::Contested : QuietSafety::Loose;
}

QuietMoveContext quiet_move_context(const Position& pos, Move move, bool enabled) {
    if (!enabled || !pos.move_dark(move) || pos.capture(move))
        return {};

    return {true, quiet_safety(pos, move)};
}

bool qsearch_quiet_context_enabled(bool useReveal, bool inCheck, bool safetyNeeded) {
    return useReveal && inCheck && safetyNeeded;
}

Depth adjust_reduction(Depth             reduction,
                       bool              darkMove,
                       bool              safeQuiet,
                       const Parameters& parameters) {
    if (darkMove)
        reduction -= parameters.reduction;
    if (safeQuiet)
        reduction -= parameters.quietReduction;
    return reduction;
}

bool bonus_enabled(const Parameters& parameters) {
    return parameters.bonusBase != 0 || parameters.bonusPhase != 0 || parameters.bonusPool != 0
        || parameters.bonusUnknown != 0 || parameters.bonusComeback != 0;
}

int bonus(const Parameters& parameters, const Features& features) {
    if (!bonus_enabled(parameters))
        return 0;

    const std::int64_t weighted = std::int64_t(parameters.bonusPhase) * features.phase
                                + std::int64_t(parameters.bonusPool) * features.pool
                                + std::int64_t(parameters.bonusUnknown) * features.unknown
                                + std::int64_t(parameters.bonusComeback) * features.comeback;

    const std::int64_t result = std::int64_t(parameters.bonusBase) + round_divide(weighted, 1024);
    return int(std::clamp<std::int64_t>(result, -512, 512));
}

bool quiet_bonus_enabled(const Parameters& parameters) {
    return parameters.quietBase || parameters.quietPhase || parameters.quietSafety
        || parameters.quietHighValue || parameters.quietDiversity || parameters.quietComeback;
}

int quiet_bonus(const Parameters& parameters, const QuietFeatures& features) {
    if (!quiet_bonus_enabled(parameters))
        return 0;

    const std::int64_t weighted = std::int64_t(parameters.quietPhase) * features.phase
                                + std::int64_t(parameters.quietSafety) * features.safety
                                + std::int64_t(parameters.quietHighValue) * features.highValue
                                + std::int64_t(parameters.quietDiversity) * features.diversity
                                + std::int64_t(parameters.quietComeback) * features.comeback;

    const std::int64_t result =
      std::int64_t(parameters.quietBase) + round_divide(weighted, 1024);
    return int(std::clamp<std::int64_t>(result, -192, 192));
}

bool raw_eval_needed(const Parameters& parameters, const QuietMoveContext& quietContext) {
    const bool useV1 = bonus_enabled(parameters);
    const bool useV2 = quietContext.eligible && quiet_bonus_enabled(parameters);
    return (useV1 && parameters.bonusComeback != 0)
        || (useV2 && parameters.quietComeback != 0);
}

int combine_bonus(int v1, int v2, bool quietEligible) {
    return quietEligible ? std::clamp(v1 + v2, -512, 512) : v1;
}

Value apply_bonus(Value rawValue, int revealBonus) {
    if (is_decisive(rawValue))
        return rawValue;

    return Value(std::clamp(int(rawValue) + revealBonus, int(VALUE_MATED_IN_MAX_PLY) + 1,
                            int(VALUE_MATE_IN_MAX_PLY) - 1));
}

Value first_raw(Value adjustedThreshold, int revealBonus) {
    constexpr int Lower  = VALUE_MATED_IN_MAX_PLY;
    constexpr int Upper  = VALUE_MATE_IN_MAX_PLY;
    const int     target = adjustedThreshold;

    if (target <= Lower || target >= Upper)
        return adjustedThreshold;
    if (target == Lower + 1)
        return Value(Lower + 1);

    const int candidate = std::max(Lower + 1, target - revealBonus);
    return Value(candidate <= Upper - 1 ? candidate : Upper);
}

Window child_window(Value alpha, Value beta, int revealBonus) {
    const Value rawAlpha = Value(first_raw(Value(alpha + 1), revealBonus) - 1);
    const Value rawBeta  = first_raw(beta, revealBonus);
    return {Value(-rawBeta), Value(-rawAlpha)};
}

int pruning_slots(int pruningMargin) {
    if (pruningMargin == 0)
        return 0;

    const int magnitude = (std::abs(pruningMargin) + 63) / 64;
    return pruningMargin < 0 ? -magnitude : magnitude;
}

QuietPruningState quiet_pruning_after_move(int moveCount, int baseLimit, int revealSlots) {
    assert(revealSlots != 0);
    return {moveCount >= baseLimit, moveCount >= baseLimit + revealSlots};
}

int move_order_score(int rawScore, bool darkMove, int moveOrder) {
    return ordered_score(rawScore, darkMove, false, {moveOrder, 0, 0});
}

int raw_move_score(int orderedScore, bool darkMove, int moveOrder) {
    return raw_score(orderedScore, darkMove, false, {moveOrder, 0, 0});
}

int ordering_adjustment(bool darkMove,
                        bool safeQuiet,
                        const OrderingParameters& parameters) {
    return (darkMove ? parameters.darkMoveOrder : 0)
         + (safeQuiet ? parameters.quietMoveOrder : 0);
}

int ordered_score(int rawScore,
                  bool darkMove,
                  bool safeQuiet,
                  const OrderingParameters& parameters) {
    return rawScore + ordering_adjustment(darkMove, safeQuiet, parameters);
}

int raw_score(int orderedScore,
              bool darkMove,
              bool safeQuiet,
              const OrderingParameters& parameters) {
    return orderedScore - ordering_adjustment(darkMove, safeQuiet, parameters);
}

Value raw_eval_for_research(Value ttEval, Value) {
    return is_valid(ttEval) ? ttEval : VALUE_NONE;
}

}  // namespace Stockfish::Reveal
