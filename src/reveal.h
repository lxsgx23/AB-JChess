/*
  AB-JChess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef REVEAL_H_INCLUDED
#define REVEAL_H_INCLUDED

#include <array>
#include <cstdint>
#include <utility>

#include "types.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Reveal {

struct Parameters {
    int bonusBase     = 0;
    int bonusPhase    = 0;
    int bonusPool     = 0;
    int bonusUnknown  = 0;
    int bonusComeback = 0;
    int moveOrder     = 0;
    int reduction     = 0;
    int pruningMargin = 0;
    int quietBase      = 0;
    int quietPhase     = 0;
    int quietSafety    = 0;
    int quietHighValue = 0;
    int quietDiversity = 0;
    int quietComeback  = 0;
    int quietMoveOrder = 0;
    int quietReduction = 0;
};

struct OrderingParameters {
    int darkMoveOrder = 0;
    int quietMoveOrder = 0;
    int pruningMargin = 0;
};

struct FeatureInputs {
    int          globalDark;
    int          poolTotal;
    int          ownBoardDark;
    std::int64_t poolValueAbovePawn;
    Value        rawParentEval;
};

struct Features {
    int phase;
    int pool;
    int unknown;
    int comeback;
};

enum class QuietSafety : std::int8_t { Safe, Contested, Loose };

struct QuietMoveContext {
    bool        eligible = false;
    QuietSafety safety   = QuietSafety::Contested;

    bool safe() const { return eligible && safety == QuietSafety::Safe; }
};

struct QuietFeatureInputs {
    int                globalDark;
    std::array<int, 6> poolCounts;
    QuietSafety        safety;
    Value              rawParentEval;
};

struct QuietFeatures {
    int phase;
    int safety;
    int highValue;
    int diversity;
    int comeback;
};

struct Window {
    Value alpha;
    Value beta;
};

struct QuietPruningState {
    bool skipVisibleQuiets;
    bool skipDarkQuiets;
};

std::int64_t round_divide(std::int64_t numerator, std::int64_t denominator);
Features     make_features(const FeatureInputs& inputs);
Features     make_features(const Position& position, Value rawParentEval);
QuietFeatures make_quiet_features(const QuietFeatureInputs& inputs);
QuietFeatures make_quiet_features(const Position& position,
                                  QuietSafety    safety,
                                  Value          rawParentEval);
QuietSafety  quiet_safety(const Position& position, Move move);
QuietMoveContext quiet_move_context(const Position& position, Move move, bool enabled);
bool         qsearch_quiet_context_enabled(bool useReveal, bool inCheck, bool safetyNeeded);
Depth        adjust_reduction(Depth             reduction,
                              bool              darkMove,
                              bool              safeQuiet,
                              const Parameters& parameters);
bool         bonus_enabled(const Parameters& parameters);
int          bonus(const Parameters& parameters, const Features& features);
bool         quiet_bonus_enabled(const Parameters& parameters);
int          quiet_bonus(const Parameters& parameters, const QuietFeatures& features);
bool         raw_eval_needed(const Parameters& parameters, const QuietMoveContext& quietContext);
int          combine_bonus(int v1, int v2, bool quietEligible);
Value        apply_bonus(Value rawValue, int revealBonus);
Value        first_raw(Value adjustedThreshold, int revealBonus);
Window       child_window(Value alpha, Value beta, int revealBonus);
int          pruning_slots(int pruningMargin);
QuietPruningState quiet_pruning_after_move(int moveCount, int baseLimit, int revealSlots);
int          move_order_score(int rawScore, bool darkMove, int moveOrder);
int          raw_move_score(int orderedScore, bool darkMove, int moveOrder);
int          ordering_adjustment(bool darkMove,
                                 bool safeQuiet,
                                 const OrderingParameters& parameters);
int          ordered_score(int rawScore,
                           bool darkMove,
                           bool safeQuiet,
                           const OrderingParameters& parameters);
int          raw_score(int orderedScore,
                       bool darkMove,
                       bool safeQuiet,
                       const OrderingParameters& parameters);
Value        raw_eval_for_research(Value ttEval, Value correctedStaticEval);

template<typename ChildSearch>
Value search_with_bonus(Value alpha, Value beta, int revealBonus, ChildSearch&& childSearch) {
    const Window window = child_window(alpha, beta, revealBonus);
    const Value  child  = std::forward<ChildSearch>(childSearch)(window.alpha, window.beta);
    return apply_bonus(Value(-child), revealBonus);
}

}  // namespace Stockfish::Reveal

#endif  // #ifndef REVEAL_H_INCLUDED
