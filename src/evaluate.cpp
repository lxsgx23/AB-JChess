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

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "abjnnue/abjnnue_network.h"
#include "abjnnue/abjnnue_inference.h"
#include "position.h"
#include "types.h"
#include "uci.h"

namespace Stockfish {

namespace {

constexpr int RuntimePieceValue[PIECE_TYPE_NB]   = {0, 943, 190, 491, 101, 438, 168, 0};
constexpr int RuntimeInitialCount[PIECE_TYPE_NB] = {0, 2, 2, 2, 5, 2, 2, 1};

int runtime_material(const Position& pos) {
    int            material = 0;
    const Bitboard dark     = pos.pieces(DARK);
    for (Color color : {WHITE, BLACK})
        for (PieceType type = ROOK; type < KING; ++type)
        {
            const int pieceValue = RuntimePieceValue[type];
            const int restCount  = pos.rest_piece(make_piece(color, type));
            material += restCount * pieceValue;
            if (type != PAWN)
            {
                const int revealedCount = std::max(0, RuntimeInitialCount[type] - restCount);
                const int onBoardCount  = popcount(pos.pieces(color, type) & ~dark);
                material += std::min(onBoardCount, revealedCount) * pieceValue;
            }
        }
    return material;
}

Value runtime_value(const ::ABJNNUE::RawEvaluation& raw, const Position& pos, int optimism) {
    const int nnue          = (raw.psqtRaw + raw.positionalRaw) / 16;
    const int complexity    = std::abs(raw.psqtRaw - raw.positionalRaw) / 16;
    const int material      = runtime_material(pos);
    const int materialScale = 935 + material * 93 / 5116;
    const int optimismTerm  = optimism * (complexity + 333) / 256;

    int value = (nnue * materialScale + optimismTerm * (materialScale - 832)) / 1024;
    value     = value * (218 - std::min(pos.rule40_count(), 120)) / 176;
    return std::clamp(value, -29507, 29507);
}

::ABJNNUE::RawEvaluation runtime_raw(const Eval::NNUE::Networks&   networks,
                                     const Position&               pos,
                                     Eval::NNUE::AccumulatorStack& accumulators,
                                     Eval::NNUE::AccumulatorCaches& caches) {
#if defined(ABJNNUE_RUNTIME_REFRESH_CACHE)
    const auto view = accumulators.evaluate(networks.big.model(), pos, caches.refresh);
#else
    (void) caches;
    const auto view = accumulators.evaluate(networks.big.model(), pos);
#endif
    return ::ABJNNUE::Inference::evaluate_accumulated(networks.big.model(), pos, view.accumulated,
                                                      ::ABJNNUE::LayerStackSelection{
                                                        view.layerStackBucket,
                                                        view.layerStackBlendQ8});
}

}  // namespace

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Networks&    networks,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism) {
    assert(!pos.checkers());
    return runtime_value(runtime_raw(networks, pos, accumulators, caches), pos, optimism);
}

Value Eval::evaluate_for_reveal(const Eval::NNUE::Networks&    networks,
                                const Position&                pos,
                                Eval::NNUE::AccumulatorStack&  accumulators,
                                Eval::NNUE::AccumulatorCaches& caches,
                                int                            optimism) {
    return runtime_value(runtime_raw(networks, pos, accumulators, caches), pos, optimism);
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Networks& networks) {

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    auto                          accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
    Eval::NNUE::AccumulatorCaches caches(networks);
    const auto                    raw   = runtime_raw(networks, pos, *accumulators, caches);
    const auto                    value = runtime_value(raw, pos, 0);

    std::stringstream ss;
    ss << "\nABJNNUE runtime evaluation\n"
       << "PSQT raw              " << raw.psqtRaw << "\n"
       << "Positional raw        " << raw.positionalRaw << "\n"
       << "Runtime value         " << value << " (side to move)\n";

    return ss.str();
}

}  // namespace Stockfish
