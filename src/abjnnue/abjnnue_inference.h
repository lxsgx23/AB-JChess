#ifndef ABJNNUE_INFERENCE_H_INCLUDED
#define ABJNNUE_INFERENCE_H_INCLUDED

#include <array>
#include <cstdint>

#include "abjnnue_feature_encoder.h"

namespace Stockfish {
class Position;
}

namespace ABJNNUE {

struct alignas(64) TransformedFeatures
  : std::array<std::uint8_t, RuntimeLayout::AccumulatorWidth> {};

struct InferenceResult {
    EncodedPosition     encoded;
    AccumulatedPosition accumulated;
    TransformedFeatures transformed{};
    std::int32_t        psqtRaw = 0;
    std::int32_t        positionalRaw = 0;
};

struct RawEvaluation {
    std::int32_t psqtRaw = 0;
    std::int32_t positionalRaw = 0;
};

struct InferenceStats {
    std::uint64_t simdTransforms   = 0;
    std::uint64_t scalarTransforms = 0;
};

class Inference {
   public:
    static InferenceResult evaluate(const Model& model, const Stockfish::Position& position);
    static RawEvaluation evaluate_accumulated(const Model& model,
                                              const Stockfish::Position& position,
                                              const AccumulatedPosition& accumulated,
                                              std::uint32_t layerStackBucket,
                                              TransformedFeatures* transformed = nullptr);
    static RawEvaluation evaluate_accumulated(const Model& model,
                                              const Stockfish::Position& position,
                                              const AccumulatedPosition& accumulated,
                                              LayerStackSelection selection,
                                              TransformedFeatures* transformed = nullptr);

    static const InferenceStats& stats() noexcept;
    static void reset_stats() noexcept;
};

}  // namespace ABJNNUE

#endif
