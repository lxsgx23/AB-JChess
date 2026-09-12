#ifndef ABJNNUE_RUNTIME_LAYOUT_H_INCLUDED
#define ABJNNUE_RUNTIME_LAYOUT_H_INCLUDED

#include "abjnnue_package.h"

namespace ABJNNUE {

struct RuntimeLayout {
    static constexpr std::size_t AccumulatorWidth         = 2048;
    static constexpr std::size_t PerspectiveOutputWidth   = AccumulatorWidth / 2;
    static constexpr std::size_t PieceSquarePlanes        = 14;
    static constexpr std::size_t Squares                 = 90;
    static constexpr std::size_t BasePieceSquareDimensions = PieceSquarePlanes * Squares;
    static constexpr std::size_t MetaDimensions           = 64;
    static constexpr std::size_t PieceSquareDimensions    = BasePieceSquareDimensions + MetaDimensions;
    static constexpr std::size_t KingBuckets              = 6;
    static constexpr std::size_t AttackBuckets            = 4;
    static constexpr std::size_t FeatureBuckets           = KingBuckets * AttackBuckets;
    static constexpr std::size_t BoardFeatureDimensions   = BasePieceSquareDimensions * FeatureBuckets;
    static constexpr std::size_t DarkRestDimensions       = MetaDimensions;
    static constexpr std::size_t VariantFeatureDimensions = PieceSquareDimensions * FeatureBuckets;
    static constexpr std::size_t FeatureDimensions        = VariantFeatureDimensions;
    static constexpr std::size_t PSQTBuckets              = 16;
    static constexpr std::size_t LayerStacks              = 16;
    static constexpr std::int64_t InterpolationQBits       = 8;
    static constexpr std::int64_t InterpolationDenominator = (1LL << InterpolationQBits) - 1LL;

    static constexpr std::int64_t interpolate_q8(std::int64_t first,
                                                  std::int64_t second,
                                                  std::uint8_t blendQ8) noexcept {
        if (blendQ8 == 0) return first;
        if (blendQ8 == InterpolationDenominator) return second;
        const std::int64_t q = blendQ8;
        return (first * (InterpolationDenominator - q) + second * q
                + InterpolationDenominator / 2)
             / InterpolationDenominator;
    }

    // V8 deliberately uses a separate, larger primary chunk.  Keeping the
    // exact payload size avoids a second legacy capacity contract and makes
    // every exported weight package self-describing through its chunk sizes.
    static constexpr std::size_t TransformerBiasesSize = AccumulatorWidth * sizeof(std::int16_t);
    static constexpr std::size_t FeatureWeightsSize =
      FeatureDimensions * AccumulatorWidth * sizeof(std::int16_t);
    static constexpr std::size_t PSQTWeightsSize =
      FeatureDimensions * PSQTBuckets * sizeof(std::int32_t);
    static constexpr std::size_t TransformerPayloadSize =
      TransformerBiasesSize + FeatureWeightsSize + PSQTWeightsSize;
    static constexpr std::size_t PrimarySize = TransformerPayloadSize;
    static constexpr std::size_t EvalHeadBucketSize = 34208;
    static constexpr std::size_t EvalHeadsSize = LayerStacks * EvalHeadBucketSize;
    static constexpr std::size_t ProbabilityScoreToMassSize = 4001;
    // Probability mass now covers the symmetric inclusive range [-950, 950].
    // Keep the inverse table length in lockstep with the ABI range.
    static constexpr std::size_t ProbabilityMassToScoreSize = 1901;

    static_assert(FeatureDimensions == 31776);
    static_assert(TransformerPayloadSize == 132192256);
    static_assert(EvalHeadBucketSize == 34208);

    ByteView transformerBiases;
    ByteView featureWeights;
    ByteView psqtWeights;
    ByteView evalHeads;
    ByteView probabilityScoreToMass;
    ByteView probabilityMassToScore;

    static RuntimeLayout bind(const Package& package);
    static RuntimeLayout bind(Package&&) = delete;
    static RuntimeLayout bind(const Package&&) = delete;
    bool inference_complete() const noexcept;
    bool probability_complete() const noexcept;
};

}  // namespace ABJNNUE

#endif
