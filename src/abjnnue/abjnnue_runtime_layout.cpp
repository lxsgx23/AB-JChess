#include "abjnnue_runtime_layout.h"

namespace ABJNNUE {

RuntimeLayout RuntimeLayout::bind(const Package& package) {
    RuntimeLayout layout;
    const auto primary = package.view("primary_runtime_nnue_container.bin");
    const auto heads = package.view("eval_heads_runtime.bin");
    const auto scoreToMass = package.view("probability_score_to_mass.i32le");
    const auto massToScore = package.view("probability_mass_to_score.i32le");

    if (primary.size != PrimarySize || heads.size != EvalHeadsSize
        || scoreToMass.size != ProbabilityScoreToMassSize * sizeof(std::int32_t)
        || massToScore.size != ProbabilityMassToScoreSize * sizeof(std::int32_t))
        throw std::runtime_error("ABJCHESSV82 runtime chunk size mismatch");
    layout.transformerBiases = {primary.data, TransformerBiasesSize};
    layout.featureWeights = {primary.data + TransformerBiasesSize, FeatureWeightsSize};
    layout.psqtWeights = {primary.data + TransformerBiasesSize + FeatureWeightsSize, PSQTWeightsSize};
    layout.evalHeads = heads;
    layout.probabilityScoreToMass = scoreToMass;
    layout.probabilityMassToScore = massToScore;
    return layout;
}

bool RuntimeLayout::inference_complete() const noexcept {
    return transformerBiases.size == TransformerBiasesSize
        && featureWeights.size == FeatureWeightsSize && psqtWeights.size == PSQTWeightsSize
        && evalHeads.size == EvalHeadsSize;
}

bool RuntimeLayout::probability_complete() const noexcept {
    return probabilityScoreToMass.size == ProbabilityScoreToMassSize * sizeof(std::int32_t)
        && probabilityMassToScore.size == ProbabilityMassToScoreSize * sizeof(std::int32_t);
}

}  // namespace ABJNNUE
