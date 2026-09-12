#ifndef ABJNNUE_MODEL_H_INCLUDED
#define ABJNNUE_MODEL_H_INCLUDED

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "abjnnue_runtime_layout.h"

namespace ABJNNUE {

class Model {
   public:
    static std::shared_ptr<const Model> load(const std::filesystem::path& path);

    const std::filesystem::path& path() const noexcept { return path_; }
    const std::vector<std::int16_t>& transformer_biases() const noexcept { return transformerBiases_; }
    const std::vector<std::int16_t>& feature_weights() const noexcept { return featureWeights_; }
    const std::vector<std::int32_t>& psqt_weights() const noexcept { return psqtWeights_; }
    const std::vector<std::uint8_t>& eval_heads() const noexcept { return evalHeads_; }
    const std::vector<std::int32_t>& probability_score_to_mass() const noexcept {
        return probabilityScoreToMass_;
    }
    const std::vector<std::int32_t>& probability_mass_to_score() const noexcept {
        return probabilityMassToScore_;
    }
    std::int32_t probability_mass_min() const noexcept { return ProbabilityMassMin; }

    bool inference_complete() const noexcept;
    bool probability_complete() const noexcept;

    // Run one of the 16 fixed 2048 -> 16 -> 32 -> 1 runtime heads.  Input is
    // the 2048-byte transformed feature vector (two 1024-wide perspectives).
    std::int32_t propagate(std::uint32_t bucket, const std::uint8_t* transformed) const;
    std::int32_t propagate_interpolated(std::uint32_t floor,
                                        std::uint8_t blendQ8,
                                        const std::uint8_t* transformed) const;

    // Combine child scores using the package's score/mass lookup tables.
    std::int32_t aggregate_probability(const std::vector<std::pair<int, int>>& samples,
                                       int aggressiveLevel) const;

    std::string summary() const;

    static constexpr std::int32_t ProbabilityScoreMin = -2000;
    static constexpr std::int32_t ProbabilityScoreMax = 2000;
    static constexpr std::int32_t ProbabilityMassMin = -950;
    static constexpr std::int32_t ProbabilityMassMax = 950;
    static constexpr std::int32_t MaximumProbabilityCount = 32;

   private:
    explicit Model(std::filesystem::path path) : path_(std::move(path)) {}

    std::filesystem::path     path_;
    std::vector<std::int16_t> transformerBiases_;
    std::vector<std::int16_t> featureWeights_;
    std::vector<std::int32_t> psqtWeights_;
    std::vector<std::uint8_t> evalHeads_;
    std::vector<std::int32_t> probabilityScoreToMass_;
    std::vector<std::int32_t> probabilityMassToScore_;
};

}  // namespace ABJNNUE

#endif
