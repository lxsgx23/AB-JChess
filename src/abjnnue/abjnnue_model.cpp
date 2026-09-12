#include "abjnnue_model.h"

#include "abjnnue_layers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ABJNNUE {
namespace {

std::vector<std::int16_t> decode_i16(ByteView view) {
    if (view.size % sizeof(std::int16_t) != 0) throw std::runtime_error("ABJNNUE int16 tensor is unaligned");
    std::vector<std::int16_t> out(view.size / sizeof(std::int16_t));
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        const std::uint16_t value = std::uint16_t(view.data[i * 2])
                                  | (std::uint16_t(view.data[i * 2 + 1]) << 8);
        out[i] = static_cast<std::int16_t>(value);
    }
    return out;
}

std::vector<std::int32_t> decode_i32(ByteView view) {
    if (view.size % sizeof(std::int32_t) != 0) throw std::runtime_error("ABJNNUE int32 tensor is unaligned");
    std::vector<std::int32_t> out(view.size / sizeof(std::int32_t));
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        const std::uint32_t value = std::uint32_t(view.data[i * 4])
                                  | (std::uint32_t(view.data[i * 4 + 1]) << 8)
                                  | (std::uint32_t(view.data[i * 4 + 2]) << 16)
                                  | (std::uint32_t(view.data[i * 4 + 3]) << 24);
        out[i] = static_cast<std::int32_t>(value);
    }
    return out;
}

std::int32_t trunc_div(std::int64_t numerator, std::int32_t denominator) {
    if (denominator == 0) throw std::logic_error("ABJNNUE probability divisor is zero");
    return static_cast<std::int32_t>(numerator / denominator);
}

}  // namespace

std::shared_ptr<const Model> Model::load(const std::filesystem::path& path) {
    auto model = std::shared_ptr<Model>(new Model(std::filesystem::absolute(path)));
    const auto package = Package::load(model->path_);
    const auto layout = RuntimeLayout::bind(package);

    model->transformerBiases_ = decode_i16(layout.transformerBiases);
    model->featureWeights_ = decode_i16(layout.featureWeights);
    model->psqtWeights_ = decode_i32(layout.psqtWeights);
    model->evalHeads_ = {layout.evalHeads.data, layout.evalHeads.data + layout.evalHeads.size};
    model->probabilityScoreToMass_ = decode_i32(layout.probabilityScoreToMass);
    model->probabilityMassToScore_ = decode_i32(layout.probabilityMassToScore);

    if (!model->inference_complete() || !model->probability_complete())
        throw std::runtime_error("ABJCHESSV82 schema is incomplete");
    return model;
}

bool Model::inference_complete() const noexcept {
    return transformerBiases_.size() == RuntimeLayout::AccumulatorWidth
        && featureWeights_.size() == RuntimeLayout::FeatureDimensions * RuntimeLayout::AccumulatorWidth
        && psqtWeights_.size() == RuntimeLayout::FeatureDimensions * RuntimeLayout::PSQTBuckets
        && evalHeads_.size() == RuntimeLayout::EvalHeadsSize;
}

bool Model::probability_complete() const noexcept {
    return probabilityScoreToMass_.size() == RuntimeLayout::ProbabilityScoreToMassSize
        && probabilityMassToScore_.size() == RuntimeLayout::ProbabilityMassToScoreSize;
}

std::int32_t Model::propagate(std::uint32_t bucket, const std::uint8_t* transformed) const {
    if (!inference_complete() || transformed == nullptr || bucket >= RuntimeLayout::LayerStacks)
        throw std::logic_error("ABJCHESSV82 eval head is unavailable");

    const auto* head = evalHeads_.data() + bucket * RuntimeLayout::EvalHeadBucketSize;
    return Layers::propagate(head, transformed);
}

std::int32_t Model::propagate_interpolated(std::uint32_t floor,
                                           std::uint8_t blendQ8,
                                           const std::uint8_t* transformed) const {
    if (!inference_complete() || transformed == nullptr || floor >= RuntimeLayout::LayerStacks)
        throw std::logic_error("ABJCHESSV82 eval head is unavailable");
    if (floor == RuntimeLayout::LayerStacks - 1 || blendQ8 == 0)
        return propagate(floor, transformed);
    const auto* first = evalHeads_.data() + floor * RuntimeLayout::EvalHeadBucketSize;
    const auto* second = evalHeads_.data() + (floor + 1) * RuntimeLayout::EvalHeadBucketSize;
    const auto a = static_cast<std::int64_t>(Layers::propagate(first, transformed));
    const auto b = static_cast<std::int64_t>(Layers::propagate(second, transformed));
    const auto value = RuntimeLayout::interpolate_q8(a, b, blendQ8);
    if (value < std::numeric_limits<std::int32_t>::min()
        || value > std::numeric_limits<std::int32_t>::max())
        throw std::overflow_error("ABJCHESSV82 interpolated head output overflow");
    return static_cast<std::int32_t>(value);
}

std::int32_t Model::aggregate_probability(const std::vector<std::pair<int, int>>& samples,
                                          int aggressiveLevel) const {
    if (aggressiveLevel < 0 || aggressiveLevel > 10)
        throw std::invalid_argument(
          "ABJNNUE AggressiveLevel must be in the range 0..10");
    if (!probability_complete() || samples.empty())
        throw std::logic_error("ABJCHESSV82 probability tables are unavailable");
    std::int32_t minimum = samples.front().first;
    std::int32_t maximum = minimum;
    std::int64_t total = 0;
    std::int64_t weightedSum = 0;
    for (const auto& [value, count] : samples)
    {
        if (count <= 0 || count > MaximumProbabilityCount)
            throw std::invalid_argument("ABJNNUE probability sample count is outside 1..32");
        if (total > MaximumProbabilityCount - count)
            throw std::invalid_argument("ABJNNUE probability sample total exceeds 32");
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        total += count;
        weightedSum += std::int64_t(value) * count;
    }
    if (total <= 0) throw std::invalid_argument("ABJNNUE probability sample total is zero");
    if (aggressiveLevel == 10)
    {
        // Level 10 is a bounded risk-sensitive aggregate.  Clamp before any
        // decisive fast path so raw out-of-range scores cannot leak through
        // or influence the variance calculation.
        std::int64_t boundedSum = 0;
        for (const auto& [value, count] : samples)
            boundedSum += std::int64_t(std::clamp(value, ProbabilityScoreMin, ProbabilityScoreMax)) * count;
        const auto average = trunc_div(boundedSum, static_cast<std::int32_t>(total));
        std::uint64_t variance = 0;
        for (const auto& [value, count] : samples)
        {
            const auto bounded = std::clamp(value, ProbabilityScoreMin, ProbabilityScoreMax);
            const auto delta = std::int64_t(bounded) - average;
            variance += static_cast<std::uint64_t>(delta * delta) * count;
        }
        const auto penalty = static_cast<std::int64_t>(std::sqrt(double(variance))) * 4 / 128;
        return average - static_cast<std::int32_t>(penalty);
    }

    if (minimum > 10000) return minimum;
    if (maximum < -10000) return maximum;
    if (minimum > ProbabilityScoreMax || maximum < ProbabilityScoreMin)
        return trunc_div(weightedSum, static_cast<std::int32_t>(total));
    if (minimum == maximum) return minimum;

    std::int64_t massSum = 0;
    for (const auto& [value, count] : samples)
    {
        const auto bounded = std::clamp(value, ProbabilityScoreMin, ProbabilityScoreMax);
        massSum += std::int64_t(probabilityScoreToMass_[bounded - ProbabilityScoreMin]) * count;
    }
    const auto mass = trunc_div(massSum, static_cast<std::int32_t>(total));
    const auto index = std::clamp(mass - ProbabilityMassMin, std::int32_t(0),
                                  static_cast<std::int32_t>(probabilityMassToScore_.size() - 1));
    return probabilityMassToScore_[index];
}

std::string Model::summary() const {
    std::ostringstream out;
    out << "ABJNNUE model loaded: " << path_.string()
        << " features=" << featureWeights_.size() / RuntimeLayout::AccumulatorWidth
        << " accumulator=" << transformerBiases_.size()
        << " psqt_buckets=" << RuntimeLayout::PSQTBuckets
        << " heads=" << evalHeads_.size() / RuntimeLayout::EvalHeadBucketSize
        << " head_backend=" << Layers::backend_name()
        << " inference_complete=" << (inference_complete() ? "true" : "false")
        << " probability_tables=" << (probability_complete() ? "true" : "false");
    return out.str();
}

}  // namespace ABJNNUE
