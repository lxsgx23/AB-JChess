#ifndef ABJNNUE_FEATURE_ENCODER_H_INCLUDED
#define ABJNNUE_FEATURE_ENCODER_H_INCLUDED

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "../types.h"
#include "abjnnue_model.h"

namespace Stockfish {
class Position;
}

namespace ABJNNUE {

class FeatureIndexList {
   public:
    // A hidden position can contain visible board rows, one row per
    // remaining identity, and nine one-hot meta rows for each perspective.
    // Keep a fixed stack-friendly bound and fail closed if a malformed FEN
    // exceeds it.
    static constexpr std::size_t Capacity = 160;

    void clear() noexcept { size_ = 0; }

    void push_back(std::uint32_t value) {
        if (size_ >= Capacity) throw std::logic_error("ABJNNUE active feature list overflow");
        values_[size_++] = value;
    }

    void sort() noexcept { std::sort(values_.begin(), values_.begin() + size_); }

    std::size_t size() const noexcept { return size_; }
    constexpr std::size_t capacity() const noexcept { return Capacity; }

    auto begin() noexcept { return values_.begin(); }
    auto end() noexcept { return values_.begin() + size_; }
    auto begin() const noexcept { return values_.begin(); }
    auto end() const noexcept { return values_.begin() + size_; }

    std::uint32_t& operator[](std::size_t index) noexcept { return values_[index]; }
    const std::uint32_t& operator[](std::size_t index) const noexcept { return values_[index]; }

   private:
    std::array<std::uint32_t, Capacity> values_{};
    std::size_t                         size_ = 0;
};

struct PerspectiveFeatures {
    Stockfish::Color           perspective = Stockfish::WHITE;
    std::uint32_t              kingBucket  = 0;
    std::uint32_t              attackBucket = 0;
    bool                       mirror      = false;
    bool                       midMirror   = false;
    std::array<std::uint8_t, 6> restCounts{};
    // The nine one-hot metadata offsets are retained so the accumulator can
    // reuse an encoded position without rescanning the board after a full
    // refresh or refresh-cache hit.
    std::array<std::uint8_t, 9> metaOffsets{};
    FeatureIndexList           active;
};

struct LayerStackSelection {
    std::uint32_t floor = 0;
    union {
        std::uint8_t blend_q8;
        std::uint8_t blendQ8;
    };

    constexpr LayerStackSelection(std::uint32_t floorValue = 0,
                                  std::uint8_t blendValue = 0)
        : floor(floorValue), blend_q8(blendValue) {}
};

struct EncodedPosition {
    std::uint32_t layerStackBucket = 0;
    LayerStackSelection layerStackSelection{};
    std::size_t   darkSquares      = 0;
    std::array<PerspectiveFeatures, Stockfish::COLOR_NB> perspectives;
};

struct KingTransform {
    std::uint32_t bucket = 0;
    bool          mirror = false;
};

using AccumulatorVector = std::array<std::int16_t, RuntimeLayout::AccumulatorWidth>;
using PSQTVector = std::array<std::int32_t, RuntimeLayout::PSQTBuckets>;

struct PerspectiveAccumulation {
    AccumulatorVector values{};
    PSQTVector        psqt{};
};

struct AccumulatedPosition {
    std::array<PerspectiveAccumulation, Stockfish::COLOR_NB> perspectives;
};

class FeatureEncoder {
   public:
    static constexpr std::size_t MaxActiveFeatures = FeatureIndexList::Capacity;
    static std::size_t dark_square_count(const Stockfish::Position& position);
    static std::uint32_t layer_stack_bucket(const Stockfish::Position& position);
    static LayerStackSelection layer_stack_selection(const Stockfish::Position& position);
    static std::uint32_t attack_bucket(const Stockfish::Position& position,
                                       Stockfish::Color perspective);
    static bool requires_mid_mirror(const Stockfish::Position& position,
                                    Stockfish::Color perspective);
    static KingTransform king_transform(const Stockfish::Position& position,
                                        Stockfish::Color perspective);
    static std::uint32_t board_index(Stockfish::Color perspective,
                                     Stockfish::Square square,
                                     Stockfish::Piece piece,
                                     std::uint32_t bucket,
                                     bool mirror);
    static std::uint32_t meta_index(Stockfish::Color perspective,
                                    std::uint32_t bucket,
                                    std::uint32_t offset);
    static std::uint32_t dark_rest_index(Stockfish::Color perspective,
                                         Stockfish::Color owner,
                                         Stockfish::PieceType type,
                                         std::uint32_t bucket);

    static EncodedPosition encode(const Stockfish::Position& position);
    static AccumulatedPosition accumulate(const Model& model,
                                          const EncodedPosition& encoded);
};

}  // namespace ABJNNUE

#endif
