#include "abjnnue_accumulator.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <sstream>
#include <stdexcept>

#if defined(USE_AVX2) || defined(USE_SSE2)
    #include <immintrin.h>
#endif

#include "../position.h"

namespace ABJNNUE {

#if defined(ABJNNUE_RUNTIME_STATS)
    #define ABJNNUE_RUNTIME_STAT(statement) \
        do                                      \
        {                                       \
            statement;                           \
        } while (false)
#else
    #define ABJNNUE_RUNTIME_STAT(statement) \
        do                                      \
        {                                       \
        } while (false)
#endif

namespace {

using namespace Stockfish;
constexpr PieceType RestOrder[] = {ROOK, CANNON, KNIGHT, BISHOP, ADVISOR, PAWN};
using FeatureIndices = std::array<std::uint32_t, RefreshCache::MaxDeltaRows>;

bool valid_feature_piece(Piece piece) {
    const int value = int(piece);
    return value == int(DARK_PIECE)
        || (value > int(NO_PIECE) && value < int(DARK_PIECE) && value != 8);
}

void validate_dirty_piece(const DirtyPiece& dirty, std::size_t stackSize) {
    const auto fail = [&](const char* field, int value) {
        std::ostringstream message;
        message << "invalid ABJNNUE DirtyPiece at push: field=" << field << " value=" << value
                << " stack_size=" << stackSize;
        throw std::logic_error(message.str());
    };
    if (!valid_feature_piece(dirty.pc)) fail("pc", int(dirty.pc));
    if (!is_ok(dirty.from)) fail("from", int(dirty.from));
    if (dirty.to != SQ_NONE && !is_ok(dirty.to)) fail("to", int(dirty.to));
    if (dirty.remove_sq != SQ_NONE)
    {
        if (!is_ok(dirty.remove_sq)) fail("remove_sq", int(dirty.remove_sq));
        if (!valid_feature_piece(dirty.remove_pc)) fail("remove_pc", int(dirty.remove_pc));
    }
    if (dirty.add_sq != SQ_NONE)
    {
        if (!is_ok(dirty.add_sq)) fail("add_sq", int(dirty.add_sq));
        if (!valid_feature_piece(dirty.add_pc) || dirty.add_pc == DARK_PIECE)
            fail("add_pc", int(dirty.add_pc));
    }
}

#if !defined(USE_SSE2)
std::int16_t add_wrapped(std::int16_t lhs, std::int16_t rhs) {
    return static_cast<std::int16_t>(static_cast<std::uint16_t>(lhs)
                                   + static_cast<std::uint16_t>(rhs));
}
std::int16_t sub_wrapped(std::int16_t lhs, std::int16_t rhs) {
    return static_cast<std::int16_t>(static_cast<std::uint16_t>(lhs)
                                   - static_cast<std::uint16_t>(rhs));
}
std::int32_t add_wrapped(std::int32_t lhs, std::int32_t rhs) {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(lhs)
                                   + static_cast<std::uint32_t>(rhs));
}
std::int32_t sub_wrapped(std::int32_t lhs, std::int32_t rhs) {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(lhs)
                                   - static_cast<std::uint32_t>(rhs));
}
#endif

void validate_indices(const FeatureIndices& added, std::size_t addedCount,
                      const FeatureIndices& removed, std::size_t removedCount) {
    for (std::size_t i = 0; i < addedCount; ++i)
        if (added[i] >= RuntimeLayout::FeatureDimensions)
            throw std::logic_error("ABJNNUE incremental feature index is out of range");
    for (std::size_t i = 0; i < removedCount; ++i)
        if (removed[i] >= RuntimeLayout::FeatureDimensions)
            throw std::logic_error("ABJNNUE incremental feature index is out of range");
}

#if !defined(ABJNNUE_RUNTIME_FUSED_UPDATE)
void apply_features_legacy(const Model& model,
                           PerspectiveAccumulation& destination,
                           const FeatureIndices& added, std::size_t addedCount,
                           const FeatureIndices& removed, std::size_t removedCount) {
    for (std::size_t i = 0; i < addedCount; ++i)
    {
        const auto* weights = model.feature_weights().data()
                            + std::size_t(added[i]) * RuntimeLayout::AccumulatorWidth;
        const auto* psqt = model.psqt_weights().data()
                         + std::size_t(added[i]) * RuntimeLayout::PSQTBuckets;
#if defined(USE_AVX2)
        for (std::size_t column = 0; column < RuntimeLayout::AccumulatorWidth; column += 16)
        {
            auto value = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(destination.values.data() + column));
            value = _mm256_add_epi16(
              value, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + column)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination.values.data() + column),
                                value);
        }
        for (std::size_t bucket = 0; bucket < RuntimeLayout::PSQTBuckets; bucket += 8)
        {
            auto psqtValue = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(destination.psqt.data() + bucket));
            psqtValue = _mm256_add_epi32(
              psqtValue,
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(psqt + bucket)));
            _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(destination.psqt.data() + bucket), psqtValue);
        }
#elif defined(USE_SSE2)
        for (std::size_t column = 0; column < RuntimeLayout::AccumulatorWidth; column += 8)
        {
            auto value = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(destination.values.data() + column));
            value = _mm_add_epi16(
              value, _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + column)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(destination.values.data() + column),
                             value);
        }
        for (std::size_t bucket = 0; bucket < RuntimeLayout::PSQTBuckets; bucket += 4)
        {
            auto psqtValue = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(destination.psqt.data() + bucket));
            psqtValue = _mm_add_epi32(
              psqtValue, _mm_loadu_si128(reinterpret_cast<const __m128i*>(psqt + bucket)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(destination.psqt.data() + bucket),
                             psqtValue);
        }
#else
        for (std::size_t column = 0; column < RuntimeLayout::AccumulatorWidth; ++column)
            destination.values[column] = add_wrapped(destination.values[column], weights[column]);
        for (std::size_t bucket = 0; bucket < RuntimeLayout::PSQTBuckets; ++bucket)
            destination.psqt[bucket] = add_wrapped(destination.psqt[bucket], psqt[bucket]);
#endif
    }
    for (std::size_t i = 0; i < removedCount; ++i)
    {
        const auto* weights = model.feature_weights().data()
                            + std::size_t(removed[i]) * RuntimeLayout::AccumulatorWidth;
        const auto* psqt = model.psqt_weights().data()
                         + std::size_t(removed[i]) * RuntimeLayout::PSQTBuckets;
#if defined(USE_AVX2)
        for (std::size_t column = 0; column < RuntimeLayout::AccumulatorWidth; column += 16)
        {
            auto value = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(destination.values.data() + column));
            value = _mm256_sub_epi16(
              value, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + column)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination.values.data() + column),
                                value);
        }
        for (std::size_t bucket = 0; bucket < RuntimeLayout::PSQTBuckets; bucket += 8)
        {
            auto psqtValue = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(destination.psqt.data() + bucket));
            psqtValue = _mm256_sub_epi32(
              psqtValue,
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(psqt + bucket)));
            _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(destination.psqt.data() + bucket), psqtValue);
        }
#elif defined(USE_SSE2)
        for (std::size_t column = 0; column < RuntimeLayout::AccumulatorWidth; column += 8)
        {
            auto value = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(destination.values.data() + column));
            value = _mm_sub_epi16(
              value, _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + column)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(destination.values.data() + column),
                             value);
        }
        for (std::size_t bucket = 0; bucket < RuntimeLayout::PSQTBuckets; bucket += 4)
        {
            auto psqtValue = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(destination.psqt.data() + bucket));
            psqtValue = _mm_sub_epi32(
              psqtValue, _mm_loadu_si128(reinterpret_cast<const __m128i*>(psqt + bucket)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(destination.psqt.data() + bucket),
                             psqtValue);
        }
#else
        for (std::size_t column = 0; column < RuntimeLayout::AccumulatorWidth; ++column)
            destination.values[column] = sub_wrapped(destination.values[column], weights[column]);
        for (std::size_t bucket = 0; bucket < RuntimeLayout::PSQTBuckets; ++bucket)
            destination.psqt[bucket] = sub_wrapped(destination.psqt[bucket], psqt[bucket]);
#endif
    }
}
#endif

void apply_features(const Model& model,
                    PerspectiveAccumulation& destination,
                    const FeatureIndices& added, std::size_t addedCount,
                    const FeatureIndices& removed, std::size_t removedCount) {
    validate_indices(added, addedCount, removed, removedCount);
#if defined(ABJNNUE_RUNTIME_FUSED_UPDATE)
#if defined(USE_AVX2)
    for (std::size_t column = 0; column < RuntimeLayout::AccumulatorWidth; column += 16)
    {
        auto value = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(destination.values.data() + column));
        for (std::size_t i = 0; i < addedCount; ++i)
        {
            const auto* weights = model.feature_weights().data()
                                + std::size_t(added[i]) * RuntimeLayout::AccumulatorWidth;
            value = _mm256_add_epi16(
              value, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + column)));
        }
        for (std::size_t i = 0; i < removedCount; ++i)
        {
            const auto* weights = model.feature_weights().data()
                                + std::size_t(removed[i]) * RuntimeLayout::AccumulatorWidth;
            value = _mm256_sub_epi16(
              value, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + column)));
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination.values.data() + column), value);
    }
    for (std::size_t bucket = 0; bucket < RuntimeLayout::PSQTBuckets; bucket += 8)
    {
        auto value = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(destination.psqt.data() + bucket));
        for (std::size_t i = 0; i < addedCount; ++i)
        {
            const auto* psqt = model.psqt_weights().data()
                             + std::size_t(added[i]) * RuntimeLayout::PSQTBuckets;
            value = _mm256_add_epi32(
              value, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(psqt + bucket)));
        }
        for (std::size_t i = 0; i < removedCount; ++i)
        {
            const auto* psqt = model.psqt_weights().data()
                             + std::size_t(removed[i]) * RuntimeLayout::PSQTBuckets;
            value = _mm256_sub_epi32(
              value, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(psqt + bucket)));
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination.psqt.data() + bucket), value);
    }
#elif defined(USE_SSE2)
    for (std::size_t column = 0; column < RuntimeLayout::AccumulatorWidth; column += 8)
    {
        auto value = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(destination.values.data() + column));
        for (std::size_t i = 0; i < addedCount; ++i)
        {
            const auto* weights = model.feature_weights().data()
                                + std::size_t(added[i]) * RuntimeLayout::AccumulatorWidth;
            value = _mm_add_epi16(
              value, _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + column)));
        }
        for (std::size_t i = 0; i < removedCount; ++i)
        {
            const auto* weights = model.feature_weights().data()
                                + std::size_t(removed[i]) * RuntimeLayout::AccumulatorWidth;
            value = _mm_sub_epi16(
              value, _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + column)));
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination.values.data() + column), value);
    }
    for (std::size_t bucket = 0; bucket < RuntimeLayout::PSQTBuckets; bucket += 4)
    {
        auto value = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(destination.psqt.data() + bucket));
        for (std::size_t i = 0; i < addedCount; ++i)
        {
            const auto* psqt = model.psqt_weights().data()
                             + std::size_t(added[i]) * RuntimeLayout::PSQTBuckets;
            value = _mm_add_epi32(
              value, _mm_loadu_si128(reinterpret_cast<const __m128i*>(psqt + bucket)));
        }
        for (std::size_t i = 0; i < removedCount; ++i)
        {
            const auto* psqt = model.psqt_weights().data()
                             + std::size_t(removed[i]) * RuntimeLayout::PSQTBuckets;
            value = _mm_sub_epi32(
              value, _mm_loadu_si128(reinterpret_cast<const __m128i*>(psqt + bucket)));
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination.psqt.data() + bucket), value);
    }
#else
    for (std::size_t column = 0; column < RuntimeLayout::AccumulatorWidth; ++column)
    {
        for (std::size_t i = 0; i < addedCount; ++i)
            destination.values[column] = add_wrapped(
              destination.values[column],
              model.feature_weights()[std::size_t(added[i]) * RuntimeLayout::AccumulatorWidth
                                      + column]);
        for (std::size_t i = 0; i < removedCount; ++i)
            destination.values[column] = sub_wrapped(
              destination.values[column],
              model.feature_weights()[std::size_t(removed[i]) * RuntimeLayout::AccumulatorWidth
                                      + column]);
    }
    for (std::size_t bucket = 0; bucket < RuntimeLayout::PSQTBuckets; ++bucket)
    {
        for (std::size_t i = 0; i < addedCount; ++i)
            destination.psqt[bucket] = add_wrapped(
              destination.psqt[bucket],
              model.psqt_weights()[std::size_t(added[i]) * RuntimeLayout::PSQTBuckets + bucket]);
        for (std::size_t i = 0; i < removedCount; ++i)
            destination.psqt[bucket] = sub_wrapped(
              destination.psqt[bucket],
              model.psqt_weights()[std::size_t(removed[i]) * RuntimeLayout::PSQTBuckets + bucket]);
    }
#endif
#else
    apply_features_legacy(model, destination, added, addedCount, removed, removedCount);
#endif
}

}  // namespace

void RefreshCache::clear() noexcept {
    for (auto& entry : entries_)
        entry = {};
    nextAge_ = 0;
    stats_   = {};
}

namespace {

using namespace Stockfish;

struct FeatureDelta {
    FeatureIndices added{};
    FeatureIndices removed{};
    std::size_t    addedCount = 0;
    std::size_t    removedCount = 0;
    bool           withinLimit = true;
};

FeatureDelta diff_features(const FeatureIndexList& before, const FeatureIndexList& after) {
    FeatureDelta delta;
    std::size_t  beforeIndex = 0;
    std::size_t  afterIndex  = 0;
    while (beforeIndex < before.size() || afterIndex < after.size())
    {
        if (beforeIndex == before.size())
        {
            if (delta.addedCount >= delta.added.size())
            {
                delta.withinLimit = false;
                return delta;
            }
            delta.added[delta.addedCount++] = after[afterIndex++];
        }
        else if (afterIndex == after.size())
        {
            if (delta.removedCount >= delta.removed.size())
            {
                delta.withinLimit = false;
                return delta;
            }
            delta.removed[delta.removedCount++] = before[beforeIndex++];
        }
        else if (before[beforeIndex] == after[afterIndex])
        {
            ++beforeIndex;
            ++afterIndex;
        }
        else if (before[beforeIndex] < after[afterIndex])
        {
            if (delta.removedCount >= delta.removed.size())
            {
                delta.withinLimit = false;
                return delta;
            }
            delta.removed[delta.removedCount++] = before[beforeIndex++];
        }
        else
        {
            if (delta.addedCount >= delta.added.size())
            {
                delta.withinLimit = false;
                return delta;
            }
            delta.added[delta.addedCount++] = after[afterIndex++];
        }
    }
    return delta;
}

bool same_cache_shape(const EncodedPosition& lhs, const EncodedPosition& rhs) {
    for (Color perspective : {WHITE, BLACK})
    {
        const auto& a = lhs.perspectives[perspective];
        const auto& b = rhs.perspectives[perspective];
        if (a.kingBucket != b.kingBucket || a.attackBucket != b.attackBucket
            || a.mirror != b.mirror || a.midMirror != b.midMirror)
            return false;
    }
    return true;
}

}  // namespace

bool RefreshCache::try_refresh(const Model& model,
                               const EncodedPosition& encoded,
                               AccumulatedPosition& accumulated) {
    ABJNNUE_RUNTIME_STAT(++stats_.lookups);

    Entry* candidate = nullptr;
    for (auto& entry : entries_)
        if (entry.valid && same_cache_shape(entry.encoded, encoded))
        {
            if (!candidate || entry.age > candidate->age)
                candidate = &entry;
        }

    if (!candidate)
    {
        ABJNNUE_RUNTIME_STAT(++stats_.misses);
        return false;
    }

    std::array<FeatureDelta, COLOR_NB> deltas{};
    std::size_t totalRows = 0;
    for (Color perspective : {WHITE, BLACK})
    {
        deltas[perspective] = diff_features(candidate->encoded.perspectives[perspective].active,
                                             encoded.perspectives[perspective].active);
        if (!deltas[perspective].withinLimit)
        {
            ABJNNUE_RUNTIME_STAT(++stats_.fallbacks);
            return false;
        }
        totalRows += deltas[perspective].addedCount + deltas[perspective].removedCount;
    }
    if (totalRows > MaxDeltaRows)
    {
        ABJNNUE_RUNTIME_STAT(++stats_.fallbacks);
        return false;
    }

    AccumulatedPosition updated = candidate->accumulated;
    for (Color perspective : {WHITE, BLACK})
        apply_features(model, updated.perspectives[perspective], deltas[perspective].added,
                       deltas[perspective].addedCount, deltas[perspective].removed,
                       deltas[perspective].removedCount);

    candidate->accumulated = updated;
    candidate->encoded     = encoded;
    candidate->age         = ++nextAge_;
    accumulated            = updated;
    ABJNNUE_RUNTIME_STAT(++stats_.hits);
    ABJNNUE_RUNTIME_STAT(stats_.deltaRows += totalRows);
    return true;
}

void RefreshCache::store(const EncodedPosition& encoded,
                         const AccumulatedPosition& accumulated) {
    Entry* target = nullptr;
    for (auto& entry : entries_)
        if (!entry.valid)
        {
            target = &entry;
            break;
        }
    if (!target)
        target = &*std::min_element(entries_.begin(), entries_.end(),
                                    [](const Entry& lhs, const Entry& rhs) {
                                        return lhs.age < rhs.age;
                                    });
    target->encoded     = encoded;
    target->accumulated = accumulated;
    target->age         = ++nextAge_;
    target->valid       = true;
}

AccumulatorStack::AccumulatorStack() { reset(); }

void AccumulatorStack::reset() {
    size_ = 1;
    entries_[0].computed = false;
    entries_[0].metadata.valid = false;
    stats_ = {};
}

void AccumulatorStack::push(const DirtyPiece& dirtyPiece, const Position& position) {
    if (size_ >= entries_.size()) throw std::runtime_error("ABJNNUE accumulator stack overflow");
    validate_dirty_piece(dirtyPiece, size_);
    Entry& entry = entries_[size_++];
    entry.dirtyPiece = dirtyPiece;
    entry.metadata = capture_metadata(position);
    entry.computed = false;
}

const DirtyPiece& AccumulatorStack::latest_dirty_piece() const {
    if (size_ == 0) throw std::logic_error("ABJNNUE accumulator stack underflow");
    return entries_[size_ - 1].dirtyPiece;
}

void AccumulatorStack::pop() {
    if (size_ <= 1) throw std::logic_error("ABJNNUE accumulator root pop");
    --size_;
}

AccumulatorStack::Metadata AccumulatorStack::capture_metadata(const Position& position) {
    Metadata metadata;
    metadata.layerStackBucket = FeatureEncoder::layer_stack_bucket(position);
    metadata.blendQ8 = FeatureEncoder::layer_stack_selection(position).blendQ8;
    metadata.darkSquares = FeatureEncoder::dark_square_count(position);
    const auto restCount = [&](Color color) {
        int total = 0;
        for (PieceType type : RestOrder)
            total += position.rest_piece(make_piece(color, type));
        return total;
    };
    const auto strongCount = [&](Color color) {
        return position.rest_piece(make_piece(color, ROOK))
             + position.rest_piece(make_piece(color, CANNON))
             + position.rest_piece(make_piece(color, KNIGHT));
    };
    const auto density = [](std::size_t count) {
        if (count == 0) return 0;
        if (count <= 2) return 1;
        if (count <= 4) return 2;
        if (count <= 8) return 3;
        if (count <= 12) return 4;
        if (count <= 16) return 5;
        if (count <= 24) return 6;
        return 7;
    };
    const auto darkByColor = [&](Color color) {
        int count = 0;
        for (Square square = SQ_A0; square <= SQ_I9; ++square)
            if (position.is_dark(square) && color_of(position.piece_on(square)) == color)
                ++count;
        return count;
    };
    const auto threatSummary = [&](Color color) {
        std::array<int, 2> result{};
        const Color enemy = ~color;
        for (Square square = SQ_A0; square <= SQ_I9; ++square)
        {
            const Piece target = position.piece_on(square);
            if (target == NO_PIECE || position.is_dark(square)) continue;
            const auto attackers = position.attackers_to(square)
                                 & position.pieces(color) & ~position.pieces(DARK);
            const auto enemyAttackers = position.attackers_to(square)
                                      & position.pieces(enemy) & ~position.pieces(DARK);
            if (color_of(target) == enemy && attackers) ++result[0];
            if (color_of(target) == color && enemyAttackers) ++result[1];
        }
        result[0] = std::clamp(result[0], 0, 7);
        result[1] = std::clamp(result[1], 0, 7);
        return result;
    };
    for (Color perspective : {WHITE, BLACK})
    {
        metadata.kingTransforms[perspective] = FeatureEncoder::king_transform(position, perspective);
        metadata.attackBuckets[perspective] = static_cast<std::uint8_t>(
          FeatureEncoder::attack_bucket(position, perspective));
        metadata.midMirrors[perspective] = FeatureEncoder::requires_mid_mirror(position, perspective);
        for (std::size_t typeIndex = 0; typeIndex < std::size(RestOrder); ++typeIndex)
        {
            const int count = position.rest_piece(make_piece(perspective, RestOrder[typeIndex]));
            if (count < 0 || count > 16)
                throw std::runtime_error("ABJNNUE V8 rest-piece count is outside its domain");
            metadata.restCounts[perspective][typeIndex] = static_cast<std::uint8_t>(count);
        }
        const Color enemy = ~perspective;
        const auto threats = threatSummary(perspective);
        const int unknownLoss = restCount(perspective) - darkByColor(perspective);
        const int enemyUnknownLoss = restCount(enemy) - darkByColor(enemy);
        if (unknownLoss < 0 || unknownLoss > 15
            || enemyUnknownLoss < 0 || enemyUnknownLoss > 15)
            throw std::runtime_error(
              "ABJNNUE V8.1 unknown_loss is outside the observation domain");
        metadata.metaOffsets[perspective] = {
          static_cast<std::uint8_t>(std::clamp(strongCount(perspective), 0, 7)),
          static_cast<std::uint8_t>(8 + std::clamp(strongCount(enemy), 0, 7)),
          static_cast<std::uint8_t>(16 + std::clamp(restCount(perspective), 0, 7)),
          static_cast<std::uint8_t>(24 + std::clamp(restCount(enemy), 0, 7)),
          static_cast<std::uint8_t>(32 + density(metadata.darkSquares)),
          static_cast<std::uint8_t>(40 + std::min(3, unknownLoss)),
          static_cast<std::uint8_t>(44 + std::min(3, enemyUnknownLoss)),
          static_cast<std::uint8_t>(48 + threats[0]),
          static_cast<std::uint8_t>(56 + threats[1])};
    }
    metadata.valid = true;
    return metadata;
}

AccumulatorStack::Metadata
AccumulatorStack::metadata_from_encoded(const EncodedPosition& encoded) {
    Metadata metadata;
    metadata.layerStackBucket = encoded.layerStackBucket;
    metadata.blendQ8 = encoded.layerStackSelection.blendQ8;
    metadata.darkSquares = encoded.darkSquares;
    for (Color perspective : {WHITE, BLACK})
    {
        const auto& side = encoded.perspectives[perspective];
        metadata.kingTransforms[perspective] = {side.kingBucket, side.mirror};
        metadata.attackBuckets[perspective] = static_cast<std::uint8_t>(side.attackBucket);
        metadata.midMirrors[perspective] = side.midMirror;
        metadata.restCounts[perspective] = side.restCounts;
        metadata.metaOffsets[perspective] = side.metaOffsets;
    }
    metadata.valid = true;
    return metadata;
}

bool AccumulatorStack::compatible(const Metadata& before, const Metadata& after) {
    if (!before.valid || !after.valid) return false;
    for (Color perspective : {WHITE, BLACK})
        if (before.kingTransforms[perspective].bucket != after.kingTransforms[perspective].bucket
            || before.kingTransforms[perspective].mirror != after.kingTransforms[perspective].mirror
            || before.attackBuckets[perspective] != after.attackBuckets[perspective]
            || before.midMirrors[perspective] != after.midMirrors[perspective])
            return false;
    return true;
}

void AccumulatorStack::apply_transition(const Model& model,
                                        const Entry& before,
                                        const Entry& after,
                                        AccumulatedPosition& accumulated) {
    const DirtyPiece& dirty = after.dirtyPiece;
    if (dirty.pc == DARK_PIECE && dirty.to == SQ_NONE && dirty.add_sq == SQ_NONE)
        throw std::logic_error("unresolved ABJNNUE dark move requires refresh");
    for (Color perspective : {WHITE, BLACK})
    {
        auto& destination = accumulated.perspectives[perspective];
        const auto transform = after.metadata.kingTransforms[perspective];
        const auto bucket = transform.bucket * RuntimeLayout::AttackBuckets
                          + after.metadata.attackBuckets[perspective];
        const auto boardIndex = [&](Square square, Piece piece) {
            return FeatureEncoder::board_index(perspective, square, piece, bucket,
                                               transform.mirror);
        };
        FeatureIndices added{}, removed{};
        std::size_t addedCount = 0, removedCount = 0;
        const auto append = [&](std::uint32_t index, bool add) {
            if (add)
            {
                if (addedCount >= added.size())
                    throw std::logic_error("ABJNNUE V8 incremental add list overflow");
                added[addedCount++] = index;
            }
            else
            {
                if (removedCount >= removed.size())
                    throw std::logic_error("ABJNNUE V8 incremental remove list overflow");
                removed[removedCount++] = index;
            }
        };
        append(boardIndex(dirty.from, dirty.pc), false);
        if (dirty.to != SQ_NONE) append(boardIndex(dirty.to, dirty.pc), true);
        if (dirty.add_sq != SQ_NONE) append(boardIndex(dirty.add_sq, dirty.add_pc), true);
        if (dirty.remove_sq != SQ_NONE) append(boardIndex(dirty.remove_sq, dirty.remove_pc), false);
        for (Color owner : {WHITE, BLACK})
            for (std::size_t typeIndex = 0; typeIndex < std::size(RestOrder); ++typeIndex)
            {
                const int oldCount = before.metadata.restCounts[owner][typeIndex];
                const int newCount = after.metadata.restCounts[owner][typeIndex];
                const auto index = FeatureEncoder::dark_rest_index(
                  perspective, owner, RestOrder[typeIndex], bucket);
                if (newCount > oldCount)
                    for (int n = oldCount; n < newCount; ++n) append(index, true);
                else if (oldCount > newCount)
                    for (int n = newCount; n < oldCount; ++n) append(index, false);
            }
        for (std::size_t i = 0; i < before.metadata.metaOffsets[perspective].size(); ++i)
        {
            const auto oldOffset = before.metadata.metaOffsets[perspective][i];
            const auto newOffset = after.metadata.metaOffsets[perspective][i];
            if (oldOffset == newOffset) continue;
            append(FeatureEncoder::meta_index(perspective, bucket, oldOffset), false);
            append(FeatureEncoder::meta_index(perspective, bucket, newOffset), true);
        }
        apply_features(model, destination, added, addedCount, removed, removedCount);
    }
}

void AccumulatorStack::refresh(const Model& model,
                               const Position& position,
                               Entry& entry,
                               RefreshCache* cache) {
    const auto encoded = FeatureEncoder::encode(position);
    if (cache && cache->try_refresh(model, encoded, entry.accumulated))
    {
        entry.metadata = metadata_from_encoded(encoded);
        entry.computed = true;
        ABJNNUE_RUNTIME_STAT(++stats_.refreshes);
        ABJNNUE_RUNTIME_STAT(++stats_.cacheRefreshes);
        return;
    }
    entry.accumulated = FeatureEncoder::accumulate(model, encoded);
    if (cache) cache->store(encoded, entry.accumulated);
    entry.metadata = metadata_from_encoded(encoded);
    entry.computed = true;
    ABJNNUE_RUNTIME_STAT(++stats_.refreshes);
    ABJNNUE_RUNTIME_STAT(++stats_.fullRefreshes);
}

AccumulatorStack::View AccumulatorStack::evaluate(const Model& model, const Position& position) {
    Entry& current = entries_[size_ - 1];
    if (!current.metadata.valid) current.metadata = capture_metadata(position);
    if (!current.computed)
    {
        std::size_t source = size_ - 1;
        while (source > 0 && !entries_[source - 1].computed) --source;
        const bool foundComputed = source > 0 && entries_[source - 1].computed;
        if (foundComputed) --source;
        bool canIncrement = foundComputed;
        if (canIncrement)
            for (std::size_t next = source + 1; next < size_; ++next)
                if (!compatible(entries_[next - 1].metadata, entries_[next].metadata)
                    || (entries_[next].dirtyPiece.pc == DARK_PIECE
                        && entries_[next].dirtyPiece.to == SQ_NONE
                        && entries_[next].dirtyPiece.add_sq == SQ_NONE))
                { canIncrement = false; break; }
        if (!canIncrement) refresh(model, position, current, nullptr);
        else
        {
            current.accumulated = entries_[source].accumulated;
            for (std::size_t next = source + 1; next < size_; ++next)
                apply_transition(model, entries_[next - 1], entries_[next], current.accumulated);
            current.computed = true;
            ABJNNUE_RUNTIME_STAT(stats_.incrementalUpdates += size_ - source - 1);
            ABJNNUE_RUNTIME_STAT(stats_.fusedUpdates += size_ - source - 1);
        }
    }
    return {current.accumulated, current.metadata.layerStackBucket,
            current.metadata.blendQ8, current.metadata.darkSquares};
}

AccumulatorStack::View AccumulatorStack::evaluate(const Model& model,
                                                  const Position& position,
                                                  RefreshCache& cache) {
    Entry& current = entries_[size_ - 1];
    if (!current.metadata.valid) current.metadata = capture_metadata(position);
    if (!current.computed)
    {
        std::size_t source = size_ - 1;
        while (source > 0 && !entries_[source - 1].computed) --source;
        const bool foundComputed = source > 0 && entries_[source - 1].computed;
        if (foundComputed) --source;
        bool canIncrement = foundComputed;
        if (canIncrement)
            for (std::size_t next = source + 1; next < size_; ++next)
                if (!compatible(entries_[next - 1].metadata, entries_[next].metadata)
                    || (entries_[next].dirtyPiece.pc == DARK_PIECE
                        && entries_[next].dirtyPiece.to == SQ_NONE
                        && entries_[next].dirtyPiece.add_sq == SQ_NONE))
                { canIncrement = false; break; }
        if (!canIncrement) refresh(model, position, current, &cache);
        else
        {
            current.accumulated = entries_[source].accumulated;
            for (std::size_t next = source + 1; next < size_; ++next)
                apply_transition(model, entries_[next - 1], entries_[next], current.accumulated);
            current.computed = true;
            ABJNNUE_RUNTIME_STAT(stats_.incrementalUpdates += size_ - source - 1);
            ABJNNUE_RUNTIME_STAT(stats_.fusedUpdates += size_ - source - 1);
        }
    }
    return {current.accumulated, current.metadata.layerStackBucket,
            current.metadata.blendQ8, current.metadata.darkSquares};
}

}  // namespace ABJNNUE

#undef ABJNNUE_RUNTIME_STAT
