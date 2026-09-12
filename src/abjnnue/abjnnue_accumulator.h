#ifndef ABJNNUE_ACCUMULATOR_H_INCLUDED
#define ABJNNUE_ACCUMULATOR_H_INCLUDED

#include <array>
#include <cstddef>
#include <cstdint>

#include "../types.h"
#include "abjnnue_feature_encoder.h"

namespace Stockfish {
class Position;
}

namespace ABJNNUE {

struct AccumulatorStats {
    std::uint64_t refreshes = 0;
    std::uint64_t fullRefreshes = 0;
    std::uint64_t cacheRefreshes = 0;
    std::uint64_t incrementalUpdates = 0;
    std::uint64_t fusedUpdates = 0;
};

struct RefreshCacheStats {
    std::uint64_t lookups = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t fallbacks = 0;
    std::uint64_t deltaRows = 0;
};

class RefreshCache {
   public:
    static constexpr std::size_t EntryCount = 32;
    static constexpr std::size_t MaxDeltaRows = 128;

    RefreshCache() { clear(); }

    void clear() noexcept;
    const RefreshCacheStats& stats() const noexcept { return stats_; }

   private:
    struct Entry {
        AccumulatedPosition accumulated;
        EncodedPosition     encoded;
        std::uint64_t       age = 0;
        bool                valid = false;
    };

    std::array<Entry, EntryCount> entries_{};
    std::uint64_t                 nextAge_ = 0;
    RefreshCacheStats             stats_{};

    bool try_refresh(const Model& model,
                     const EncodedPosition& encoded,
                     AccumulatedPosition& accumulated);
    void store(const EncodedPosition& encoded, const AccumulatedPosition& accumulated);

    friend class AccumulatorStack;
};

class AccumulatorStack {
   public:
    struct View {
        const AccumulatedPosition& accumulated;
        std::uint32_t              layerStackBucket;
        std::uint8_t               layerStackBlendQ8;
        std::size_t                darkSquares;
    };

    AccumulatorStack();
    void reset();
    void push(const Stockfish::DirtyPiece& dirtyPiece, const Stockfish::Position& position);
    void pop();
    const Stockfish::DirtyPiece& latest_dirty_piece() const;
    View evaluate(const Model& model, const Stockfish::Position& position);
    View evaluate(const Model& model,
                  const Stockfish::Position& position,
                  RefreshCache& cache);
    const AccumulatorStats& stats() const noexcept { return stats_; }

   private:
    static constexpr std::size_t RestTypes = 6;

    struct Metadata {
        std::uint32_t layerStackBucket = 0;
        std::uint8_t blendQ8 = 0;
        std::size_t darkSquares = 0;
        std::array<KingTransform, Stockfish::COLOR_NB> kingTransforms{};
        std::array<std::uint8_t, Stockfish::COLOR_NB> attackBuckets{};
        std::array<bool, Stockfish::COLOR_NB> midMirrors{};
        std::array<std::array<std::uint8_t, RestTypes>, Stockfish::COLOR_NB> restCounts{};
        std::array<std::array<std::uint8_t, 9>, Stockfish::COLOR_NB> metaOffsets{};
        bool valid = false;
        bool dark_variant() const noexcept { return darkSquares != 0; }
    };

    struct Entry {
        AccumulatedPosition accumulated;
        Stockfish::DirtyPiece dirtyPiece{};
        Metadata metadata;
        bool computed = false;
    };

    static Metadata capture_metadata(const Stockfish::Position& position);
    static Metadata metadata_from_encoded(const EncodedPosition& encoded);
    static bool compatible(const Metadata& before, const Metadata& after);
    static void apply_transition(const Model& model,
                                 const Entry& before,
                                 const Entry& after,
                                 AccumulatedPosition& accumulated);
    void refresh(const Model& model,
                 const Stockfish::Position& position,
                 Entry& entry,
                 RefreshCache* cache);

    std::array<Entry, Stockfish::MAX_PLY + 10> entries_{};
    std::size_t size_ = 1;
    AccumulatorStats stats_{};
};

}  // namespace ABJNNUE

#endif
