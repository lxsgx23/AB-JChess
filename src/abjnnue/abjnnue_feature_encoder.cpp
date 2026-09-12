#include "abjnnue_feature_encoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(USE_AVX2) || defined(USE_SSE2)
    #include <immintrin.h>
#endif

#include "../bitboard.h"
#include "../position.h"

namespace ABJNNUE {
namespace {

using namespace Stockfish;

enum : std::uint32_t {
    // Keep this order byte-for-byte identical to HalfKAv2_hm_jieqi_v3 in
    // training_data_loader.cpp.  V6 used a different alternating piece order;
    // changing it here would make every exported V8 row read the wrong weight.
    PS_US_ROOK      = 0 * SQUARE_NB,
    PS_THEM_ROOK    = 1 * SQUARE_NB,
    PS_US_CANNON    = 2 * SQUARE_NB,
    PS_THEM_CANNON  = 3 * SQUARE_NB,
    PS_US_KNIGHT    = 4 * SQUARE_NB,
    PS_THEM_KNIGHT  = 5 * SQUARE_NB,
    PS_US_PAWN      = 6 * SQUARE_NB,
    PS_THEM_PAWN    = 7 * SQUARE_NB,
    PS_US_ADVISOR   = 8 * SQUARE_NB,
    PS_THEM_ADVISOR = 9 * SQUARE_NB,
    PS_US_BISHOP    = 10 * SQUARE_NB,
    PS_THEM_BISHOP  = 11 * SQUARE_NB,
    PS_KING         = 12 * SQUARE_NB,
    PS_DARK         = 13 * SQUARE_NB,
};

constexpr std::uint32_t PieceSquareIndex[COLOR_NB][PIECE_NB + 1] = {
  {PS_KING, PS_US_ROOK, PS_US_ADVISOR, PS_US_CANNON, PS_US_PAWN, PS_US_KNIGHT,
   PS_US_BISHOP, PS_KING, PS_KING, PS_THEM_ROOK, PS_THEM_ADVISOR,
   PS_THEM_CANNON, PS_THEM_PAWN, PS_THEM_KNIGHT, PS_THEM_BISHOP, PS_KING,
   PS_DARK},
  {PS_KING, PS_THEM_ROOK, PS_THEM_ADVISOR, PS_THEM_CANNON, PS_THEM_PAWN,
   PS_THEM_KNIGHT, PS_THEM_BISHOP, PS_KING, PS_KING, PS_US_ROOK, PS_US_ADVISOR,
   PS_US_CANNON, PS_US_PAWN, PS_US_KNIGHT, PS_US_BISHOP, PS_KING, PS_DARK},
};

// The high bit is the canonical file-mirror hint used by AB-JChess's
// HalfKAv2_hm encoder. Only the six king buckets are part of the ABI.
constexpr std::uint8_t KingBucket[SQUARE_NB] = {
  0, 0, 0, 0, 1, std::uint8_t(8 | 0), 0, 0, 0,
  0, 0, 0, 2, 3, std::uint8_t(8 | 2), 0, 0, 0,
  0, 0, 0, 4, 5, std::uint8_t(8 | 4), 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 4, 5, std::uint8_t(8 | 4), 0, 0, 0,
  0, 0, 0, 2, 3, std::uint8_t(8 | 2), 0, 0, 0,
  0, 0, 0, 0, 1, std::uint8_t(8 | 0), 0, 0, 0,
};
static_assert(std::size(KingBucket) == SQUARE_NB);

constexpr PieceType RestOrder[] = {ROOK, CANNON, KNIGHT, BISHOP, ADVISOR, PAWN};
constexpr std::uint64_t MidBalanceEncoding = 0xa4a92a74e989d3a7ULL;

bool valid_feature_piece(Piece piece) {
    const int value = int(piece);
    return value == int(DARK_PIECE)
        || (value > int(NO_PIECE) && value < int(DARK_PIECE) && value != 8);
}

std::uint64_t mid_mirror_encoding(Piece piece, Square square) {
    static constexpr std::uint8_t shifts[8][2] = {
      {0, 0}, {44, 0}, {60, 36}, {47, 7},
      {53, 21}, {50, 14}, {57, 29}, {0, 0}};
    const int type = int(type_of(piece));
    if (type < 1 || type > 7 || file_of(square) == FILE_E)
        return 0;
    if (type == KING)
        return 1ULL << 63;
    const int file = int(file_of(square));
    const int rank = int(rank_of(square));
    const int relativeRank = color_of(piece) == WHITE ? rank : 9 - rank;
    const int relativeFile = file < int(FILE_E) ? file : 8 - file;
    const auto s1 = shifts[type][0];
    const auto s2 = shifts[type][1];
    const std::uint64_t value = (1ULL << s1)
                              | ((std::uint64_t(3 - relativeFile) * 10
                                  + std::uint64_t(relativeRank)) << s2);
    return file < int(FILE_E) ? value : std::uint64_t(-std::int64_t(value));
}

std::uint64_t mid_encoding(const Position& position, Color color) {
    std::uint64_t encoding = MidBalanceEncoding;
    for (Square square = SQ_A0; square <= SQ_I9; ++square)
    {
        const Piece piece = position.piece_on(square);
        if (piece == NO_PIECE || position.is_dark(square) || color_of(piece) != color)
            continue;
        encoding += mid_mirror_encoding(piece, square);
    }
    return encoding;
}

bool requires_mid_mirror_impl(const Position& position, Color color) {
    const auto own = mid_encoding(position, color);
    const auto enemy = mid_encoding(position, ~color);
    return ((1ULL << 63) & own & enemy)
        && (own < MidBalanceEncoding
            || (own == MidBalanceEncoding && enemy < MidBalanceEncoding));
}

KingTransform make_king_transform(const Position& position,
                                  Color perspective,
                                  bool midMirror) {
    const Square own = position.king_square(perspective);
    const Square opp = position.king_square(~perspective);
    const auto ownCode = KingBucket[own];
    const auto oppCode = KingBucket[opp];
    const auto ownBucket = ownCode & 0x7;
    const auto oppBucket = oppCode & 0x7;
    const bool mirror = (ownCode >> 3)
                     || ((ownBucket & 1)
                         && ((oppCode >> 3) || ((oppBucket & 1) && midMirror)));
    return {static_cast<std::uint32_t>(ownBucket), mirror};
}

std::uint32_t map_square(Color perspective, Square square, bool mirror) {
    if (mirror) square = flip_file(square);
    if (perspective == BLACK) square = flip_rank(square);
    return static_cast<std::uint32_t>(square);
}

std::uint32_t make_board_index(Color perspective,
                               Square square,
                               Piece piece,
                               std::uint32_t bucket,
                               bool mirror) {
    if (!valid_feature_piece(piece) || bucket >= RuntimeLayout::FeatureBuckets)
        throw std::logic_error("invalid ABJNNUE V8 board feature input");
    const auto index = bucket * RuntimeLayout::PieceSquareDimensions
                     + PieceSquareIndex[perspective][piece]
                     + map_square(perspective, square, mirror);
    if (index >= RuntimeLayout::FeatureDimensions)
        throw std::logic_error("ABJNNUE V8 board feature index is out of range");
    return index;
}

int bucket8(int value) { return std::clamp(value, 0, 7); }

int rest_count(const Position& position, Color color) {
    int count = 0;
    for (PieceType type : RestOrder)
        count += position.rest_piece(make_piece(color, type));
    return count;
}

int strong_rest_count(const Position& position, Color color) {
    return position.rest_piece(make_piece(color, ROOK))
         + position.rest_piece(make_piece(color, CANNON))
         + position.rest_piece(make_piece(color, KNIGHT));
}

int dark_count_for_color(const Position& position, Color color) {
    int count = 0;
    for (Square square = SQ_A0; square <= SQ_I9; ++square)
        if (position.is_dark(square) && color_of(position.piece_on(square)) == color)
            ++count;
    return count;
}

std::array<int, 2> visible_threat_summary(const Position& position, Color color) {
    const Color enemy = ~color;
    int enemyTargets = 0;
    int ownTargets = 0;
    const auto dark = position.pieces(DARK);
    for (Square square = SQ_A0; square <= SQ_I9; ++square)
    {
        const Piece target = position.piece_on(square);
        if (target == NO_PIECE || position.is_dark(square)) continue;
        const auto attackers = position.attackers_to(square) & position.pieces(color) & ~dark;
        const auto enemyAttackers = position.attackers_to(square) & position.pieces(enemy) & ~dark;
        if (color_of(target) == enemy && attackers) ++enemyTargets;
        if (color_of(target) == color && enemyAttackers) ++ownTargets;
    }
    return {std::clamp(enemyTargets, 0, 7), std::clamp(ownTargets, 0, 7)};
}

int dark_density_bucket(int count) {
    if (count == 0) return 0;
    if (count <= 2) return 1;
    if (count <= 4) return 2;
    if (count <= 8) return 3;
    if (count <= 12) return 4;
    if (count <= 16) return 5;
    if (count <= 24) return 6;
    return 7;
}

int uncertainty_phase_bucket(int darkCount, int strongRest) {
    const int darkBucket = darkCount >= 17 ? 4 : darkCount >= 9 ? 3 : darkCount >= 5 ? 2
                                  : darkCount >= 1 ? 1 : 0;
    const int restBucket = strongRest >= 9 ? 3 : strongRest >= 5 ? 2 : strongRest >= 2 ? 1 : 0;
    return darkBucket + restBucket;
}

double interval_residual(int value, const int* thresholds, std::size_t count) {
    value = std::max(0, value);
    std::size_t bucket = 0;
    while (bucket + 1 < count && value >= thresholds[bucket + 1]) ++bucket;
    if (bucket + 1 >= count) return 0.0;
    const int lower = thresholds[bucket];
    const int upper = thresholds[bucket + 1];
    if (upper <= lower) return 0.0;
    return double(std::clamp(value - lower, 0, upper - lower - 1))
         / double(upper - lower);
}

double piece_residual(int pieceCount) {
    static constexpr int lower[16] = {2, 7, 9, 11, 13, 15, 17, 19,
                                      21, 23, 25, 27, 29, 31, 32, 33};
    static constexpr int upper[16] = {7, 9, 11, 13, 15, 17, 19, 21,
                                      23, 25, 27, 29, 31, 32, 33, 34};
    static constexpr int bases[33] = {
      -1, -1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
      6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 14, 15};
    const int count = std::clamp(pieceCount, 2, 32);
    const int bucket = std::clamp(bases[count], 0, 15);
    return double(std::clamp(count - lower[bucket], 0, upper[bucket] - lower[bucket] - 1))
         / double(upper[bucket] - lower[bucket]);
}

void append_rest_features(const Position& position,
                          Color perspective,
                          std::uint32_t bucket,
                          FeatureIndexList& active) {
    for (Color owner : {WHITE, BLACK})
        for (PieceType type : RestOrder)
        {
            const int count = position.rest_piece(make_piece(owner, type));
            if (count < 0 || count > 16)
                throw std::runtime_error("ABJNNUE V8 rest-piece count is invalid");
            const auto index = FeatureEncoder::dark_rest_index(perspective, owner, type, bucket);
            for (int n = 0; n < count; ++n)
                active.push_back(index);
        }
}

void append_meta_features(const Position& position,
                          Color perspective,
                          std::uint32_t bucket,
                          FeatureIndexList& active,
                          std::array<std::uint8_t, 9>* offsets = nullptr) {
    const Color enemy = ~perspective;
    std::size_t offsetIndex = 0;
    const auto emit = [&](std::uint32_t offset) {
        if (offsets != nullptr)
            (*offsets)[offsetIndex++] = static_cast<std::uint8_t>(offset);
        active.push_back(FeatureEncoder::meta_index(perspective, bucket, offset));
    };
    const int darkCount = static_cast<int>(FeatureEncoder::dark_square_count(position));
    const auto threats = visible_threat_summary(position, perspective);
    const int unknownLoss = rest_count(position, perspective) - dark_count_for_color(position, perspective);
    const int enemyUnknownLoss = rest_count(position, enemy) - dark_count_for_color(position, enemy);
    if (unknownLoss < 0 || unknownLoss > 15
        || enemyUnknownLoss < 0 || enemyUnknownLoss > 15)
        throw std::runtime_error(
          "ABJNNUE V8.1 unknown_loss is outside the observation domain");
    emit(bucket8(strong_rest_count(position, perspective)));
    emit(8 + bucket8(strong_rest_count(position, enemy)));
    emit(16 + bucket8(rest_count(position, perspective)));
    emit(24 + bucket8(rest_count(position, enemy)));
    emit(32 + dark_density_bucket(darkCount));
    emit(40 + std::min(3, unknownLoss));
    emit(44 + std::min(3, enemyUnknownLoss));
    emit(48 + threats[0]);
    emit(56 + threats[1]);
}

#if !defined(USE_SSE2)
std::int16_t add_i16(std::int16_t lhs, std::int16_t rhs) {
    return static_cast<std::int16_t>(static_cast<std::uint16_t>(lhs)
                                   + static_cast<std::uint16_t>(rhs));
}
std::int32_t add_i32(std::int32_t lhs, std::int32_t rhs) {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(lhs)
                                   + static_cast<std::uint32_t>(rhs));
}
#endif

}  // namespace

std::size_t FeatureEncoder::dark_square_count(const Position& position) {
    return static_cast<std::size_t>(popcount(position.pieces(DARK)));
}

std::uint32_t FeatureEncoder::attack_bucket(const Position& position, Color perspective) {
    bool hasRook = false;
    bool hasKnightOrCannon = false;
    for (Square square = SQ_A0; square <= SQ_I9; ++square)
    {
        const Piece piece = position.piece_on(square);
        if (piece == NO_PIECE || position.is_dark(square) || color_of(piece) != perspective)
            continue;
        switch (type_of(piece))
        {
        case ROOK: hasRook = true; break;
        case KNIGHT:
        case CANNON: hasKnightOrCannon = true; break;
        default: break;
        }
    }
    return static_cast<std::uint32_t>(hasRook ? 2 : 0) + (hasKnightOrCannon ? 1 : 0);
}

bool FeatureEncoder::requires_mid_mirror(const Position& position, Color perspective) {
    return requires_mid_mirror_impl(position, perspective);
}

std::uint32_t FeatureEncoder::layer_stack_bucket(const Position& position) {
    static constexpr int baseBuckets[33] = {
      -1, -1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
      6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 14, 15};
    const int pieceCount = std::clamp(position.count<ALL_PIECES>(), 2, 32);
    const int darkCount = static_cast<int>(dark_square_count(position));
    const int strongRest = strong_rest_count(position, WHITE) + strong_rest_count(position, BLACK);
    const int bucket = baseBuckets[pieceCount] + uncertainty_phase_bucket(darkCount, strongRest);
    return static_cast<std::uint32_t>(std::clamp(bucket, 0, 15));
}

LayerStackSelection FeatureEncoder::layer_stack_selection(const Position& position) {
    static constexpr int darkThresholds[] = {0, 1, 5, 9, 17, 33};
    static constexpr int restThresholds[] = {0, 2, 5, 9, 19};
    const int pieceCount = std::clamp(position.count<ALL_PIECES>(), 2, 32);
    const int darkCount = static_cast<int>(dark_square_count(position));
    const int strongRest = strong_rest_count(position, WHITE)
                         + strong_rest_count(position, BLACK);
    const auto floor = layer_stack_bucket(position);
    if (floor >= 15) return {15, 0};
    const double fraction = (piece_residual(pieceCount)
                           + interval_residual(darkCount, darkThresholds,
                                                std::size(darkThresholds))
                           + interval_residual(strongRest, restThresholds,
                                                std::size(restThresholds))) / 3.0;
    const int rounded = static_cast<int>(std::floor(fraction * 255.0 + 0.5));
    if (rounded >= 255) return {std::min<std::uint32_t>(15, floor + 1), 0};
    return {floor, static_cast<std::uint8_t>(std::clamp(rounded, 0, 254))};
}

KingTransform FeatureEncoder::king_transform(const Position& position, Color perspective) {
    return make_king_transform(position, perspective, requires_mid_mirror(position, perspective));
}

std::uint32_t FeatureEncoder::board_index(Color perspective,
                                          Square square,
                                          Piece piece,
                                          std::uint32_t bucket,
                                          bool mirror) {
    return make_board_index(perspective, square, piece, bucket, mirror);
}

std::uint32_t FeatureEncoder::meta_index(Color /*perspective*/,
                                         std::uint32_t bucket,
                                         std::uint32_t offset) {
    if (bucket >= RuntimeLayout::FeatureBuckets || offset >= RuntimeLayout::MetaDimensions)
        throw std::logic_error("ABJNNUE V8 meta feature index is out of range");
    return bucket * RuntimeLayout::PieceSquareDimensions
         + RuntimeLayout::BasePieceSquareDimensions + offset;
}

std::uint32_t FeatureEncoder::dark_rest_index(Color perspective,
                                              Color owner,
                                              PieceType type,
                                              std::uint32_t bucket) {
    std::uint32_t offset = 0;
    switch (type)
    {
    case ROOK: offset = 0; break;
    case ADVISOR: offset = 1; break;
    case CANNON: offset = 2; break;
    case PAWN: offset = 3; break;
    case KNIGHT: offset = 4; break;
    case BISHOP: offset = 5; break;
    default: throw std::invalid_argument("piece type has no ABJNNUE V8 rest feature");
    }
    const std::uint32_t base = owner == perspective ? 9 : 72;
    if (bucket >= RuntimeLayout::FeatureBuckets)
        throw std::logic_error("ABJNNUE V8 rest feature bucket is out of range");
    return bucket * RuntimeLayout::PieceSquareDimensions + PS_DARK + base + offset;
}

EncodedPosition FeatureEncoder::encode(const Position& position) {
    EncodedPosition output;
    output.darkSquares = dark_square_count(position);
    output.layerStackBucket = layer_stack_bucket(position);
    output.layerStackSelection = layer_stack_selection(position);

    for (Color perspective : {WHITE, BLACK})
    {
        auto& side = output.perspectives[perspective];
        const bool midMirror = requires_mid_mirror(position, perspective);
        const auto transform = make_king_transform(position, perspective, midMirror);
        const auto attack = attack_bucket(position, perspective);
        const auto bucket = transform.bucket * RuntimeLayout::AttackBuckets + attack;
        side.perspective = perspective;
        side.kingBucket = transform.bucket;
        side.attackBucket = attack;
        side.mirror = transform.mirror;
        side.midMirror = midMirror;

        for (Square square = SQ_A0; square <= SQ_I9; ++square)
        {
            const Piece piece = position.piece_on(square);
            if (piece == NO_PIECE) continue;
            const Piece featurePiece = position.is_dark(square) ? DARK_PIECE : piece;
            side.active.push_back(board_index(perspective, square, featurePiece, bucket,
                                              transform.mirror));
        }
        append_rest_features(position, perspective, bucket, side.active);
        append_meta_features(position, perspective, bucket, side.active, &side.metaOffsets);
        side.active.sort();
        for (std::size_t i = 0; i < std::size(side.restCounts); ++i)
        {
            const int count = position.rest_piece(make_piece(perspective, RestOrder[i]));
            side.restCounts[i] = static_cast<std::uint8_t>(std::clamp(count, 0, 255));
        }
    }
    return output;
}

AccumulatedPosition FeatureEncoder::accumulate(const Model& model,
                                               const EncodedPosition& encoded) {
    if (!model.inference_complete()) throw std::invalid_argument("ABJNNUE V8 model is incomplete");
    AccumulatedPosition output;
    for (Color perspective : {WHITE, BLACK})
    {
        auto& dst = output.perspectives[perspective];
        std::copy(model.transformer_biases().begin(), model.transformer_biases().end(),
                  dst.values.begin());
        for (const std::uint32_t index : encoded.perspectives[perspective].active)
        {
            if (index >= RuntimeLayout::FeatureDimensions)
                throw std::logic_error("ABJNNUE V8 active feature index is out of range");
            const std::size_t weightOffset = std::size_t(index) * RuntimeLayout::AccumulatorWidth;
#if defined(USE_AVX2)
            for (std::size_t column = 0; column < RuntimeLayout::AccumulatorWidth; column += 16)
            {
                auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst.values.data() + column));
                value = _mm256_add_epi16(value, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    model.feature_weights().data() + weightOffset + column)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.values.data() + column), value);
            }
#elif defined(USE_SSE2)
            for (std::size_t column = 0; column < RuntimeLayout::AccumulatorWidth; column += 8)
            {
                auto value = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst.values.data() + column));
                value = _mm_add_epi16(value, _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    model.feature_weights().data() + weightOffset + column)));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst.values.data() + column), value);
            }
#else
            for (std::size_t column = 0; column < RuntimeLayout::AccumulatorWidth; ++column)
                dst.values[column] = add_i16(dst.values[column],
                    model.feature_weights()[weightOffset + column]);
#endif
            const std::size_t psqtOffset = std::size_t(index) * RuntimeLayout::PSQTBuckets;
#if defined(USE_AVX2)
            for (std::size_t bucket = 0; bucket < RuntimeLayout::PSQTBuckets; bucket += 8)
            {
                auto psqt = _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(dst.psqt.data() + bucket));
                psqt = _mm256_add_epi32(
                  psqt, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                          model.psqt_weights().data() + psqtOffset + bucket)));
                _mm256_storeu_si256(
                  reinterpret_cast<__m256i*>(dst.psqt.data() + bucket), psqt);
            }
#elif defined(USE_SSE2)
            for (std::size_t bucket = 0; bucket < RuntimeLayout::PSQTBuckets; bucket += 4)
            {
                auto psqt = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst.psqt.data() + bucket));
                psqt = _mm_add_epi32(psqt, _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    model.psqt_weights().data() + psqtOffset + bucket)));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst.psqt.data() + bucket), psqt);
            }
#else
            for (std::size_t bucket = 0; bucket < RuntimeLayout::PSQTBuckets; ++bucket)
                dst.psqt[bucket] = add_i32(dst.psqt[bucket],
                    model.psqt_weights()[psqtOffset + bucket]);
#endif
        }
    }
    return output;
}

}  // namespace ABJNNUE
