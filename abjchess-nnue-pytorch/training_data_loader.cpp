#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <filesystem>
#include <charconv>
#include <mutex>
#include <thread>
#include <deque>
#include <random>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <unordered_map>
#include <optional>
#include <type_traits>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"

#include "jqv4/jqv4_hash.h"
#include "jqv4/jqv4_json.h"
#include "jqv4/jqv4_reader.h"
#include "jqv4/jqv4_rng.h"
#include "jqv4/jqv4_sha256.h"
#include "jqv4/jqv4_wire.h"
#include "v8_jqv4_shuffle.h"

#if defined (_WIN32)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#elif defined (__GNUC__) || defined (__clang__)
#define EXPORT __attribute__ ((visibility ("default")))
#define CDECL
#else
#define EXPORT
#define CDECL
#endif

using namespace bin;
using namespace chess;

namespace jieqi {

static constexpr std::uint32_t MAGIC = 0x514A4B50u;
static constexpr std::uint32_t VERSION = 3;
static constexpr int FEN_BYTES = 256;
static constexpr int SQUARE_NB = 90;
static constexpr int PS_NB = 14 * SQUARE_NB;

enum Color { WHITE = 0, BLACK = 1 };

enum Piece : int {
    NO_PIECE = 0,
    W_ROOK, W_ADVISOR, W_CANNON, W_PAWN, W_KNIGHT, W_BISHOP, W_KING,
    B_ROOK = 9, B_ADVISOR, B_CANNON, B_PAWN, B_KNIGHT, B_BISHOP, B_KING,
    DARK_PIECE = 16
};

#pragma pack(push, 1)
struct PackedSfenValue {
    char fen[FEN_BYTES];
    std::int16_t score;
    std::uint32_t move;
    std::uint16_t gamePly;
    std::int8_t game_result;
    std::uint8_t reserved[5];
};
#pragma pack(pop)

static_assert(sizeof(PackedSfenValue) == 270);

struct Position {
    Piece board[SQUARE_NB]{};
    bool dark[SQUARE_NB]{};
    int rest[16]{};
    int unknown_loss[2]{};
    // Raw binary FEN may include an optional ``|U...u...`` suffix. V8 derives
    // the same values from rest/coloured-dark inventory and treats an explicit
    // suffix only as a consistency assertion.
    bool unknown_loss_explicit = false;
    Color side = WHITE;
    std::uint16_t gamePly = 0;

    int piece_count() const {
        int n = 0;
        for (auto p : board)
            n += p != NO_PIECE;
        return n;
    }
};

struct TrainingDataEntry {
    Position pos;
    std::uint32_t move;
    std::int16_t score;
    std::uint16_t ply;
    std::int16_t result;
};

static Piece piece_from_char(char c) {
    switch (c)
    {
    case 'R': return W_ROOK;
    case 'A': return W_ADVISOR;
    case 'C': return W_CANNON;
    case 'P': return W_PAWN;
    case 'N': return W_KNIGHT;
    case 'B': return W_BISHOP;
    case 'K': return W_KING;
    case 'r': return B_ROOK;
    case 'a': return B_ADVISOR;
    case 'c': return B_CANNON;
    case 'p': return B_PAWN;
    case 'n': return B_KNIGHT;
    case 'b': return B_BISHOP;
    case 'k': return B_KING;
    default: return NO_PIECE;
    }
}

static Piece dark_piece_at_square(int sq, Color c) {
    static constexpr char darkPieces[] =
        "RNBAKABNR"
        "........."
        ".C.....C."
        "P.P.P.P.P"
        "........."
        "........."
        "p.p.p.p.p"
        ".c.....c."
        "........."
        "rnbakabnr";
    Piece pc = piece_from_char(darkPieces[sq]);
    if (pc == NO_PIECE)
        return NO_PIECE;
    if (c == WHITE && pc >= B_ROOK)
        pc = Piece(pc - 8);
    if (c == BLACK && pc < B_ROOK)
        pc = Piece(pc + 8);
    return pc;
}

static Color color_of(Piece pc) {
    return pc >= B_ROOK ? BLACK : WHITE;
}

static int type_of(Piece pc) {
    return int(pc) & 7;
}

static int make_piece(Color c, int pt) {
    return (c == BLACK ? 8 : 0) + pt;
}

static int file_of(int sq) { return sq % 9; }
static int rank_of(int sq) { return sq / 9; }
static int make_square(int file, int rank) { return rank * 9 + file; }
static int flip_rank(int sq) { return make_square(file_of(sq), 9 - rank_of(sq)); }
static int flip_file(int sq) { return make_square(8 - file_of(sq), rank_of(sq)); }

[[noreturn]] static void throw_fen_error(const std::string& detail) {
    throw std::runtime_error("Jieqi v3 FEN: " + detail);
}

static std::uint64_t parse_fen_decimal(
    std::string_view text,
    std::uint64_t maximum,
    const char* field)
{
    if (text.empty())
        throw_fen_error(std::string(field) + " is empty");

    std::uint64_t value = 0;
    for (char c : text)
    {
        if (c < '0' || c > '9')
            throw_fen_error(std::string(field) + " must be decimal");
        const auto digit = std::uint64_t(c - '0');
        if (digit > maximum || value > (maximum - digit) / 10)
            throw_fen_error(std::string(field) + " is out of range");
        value = value * 10 + digit;
    }
    return value;
}

static int parse_fen_count(
    const std::string& field,
    std::size_t& offset,
    int maximum,
    const char* label)
{
    const auto begin = offset;
    while (offset < field.size()
           && std::isdigit(static_cast<unsigned char>(field[offset])))
        ++offset;
    if (begin == offset)
        return 1;
    return int(parse_fen_decimal(
        std::string_view(field).substr(begin, offset - begin),
        std::uint64_t(maximum),
        label));
}

static int initial_piece_count(Piece pc) {
    switch (type_of(pc))
    {
    case 1: // rook
    case 2: // advisor
    case 3: // cannon
    case 5: // knight
    case 6: // bishop
        return 2;
    case 4: // pawn
        return 5;
    case 7: // king
        return 1;
    default:
        return 0;
    }
}

static Position parse_fen(const std::string& fen) {
    Position pos;
    std::fill(std::begin(pos.board), std::end(pos.board), NO_PIECE);
    std::fill(std::begin(pos.dark), std::end(pos.dark), false);
    std::fill(std::begin(pos.rest), std::end(pos.rest), 0);
    std::fill(std::begin(pos.unknown_loss), std::end(pos.unknown_loss), 0);

    std::istringstream ss(fen);
    std::string board, stm, rest, rule50_text, fullmove_text, extra;
    if (!(ss >> board >> stm >> rest >> rule50_text >> fullmove_text))
        throw_fen_error("expected exactly five fields");
    if (ss >> extra)
        throw_fen_error("contains extra fields");

    if (stm == "w")
        pos.side = WHITE;
    else if (stm == "b")
        pos.side = BLACK;
    else
        throw_fen_error("side-to-move must be w or b");

    const auto rule50 = parse_fen_decimal(
        rule50_text,
        std::uint64_t(std::numeric_limits<int>::max()),
        "rule counter");
    (void)rule50;

    const auto maximum_fullmove =
        (std::uint64_t(std::numeric_limits<std::uint16_t>::max())
         - std::uint64_t(pos.side == BLACK))
        / 2 + 1;
    const auto fullmove = parse_fen_decimal(
        fullmove_text, maximum_fullmove, "fullmove number");
    if (fullmove == 0)
        throw_fen_error("fullmove number must be positive");
    pos.gamePly = std::uint16_t(
        2 * (fullmove - 1) + std::uint64_t(pos.side == BLACK));

    int rank = 9;
    int file = 0;
    int board_count[2]{};
    int known_count[16]{};
    for (char c : board)
    {
        if (c == '/')
        {
            if (file != 9 || rank <= 0)
                throw_fen_error("board has an invalid rank separator");
            --rank;
            file = 0;
        }
        else if (std::isdigit(static_cast<unsigned char>(c)))
        {
            const int empty_count = c - '0';
            if (empty_count < 1 || empty_count > 9
                || file + empty_count > 9)
                throw_fen_error("board has an invalid empty-square count");
            file += empty_count;
        }
        else
        {
            if (rank < 0 || rank >= 10 || file < 0 || file >= 9)
                throw_fen_error("board square is out of range");
            const int sq = make_square(file, rank);
            if (sq < 0 || sq >= SQUARE_NB)
                throw_fen_error("board square is out of range");

            Piece pc = NO_PIECE;
            if (c == 'X' || c == 'x')
            {
                const Color color = c == 'X' ? WHITE : BLACK;
                pc = dark_piece_at_square(sq, color);
                if (pc == NO_PIECE)
                    throw_fen_error(
                        "board dark piece has no valid placeholder");
                pos.dark[sq] = true;
            }
            else
            {
                pc = piece_from_char(c);
                if (pc == NO_PIECE)
                    throw_fen_error("board contains an invalid piece");
            }

            pos.board[sq] = pc;
            const Color color = color_of(pc);
            if (++board_count[color] > 16)
                throw_fen_error("board has more than 16 pieces for one side");
            if (!pos.dark[sq]
                && ++known_count[pc] > initial_piece_count(pc))
                throw_fen_error("board exceeds the initial piece inventory");
            ++file;
        }
    }
    if (rank != 0 || file != 9)
        throw_fen_error("board must contain exactly 10 ranks of 9 files");

    const std::size_t restSep = rest.find('|');
    if (restSep != std::string::npos
        && rest.find('|', restSep + 1) != std::string::npos)
        throw_fen_error("rest field has multiple separators");
    const std::string restPieces =
        restSep == std::string::npos ? rest : rest.substr(0, restSep);
    const std::string unknowns =
        restSep == std::string::npos ? "" : rest.substr(restSep + 1);
    pos.unknown_loss_explicit = !unknowns.empty();

    if (restPieces.empty())
        throw_fen_error("rest piece field is empty");
    bool seen_rest_piece[16]{};
    if (restPieces != "-")
    {
        for (std::size_t i = 0; i < restPieces.size();)
        {
            const Piece pc = piece_from_char(restPieces[i++]);
            if (pc == NO_PIECE || type_of(pc) == 7)
                throw_fen_error(
                    "rest field contains an invalid or king piece");
            if (seen_rest_piece[pc])
                throw_fen_error("rest field contains a duplicate piece");
            seen_rest_piece[pc] = true;

            const int maximum = initial_piece_count(pc);
            const int count = parse_fen_count(
                restPieces, i, maximum, "rest piece count");
            if (known_count[pc] + count > maximum)
                throw_fen_error(
                    "known and rest pieces exceed initial inventory");
            pos.rest[pc] = count;
        }
    }

    bool seen_unknown[2]{};
    for (std::size_t i = 0; i < unknowns.size();)
    {
        const char marker = unknowns[i++];
        Color color;
        if (marker == 'U')
            color = WHITE;
        else if (marker == 'u')
            color = BLACK;
        else
            throw_fen_error("unknown-loss marker must be U or u");
        if (seen_unknown[color])
            throw_fen_error("unknown-loss marker is duplicated");
        seen_unknown[color] = true;
        pos.unknown_loss[color] = parse_fen_count(
            unknowns, i, 15, "unknown-loss count");
    }

    return pos;
}

static TrainingDataEntry from_record(const PackedSfenValue& psv) {
    TrainingDataEntry e;
    std::size_t len = 0;
    while (len < FEN_BYTES && psv.fen[len])
        ++len;
    if (len == FEN_BYTES)
        throw_fen_error("record is not NUL-terminated");
    e.pos = parse_fen(std::string(psv.fen, psv.fen + len));
    e.move = psv.move;
    e.score = psv.score;
    e.ply = psv.gamePly;
    e.result = psv.game_result;
    return e;
}

} // namespace jieqi

static constexpr int MAX_PIECES = PIECE_COUNT;
static constexpr int MAX_HAND_PIECES = POCKETS ? 2 * static_cast<int>(File::FILE_NB) : 0;

static Square orient(Color color, Square sq)
{
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        // Use a 180-degree rotation for the black orientation.
        return flip_horizontally(flip_vertically(sq));
    }
}

static Square orient_flip(Color color, Square sq)
{
    if (sq == Square::NB)
        // map missing king to zero
        return Square::MIN;
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        return flip_vertically(sq);
    }
}

static int map_king(Square sq)
{
    // palace squares for Xiangi/Janggi
    // map accessible king squares skipping the gaps
    if (Square::KNB == Square(9) && Square::KNB != Square::NB)
        return (int(sq) - 6 * (int(sq) / int(File::FILE_NB)) - 3) % int(Square::KNB);

    return int(sq) % int(Square::KNB);
}

struct HalfKP {
    static constexpr int NUM_SQ = static_cast<int>(Square::NB);
    static constexpr int NUM_PT = static_cast<int>(PieceType::MaxPiece) * 2;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = MAX_PIECES;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + map_king(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto ksq = pos.kingSquare(color);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int j = 0;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None || type_of(p) == PieceType::King)
                continue;
            values[j] = 1.0f;
            features[j] = feature_index(color, orient(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKPFactorized {
    // Factorized features
    static constexpr int K_INPUTS = HalfKP::NUM_SQ;
    static constexpr int PIECE_INPUTS = HalfKP::NUM_SQ * HalfKP::NUM_PT;
    static constexpr int INPUTS = HalfKP::INPUTS + K_INPUTS + PIECE_INPUTS;

    static constexpr int MAX_K_FEATURES = 1;
    static constexpr int MAX_PIECE_FEATURES = MAX_PIECES;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKP::MAX_ACTIVE_FEATURES + MAX_K_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto [start_j, offset] = HalfKP::fill_features_sparse(e, features, values, color);
        int j = start_j;
        auto& pos = e.pos;
        {
            // king square factor
            auto ksq = pos.kingSquare(color);
            features[j] = offset + static_cast<int>(orient(color, ksq));
            values[j] = static_cast<float>(start_j);
            ++j;
        }
        offset += K_INPUTS;

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None || type_of(p) == PieceType::King)
                continue;
            auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKP::NUM_SQ) + static_cast<int>(orient(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKA {
    static constexpr int NUM_SQ = static_cast<int>(Square::NB);
    static constexpr int NUM_PT = (static_cast<int>(PieceType::MaxPiece) + 1) * 2;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = MAX_PIECES;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        return 1 + static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + map_king(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None)
                continue;
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAFactorized {
    // Factorized features
    static constexpr int PIECE_INPUTS = HalfKA::NUM_SQ * HalfKA::NUM_PT;
    static constexpr int INPUTS = HalfKA::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = MAX_PIECES;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKA::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKA::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;

        int j = start_j;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None)
                continue;
            auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKA::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2 {
    static constexpr int NUM_KSQ = static_cast<int>(Square::KNB);
    static constexpr int NUM_SQ = static_cast<int>(Square::NB);
    static constexpr int NUM_PT = (static_cast<int>(PieceType::MaxPiece) + 1) * 2 - (NUM_KSQ > 1);
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT + MAX_HAND_PIECES * (NUM_PT - (NUM_KSQ > 1));
    static constexpr int INPUTS = NUM_PLANES * NUM_KSQ;

    static constexpr int MAX_ACTIVE_FEATURES = MAX_PIECES;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        if (NUM_PT % 2 && p_idx == NUM_PT)
            --p_idx; // pack the opposite king into the same NUM_SQ * NUM_SQ
        return static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + map_king(ksq) * NUM_PLANES;
    }

    static int feature_index(Color color, Square ksq, int handCount, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        return handCount + p_idx * MAX_HAND_PIECES + NUM_SQ * NUM_PT + map_king(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None)
                continue;
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        for (PieceType pt = PieceType::Pawn; pt < PieceType::King; ++pt)
            for (Color c : { Color::White, Color::Black })
                for (int i = 0; i < pos.getHandCount(make_piece(pt, c)); i++)
                {
                    values[j] = 1.0f;
                    features[j] = feature_index(color, orient_flip(color, ksq), i, make_piece(pt, c));
                    ++j;
                }

        return { j, INPUTS };
    }
};

struct HalfKAv2Factorized {
    // Factorized features
    static constexpr int NUM_PT = (static_cast<int>(PieceType::MaxPiece) + 1) * 2;
    static constexpr int PIECE_INPUTS = HalfKAv2::NUM_SQ * NUM_PT + MAX_HAND_PIECES * (NUM_PT - 2 * (HalfKAv2::NUM_KSQ > 1));
    static constexpr int INPUTS = HalfKAv2::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = MAX_PIECES;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAv2::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKAv2::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;

        int j = start_j;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None)
                continue;
            auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKAv2::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        for (PieceType pt = PieceType::Pawn; pt < PieceType::King; ++pt)
            for (Color c : { Color::White, Color::Black })
                for (int i = 0; i < pos.getHandCount(make_piece(pt, c)); i++)
                {
                    values[j] = 1.0f;
                    auto p_idx = static_cast<int>(pt) * 2 + (c != color);
                    features[j] = offset + i + p_idx * MAX_HAND_PIECES + HalfKAv2::NUM_SQ * NUM_PT;
                    ++j;
                }

        return { j, INPUTS };
    }
};

struct HalfKAv2HmJieqi {
    static constexpr int NUM_SQ = jieqi::SQUARE_NB;
    static constexpr int PS_NB = jieqi::PS_NB;
    static constexpr int INPUTS = 6 * PS_NB;
    static constexpr int MAX_ACTIVE_FEATURES = 64;

    static constexpr int PS_W_ROOK    = 0 * NUM_SQ;
    static constexpr int PS_B_ROOK    = 1 * NUM_SQ;
    static constexpr int PS_W_CANNON  = 2 * NUM_SQ;
    static constexpr int PS_B_CANNON  = 3 * NUM_SQ;
    static constexpr int PS_W_KNIGHT  = 4 * NUM_SQ;
    static constexpr int PS_B_KNIGHT  = 5 * NUM_SQ;
    static constexpr int PS_W_PAWN    = 6 * NUM_SQ;
    static constexpr int PS_B_PAWN    = 7 * NUM_SQ;
    static constexpr int PS_W_ADVISOR = 8 * NUM_SQ;
    static constexpr int PS_B_ADVISOR = 9 * NUM_SQ;
    static constexpr int PS_W_BISHOP  = 10 * NUM_SQ;
    static constexpr int PS_B_BISHOP  = 11 * NUM_SQ;
    static constexpr int PS_WB_KING   = 12 * NUM_SQ;
    static constexpr int PS_DARK      = 13 * NUM_SQ;
    static constexpr int PS_UNKNOWN_OWN   = PS_DARK + 15;
    static constexpr int PS_UNKNOWN_ENEMY = PS_DARK + 78;

    static int piece_square_index(jieqi::Color perspective, jieqi::Piece pc) {
        const bool own = jieqi::color_of(pc) == perspective;
        switch (jieqi::type_of(pc))
        {
        case 1: return own ? PS_W_ROOK : PS_B_ROOK;
        case 2: return own ? PS_W_ADVISOR : PS_B_ADVISOR;
        case 3: return own ? PS_W_CANNON : PS_B_CANNON;
        case 4: return own ? PS_W_PAWN : PS_B_PAWN;
        case 5: return own ? PS_W_KNIGHT : PS_B_KNIGHT;
        case 6: return own ? PS_W_BISHOP : PS_B_BISHOP;
        case 7: return PS_WB_KING;
        default: return 0;
        }
    }

    static int dark_rest_index(jieqi::Color perspective, jieqi::Piece pc) {
        const bool own = jieqi::color_of(pc) == perspective;
        const int base = own ? 9 : 72;
        switch (jieqi::type_of(pc))
        {
        case 1: return PS_DARK + base + 0;
        case 2: return PS_DARK + base + 1;
        case 3: return PS_DARK + base + 2;
        case 4: return PS_DARK + base + 3;
        case 5: return PS_DARK + base + 4;
        case 6: return PS_DARK + base + 5;
        default: return 0;
        }
    }

    static int unknown_loss_index(jieqi::Color perspective, jieqi::Color c) {
        return c == perspective ? PS_UNKNOWN_OWN : PS_UNKNOWN_ENEMY;
    }

    static std::pair<int, bool> king_bucket(int ksq, int oksq) {
        static constexpr std::uint8_t M = 1 << 3;
        static constexpr std::uint8_t buckets[NUM_SQ] = {
          0, 0, 0, 0, 1, std::uint8_t(M | 0), 0, 0, 0,
          0, 0, 0, 2, 3, std::uint8_t(M | 2), 0, 0, 0,
          0, 0, 0, 4, 5, std::uint8_t(M | 4), 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 4, 5, std::uint8_t(M | 4), 0, 0, 0,
          0, 0, 0, 2, 3, std::uint8_t(M | 2), 0, 0, 0,
          0, 0, 0, 0, 1, std::uint8_t(M | 0), 0, 0, 0,
        };

        const int kb = buckets[ksq] & 0x7;
        const bool mirror = (buckets[ksq] >> 3) || ((kb & 1) && (buckets[oksq] >> 3));
        return { kb, mirror };
    }

    static int map_square(jieqi::Color perspective, int sq, bool mirror) {
        if (mirror)
            sq = jieqi::flip_file(sq);
        if (perspective == jieqi::BLACK)
            sq = jieqi::flip_rank(sq);
        return sq;
    }

    static int make_index(jieqi::Color perspective, int sq, jieqi::Piece pc, int bucket, bool mirror) {
        if (pc == jieqi::DARK_PIECE)
            return PS_NB * bucket + PS_DARK + map_square(perspective, sq, mirror);
        return PS_NB * bucket + piece_square_index(perspective, pc) + map_square(perspective, sq, mirror);
    }

    static int layer_stack_bucket(int pieceCount) {
        static constexpr int buckets[33] = {
          -1, -1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
          6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 14, 15,
        };
        return buckets[std::clamp(pieceCount, 2, 32)];
    }

    static int layer_stack_bucket(const jieqi::Position& pos) {
        return layer_stack_bucket(pos.piece_count());
    }

    static int king_square(const jieqi::Position& pos, jieqi::Color c) {
        const jieqi::Piece king = jieqi::Piece(jieqi::make_piece(c, 7));
        for (int sq = 0; sq < NUM_SQ; ++sq)
            if (pos.board[sq] == king)
                return sq;
        return c == jieqi::WHITE ? 4 : 85;
    }

    static std::pair<int, int> fill_features_sparse(
      const jieqi::TrainingDataEntry& e, int* features, float* values, jieqi::Color color) {

        const int ksq = king_square(e.pos, color);
        const int oksq = king_square(e.pos, color == jieqi::WHITE ? jieqi::BLACK : jieqi::WHITE);
        const auto [bucket, mirror] = king_bucket(ksq, oksq);

        int j = 0;
        for (int sq = 0; sq < NUM_SQ; ++sq)
        {
            if (e.pos.board[sq] == jieqi::NO_PIECE)
                continue;
            values[j] = 1.0f;
            features[j] = make_index(
              color, sq, e.pos.dark[sq] ? jieqi::DARK_PIECE : e.pos.board[sq], bucket, mirror);
            ++j;
        }

        for (int pc = 1; pc < 16; ++pc)
        {
            if (pc == 7 || pc == 8 || pc == 15)
                continue;
            for (int n = 0; n < e.pos.rest[pc]; ++n)
            {
                values[j] = 1.0f;
                features[j] = PS_NB * bucket + dark_rest_index(color, jieqi::Piece(pc));
                ++j;
            }
        }

        for (jieqi::Color c : {jieqi::WHITE, jieqi::BLACK})
            for (int n = 0; n < e.pos.unknown_loss[c]; ++n)
            {
                values[j] = 1.0f;
                features[j] = PS_NB * bucket + unknown_loss_index(color, c);
                ++j;
            }

        return { j, INPUTS };
    }
};

struct HalfKAv2HmJieqiV2 {
    static constexpr int NUM_SQ = jieqi::SQUARE_NB;
    static constexpr int PS_NB = jieqi::PS_NB;
    static constexpr int ATTACK_BUCKETS = 4;
    static constexpr int INPUTS = 6 * ATTACK_BUCKETS * PS_NB;
    static constexpr int MAX_ACTIVE_FEATURES = 64;

    static constexpr int PS_W_ROOK    = 0 * NUM_SQ;
    static constexpr int PS_B_ROOK    = 1 * NUM_SQ;
    static constexpr int PS_W_CANNON  = 2 * NUM_SQ;
    static constexpr int PS_B_CANNON  = 3 * NUM_SQ;
    static constexpr int PS_W_KNIGHT  = 4 * NUM_SQ;
    static constexpr int PS_B_KNIGHT  = 5 * NUM_SQ;
    static constexpr int PS_W_PAWN    = 6 * NUM_SQ;
    static constexpr int PS_B_PAWN    = 7 * NUM_SQ;
    static constexpr int PS_W_ADVISOR = 8 * NUM_SQ;
    static constexpr int PS_B_ADVISOR = 9 * NUM_SQ;
    static constexpr int PS_W_BISHOP  = 10 * NUM_SQ;
    static constexpr int PS_B_BISHOP  = 11 * NUM_SQ;
    static constexpr int PS_WB_KING   = 12 * NUM_SQ;
    static constexpr int PS_DARK      = 13 * NUM_SQ;
    static constexpr int PS_UNKNOWN_OWN   = PS_DARK + 15;
    static constexpr int PS_UNKNOWN_ENEMY = PS_DARK + 78;
    static constexpr std::uint64_t BALANCE_ENCODING = 0xa4a92a74e989d3a7ULL;

    static int piece_square_index(jieqi::Color perspective, jieqi::Piece pc) {
        const bool own = jieqi::color_of(pc) == perspective;
        switch (jieqi::type_of(pc))
        {
        case 1: return own ? PS_W_ROOK : PS_B_ROOK;
        case 2: return own ? PS_W_ADVISOR : PS_B_ADVISOR;
        case 3: return own ? PS_W_CANNON : PS_B_CANNON;
        case 4: return own ? PS_W_PAWN : PS_B_PAWN;
        case 5: return own ? PS_W_KNIGHT : PS_B_KNIGHT;
        case 6: return own ? PS_W_BISHOP : PS_B_BISHOP;
        case 7: return PS_WB_KING;
        default: return 0;
        }
    }

    static int dark_rest_index(jieqi::Color perspective, jieqi::Piece pc) {
        const bool own = jieqi::color_of(pc) == perspective;
        const int base = own ? 9 : 72;
        switch (jieqi::type_of(pc))
        {
        case 1: return PS_DARK + base + 0;
        case 2: return PS_DARK + base + 1;
        case 3: return PS_DARK + base + 2;
        case 4: return PS_DARK + base + 3;
        case 5: return PS_DARK + base + 4;
        case 6: return PS_DARK + base + 5;
        default: return 0;
        }
    }

    static int unknown_loss_index(jieqi::Color perspective, jieqi::Color c) {
        return c == perspective ? PS_UNKNOWN_OWN : PS_UNKNOWN_ENEMY;
    }

    static std::uint64_t mid_mirror_encoding(jieqi::Piece pc, int sq) {
        static constexpr std::uint8_t shifts[8][2] = {
            {0, 0}, {44, 0}, {60, 36}, {47, 7},
            {53, 21}, {50, 14}, {57, 29}, {0, 0}
        };

        const int pt = jieqi::type_of(pc);
        if (pt < 1 || pt > 7)
            return 0;

        const int f = jieqi::file_of(sq);
        if (f == 4)
            return 0;

        if (pt == 7)
            return 1ULL << 63;

        const jieqi::Color c = jieqi::color_of(pc);
        const int r = jieqi::rank_of(sq);
        const std::uint8_t r_ = c == jieqi::WHITE ? std::uint8_t(r) : std::uint8_t(9 - r);
        const std::uint8_t f_ = f < 4 ? std::uint8_t(f) : std::uint8_t(8 - f);
        const auto s1 = shifts[pt][0];
        const auto s2 = shifts[pt][1];
        std::uint64_t encoding =
            (1ULL << s1) | ((std::uint64_t(3 - f_) * 10 + std::uint64_t(r_)) << s2);

        return f < 4 ? encoding : std::uint64_t(-std::int64_t(encoding));
    }

    static std::uint64_t mid_encoding(const jieqi::Position& pos, jieqi::Color c) {
        std::uint64_t encoding = BALANCE_ENCODING;
        for (int sq = 0; sq < NUM_SQ; ++sq)
        {
            const auto pc = pos.board[sq];
            if (pc == jieqi::NO_PIECE || pos.dark[sq] || jieqi::color_of(pc) != c)
                continue;
            encoding += mid_mirror_encoding(pc, sq);
        }
        return encoding;
    }

    static bool requires_mid_mirror(const jieqi::Position& pos, jieqi::Color c) {
        const auto own = mid_encoding(pos, c);
        const auto enemy = mid_encoding(pos, c == jieqi::WHITE ? jieqi::BLACK : jieqi::WHITE);
        return ((1ULL << 63) & own & enemy)
            && (own < BALANCE_ENCODING || (own == BALANCE_ENCODING && enemy < BALANCE_ENCODING));
    }

    static int attack_bucket(const jieqi::Position& pos, jieqi::Color c) {
        int rooks = 0;
        int knights = 0;
        int cannons = 0;

        for (int sq = 0; sq < NUM_SQ; ++sq)
        {
            const auto pc = pos.board[sq];
            if (pc == jieqi::NO_PIECE || pos.dark[sq] || jieqi::color_of(pc) != c)
                continue;

            switch (jieqi::type_of(pc))
            {
            case 1: rooks = std::min(2, rooks + 1); break;
            case 3: cannons = std::min(2, cannons + 1); break;
            case 5: knights = std::min(2, knights + 1); break;
            default: break;
            }
        }

        return int(bool(rooks)) * 2 + int(bool(knights + cannons));
    }

    static std::pair<int, bool> king_bucket(int ksq, int oksq, bool midMirror) {
        static constexpr std::uint8_t M = 1 << 3;
        static constexpr std::uint8_t buckets[NUM_SQ] = {
          0, 0, 0, 0, 1, std::uint8_t(M | 0), 0, 0, 0,
          0, 0, 0, 2, 3, std::uint8_t(M | 2), 0, 0, 0,
          0, 0, 0, 4, 5, std::uint8_t(M | 4), 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 4, 5, std::uint8_t(M | 4), 0, 0, 0,
          0, 0, 0, 2, 3, std::uint8_t(M | 2), 0, 0, 0,
          0, 0, 0, 0, 1, std::uint8_t(M | 0), 0, 0, 0,
        };

        const int kb = buckets[ksq] & 0x7;
        const int okb = buckets[oksq] & 0x7;
        const bool mirror =
            (buckets[ksq] >> 3)
            || ((kb & 1) && ((buckets[oksq] >> 3) || ((okb & 1) && midMirror)));
        return { kb, mirror };
    }

    static int map_square(jieqi::Color perspective, int sq, bool mirror) {
        if (mirror)
            sq = jieqi::flip_file(sq);
        if (perspective == jieqi::BLACK)
            sq = jieqi::flip_rank(sq);
        return sq;
    }

    static int make_index(jieqi::Color perspective, int sq, jieqi::Piece pc, int bucket, bool mirror) {
        if (pc == jieqi::DARK_PIECE)
            return PS_NB * bucket + PS_DARK + map_square(perspective, sq, mirror);
        return PS_NB * bucket + piece_square_index(perspective, pc) + map_square(perspective, sq, mirror);
    }

    static int layer_stack_bucket(int pieceCount) {
        return HalfKAv2HmJieqi::layer_stack_bucket(pieceCount);
    }

    static int layer_stack_bucket(const jieqi::Position& pos) {
        return layer_stack_bucket(pos.piece_count());
    }

    static int king_square(const jieqi::Position& pos, jieqi::Color c) {
        return HalfKAv2HmJieqi::king_square(pos, c);
    }

    static std::pair<int, int> fill_features_sparse(
      const jieqi::TrainingDataEntry& e, int* features, float* values, jieqi::Color color) {

        const int ksq = king_square(e.pos, color);
        const int oksq = king_square(e.pos, color == jieqi::WHITE ? jieqi::BLACK : jieqi::WHITE);
        const auto [kingBucket, mirror] = king_bucket(ksq, oksq, requires_mid_mirror(e.pos, color));
        const int bucket = kingBucket * ATTACK_BUCKETS + attack_bucket(e.pos, color);

        int j = 0;
        for (int sq = 0; sq < NUM_SQ; ++sq)
        {
            if (e.pos.board[sq] == jieqi::NO_PIECE)
                continue;
            values[j] = 1.0f;
            features[j] = make_index(
              color, sq, e.pos.dark[sq] ? jieqi::DARK_PIECE : e.pos.board[sq], bucket, mirror);
            ++j;
        }

        for (int pc = 1; pc < 16; ++pc)
        {
            if (pc == 7 || pc == 8 || pc == 15)
                continue;
            for (int n = 0; n < e.pos.rest[pc]; ++n)
            {
                values[j] = 1.0f;
                features[j] = PS_NB * bucket + dark_rest_index(color, jieqi::Piece(pc));
                ++j;
            }
        }

        for (jieqi::Color c : {jieqi::WHITE, jieqi::BLACK})
            for (int n = 0; n < e.pos.unknown_loss[c]; ++n)
            {
                values[j] = 1.0f;
                features[j] = PS_NB * bucket + unknown_loss_index(color, c);
                ++j;
            }

        return { j, INPUTS };
    }
};

struct HalfKAv2HmJieqiV3 : HalfKAv2HmJieqiV2 {
    static constexpr int BASE_PS_NB = HalfKAv2HmJieqiV2::PS_NB;
    static constexpr int META_NB = 64;
    static constexpr int PS_NB = BASE_PS_NB + META_NB;
    static constexpr int INPUTS = 6 * ATTACK_BUCKETS * PS_NB;
    static constexpr int MAX_ACTIVE_FEATURES = 128;

    static int make_index(jieqi::Color perspective, int sq, jieqi::Piece pc, int bucket, bool mirror) {
        if (pc == jieqi::DARK_PIECE)
            return PS_NB * bucket + PS_DARK + map_square(perspective, sq, mirror);
        return PS_NB * bucket + piece_square_index(perspective, pc) + map_square(perspective, sq, mirror);
    }

    static int bucket8(int value) {
        return std::min(value, 7);
    }

    static int rest_count(const jieqi::Position& pos, jieqi::Color c) {
        int count = 0;
        for (int pt = 1; pt <= 6; ++pt)
            count += pos.rest[jieqi::make_piece(c, pt)];
        return count;
    }

    static int strong_rest_count(const jieqi::Position& pos, jieqi::Color c) {
        return pos.rest[jieqi::make_piece(c, 1)]
             + pos.rest[jieqi::make_piece(c, 3)]
             + pos.rest[jieqi::make_piece(c, 5)];
    }

    static int strong_rest_profile(const jieqi::Position& pos, jieqi::Color c) {
        return (pos.rest[jieqi::make_piece(c, 1)] > 0 ? 1 : 0)
             | (pos.rest[jieqi::make_piece(c, 3)] > 0 ? 2 : 0)
             | (pos.rest[jieqi::make_piece(c, 5)] > 0 ? 4 : 0);
    }

    static int dark_count(const jieqi::Position& pos) {
        int count = 0;
        for (bool dark : pos.dark)
            count += int(dark);
        return count;
    }

    static int dark_density_bucket(int count) {
        if (count == 0) return 0;
        if (count <= 2) return 1;
        if (count <= 4) return 2;
        if (count <= 8) return 3;
        if (count <= 12) return 4;
        if (count <= 16) return 5;
        if (count <= 24) return 6;
        return 7;
    }

    static int uncertainty_phase_bucket(int darkCount, int strongRest, int unknownLoss) {
        const int darkBucket =
            darkCount >= 17 ? 4 : darkCount >= 9 ? 3 : darkCount >= 5 ? 2 : darkCount >= 1 ? 1 : 0;
        const int restBucket = strongRest >= 9 ? 3 : strongRest >= 5 ? 2 : strongRest >= 2 ? 1 : 0;
        return darkBucket + restBucket + (unknownLoss > 0 ? 1 : 0);
    }

    static int layer_stack_bucket(const jieqi::Position& pos) {
        const int base = HalfKAv2HmJieqi::layer_stack_bucket(pos.piece_count());
        const int strongRest = strong_rest_count(pos, jieqi::WHITE) + strong_rest_count(pos, jieqi::BLACK);
        const int unknownLoss = pos.unknown_loss[jieqi::WHITE] + pos.unknown_loss[jieqi::BLACK];
        return std::clamp(base + uncertainty_phase_bucket(dark_count(pos), strongRest, unknownLoss), 0, 15);
    }

    static void append_meta_features(
      const jieqi::Position& pos, int bucket, jieqi::Color color, int& j, int* features, float* values) {
        const auto enemy = color == jieqi::WHITE ? jieqi::BLACK : jieqi::WHITE;
        const int metaBase = PS_NB * bucket + BASE_PS_NB;

        auto emit = [&](int offset) {
            values[j] = 1.0f;
            features[j] = metaBase + offset;
            ++j;
        };

        emit(0 + bucket8(strong_rest_count(pos, color)));
        emit(8 + bucket8(strong_rest_count(pos, enemy)));
        emit(16 + bucket8(rest_count(pos, color)));
        emit(24 + bucket8(rest_count(pos, enemy)));
        emit(32 + dark_density_bucket(dark_count(pos)));
        emit(40 + std::min(pos.unknown_loss[color], 3));
        emit(44 + std::min(pos.unknown_loss[enemy], 3));
        emit(48 + strong_rest_profile(pos, color));
        emit(56 + strong_rest_profile(pos, enemy));
    }

    static std::pair<int, int> fill_features_sparse(
      const jieqi::TrainingDataEntry& e, int* features, float* values, jieqi::Color color) {

        const int ksq = king_square(e.pos, color);
        const int oksq = king_square(e.pos, color == jieqi::WHITE ? jieqi::BLACK : jieqi::WHITE);
        const auto [kingBucket, mirror] = king_bucket(ksq, oksq, requires_mid_mirror(e.pos, color));
        const int bucket = kingBucket * ATTACK_BUCKETS + attack_bucket(e.pos, color);

        int j = 0;
        for (int sq = 0; sq < NUM_SQ; ++sq)
        {
            if (e.pos.board[sq] == jieqi::NO_PIECE)
                continue;
            values[j] = 1.0f;
            features[j] = make_index(
              color, sq, e.pos.dark[sq] ? jieqi::DARK_PIECE : e.pos.board[sq], bucket, mirror);
            ++j;
        }

        for (int pc = 1; pc < 16; ++pc)
        {
            if (pc == 7 || pc == 8 || pc == 15)
                continue;
            for (int n = 0; n < e.pos.rest[pc]; ++n)
            {
                values[j] = 1.0f;
                features[j] = PS_NB * bucket + dark_rest_index(color, jieqi::Piece(pc));
                ++j;
            }
        }

        for (jieqi::Color c : {jieqi::WHITE, jieqi::BLACK})
            for (int n = 0; n < e.pos.unknown_loss[c]; ++n)
            {
                values[j] = 1.0f;
                features[j] = PS_NB * bucket + unknown_loss_index(color, c);
                ++j;
            }

        append_meta_features(e.pos, bucket, color, j, features, values);
        return { j, INPUTS };
    }
};

// V8 reuses the compact jqv4 board/rest ABI (without the optional full-threat
// block) while owning its metadata policy. The two unknown-loss groups remain
// reserved rows and are never active V8 features. Keep a named type and
// dispatch string so an unrelated feature name cannot be accepted as V8 data.
struct HalfKAv2HmJieqiV8 : HalfKAv2HmJieqiV3 {
    static constexpr int INPUTS = 6 * ATTACK_BUCKETS * PS_NB;
    static constexpr int MAX_ACTIVE_FEATURES = 160;

    struct LayerStackSelection {
        int floor = 0;
        std::uint8_t blend_q8 = 0;
    };

    static int interval_residual_num(int value, const int* thresholds,
                                     std::size_t count, int& denominator) {
        value = std::max(0, value);
        std::size_t bucket = 0;
        while (bucket + 1 < count && value >= thresholds[bucket + 1]) ++bucket;
        if (bucket + 1 >= count) { denominator = 1; return 0; }
        const int lower = thresholds[bucket];
        const int upper = thresholds[bucket + 1];
        denominator = std::max(1, upper - lower);
        return std::clamp(value - lower, 0, denominator - 1);
    }

    static double piece_residual(int pieceCount) {
        static constexpr int lower[16] = {2, 7, 9, 11, 13, 15, 17, 19,
                                          21, 23, 25, 27, 29, 31, 32, 33};
        static constexpr int upper[16] = {7, 9, 11, 13, 15, 17, 19, 21,
                                          23, 25, 27, 29, 31, 32, 33, 34};
        const int count = std::clamp(pieceCount, 2, 32);
        const int bucket = std::clamp(HalfKAv2HmJieqi::layer_stack_bucket(count), 0, 15);
        const int den = std::max(1, upper[bucket] - lower[bucket]);
        return double(std::clamp(count - lower[bucket], 0, den - 1)) / double(den);
    }

    static HalfKAv2HmJieqiV8::LayerStackSelection layer_stack_selection(const jieqi::Position& pos) {
        static constexpr int darkThresholds[] = {0, 1, 5, 9, 17, 33};
        static constexpr int restThresholds[] = {0, 2, 5, 9, 19};
        const int floor = layer_stack_bucket(pos);
        if (floor >= 15) return {15, 0};
        int darkDen = 1, restDen = 1;
        const double darkResidual = double(interval_residual_num(
            dark_count(pos), darkThresholds, std::size(darkThresholds), darkDen)) / darkDen;
        const int strongRest = strong_rest_count(pos, jieqi::WHITE)
                             + strong_rest_count(pos, jieqi::BLACK);
        const double restResidual = double(interval_residual_num(
            strongRest, restThresholds, std::size(restThresholds), restDen)) / restDen;
        const double fraction = (piece_residual(pos.piece_count()) + darkResidual + restResidual) / 3.0;
        const int rounded = static_cast<int>(std::floor(fraction * 255.0 + 0.5));
        if (rounded >= 255) return {std::min(15, floor + 1), 0};
        return {floor, static_cast<std::uint8_t>(std::clamp(rounded, 0, 254))};
    }

    static std::array<int, 2> visible_threat_summary(const jieqi::Position& pos,
                                                       jieqi::Color color) {
        const auto enemy = color == jieqi::WHITE ? jieqi::BLACK : jieqi::WHITE;
        std::array<bool, jieqi::SQUARE_NB> enemyTargets{};
        std::array<bool, jieqi::SQUARE_NB> ownTargets{};
        const auto onBoard = [](int f, int r) { return f >= 0 && f < 9 && r >= 0 && r < 10; };
        const auto occupied = [&](int sq) { return pos.board[sq] != jieqi::NO_PIECE; };
        const auto visible = [&](int sq) { return occupied(sq) && !pos.dark[sq]; };
        const auto add = [&](jieqi::Color attacker_color, int to) {
            if (!onBoard(jieqi::file_of(to), jieqi::rank_of(to)) || !visible(to)) return;
            const auto target = pos.board[to];
            if (attacker_color == color && jieqi::color_of(target) == enemy)
                enemyTargets[to] = true;
            if (attacker_color == enemy && jieqi::color_of(target) == color)
                ownTargets[to] = true;
        };
        for (const auto attacker_color : {color, enemy}) {
            for (int from = 0; from < jieqi::SQUARE_NB; ++from) {
                if (!visible(from)
                    || jieqi::color_of(pos.board[from]) != attacker_color)
                    continue;
                const int f = jieqi::file_of(from), r = jieqi::rank_of(from);
                const int type = jieqi::type_of(pos.board[from]);
                if (type == 1 || type == 3) {
                    for (const auto [df, dr] : {std::pair{1, 0}, std::pair{-1, 0},
                                                 std::pair{0, 1}, std::pair{0, -1}}) {
                        int nf = f + df, nr = r + dr; bool screen = false;
                        while (onBoard(nf, nr)) {
                            const int to = jieqi::make_square(nf, nr);
                            if (type == 1) add(attacker_color, to);
                            else if (screen) add(attacker_color, to);
                            if (occupied(to)) {
                                if (type == 1) break;
                                if (!screen) screen = true; else break;
                            }
                            nf += df; nr += dr;
                        }
                    }
                } else if (type == 5) {
                    for (const auto [df, dr] : {std::pair{1, 2}, std::pair{-1, 2},
                                                 std::pair{1, -2}, std::pair{-1, -2},
                                                 std::pair{2, 1}, std::pair{2, -1},
                                                 std::pair{-2, 1}, std::pair{-2, -1}}) {
                        const int legF = f + (std::abs(df) == 2 ? df / 2 : 0);
                        const int legR = r + (std::abs(dr) == 2 ? dr / 2 : 0);
                        if (onBoard(f + df, r + dr)
                            && !occupied(jieqi::make_square(legF, legR)))
                            add(attacker_color, jieqi::make_square(f + df, r + dr));
                    }
                } else if (type == 6) {
                    for (const auto [df, dr] : {std::pair{2, 2}, std::pair{2, -2},
                                                 std::pair{-2, 2}, std::pair{-2, -2}}) {
                        const int nf = f + df, nr = r + dr;
                        const int ef = f + df / 2, er = r + dr / 2;
                        if (onBoard(nf, nr) && !occupied(jieqi::make_square(ef, er))
                            && ((r <= 4 && nr <= 4) || (r >= 5 && nr >= 5)))
                            add(attacker_color, jieqi::make_square(nf, nr));
                    }
                } else if (type == 2 || type == 7) {
                    const bool palace = f >= 3 && f <= 5 && (r <= 2 || r >= 7);
                    if (!palace) continue;
                    for (const auto [df, dr] : (type == 2
                        ? std::array<std::pair<int, int>, 4>{{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}}
                        : std::array<std::pair<int, int>, 4>{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}})) {
                        const int nf = f + df, nr = r + dr;
                        if (onBoard(nf, nr) && nf >= 3 && nf <= 5
                            && (nr <= 2 || nr >= 7))
                            add(attacker_color, jieqi::make_square(nf, nr));
                    }
                } else if (type == 4) {
                    const int dr = attacker_color == jieqi::WHITE ? 1 : -1;
                    if (onBoard(f, r + dr))
                        add(attacker_color, jieqi::make_square(f, r + dr));
                    const bool crossed = attacker_color == jieqi::WHITE ? r > 4 : r < 5;
                    if (crossed)
                        for (int df : {-1, 1})
                            if (onBoard(f + df, r))
                                add(attacker_color, jieqi::make_square(f + df, r));
                }
            }
        }
        return {static_cast<int>(std::count(enemyTargets.begin(), enemyTargets.end(), true)),
                static_cast<int>(std::count(ownTargets.begin(), ownTargets.end(), true))};
    }

    static int layer_stack_bucket(const jieqi::Position& pos) {
        const int base = HalfKAv2HmJieqi::layer_stack_bucket(pos.piece_count());
        const int strongRest = strong_rest_count(pos, jieqi::WHITE)
                              + strong_rest_count(pos, jieqi::BLACK);
        const int darkCount = dark_count(pos);
        const int darkBucket = darkCount >= 17 ? 4 : darkCount >= 9 ? 3
                              : darkCount >= 5 ? 2 : darkCount >= 1 ? 1 : 0;
        const int restBucket = strongRest >= 9 ? 3 : strongRest >= 5 ? 2
                              : strongRest >= 2 ? 1 : 0;
        return std::clamp(base + darkBucket + restBucket, 0, 15);
    }

    static void append_meta_features_v8(
      const jieqi::Position& pos, int bucket, jieqi::Color color,
      int& j, int* features, float* values) {
        const auto enemy = color == jieqi::WHITE ? jieqi::BLACK : jieqi::WHITE;
        const int metaBase = PS_NB * bucket + BASE_PS_NB;
        auto emit = [&](int offset) {
            values[j] = 1.0f;
            features[j] = metaBase + offset;
            ++j;
        };
        emit(0 + bucket8(strong_rest_count(pos, color)));
        emit(8 + bucket8(strong_rest_count(pos, enemy)));
        emit(16 + bucket8(rest_count(pos, color)));
        emit(24 + bucket8(rest_count(pos, enemy)));
        emit(32 + dark_density_bucket(dark_count(pos)));
        emit(40 + std::min(pos.unknown_loss[color], 3));
        emit(44 + std::min(pos.unknown_loss[enemy], 3));
        const auto threats = visible_threat_summary(pos, color);
        emit(48 + std::min(threats[0], 7));
        emit(56 + std::min(threats[1], 7));
    }

    static std::pair<int, int> fill_features_sparse(
      const jieqi::TrainingDataEntry& e, int* features, float* values, jieqi::Color color) {
        const int ksq = king_square(e.pos, color);
        const int oksq = king_square(e.pos, color == jieqi::WHITE ? jieqi::BLACK : jieqi::WHITE);
        const auto [kingBucket, mirror] = king_bucket(
            ksq, oksq, requires_mid_mirror(e.pos, color));
        const int bucket = kingBucket * ATTACK_BUCKETS + attack_bucket(e.pos, color);

        int j = 0;
        for (int sq = 0; sq < NUM_SQ; ++sq)
        {
            if (e.pos.board[sq] == jieqi::NO_PIECE)
                continue;
            values[j] = 1.0f;
            features[j] = make_index(
                color, sq, e.pos.dark[sq] ? jieqi::DARK_PIECE : e.pos.board[sq],
                bucket, mirror);
            ++j;
        }

        for (int pc = 1; pc < 16; ++pc)
        {
            if (pc == 7 || pc == 8 || pc == 15)
                continue;
            for (int n = 0; n < e.pos.rest[pc]; ++n)
            {
                values[j] = 1.0f;
                features[j] = PS_NB * bucket + dark_rest_index(color, jieqi::Piece(pc));
                ++j;
            }
        }

        append_meta_features_v8(e.pos, bucket, color, j, features, values);
        return {j, INPUTS};
    }
};

struct HalfKAv2HmJieqiV8Factorized {
    static constexpr int PSQ_FACTOR_INPUTS = HalfKAv2HmJieqiV8::PS_NB;
    static constexpr int BUCKET_FACTOR_INPUTS = 24;
    static constexpr int VIRTUAL_INPUTS = PSQ_FACTOR_INPUTS + BUCKET_FACTOR_INPUTS;
    static constexpr int INPUTS = HalfKAv2HmJieqiV8::INPUTS + VIRTUAL_INPUTS;
    // Each real row contributes its PSQ and bucket factors.
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAv2HmJieqiV8::MAX_ACTIVE_FEATURES * 3;

    static constexpr int PSQ_FACTOR_BASE = HalfKAv2HmJieqiV8::INPUTS;
    static constexpr int BUCKET_FACTOR_BASE = PSQ_FACTOR_BASE + PSQ_FACTOR_INPUTS;

    static int layer_stack_bucket(const jieqi::Position& pos) {
        return HalfKAv2HmJieqiV8::layer_stack_bucket(pos);
    }

    static HalfKAv2HmJieqiV8::LayerStackSelection layer_stack_selection(const jieqi::Position& pos) {
        return HalfKAv2HmJieqiV8::layer_stack_selection(pos);
    }

    static void append_virtual(int realFeature, int& j, int* features, float* values) {
        values[j] = 1.0f;
        features[j++] = PSQ_FACTOR_BASE + (realFeature % HalfKAv2HmJieqiV8::PS_NB);
        values[j] = 1.0f;
        features[j++] = BUCKET_FACTOR_BASE + (realFeature / HalfKAv2HmJieqiV8::PS_NB);
    }

    static std::pair<int, int> fill_features_sparse(
      const jieqi::TrainingDataEntry& e, int* features, float* values, jieqi::Color color) {
        auto [realCount, _] = HalfKAv2HmJieqiV8::fill_features_sparse(e, features, values, color);
        int j = realCount;
        // Append after the real rows; the sparse CUDA kernel only requires a
        // contiguous list and does not require sorting.
        for (int i = 0; i < realCount; ++i)
            append_virtual(features[i], j, features, values);
        return {j, INPUTS};
    }
};

template <typename Feature>
struct V81RawFeaturePolicy : std::false_type {};

template <>
struct V81RawFeaturePolicy<HalfKAv2HmJieqiV8> : std::true_type {};

template <>
struct V81RawFeaturePolicy<HalfKAv2HmJieqiV8Factorized> : std::true_type {};

static void normalize_v81_raw_unknown_loss(jieqi::Position& position) {
    int darkByColor[2] = {0, 0};
    for (int sq = 0; sq < jieqi::SQUARE_NB; ++sq) {
        if (!position.dark[sq])
            continue;
        const auto piece = position.board[sq];
        if (piece == jieqi::NO_PIECE || piece == jieqi::DARK_PIECE)
            throw std::runtime_error(
                "V8 raw FEN cannot validate an uncoloured dark piece");
        ++darkByColor[jieqi::color_of(piece)];
    }
    for (const auto color : {jieqi::WHITE, jieqi::BLACK}) {
        int restCount = 0;
        for (int type = 1; type <= 6; ++type)
            restCount += position.rest[jieqi::make_piece(color, type)];
        const int expectedUnknownLoss = restCount - darkByColor[color];
        if (expectedUnknownLoss < 0 || expectedUnknownLoss > 15)
            throw std::runtime_error(
                "V8 raw FEN unknown_loss is outside the observation domain");
        if (position.unknown_loss_explicit
            && position.unknown_loss[color] != expectedUnknownLoss)
            throw std::runtime_error(
                "V8 raw FEN unknown_loss does not match rest/dark inventory");
        position.unknown_loss[color] = expectedUnknownLoss;
    }
}

struct JieqiFullThreats {
    static constexpr int THREAT_INPUTS = 90857;
    static constexpr int MAX_ACTIVE_FEATURES = 256;

    static constexpr int ROOK = 1;
    static constexpr int ADVISOR = 2;
    static constexpr int CANNON = 3;
    static constexpr int PAWN = 4;
    static constexpr int KNIGHT = 5;
    static constexpr int BISHOP = 6;
    static constexpr int KING = 7;

    struct ThreatRecord {
        int attacker;
        int from;
        int to;
        int attacked;
    };

    static int flip_piece(int pc) {
        return pc ^ 8;
    }

    static bool is_visible_piece(const jieqi::Position& pos, int sq) {
        return pos.board[sq] != jieqi::NO_PIECE && !pos.dark[sq];
    }

    static bool on_board(int file, int rank) {
        return 0 <= file && file < 9 && 0 <= rank && rank < 10;
    }

    static bool occupied(const jieqi::Position& pos, int sq) {
        return pos.board[sq] != jieqi::NO_PIECE;
    }

    template <typename Emit>
    static void rook_attacks(int sq, Emit emit) {
        const int f = jieqi::file_of(sq);
        const int r = jieqi::rank_of(sq);
        for (const auto [df, dr] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}})
        {
            int nf = f + df;
            int nr = r + dr;
            while (on_board(nf, nr))
            {
                emit(jieqi::make_square(nf, nr));
                nf += df;
                nr += dr;
            }
        }
    }

    template <typename Emit>
    static void cannon_possible_attacks(int sq, Emit emit) {
        const int f = jieqi::file_of(sq);
        const int r = jieqi::rank_of(sq);
        for (const auto [df, dr] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}})
        {
            int nf = f + 2 * df;
            int nr = r + 2 * dr;
            while (on_board(nf, nr))
            {
                emit(jieqi::make_square(nf, nr));
                nf += df;
                nr += dr;
            }
        }
    }

    template <typename Emit>
    static void advisor_attacks(int sq, Emit emit) {
        const int f = jieqi::file_of(sq);
        const int r = jieqi::rank_of(sq);
        for (const auto [df, dr] : {std::pair{1, 1}, std::pair{1, -1}, std::pair{-1, 1}, std::pair{-1, -1}})
            if (on_board(f + df, r + dr))
                emit(jieqi::make_square(f + df, r + dr));
    }

    static bool in_palace(int sq) {
        const int f = jieqi::file_of(sq);
        const int r = jieqi::rank_of(sq);
        return 3 <= f && f <= 5 && (r <= 2 || r >= 7);
    }

    template <typename Emit>
    static void king_attacks(int sq, Emit emit) {
        if (!in_palace(sq))
            return;
        const int f = jieqi::file_of(sq);
        const int r = jieqi::rank_of(sq);
        for (const auto [df, dr] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}})
        {
            const int to = jieqi::make_square(f + df, r + dr);
            if (on_board(f + df, r + dr) && in_palace(to))
                emit(to);
        }
    }

    template <typename Emit>
    static void knight_attacks(int sq, Emit emit) {
        const int f = jieqi::file_of(sq);
        const int r = jieqi::rank_of(sq);
        for (const auto [df, dr] : {std::pair{1, 2}, std::pair{-1, 2}, std::pair{1, -2}, std::pair{-1, -2},
                                    std::pair{2, 1}, std::pair{2, -1}, std::pair{-2, 1}, std::pair{-2, -1}})
            if (on_board(f + df, r + dr))
                emit(jieqi::make_square(f + df, r + dr));
    }

    template <typename Emit>
    static void bishop_attacks(int sq, Emit emit) {
        const int f = jieqi::file_of(sq);
        const int r = jieqi::rank_of(sq);
        for (const auto [df, dr] : {std::pair{2, 2}, std::pair{2, -2}, std::pair{-2, 2}, std::pair{-2, -2}})
            if (on_board(f + df, r + dr))
                emit(jieqi::make_square(f + df, r + dr));
    }

    template <typename Emit>
    static void pawn_attacks(int sq, jieqi::Color c, Emit emit) {
        const int f = jieqi::file_of(sq);
        const int r = jieqi::rank_of(sq);
        const int forward = c == jieqi::WHITE ? 1 : -1;
        if (on_board(f, r + forward))
            emit(jieqi::make_square(f, r + forward));
        const bool crossed = (c == jieqi::WHITE && r > 4) || (c == jieqi::BLACK && r < 5);
        if (crossed)
            for (int df : {-1, 1})
                if (on_board(f + df, r))
                    emit(jieqi::make_square(f + df, r));
    }

    template <typename Emit>
    static void possible_attacks(jieqi::Piece pc, int sq, Emit emit) {
        switch (jieqi::type_of(pc))
        {
        case ROOK: rook_attacks(sq, emit); break;
        case ADVISOR: advisor_attacks(sq, emit); break;
        case CANNON: cannon_possible_attacks(sq, emit); break;
        case PAWN: pawn_attacks(sq, jieqi::color_of(pc), emit); break;
        case KNIGHT: knight_attacks(sq, emit); break;
        case BISHOP: bishop_attacks(sq, emit); break;
        case KING: king_attacks(sq, emit); break;
        default: break;
        }
    }

    template <typename Emit>
    static void actual_attacks(const jieqi::Position& pos, jieqi::Piece pc, int sq, Emit emit) {
        const int f = jieqi::file_of(sq);
        const int r = jieqi::rank_of(sq);

        switch (jieqi::type_of(pc))
        {
        case ROOK:
            for (const auto [df, dr] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}})
            {
                int nf = f + df;
                int nr = r + dr;
                while (on_board(nf, nr))
                {
                    const int to = jieqi::make_square(nf, nr);
                    emit(to);
                    if (occupied(pos, to))
                        break;
                    nf += df;
                    nr += dr;
                }
            }
            break;
        case CANNON:
            for (const auto [df, dr] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}})
            {
                int nf = f + df;
                int nr = r + dr;
                bool hurdle = false;
                while (on_board(nf, nr))
                {
                    const int to = jieqi::make_square(nf, nr);
                    if (hurdle)
                        emit(to);
                    if (occupied(pos, to))
                    {
                        if (!hurdle)
                            hurdle = true;
                        else
                            break;
                    }
                    nf += df;
                    nr += dr;
                }
            }
            break;
        case KNIGHT:
            for (const auto [df, dr] : {std::pair{1, 2}, std::pair{-1, 2}, std::pair{1, -2}, std::pair{-1, -2},
                                        std::pair{2, 1}, std::pair{2, -1}, std::pair{-2, 1}, std::pair{-2, -1}})
            {
                const int lf = f + (std::abs(df) == 2 ? df / 2 : 0);
                const int lr = r + (std::abs(dr) == 2 ? dr / 2 : 0);
                if (on_board(f + df, r + dr) && !occupied(pos, jieqi::make_square(lf, lr)))
                    emit(jieqi::make_square(f + df, r + dr));
            }
            break;
        case BISHOP:
            for (const auto [df, dr] : {std::pair{2, 2}, std::pair{2, -2}, std::pair{-2, 2}, std::pair{-2, -2}})
                if (on_board(f + df, r + dr) && !occupied(pos, jieqi::make_square(f + df / 2, r + dr / 2)))
                    emit(jieqi::make_square(f + df, r + dr));
            break;
        default:
            possible_attacks(pc, sq, emit);
            break;
        }
    }

    static const std::array<ThreatRecord, THREAT_INPUTS>& records() {
        static const auto table = [] {
            std::array<ThreatRecord, THREAT_INPUTS> out{};
            int j = 0;
            for (int attacker : {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15})
            {
                const int pt = jieqi::type_of(jieqi::Piece(attacker));
                for (int from = 0; from < jieqi::SQUARE_NB; ++from)
                    for (int attacked : {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15})
                    {
                        if (!valid_pair(attacker, attacked))
                            continue;
                        possible_attacks(jieqi::Piece(attacker), from, [&](int to) {
                            const bool enemy = jieqi::color_of(jieqi::Piece(attacker)) != jieqi::color_of(jieqi::Piece(attacked));
                            const bool sameFile = jieqi::file_of(from) == jieqi::file_of(to);
                            const bool sameRank = jieqi::rank_of(from) == jieqi::rank_of(to);
                            const bool semiExcluded =
                                pt == jieqi::type_of(jieqi::Piece(attacked))
                                && (pt != PAWN || (enemy && sameFile) || (!enemy && sameRank))
                                && pt != KNIGHT;
                            if (!semiExcluded || from > to)
                                out[j++] = {attacker, from, to, attacked};
                        });
                    }
            }
            if (j != THREAT_INPUTS)
                std::abort();
            return out;
        }();
        return table;
    }

    static bool valid_pair(int attacker, int attacked) {
        static constexpr bool pairs[16][16] = {
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0},
            {0,1,1,1,0,1,0,0,0,1,0,1,1,1,0,0},
            {0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0},
            {0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,0},
            {0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0},
            {0,1,0,1,1,1,1,1,0,1,0,1,1,1,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1},
            {0,1,0,1,1,1,0,0,0,1,1,1,0,1,0,1},
            {0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1},
            {0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,0},
            {0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1},
            {0,1,0,1,1,1,0,0,0,1,0,1,1,1,1,1},
            {0,0,0,1,1,1,0,0,0,0,1,1,0,1,1,0},
        };
        return pairs[attacker][attacked];
    }

    static const std::vector<int>& offsets() {
        static const auto table = [] {
            std::vector<int> out(16 * jieqi::SQUARE_NB * jieqi::SQUARE_NB * 16, THREAT_INPUTS);
            const auto& recs = records();
            for (int i = 0; i < THREAT_INPUTS; ++i)
            {
                const auto& r = recs[i];
                out[offset_slot(r.attacker, r.from, r.to, r.attacked)] = i;
            }
            return out;
        }();
        return table;
    }

    static int offset_slot(int attacker, int from, int to, int attacked) {
        return (((attacker * jieqi::SQUARE_NB + from) * jieqi::SQUARE_NB + to) * 16 + attacked);
    }

    static int make_index(jieqi::Color perspective, int attacker, int from, int to, int attacked, bool mirror) {
        from = HalfKAv2HmJieqiV2::map_square(perspective, from, mirror);
        to = HalfKAv2HmJieqiV2::map_square(perspective, to, mirror);
        if (perspective == jieqi::BLACK)
        {
            attacker = flip_piece(attacker);
            attacked = flip_piece(attacked);
        }
        return offsets()[offset_slot(attacker, from, to, attacked)];
    }

    static int append_features(const jieqi::TrainingDataEntry& e,
                               int* features,
                               float* values,
                               jieqi::Color color,
                               int start,
                               int featureBase = HalfKAv2HmJieqiV2::INPUTS) {
        const int ksq = HalfKAv2HmJieqiV2::king_square(e.pos, color);
        const int oksq = HalfKAv2HmJieqiV2::king_square(e.pos, color == jieqi::WHITE ? jieqi::BLACK : jieqi::WHITE);
        const auto [kingBucket, mirror] = HalfKAv2HmJieqiV2::king_bucket(ksq, oksq, HalfKAv2HmJieqiV2::requires_mid_mirror(e.pos, color));

        int j = start;
        for (int from = 0; from < jieqi::SQUARE_NB; ++from)
        {
            if (!is_visible_piece(e.pos, from))
                continue;
            const int attacker = e.pos.board[from];
            actual_attacks(e.pos, jieqi::Piece(attacker), from, [&](int to) {
                if (!is_visible_piece(e.pos, to))
                    return;
                const int attacked = e.pos.board[to];
                const int index = make_index(color, attacker, from, to, attacked, mirror);
                if (index < THREAT_INPUTS)
                {
                    values[j] = 1.0f;
                    features[j] = featureBase + index;
                    ++j;
                }
            });
        }
        (void) kingBucket;
        return j;
    }
};

struct HalfKAv2HmJieqiV3FullThreats {
    static constexpr int INPUTS = HalfKAv2HmJieqiV3::INPUTS + JieqiFullThreats::THREAT_INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAv2HmJieqiV3::MAX_ACTIVE_FEATURES + JieqiFullThreats::MAX_ACTIVE_FEATURES;

    static int layer_stack_bucket(const jieqi::Position& pos) {
        return HalfKAv2HmJieqiV3::layer_stack_bucket(pos);
    }

    static std::pair<int, int> fill_features_sparse(
      const jieqi::TrainingDataEntry& e, int* features, float* values, jieqi::Color color) {
        auto [j, _] = HalfKAv2HmJieqiV3::fill_features_sparse(e, features, values, color);
        j = JieqiFullThreats::append_features(e, features, values, color, j, HalfKAv2HmJieqiV3::INPUTS);
        return {j, INPUTS};
    }
};

struct HalfKAv2HmJieqiV3FullThreatsFactorized {
    static constexpr int PSQ_FACTOR_INPUTS = HalfKAv2HmJieqiV3::PS_NB;
    static constexpr int BUCKET_FACTOR_INPUTS = 24;
    static constexpr int THREAT_PAIR_FACTOR_INPUTS = 16 * 16;
    static constexpr int THREAT_ATTACKER_FACTOR_INPUTS = 16 * jieqi::SQUARE_NB;
    static constexpr int THREAT_TARGET_FACTOR_INPUTS = 16 * jieqi::SQUARE_NB;
    static constexpr int VIRTUAL_INPUTS = PSQ_FACTOR_INPUTS + BUCKET_FACTOR_INPUTS
                                        + THREAT_PAIR_FACTOR_INPUTS
                                        + THREAT_ATTACKER_FACTOR_INPUTS
                                        + THREAT_TARGET_FACTOR_INPUTS;
    static constexpr int INPUTS = HalfKAv2HmJieqiV3FullThreats::INPUTS + VIRTUAL_INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = 1536;

    static constexpr int PSQ_FACTOR_BASE = HalfKAv2HmJieqiV3FullThreats::INPUTS;
    static constexpr int BUCKET_FACTOR_BASE = PSQ_FACTOR_BASE + PSQ_FACTOR_INPUTS;
    static constexpr int THREAT_PAIR_FACTOR_BASE = BUCKET_FACTOR_BASE + BUCKET_FACTOR_INPUTS;
    static constexpr int THREAT_ATTACKER_FACTOR_BASE = THREAT_PAIR_FACTOR_BASE + THREAT_PAIR_FACTOR_INPUTS;
    static constexpr int THREAT_TARGET_FACTOR_BASE = THREAT_ATTACKER_FACTOR_BASE + THREAT_ATTACKER_FACTOR_INPUTS;

    static int layer_stack_bucket(const jieqi::Position& pos) {
        return HalfKAv2HmJieqiV3::layer_stack_bucket(pos);
    }

    static void append_virtual(int realFeature, int& j, int* features, float* values) {
        if (realFeature < HalfKAv2HmJieqiV3::INPUTS)
        {
            values[j] = 1.0f;
            features[j++] = PSQ_FACTOR_BASE + (realFeature % HalfKAv2HmJieqiV3::PS_NB);
            values[j] = 1.0f;
            features[j++] = BUCKET_FACTOR_BASE + (realFeature / HalfKAv2HmJieqiV3::PS_NB);
            return;
        }

        const int threat = realFeature - HalfKAv2HmJieqiV3::INPUTS;
        const auto& rec = JieqiFullThreats::records()[threat];
        values[j] = 1.0f;
        features[j++] = THREAT_PAIR_FACTOR_BASE + rec.attacker * 16 + rec.attacked;
        values[j] = 1.0f;
        features[j++] = THREAT_ATTACKER_FACTOR_BASE + rec.attacker * jieqi::SQUARE_NB + rec.from;
        values[j] = 1.0f;
        features[j++] = THREAT_TARGET_FACTOR_BASE + rec.attacked * jieqi::SQUARE_NB + rec.to;
    }

    static std::pair<int, int> fill_features_sparse(
      const jieqi::TrainingDataEntry& e, int* features, float* values, jieqi::Color color) {
        auto [realCount, _] = HalfKAv2HmJieqiV3FullThreats::fill_features_sparse(e, features, values, color);
        int j = realCount;
        for (int i = 0; i < realCount; ++i)
            append_virtual(features[i], j, features, values);
        return {j, INPUTS};
    }
};

template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        return T::fill_features_sparse(e, features, values, color);
    }
};

static std::size_t max_sparse_batch_entries(int max_active)
{
    if (max_active <= 0)
        throw std::length_error(
            "SparseBatch max_active_features must be positive");
    return std::size_t(std::numeric_limits<int>::max())
        / std::size_t(max_active);
}

struct SparseBatch
{
    static constexpr bool IS_BATCH = true;

    template <typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;
        allocate_storage(
            entries.size(), FeatureSet<Ts...>::MAX_ACTIVE_FEATURES);

        for (std::size_t i = 0; i < entries.size(); ++i)
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
    }

    template <typename JieqiFeature>
    SparseBatch(
        FeatureSet<JieqiFeature>,
        const std::vector<jieqi::TrainingDataEntry>& entries,
        const std::vector<float>* eval_weights = nullptr,
        const std::vector<float>* layer_stack_blends = nullptr)
    {
        num_inputs = FeatureSet<JieqiFeature>::INPUTS;
        allocate_storage(
            entries.size(), FeatureSet<JieqiFeature>::MAX_ACTIVE_FEATURES,
            eval_weights != nullptr, layer_stack_blends != nullptr);

        if (eval_weights != nullptr && eval_weights->size() != entries.size())
            throw std::invalid_argument("eval_weight count does not match batch");
        if (layer_stack_blends != nullptr && layer_stack_blends->size() != entries.size())
            throw std::invalid_argument("layer_stack_blend count does not match batch");

        for (std::size_t i = 0; i < entries.size(); ++i)
            fill_jieqi_entry(
                FeatureSet<JieqiFeature>{}, i, entries[i],
                eval_weights != nullptr ? (*eval_weights)[i] : 0.0f,
                layer_stack_blends != nullptr ? (*layer_stack_blends)[i] : 0.0f);
    }

    int num_inputs;
    int size;

    float* is_white = nullptr;
    float* outcome = nullptr;
    float* score = nullptr;
    int num_active_white_features;
    int num_active_black_features;
    int max_active_features;
    int* white = nullptr;
    int* black = nullptr;
    float* white_values = nullptr;
    float* black_values = nullptr;
    int* psqt_indices = nullptr;
    int* layer_stack_indices = nullptr;
    // Optional tail field.  V3 batches leave this null; JQv4 V8 batches
    // provide one nonnegative weight per row for exact CP labels.
    float* eval_weight = nullptr;
    // Append-only tail: Q0.8 selection represented as blend_q8 / 255.
    float* layer_stack_blend = nullptr;

    std::unique_ptr<float[]> is_white_storage;
    std::unique_ptr<float[]> outcome_storage;
    std::unique_ptr<float[]> score_storage;
    std::unique_ptr<int[]> white_storage;
    std::unique_ptr<int[]> black_storage;
    std::unique_ptr<float[]> white_values_storage;
    std::unique_ptr<float[]> black_values_storage;
    std::unique_ptr<int[]> psqt_indices_storage;
    std::unique_ptr<int[]> layer_stack_indices_storage;
    std::unique_ptr<float[]> eval_weight_storage;
    std::unique_ptr<float[]> layer_stack_blend_storage;

    ~SparseBatch() = default;

private:

    void allocate_storage(
        std::size_t entry_count, int max_active, bool with_eval_weight = false,
        bool with_layer_stack_blend = false)
    {
        if (entry_count
            > std::size_t(std::numeric_limits<int>::max()))
            throw std::length_error(
                "SparseBatch entry count exceeds INT_MAX");
        if (entry_count > max_sparse_batch_entries(max_active))
            throw std::length_error(
                "SparseBatch active feature count exceeds INT_MAX");

        const auto active_count = std::size_t(max_active);
        if (entry_count
            > std::numeric_limits<std::size_t>::max() / active_count)
            throw std::length_error(
                "SparseBatch feature storage size overflow");
        const auto feature_count = entry_count * active_count;

        size = int(entry_count);
        num_active_white_features = 0;
        num_active_black_features = 0;
        max_active_features = max_active;

        is_white_storage = std::make_unique<float[]>(entry_count);
        outcome_storage = std::make_unique<float[]>(entry_count);
        score_storage = std::make_unique<float[]>(entry_count);
        white_storage = std::make_unique<int[]>(feature_count);
        black_storage = std::make_unique<int[]>(feature_count);
        white_values_storage = std::make_unique<float[]>(feature_count);
        black_values_storage = std::make_unique<float[]>(feature_count);
        psqt_indices_storage = std::make_unique<int[]>(entry_count);
        layer_stack_indices_storage = std::make_unique<int[]>(entry_count);
        if (with_eval_weight)
            eval_weight_storage = std::make_unique<float[]>(entry_count);
        if (with_layer_stack_blend)
            layer_stack_blend_storage = std::make_unique<float[]>(entry_count);

        is_white = is_white_storage.get();
        outcome = outcome_storage.get();
        score = score_storage.get();
        white = white_storage.get();
        black = black_storage.get();
        white_values = white_values_storage.get();
        black_values = black_values_storage.get();
        psqt_indices = psqt_indices_storage.get();
        layer_stack_indices = layer_stack_indices_storage.get();
        eval_weight = eval_weight_storage.get();
        layer_stack_blend = layer_stack_blend_storage.get();

        std::fill_n(white, feature_count, -1);
        std::fill_n(black, feature_count, -1);
        std::fill_n(white_values, feature_count, 0.0f);
        std::fill_n(black_values, feature_count, 0.0f);
    }

    template <typename... Ts>
    void fill_entry(
        FeatureSet<Ts...>,
        std::size_t i,
        const TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        psqt_indices[i] = (e.pos.pieceCount() - 1) * 8 / MAX_PIECES;
        layer_stack_indices[i] = psqt_indices[i];
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(
        FeatureSet<Ts...>,
        std::size_t i,
        const TrainingDataEntry& e)
    {
        const auto offset =
            i * std::size_t(FeatureSet<Ts...>::MAX_ACTIVE_FEATURES);
        num_active_white_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, white + offset, white_values + offset, Color::White)
            .first;
        num_active_black_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, black + offset, black_values + offset, Color::Black)
            .first;
    }

    template <typename JieqiFeature>
    void fill_jieqi_entry(
        FeatureSet<JieqiFeature>,
        std::size_t i,
        const jieqi::TrainingDataEntry& e,
        float eval_weight_value = 0.0f,
        float layer_stack_blend_value = 0.0f)
    {
        is_white[i] = static_cast<float>(e.pos.side == jieqi::WHITE);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        psqt_indices[i] = JieqiFeature::layer_stack_bucket(e.pos);
        layer_stack_indices[i] = psqt_indices[i];
        if (eval_weight != nullptr)
            eval_weight[i] = eval_weight_value;
        if (layer_stack_blend != nullptr)
            layer_stack_blend[i] = layer_stack_blend_value;
        fill_jieqi_features(FeatureSet<JieqiFeature>{}, i, e);
    }

    template <typename JieqiFeature>
    void fill_jieqi_features(
        FeatureSet<JieqiFeature>,
        std::size_t i,
        const jieqi::TrainingDataEntry& e)
    {
        const auto offset =
            i * std::size_t(FeatureSet<JieqiFeature>::MAX_ACTIVE_FEATURES);
        num_active_white_features +=
            JieqiFeature::fill_features_sparse(e, white + offset, white_values + offset, jieqi::WHITE)
            .first;
        num_active_black_features +=
            JieqiFeature::fill_features_sparse(e, black + offset, black_values + offset, jieqi::BLACK)
            .first;
    }
};

struct AnyStream
{
    virtual ~AnyStream() = default;
};

template <typename StorageT>
struct Stream : AnyStream
{
    using StorageType = StorageT;

    Stream() = default;

    Stream(int concurrency, const char* filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filename, cyclic, skipPredicate))
    {
    }

    virtual StorageT* next() = 0;

protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

template <typename StorageT>
struct AsyncStream : Stream<StorageT>
{
    using BaseType = Stream<StorageT>;

    AsyncStream(int concurrency, const char* filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(1, filename, cyclic, skipPredicate)
    {
    }

    ~AsyncStream()
    {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

protected:
    std::future<StorageT*> m_next;
};

template <typename JieqiFeature>
struct JieqiSparseBatchStream : Stream<SparseBatch>
{
    JieqiSparseBatchStream(
        const char* filename,
        int batch_size,
        bool cyclic,
        int rank,
        int world_size) :
        m_filename(filename ? filename : ""),
        m_batch_size(batch_size),
        m_cyclic(cyclic),
        m_rank(rank),
        m_world_size(world_size)
    {
        open_and_validate();
    }

    SparseBatch* next() override
    {
        std::vector<jieqi::TrainingDataEntry> entries;
        entries.reserve(m_batch_size);

        while (entries.size() < std::size_t(m_batch_size))
        {
            if (m_next_record >= m_end_record)
            {
                if (!m_cyclic)
                    break;
                rewind_shard();
            }

            jieqi::PackedSfenValue record{};
            if (!m_stream.read(reinterpret_cast<char*>(&record), sizeof(record)))
                throw std::runtime_error(
                    "unexpected end of Jieqi v3 record data");
            ++m_next_record;
            auto entry = jieqi::from_record(record);
            if constexpr (V81RawFeaturePolicy<JieqiFeature>::value) {
                normalize_v81_raw_unknown_loss(entry.pos);
            }
            entries.push_back(std::move(entry));
        }

        if (entries.empty())
            return nullptr;

        return new SparseBatch(FeatureSet<JieqiFeature>{}, entries);
    }

private:
    static constexpr std::streamoff JIEQI_HEADER_BYTES = 16;

    static std::uint32_t read_u32(std::istream& is)
    {
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i)
        {
            const int byte = is.get();
            if (byte == std::char_traits<char>::eof())
                throw std::runtime_error("truncated Jieqi v3 header");
            value |= std::uint32_t(std::uint8_t(byte)) << (8 * i);
        }
        return value;
    }

    static std::uint64_t shard_boundary(
        std::uint64_t total_records,
        int boundary,
        int world_size)
    {
        const auto world = std::uint64_t(world_size);
        const auto index = std::uint64_t(boundary);
        const auto quotient = total_records / world;
        const auto remainder = total_records % world;
        return quotient * index + (remainder * index) / world;
    }

    void open_and_validate()
    {
        if (m_batch_size <= 0)
            throw std::runtime_error("Jieqi v3 batch_size must be positive");
        if (std::size_t(m_batch_size)
            > max_sparse_batch_entries(
                FeatureSet<JieqiFeature>::MAX_ACTIVE_FEATURES))
            throw std::runtime_error(
                "Jieqi v3 batch_size is too large for active feature counts");
        if (m_world_size <= 0 || m_rank < 0 || m_rank >= m_world_size)
            throw std::runtime_error("invalid Jieqi v3 rank/world_size");
        if (m_filename.empty())
            throw std::runtime_error("Jieqi v3 filename must not be empty");

        m_stream = std::ifstream(m_filename, std::ios::binary);
        if (!m_stream)
            throw std::runtime_error("cannot open Jieqi v3 data file");

        const auto magic = read_u32(m_stream);
        const auto version = read_u32(m_stream);
        const auto record_size = read_u32(m_stream);
        const auto fen_bytes = read_u32(m_stream);
        if (magic != jieqi::MAGIC || version != jieqi::VERSION
            || record_size != sizeof(jieqi::PackedSfenValue)
            || fen_bytes != jieqi::FEN_BYTES)
            throw std::runtime_error("invalid Jieqi v3 header");

        m_stream.seekg(0, std::ios::end);
        const auto end_position = m_stream.tellg();
        if (end_position == std::streampos(-1))
            throw std::runtime_error("cannot determine Jieqi v3 file size");
        const auto file_bytes = static_cast<std::streamoff>(end_position);
        if (file_bytes < JIEQI_HEADER_BYTES)
            throw std::runtime_error(
                "Jieqi v3 file is smaller than its header");

        const auto data_bytes = file_bytes - JIEQI_HEADER_BYTES;
        const auto record_bytes =
            std::streamoff(sizeof(jieqi::PackedSfenValue));
        if (data_bytes % record_bytes != 0)
            throw std::runtime_error("invalid Jieqi v3 payload length");

        m_total_records =
            std::uint64_t(data_bytes / record_bytes);
        if (m_total_records < std::uint64_t(m_world_size))
            throw std::runtime_error(
                "Jieqi v3 file has fewer records than ranks");

        m_start_record =
            shard_boundary(m_total_records, m_rank, m_world_size);
        m_end_record =
            shard_boundary(m_total_records, m_rank + 1, m_world_size);
        rewind_shard();
    }

    void rewind_shard()
    {
        m_stream.clear();
        m_stream.seekg(
            JIEQI_HEADER_BYTES
                + std::streamoff(
                    m_start_record * sizeof(jieqi::PackedSfenValue)),
            std::ios::beg);
        if (!m_stream)
            throw std::runtime_error("failed to seek to Jieqi v3 shard");
        m_next_record = m_start_record;
    }

    std::string m_filename;
    int m_batch_size;
    bool m_cyclic;
    int m_rank;
    int m_world_size;
    std::uint64_t m_total_records = 0;
    std::uint64_t m_start_record = 0;
    std::uint64_t m_end_record = 0;
    std::uint64_t m_next_record = 0;
    std::ifstream m_stream;
};

namespace {

namespace jqv4 = jieqi::v4;
using jqv4::JsonKind;
using jqv4::JsonValueV1;

const JsonValueV1* json_member(const JsonValueV1& object,
                               std::string_view key)
{
    if (object.kind != JsonKind::Object)
        return nullptr;
    for (const auto& member : object.object_members)
        if (member.first == key)
            return &member.second;
    return nullptr;
}

std::string json_string(const JsonValueV1& object,
                        std::string_view key,
                        const char* context)
{
    const auto* value = json_member(object, key);
    if (!value || value->kind != JsonKind::String || value->scalar_utf8.empty())
        throw std::runtime_error(
            std::string(context) + " requires a nonempty string field "
            + std::string(key));
    return value->scalar_utf8;
}

std::uint64_t json_u64(const JsonValueV1& object,
                       std::string_view key,
                       const char* context)
{
    const auto* value = json_member(object, key);
    if (!value || value->kind != JsonKind::Integer || value->integer_negative)
        throw std::runtime_error(
            std::string(context) + " requires a nonnegative integer field "
            + std::string(key));
    return value->integer_magnitude;
}

std::uint64_t json_optional_u64(const JsonValueV1& object,
                                std::string_view key,
                                std::uint64_t fallback,
                                const char* context)
{
    const auto* value = json_member(object, key);
    if (!value)
        return fallback;
    if (value->kind != JsonKind::Integer || value->integer_negative)
        throw std::runtime_error(
            std::string(context) + " field " + std::string(key)
            + " must be a nonnegative integer");
    return value->integer_magnitude;
}

struct Jqv4ManifestFile
{
    std::filesystem::path path;
    std::string shard;
    std::uint64_t record_start = 0U;
    std::uint64_t record_end = 0U;
    std::array<std::uint8_t, 32> sha256{};
    std::array<std::uint8_t, 16> file_uuid{};
};

struct Jqv4Manifest
{
    std::vector<Jqv4ManifestFile> files;
    std::string input_root;
    std::optional<std::array<std::uint8_t, 16>> dataset_uuid;
    std::string partition;
    std::string split_strategy;
    std::uint64_t seed = 0U;
    std::uint32_t numerator = 0U;
    std::uint32_t denominator = 1U;
    std::string shuffle_strategy;
    std::uint64_t shuffle_seed = 0U;
    std::uint32_t shuffle_buffer_blocks = 0U;
};

std::array<std::uint8_t, 32> parse_sha256(std::string_view text,
                                          const char* context)
{
    if (text.size() != 64U)
        throw std::runtime_error(std::string(context)
                                 + " sha256 must contain 64 hex digits");
    std::array<std::uint8_t, 32> result{};
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (std::size_t i = 0; i < result.size(); ++i) {
        const int hi = hex(text[i * 2U]);
        const int lo = hex(text[i * 2U + 1U]);
        if (hi < 0 || lo < 0)
            throw std::runtime_error(std::string(context)
                                     + " sha256 contains non-hex digits");
        result[i] = static_cast<std::uint8_t>((hi << 4) | lo);
    }
    return result;
}

std::array<std::uint8_t, 16> parse_uuid16(std::string_view text,
                                          const char* context)
{
    if (text.size() != 36U || text[8] != '-' || text[13] != '-'
        || text[18] != '-' || text[23] != '-')
        throw std::runtime_error(std::string(context)
                                 + " UUID must use canonical 36-character form");
    std::string compact;
    compact.reserve(32U);
    for (char c : text)
        if (c != '-')
            compact.push_back(c);
    const auto parsed = parse_sha256(compact + std::string(32U, '0'), context);
    std::array<std::uint8_t, 16> result{};
    std::copy_n(parsed.begin(), result.size(), result.begin());
    return result;
}

std::string uuid_key(const std::array<std::uint8_t, 16>& uuid)
{
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(32U);
    for (const auto byte : uuid) {
        result.push_back(hex[byte >> 4U]);
        result.push_back(hex[byte & 0x0fU]);
    }
    return result;
}

 jqv4::FileHeaderV4 read_jqv4_header(const std::filesystem::path& path)
{
    // Read only the fixed header.  This is intentionally much cheaper than
    // FileReaderV4::open_finalized, which scans every block index and game
    // table in a source file.
    std::array<std::uint8_t, 128> bytes{};
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open JQv4 source header: "
                                 + path.string());
    input.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size()))
        throw std::runtime_error("JQv4 source header is truncated: "
                                 + path.string());
    const auto decoded = jqv4::decode_file_header_v4(
        jqv4::ByteView{bytes.data(), bytes.size()});
    if (!decoded.ok())
        throw std::runtime_error("cannot decode JQv4 source header "
                                 + path.string() + ": "
                                 + decoded.status.message);
    return *decoded.value;
}

std::array<std::uint8_t, 16>
read_jqv4_header_file_uuid(const std::filesystem::path& path)
{
    return read_jqv4_header(path).file_uuid;
}

Jqv4Manifest parse_jqv4_manifest(const std::filesystem::path& manifest_path)
{
    std::ifstream input(manifest_path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open JQv4 manifest");
    std::vector<std::uint8_t> bytes{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad() || bytes.empty())
        throw std::runtime_error("cannot read JQv4 manifest");

    const auto parsed = jqv4::detail::parse_json_v1(
        jqv4::ByteView{bytes.data(), bytes.size()});
    if (!parsed.ok())
        throw std::runtime_error("invalid JQv4 manifest JSON: "
                                 + parsed.status.message);
    const auto& root = *parsed.value;
    if (root.kind != JsonKind::Object)
        throw std::runtime_error("JQv4 manifest root must be an object");
    if (json_string(root, "schema", "JQv4 manifest")
        != "abjchess-v8-jqv4-manifest-v1")
        throw std::runtime_error("unsupported JQv4 manifest schema");

    Jqv4Manifest result;
    if (const auto* dataset = json_member(root, "dataset_uuid")) {
        if (dataset->kind != JsonKind::String)
            throw std::runtime_error("JQv4 manifest dataset_uuid must be a string");
        result.dataset_uuid = parse_uuid16(
            dataset->scalar_utf8, "JQv4 manifest dataset_uuid");
    }
    if (const auto* input_root = json_member(root, "input_root")) {
        if (input_root->kind != JsonKind::String)
            throw std::runtime_error("JQv4 manifest input_root must be a string");
        result.input_root = input_root->scalar_utf8;
    }

    const auto* split = json_member(root, "split");
    if (!split || split->kind != JsonKind::Object)
        throw std::runtime_error("JQv4 manifest split must be an object");
    result.split_strategy = json_string(
        *split, "strategy", "JQv4 manifest split");
    if (result.split_strategy != "whole-game-h64-v1")
        throw std::runtime_error("unsupported JQv4 split strategy");
    result.partition = json_string(*split, "partition", "JQv4 manifest split");
    if (result.partition != "train" && result.partition != "validation")
        throw std::runtime_error("JQv4 split partition must be train or validation");
    const auto seed = json_u64(*split, "seed", "JQv4 manifest split");
    const auto numerator = json_u64(*split, "numerator", "JQv4 manifest split");
    const auto denominator = json_u64(*split, "denominator", "JQv4 manifest split");
    if (seed > std::numeric_limits<std::uint64_t>::max()
        || numerator > std::numeric_limits<std::uint32_t>::max()
        || denominator > std::numeric_limits<std::uint32_t>::max()
        || denominator == 0U || numerator >= denominator)
        throw std::runtime_error("invalid JQv4 split fraction");
    result.seed = seed;
    result.numerator = static_cast<std::uint32_t>(numerator);
    result.denominator = static_cast<std::uint32_t>(denominator);

    const auto* shuffle = json_member(root, "shuffle");
    if (!shuffle || shuffle->kind != JsonKind::Object)
        throw std::runtime_error("JQv4 manifest shuffle must be an object");
    result.shuffle_strategy = json_string(
        *shuffle, "strategy", "JQv4 manifest shuffle");
    if (result.shuffle_strategy != "hierarchical-block-window-v1")
        throw std::runtime_error("unsupported JQv4 shuffle strategy");
    result.shuffle_seed = json_u64(
        *shuffle, "seed", "JQv4 manifest shuffle");
    const auto buffer_blocks = json_u64(
        *shuffle, "buffer_blocks", "JQv4 manifest shuffle");
    if (buffer_blocks == 0U
        || buffer_blocks > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error(
            "JQv4 shuffle buffer_blocks must fit nonzero u32");
    result.shuffle_buffer_blocks =
        static_cast<std::uint32_t>(buffer_blocks);

    const auto* files = json_member(root, "files");
    if (!files || files->kind != JsonKind::Array || files->array_items.empty())
        throw std::runtime_error("JQv4 manifest files must be a nonempty array");
    std::unordered_set<std::string> seen_paths;
    result.files.reserve(files->array_items.size());
    for (const auto& file : files->array_items) {
        if (file.kind != JsonKind::Object)
            throw std::runtime_error("JQv4 manifest file entry must be an object");
        const auto* path_value = json_member(file, "path");
        const auto* relative_value = json_member(file, "relative_path");
        const JsonValueV1* selected_path = path_value ? path_value : relative_value;
        if (!selected_path || selected_path->kind != JsonKind::String
            || selected_path->scalar_utf8.empty())
            throw std::runtime_error(
                "JQv4 manifest file requires path or relative_path");
        auto path_text = selected_path->scalar_utf8;
        std::filesystem::path path(path_text);
        if (path.is_relative()) {
            auto base = result.input_root.empty()
                ? manifest_path.parent_path()
                : std::filesystem::path(result.input_root);
            if (base.is_relative())
                base = manifest_path.parent_path() / base;
            path = base / path;
        }
        std::error_code ec;
        path = std::filesystem::weakly_canonical(path, ec);
        if (ec || !std::filesystem::is_regular_file(path, ec) || ec)
            throw std::runtime_error("JQv4 manifest source file is missing: "
                                     + path.string());
        if (path.extension() != ".jqv4")
            throw std::runtime_error("JQv4 manifest source must end in .jqv4");
        const auto key = path.generic_u8string();
        if (!seen_paths.insert(key).second)
            throw std::runtime_error("JQv4 manifest contains a duplicate source file");
        auto shard = json_string(file, "shard", "JQv4 manifest file");
        if (shard != "1" && shard != "2" && shard != "3" && shard != "4")
            throw std::runtime_error("JQv4 manifest file has an invalid shard");
        const auto expected_sha = parse_sha256(
            json_string(file, "sha256", "JQv4 manifest file"),
            "JQv4 manifest file");
        const auto file_uuid = parse_uuid16(
            json_string(file, "file_uuid", "JQv4 manifest file"),
            "JQv4 manifest file");
        const auto record_start = json_u64(
            file, "record_start", "JQv4 manifest file");
        const auto record_end = json_u64(
            file, "record_end", "JQv4 manifest file");
        if (record_start >= record_end)
            throw std::runtime_error(
                "JQv4 manifest file record range is empty or reversed");
        const auto actual_sha = jqv4::sha256_file(path);
        if (!actual_sha.ok())
            throw std::runtime_error("cannot hash JQv4 source file "
                                     + path.string() + ": "
                                     + actual_sha.status.message);
        if (*actual_sha.value != expected_sha)
            throw std::runtime_error("JQv4 source SHA-256 mismatch: "
                                     + path.string());
        result.files.push_back(Jqv4ManifestFile{
            std::move(path), std::move(shard), record_start, record_end,
            expected_sha, file_uuid});
    }
    // The native shuffle fingerprint intentionally consumes manifest order.
    // Bind that order to the same canonical identity tuple as the Python
    // provenance checker so a direct DLL caller cannot silently reshuffle a
    // corpus while retaining a teacher digest.
    for (std::size_t i = 1U; i < result.files.size(); ++i) {
        const auto& previous = result.files[i - 1U];
        const auto& current = result.files[i];
        const bool out_of_order =
            current.record_start < previous.record_start
            || (current.record_start == previous.record_start
                && (current.record_end < previous.record_end
                    || (current.record_end == previous.record_end
                        && current.file_uuid < previous.file_uuid)));
        if (out_of_order)
            throw std::runtime_error(
                "JQv4 manifest files are not in canonical order");
        if (current.record_start != previous.record_end)
            throw std::runtime_error(
                "JQv4 manifest file record ranges are not contiguous");
    }
    if (result.partition == "validation") {
        for (const auto& file : result.files)
            if (file.shard != "4")
                throw std::runtime_error(
                    "validation JQv4 manifest may only contain shard 4");
    }
    return result;
}

std::uint64_t jqv4_split_threshold(std::uint32_t numerator,
                                   std::uint32_t denominator)
{
    if (numerator == 0U)
        return 0U;
    const std::uint64_t d = denominator;
    std::uint64_t quotient = std::numeric_limits<std::uint64_t>::max() / d;
    std::uint64_t remainder =
        std::numeric_limits<std::uint64_t>::max() % d + 1U;
    if (remainder == d) {
        ++quotient;
        remainder = 0U;
    }
    return quotient * numerator + (remainder * numerator) / d;
}

std::uint64_t jqv4_shuffle_fingerprint(
    const std::array<std::uint8_t, 16>& dataset_uuid,
    const std::string& partition,
    const std::vector<Jqv4ManifestFile>& files)
{
    jqv4::DomainHash64 hash(
        "abjchess-v8-jqv4-shuffle-fingerprint-v1");
    const auto add = [&hash](jqv4::ByteView bytes, const char* field) {
        const auto status = hash.add_bytes(bytes);
        if (!status.ok())
            throw std::runtime_error(
                std::string("cannot hash JQv4 shuffle ") + field + ": "
                + status.message);
    };
    add(jqv4::ByteView{dataset_uuid.data(), dataset_uuid.size()},
        "dataset UUID");
    add(jqv4::ByteView{
            reinterpret_cast<const std::uint8_t*>(partition.data()),
            partition.size()},
        "partition");
    const auto count_status = hash.add_u64(
        static_cast<std::uint64_t>(files.size()));
    if (!count_status.ok())
        throw std::runtime_error(
            "cannot hash JQv4 shuffle file count: "
            + count_status.message);
    for (const auto& file : files)
        add(jqv4::ByteView{file.file_uuid.data(), file.file_uuid.size()},
            "file UUID");
    return hash.finish();
}

jieqi::Piece jqv4_piece(std::uint8_t code)
{
    if (code == jqv4::kEmptyPieceCodeV1)
        return jieqi::NO_PIECE;
    if (code == 14U || code == 15U)
        return jieqi::DARK_PIECE;
    if (code <= 6U)
        return static_cast<jieqi::Piece>(1 + code);
    if (code <= 13U)
        return static_cast<jieqi::Piece>(9 + (code - 7U));
    throw std::runtime_error("JQv4 observation contains an invalid piece code");
}

jieqi::TrainingDataEntry jqv4_to_training_entry(
    const jqv4::DecodedRecordV1& record, float& eval_weight)
{
    jieqi::TrainingDataEntry result{};
    const auto& observation = record.observation;
    std::fill(std::begin(result.pos.board), std::end(result.pos.board),
              jieqi::NO_PIECE);
    std::fill(std::begin(result.pos.dark), std::end(result.pos.dark), false);
    std::fill(std::begin(result.pos.rest), std::end(result.pos.rest), 0);
    std::fill(std::begin(result.pos.unknown_loss),
              std::end(result.pos.unknown_loss), 0);
    for (int sq = 0; sq < jieqi::SQUARE_NB; ++sq) {
        const auto code = observation.board[static_cast<std::size_t>(sq)];
        if (code == jqv4::kEmptyPieceCodeV1)
            continue;
        result.pos.board[sq] = jqv4_piece(code);
        result.pos.dark[sq] = code == 14U || code == 15U;
    }
    for (int color = 0; color < 2; ++color)
        for (int type = 1; type <= 6; ++type)
            result.pos.rest[jieqi::make_piece(
                static_cast<jieqi::Color>(color), type)] =
                observation.rest[color][type - 1];
    result.pos.unknown_loss[jieqi::WHITE] = observation.unknown_loss[0];
    result.pos.unknown_loss[jieqi::BLACK] = observation.unknown_loss[1];
    // V8 keeps unknown captured identities explicit.  The observation's
    // colored dark markers are the only source for the dark cover count; do
    // not infer either side from total dark density.
    int darkByColor[2] = {0, 0};
    for (const auto code : observation.board) {
        if (code == 14U) ++darkByColor[jieqi::WHITE];
        else if (code == 15U) ++darkByColor[jieqi::BLACK];
    }
    for (int color = 0; color < 2; ++color) {
        if (result.pos.unknown_loss[color] < 0
            || result.pos.unknown_loss[color] > 15)
            throw std::runtime_error(
                "JQv4 V8 unknown_loss is outside the observation domain");
        int restCount = 0;
        for (int type = 1; type <= 6; ++type)
            restCount += result.pos.rest[jieqi::make_piece(
                static_cast<jieqi::Color>(color), type)];
        if (restCount != darkByColor[color] + result.pos.unknown_loss[color])
            throw std::runtime_error("JQv4 V8 unknown_loss does not match rest/dark inventory");
    }
    result.pos.side = observation.side_to_move == 0U
        ? jieqi::WHITE : jieqi::BLACK;
    result.pos.gamePly = record.core.game_ply;
    result.move = record.core.teacher_best_action == 0xffffU
        ? 0U : record.core.teacher_best_action;
    result.score = record.core.root_value;
    result.ply = record.core.game_ply;

    // Synthetic results are already encoded from the side-to-move
    // perspective.  Native game metadata stores the winner as white/black.
    if (record.legacy_result.has_value()) {
        result.result = *record.legacy_result;
    } else {
        const auto outcome = static_cast<std::uint16_t>(
            record.game.packed_metadata & 0x0003U);
        // GameTable outcome codes are 1=draw, 2=white win, 3=black win.
        int value = outcome == 2U ? 1 : outcome == 3U ? -1 : 0;
        if (result.pos.side == jieqi::BLACK)
            value = -value;
        result.result = static_cast<std::int16_t>(value);
    }

    const auto flags = record.core.record_flags;
    const bool cp = (flags & 0x0003U) == 1U;
    const bool exact = ((flags >> 2U) & 0x0003U) == 0U;
    const bool completed = (flags & 0x0040U) != 0U;
    const bool clipped = (flags & 0x0080U) != 0U;
    eval_weight = cp && exact && completed && !clipped ? 1.0f : 0.0f;
    return result;
}

} // namespace

template <typename JieqiFeature>
struct Jqv4SparseBatchStream : Stream<SparseBatch>
{
    Jqv4SparseBatchStream(
        const char* manifest_filename,
        int batch_size,
        bool cyclic,
        int rank,
        int world_size)
        : m_manifest_path(manifest_filename ? manifest_filename : ""),
          m_batch_size(batch_size), m_cyclic(cyclic), m_rank(rank),
          m_world_size(world_size)
    {
        open_and_validate();
    }

    SparseBatch* next() override
    {
        std::vector<jieqi::TrainingDataEntry> entries;
        std::vector<float> eval_weights;
        std::vector<float> layer_stack_blends;
        entries.reserve(static_cast<std::size_t>(m_batch_size));
        eval_weights.reserve(static_cast<std::size_t>(m_batch_size));
        layer_stack_blends.reserve(static_cast<std::size_t>(m_batch_size));

        while (entries.size() < static_cast<std::size_t>(m_batch_size)) {
            if (!m_block.has_value() ||
                m_record_index >= m_block->records.size()) {
                if (!load_next_block()) {
                    if (entries.empty())
                        return nullptr;
                    break;
                }
            }
            const auto record_ordinal = m_record_order[m_record_index++];
            if (record_ordinal >= m_block->records.size())
                throw std::runtime_error(
                    "JQv4 record shuffle produced an invalid ordinal");
            const auto& record = m_block->records[
                static_cast<std::size_t>(record_ordinal)];
            if (!include_record(record, m_current_shard))
                continue;
            float weight = 0.0f;
            entries.push_back(jqv4_to_training_entry(record, weight));
            eval_weights.push_back(weight);
            const auto selection = JieqiFeature::layer_stack_selection(entries.back().pos);
            layer_stack_blends.push_back(static_cast<float>(selection.blend_q8) / 255.0f);
        }
        if (entries.empty())
            return nullptr;
        return new SparseBatch(
            FeatureSet<JieqiFeature>{}, entries, &eval_weights, &layer_stack_blends);
    }

private:
    bool load_next_block()
    {
        m_block.reset();
        m_record_order.clear();
        for (;;) {
            // A global CatalogV1 materializes one block descriptor for every
            // block in every source file.  The production tree has millions
            // of small blocks, so retaining that vector costs gigabytes
            // before the first training row is available.  Keep one reader
            // and one decoded block alive instead, advancing lazily.
            if (!m_file_reader) {
                if (m_file_order_index >= m_file_order.size()) {
                    if (!m_cyclic)
                        return false;
                    if (m_coverage_cycle
                        == std::numeric_limits<std::uint64_t>::max())
                        throw std::runtime_error(
                            "JQv4 shuffle coverage cycle overflow");
                    ++m_coverage_cycle;
                    initialize_cycle();
                }
                m_current_file_ordinal = m_file_order[m_file_order_index++];
                if (m_current_file_ordinal >= m_manifest.files.size())
                    throw std::runtime_error(
                        "JQv4 file shuffle produced an invalid ordinal");
                const auto& file = m_manifest.files[
                    static_cast<std::size_t>(m_current_file_ordinal)];
                auto opened = jqv4::FileReaderV4::open_finalized_lazy(file.path);
                if (!opened.ok())
                    throw std::runtime_error(
                        "cannot open JQv4 source " + file.path.string()
                        + ": " + opened.status.message);
                m_file_reader = std::move(*opened.value);
                if (m_file_reader->header().file_uuid != file.file_uuid)
                    throw std::runtime_error(
                        "JQv4 source file UUID changed after manifest validation: "
                        + file.path.string());
                if (m_file_reader->index().empty())
                    throw std::runtime_error(
                        "JQv4 source contains no blocks: " + file.path.string());
                m_current_shard = file.shard;
                const auto block_window_seed = jqv4::hash_block_seed_v1(
                    m_cycle_seed, file.file_uuid,
                    std::numeric_limits<std::uint64_t>::max());
                m_block_shuffle.emplace(
                    static_cast<std::uint64_t>(m_file_reader->index().size()),
                    m_manifest.shuffle_buffer_blocks, block_window_seed);
            }

            const auto block_ordinal = m_block_shuffle->next();
            if (!block_ordinal.has_value()) {
                m_block_shuffle.reset();
                m_file_reader.reset();
                continue;
            }
            if (*block_ordinal >= m_file_reader->index().size())
                throw std::runtime_error(
                    "JQv4 block shuffle produced an invalid ordinal");
            const auto block_id = m_file_reader->index()[
                static_cast<std::size_t>(*block_ordinal)].block_id;
            auto decoded = m_file_reader->decode_block(
                block_id, jqv4::DecodeOptionsV1{false, false, true, false});
            if (!decoded.ok())
                throw std::runtime_error("cannot decode JQv4 block: "
                                         + decoded.status.message);
            m_block = std::move(*decoded.value);
            const auto& file = m_manifest.files[
                static_cast<std::size_t>(m_current_file_ordinal)];
            const auto record_seed = jqv4::hash_block_seed_v1(
                m_cycle_seed, file.file_uuid, block_id);
            m_record_order =
                abjchess::v8_training::permuted_ordinals_v1(
                    m_block->records.size(), record_seed);
            m_record_index = 0U;
            if (!m_block->records.empty())
                return true;
        }
    }

    void initialize_cycle()
    {
        m_cycle_seed = jqv4::hash_cycle_seed_v1(
            m_shuffle_fingerprint, m_manifest.shuffle_seed,
            m_coverage_cycle, 1U);
        m_file_order = abjchess::v8_training::permuted_ordinals_v1(
            m_manifest.files.size(), m_cycle_seed);
        m_file_order_index = 0U;
        m_file_reader.reset();
        m_block_shuffle.reset();
        m_block.reset();
        m_record_order.clear();
        m_record_index = 0U;
        m_current_shard.clear();
    }

    bool include_record(const jqv4::DecodedRecordV1& record,
                        const std::string& shard) const noexcept
    {
        const bool selected = jqv4::hash_split_v1(
            m_dataset_uuid, record.game.game_id, m_split_seed) < m_split_threshold;
        if (shard == "4")
            return m_partition == "validation" ? selected : !selected;
        return m_partition == "train";
    }

    void open_and_validate()
    {
        if (m_batch_size <= 0)
            throw std::runtime_error("JQv4 batch_size must be positive");
        if (std::size_t(m_batch_size) > max_sparse_batch_entries(
                FeatureSet<JieqiFeature>::MAX_ACTIVE_FEATURES))
            throw std::runtime_error("JQv4 batch_size is too large for active feature counts");
        if (m_world_size <= 0 || m_rank < 0 || m_rank >= m_world_size)
            throw std::runtime_error("invalid JQv4 rank/world_size");
        if (m_rank != 0 || m_world_size != 1)
            throw std::runtime_error("JQv4 stream currently requires one rank");
        if (m_manifest_path.empty())
            throw std::runtime_error("JQv4 manifest path must not be empty");

        m_manifest = parse_jqv4_manifest(m_manifest_path);
        m_split_seed = m_manifest.seed;
        m_partition = m_manifest.partition;
        m_split_threshold = jqv4_split_threshold(
            m_manifest.numerator, m_manifest.denominator);
        if (m_manifest.files.empty())
            throw std::runtime_error("JQv4 manifest contains no source files");

        bool have_baseline = false;
        std::array<std::uint8_t, 16> baseline_dataset{};
        jqv4::SplitMetadataV1 baseline_split{};
        std::string baseline_feature_name;
        std::array<std::uint8_t, 32> baseline_feature_identity{};
        std::unordered_set<std::string> file_uuids;
        for (std::size_t ordinal = 0U;
             ordinal < m_manifest.files.size(); ++ordinal) {
            const auto& file = m_manifest.files[ordinal];
            if (read_jqv4_header_file_uuid(file.path) != file.file_uuid)
                throw std::runtime_error(
                    "JQv4 manifest file_uuid disagrees with source header: "
                    + file.path.string());
            if (!file_uuids.emplace(uuid_key(file.file_uuid)).second)
                throw std::runtime_error(
                    "JQv4 manifest contains duplicate file_uuid values");

            // Open and validate one source at a time.  Every reader is
            // released here; the streaming loop later opens the first file
            // selected by the cycle's shuffled order.
            auto opened = jqv4::FileReaderV4::open_finalized_lazy(file.path);
            if (!opened.ok())
                throw std::runtime_error(
                    "cannot open JQv4 source " + file.path.string()
                    + ": " + opened.status.message);
            auto reader = std::move(*opened.value);
            if (reader->index().empty())
                throw std::runtime_error(
                    "JQv4 source contains no blocks: " + file.path.string());
            const auto& header = reader->header();
            const auto& metadata = reader->metadata();
            if (!have_baseline) {
                baseline_dataset = header.dataset_uuid;
                baseline_split = metadata.split;
                baseline_feature_name = metadata.feature_set.name;
                baseline_feature_identity = metadata.feature_set.identity_sha256;
                m_dataset_uuid = baseline_dataset;
                have_baseline = true;
            } else {
                if (header.dataset_uuid != baseline_dataset
                    || metadata.dataset_uuid != baseline_dataset)
                    throw std::runtime_error(
                        "JQv4 sources have mixed dataset UUIDs");
                if (metadata.split.role != baseline_split.role
                    || metadata.split.seed != baseline_split.seed
                    || metadata.split.validation_numerator
                           != baseline_split.validation_numerator
                    || metadata.split.validation_denominator
                           != baseline_split.validation_denominator
                    || metadata.split.validation_threshold
                           != baseline_split.validation_threshold)
                    throw std::runtime_error(
                        "JQv4 sources have mixed split identities");
                if (metadata.feature_set.name != baseline_feature_name
                    || metadata.feature_set.identity_sha256
                           != baseline_feature_identity)
                    throw std::runtime_error(
                        "JQv4 sources have mixed feature identities");
            }
        }
        if (m_manifest.dataset_uuid.has_value()
            && *m_manifest.dataset_uuid != m_dataset_uuid)
            throw std::runtime_error("JQv4 manifest dataset UUID disagrees with source files");
        if (!have_baseline)
            throw std::runtime_error("JQv4 manifest contains no readable source files");
        m_shuffle_fingerprint = jqv4_shuffle_fingerprint(
            m_dataset_uuid, m_partition, m_manifest.files);
        initialize_cycle();
    }

    std::filesystem::path m_manifest_path;
    Jqv4Manifest m_manifest;
    std::shared_ptr<jqv4::FileReaderV4> m_file_reader;
    std::array<std::uint8_t, 16> m_dataset_uuid{};
    std::uint64_t m_split_seed = 0U;
    std::uint64_t m_split_threshold = 0U;
    std::string m_partition;
    std::uint64_t m_shuffle_fingerprint = 0U;
    std::uint64_t m_coverage_cycle = 0U;
    std::uint64_t m_cycle_seed = 0U;
    std::vector<std::uint64_t> m_file_order;
    std::size_t m_file_order_index = 0U;
    std::uint64_t m_current_file_ordinal = 0U;
    std::optional<abjchess::v8_training::BoundedOrdinalShuffleV1>
        m_block_shuffle;
    std::vector<std::uint64_t> m_record_order;
    std::size_t m_record_index = 0U;
    std::string m_current_shard;
    std::optional<jqv4::DecodedBlockV1> m_block;
    int m_batch_size;
    bool m_cyclic;
    int m_rank;
    int m_world_size;
};

template <typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream : Stream<StorageT>
{
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(int concurrency, const char* filename, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filename,
            cyclic,
            skipPredicate
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while(!m_stop_flag.load())
            {
                entries.clear();

                {
                    std::unique_lock lock(m_stream_mutex);
                    BaseType::m_stream->fill(entries, m_batch_size);
                    if (entries.empty())
                    {
                        break;
                    }
                }

                auto batch = new StorageT(FeatureSet{}, entries);

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }

            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    StorageT* next() override
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FeaturedBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    std::deque<StorageT*> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
};


std::function<bool(const TrainingDataEntry&)> make_skip_predicate(bool filtered, int random_fen_skipping)
{
    if (filtered || random_fen_skipping)
    {
        return [
            random_fen_skipping,
            prob = double(random_fen_skipping) / (random_fen_skipping + 1),
            filtered
            ](const TrainingDataEntry& e){

            auto do_skip = [&]() {
                std::bernoulli_distribution distrib(prob);
                auto& prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_filter = [&]() {
                return false;
            };

            static thread_local std::mt19937 gen(std::random_device{}());
            return (random_fen_skipping && do_skip()) || (filtered && do_filter());
        };
    }

    return nullptr;
}

thread_local std::string training_data_loader_last_error;

static void set_training_data_loader_error(const char* message) noexcept
{
    const char* const safe_message =
        message ? message : "unknown training data loader error";
    try
    {
        training_data_loader_last_error = safe_message;
    }
    catch (...)
    {
        training_data_loader_last_error.clear();
    }
    std::fprintf(stderr, "training_data_loader: %s\n", safe_message);
}

static Stream<SparseBatch>* create_sparse_batch_stream_impl(
    const char* feature_set_c,
    int concurrency,
    const char* filename,
    int batch_size,
    bool cyclic,
    bool filtered,
    int random_fen_skipping,
    int rank,
    int world_size)
{
    if (!feature_set_c)
        throw std::runtime_error("feature_set must not be null");
    if (!filename)
        throw std::runtime_error("training data filename must not be null");

    const std::string_view feature_set(feature_set_c);
    const std::filesystem::path input_path(filename);
    const auto extension = input_path.extension().string();
    const bool jqv4_manifest = extension == ".json"
        || extension == ".jqv8.json";
    if (feature_set == "HalfKAv2_hm_jieqi_v8" && jqv4_manifest)
        return new Jqv4SparseBatchStream<HalfKAv2HmJieqiV8>(
            filename, batch_size, cyclic, rank, world_size);
    if (feature_set == "HalfKAv2_hm_jieqi_v8")
        return new JieqiSparseBatchStream<HalfKAv2HmJieqiV8>(
            filename, batch_size, cyclic, rank, world_size);
    if (feature_set == "HalfKAv2_hm_jieqi_v8^" && jqv4_manifest)
        return new Jqv4SparseBatchStream<HalfKAv2HmJieqiV8Factorized>(
            filename, batch_size, cyclic, rank, world_size);
    if (feature_set == "HalfKAv2_hm_jieqi_v8^")
        return new JieqiSparseBatchStream<HalfKAv2HmJieqiV8Factorized>(
            filename, batch_size, cyclic, rank, world_size);
    if (feature_set == "HalfKAv2_hm_jieqi_v3_fullthreats")
        return new JieqiSparseBatchStream<HalfKAv2HmJieqiV3FullThreats>(
            filename, batch_size, cyclic, rank, world_size);
    if (feature_set == "HalfKAv2_hm_jieqi_v3_fullthreats^")
        return new JieqiSparseBatchStream<
            HalfKAv2HmJieqiV3FullThreatsFactorized>(
                filename, batch_size, cyclic, rank, world_size);

    if (rank != 0 || world_size != 1)
        throw std::runtime_error(
            "rank sharding is supported only for Jieqi v3 FullThreats");

    auto skipPredicate = make_skip_predicate(filtered, random_fen_skipping);
    if (feature_set == "HalfKP")
        return new FeaturedBatchStream<FeatureSet<HalfKP>, SparseBatch>(
            concurrency, filename, batch_size, cyclic, skipPredicate);
    if (feature_set == "HalfKP^")
        return new FeaturedBatchStream<FeatureSet<HalfKPFactorized>, SparseBatch>(
            concurrency, filename, batch_size, cyclic, skipPredicate);
    if (feature_set == "HalfKA")
        return new FeaturedBatchStream<FeatureSet<HalfKA>, SparseBatch>(
            concurrency, filename, batch_size, cyclic, skipPredicate);
    if (feature_set == "HalfKA^")
        return new FeaturedBatchStream<FeatureSet<HalfKAFactorized>, SparseBatch>(
            concurrency, filename, batch_size, cyclic, skipPredicate);
    if (feature_set == "HalfKAv2")
        return new FeaturedBatchStream<FeatureSet<HalfKAv2>, SparseBatch>(
            concurrency, filename, batch_size, cyclic, skipPredicate);
    if (feature_set == "HalfKAv2^")
        return new FeaturedBatchStream<FeatureSet<HalfKAv2Factorized>, SparseBatch>(
            concurrency, filename, batch_size, cyclic, skipPredicate);

    throw std::runtime_error(
        std::string("unknown feature set: ") + feature_set_c);
}

static Stream<SparseBatch>* create_sparse_batch_stream_checked(
    const char* feature_set_c,
    int concurrency,
    const char* filename,
    int batch_size,
    bool cyclic,
    bool filtered,
    int random_fen_skipping,
    int rank,
    int world_size) noexcept
{
    training_data_loader_last_error.clear();
    try
    {
        return create_sparse_batch_stream_impl(
            feature_set_c, concurrency, filename, batch_size, cyclic,
            filtered, random_fen_skipping, rank, world_size);
    }
    catch (const std::exception& e)
    {
        set_training_data_loader_error(e.what());
        return nullptr;
    }
    catch (...)
    {
        set_training_data_loader_error(
            "unknown C++ exception while creating sparse batch stream");
        return nullptr;
    }
}

extern "C" {

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(
        const char* feature_set_c,
        int concurrency,
        const char* filename,
        int batch_size,
        bool cyclic,
        bool filtered,
        int random_fen_skipping)
    {
        return create_sparse_batch_stream_checked(
            feature_set_c, concurrency, filename, batch_size, cyclic,
            filtered, random_fen_skipping, 0, 1);
    }

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream_v2(
        const char* feature_set_c,
        int concurrency,
        const char* filename,
        int batch_size,
        bool cyclic,
        bool filtered,
        int random_fen_skipping,
        int rank,
        int world_size)
    {
        return create_sparse_batch_stream_checked(
            feature_set_c, concurrency, filename, batch_size, cyclic,
            filtered, random_fen_skipping, rank, world_size);
    }

    EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
    {
        delete stream;
    }

    EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
    {
        training_data_loader_last_error.clear();
        try
        {
            if (!stream)
                throw std::runtime_error(
                    "fetch_next_sparse_batch received a null stream");
            return stream->next();
        }
        catch (const std::exception& e)
        {
            set_training_data_loader_error(e.what());
            return nullptr;
        }
        catch (...)
        {
            set_training_data_loader_error(
                "unknown C++ exception while fetching sparse batch");
            return nullptr;
        }
    }

    EXPORT void CDECL destroy_sparse_batch(SparseBatch* e)
    {
        delete e;
    }

    EXPORT const char* CDECL get_training_data_loader_last_error()
    {
        return training_data_loader_last_error.c_str();
    }

}

/* benches */ //*
#include <chrono>

int main()
{
    auto stream = create_sparse_batch_stream("HalfKP", 4, "10m_d3_q_2.bin", 8192, true, false, 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        if (i % 100 == 0) std::cout << i << '\n';
        destroy_sparse_batch(stream->next());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}
//*/
