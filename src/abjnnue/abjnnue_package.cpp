#include "abjnnue_package.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace ABJNNUE {
namespace {

constexpr std::array<std::uint8_t, 16> Magic = {
  'A', 'B', 'J', 'C', 'H', 'E', 'S', 'S', 'V', '8', '2', 0, 0, 0, 0, 0};

struct RequiredChunk {
    const char*   name;
    std::uint64_t size;
};

constexpr std::uint64_t AccumulatorWidth = 2048;
constexpr std::uint64_t FeatureDimensions = 31776;
constexpr std::uint64_t PSQTBuckets = 16;
constexpr std::uint64_t LayerStacks = 16;
constexpr std::uint64_t TransformerBiasesSize = AccumulatorWidth * sizeof(std::int16_t);
constexpr std::uint64_t FeatureWeightsSize =
  FeatureDimensions * AccumulatorWidth * sizeof(std::int16_t);
constexpr std::uint64_t PSQTWeightsSize =
  FeatureDimensions * PSQTBuckets * sizeof(std::int32_t);
constexpr std::uint64_t PrimarySize =
  TransformerBiasesSize + FeatureWeightsSize + PSQTWeightsSize;
constexpr std::uint64_t EvalHeadBucketSize = 34208;
constexpr std::uint64_t EvalHeadsSize = LayerStacks * EvalHeadBucketSize;
constexpr std::uint64_t ProbabilityScoreToMassSize = 4001 * sizeof(std::int32_t);
constexpr std::uint64_t ProbabilityMassToScoreSize = 1901 * sizeof(std::int32_t);

constexpr std::array<RequiredChunk, 4> RequiredChunks = {{
  {"primary_runtime_nnue_container.bin", PrimarySize},
  {"eval_heads_runtime.bin", EvalHeadsSize},
  {"probability_score_to_mass.i32le", ProbabilityScoreToMassSize},
  {"probability_mass_to_score.i32le", ProbabilityMassToScoreSize},
}};

constexpr std::uint64_t RequiredPayloadSize =
  PrimarySize + EvalHeadsSize + ProbabilityScoreToMassSize + ProbabilityMassToScoreSize;

static_assert(PrimarySize == 132192256);
static_assert(EvalHeadsSize == 547328);
static_assert(RequiredPayloadSize == 132763192);

constexpr std::array<std::uint32_t, 64> K = {
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
  0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa,
  0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr std::uint32_t rotr(std::uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

class Sha256 {
   public:
    void update(const std::uint8_t* data, std::size_t size) {
        totalBytes_ += size;
        while (size != 0)
        {
            const std::size_t take = std::min(size, block_.size() - used_);
            std::memcpy(block_.data() + used_, data, take);
            used_ += take;
            data += take;
            size -= take;
            if (used_ == block_.size())
            {
                compress(block_.data());
                used_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finish() {
        const std::uint64_t bitLength = totalBytes_ * 8;
        block_[used_++]              = 0x80;
        if (used_ > 56)
        {
            std::fill(block_.begin() + used_, block_.end(), 0);
            compress(block_.data());
            used_ = 0;
        }
        std::fill(block_.begin() + used_, block_.begin() + 56, 0);
        for (int i = 0; i < 8; ++i)
            block_[63 - i] = static_cast<std::uint8_t>(bitLength >> (i * 8));
        compress(block_.data());

        std::array<std::uint8_t, 32> out{};
        for (std::size_t i = 0; i < state_.size(); ++i)
            for (int j = 0; j < 4; ++j)
                out[i * 4 + j] = static_cast<std::uint8_t>(state_[i] >> (24 - 8 * j));
        return out;
    }

   private:
    void compress(const std::uint8_t* p) {
        std::uint32_t w[64]{};
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(p[i * 4]) << 24) | (std::uint32_t(p[i * 4 + 1]) << 16)
                 | (std::uint32_t(p[i * 4 + 2]) << 8) | std::uint32_t(p[i * 4 + 3]);
        for (int i = 16; i < 64; ++i)
        {
            const auto s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const auto s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i]          = w[i - 16] + s0 + w[i - 7] + s1;
        }
        auto a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        auto e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i)
        {
            const auto s1  = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const auto ch  = (e & f) ^ (~e & g);
            const auto t1  = h + s1 + ch + K[i] + w[i];
            const auto s0  = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const auto maj = (a & b) ^ (a & c) ^ (b & c);
            const auto t2  = s0 + maj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                           0xa54ff53a, 0x510e527f, 0x9b05688c,
                                           0x1f83d9ab, 0x5be0cd19};
    std::array<std::uint8_t, 64> block_{};
    std::size_t                  used_       = 0;
    std::uint64_t                totalBytes_ = 0;
};

std::uint32_t read_u32_le(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16)
         | (std::uint32_t(p[3]) << 24);
}

std::size_t skip_ws(std::string_view s, std::size_t p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r' || s[p] == '\n'))
        ++p;
    return p;
}

std::string json_string(std::string_view object, std::string_view key, bool required = true) {
    const std::string marker = "\"" + std::string(key) + "\"";
    auto              p      = object.find(marker);
    if (p == std::string_view::npos)
    {
        if (!required) return {};
        throw std::runtime_error("missing JSON string field: " + std::string(key));
    }
    p = object.find(':', p + marker.size());
    if (p == std::string_view::npos) throw std::runtime_error("invalid JSON field");
    p = skip_ws(object, p + 1);
    if (p >= object.size() || object[p] != '"') throw std::runtime_error("JSON field is not a string");
    ++p;
    std::string out;
    bool escape = false;
    for (; p < object.size(); ++p)
    {
        const char c = object[p];
        if (escape)
        {
            if (c == '"' || c == '\\' || c == '/') out.push_back(c);
            else if (c == 'n') out.push_back('\n');
            else if (c == 'r') out.push_back('\r');
            else if (c == 't') out.push_back('\t');
            else throw std::runtime_error("unsupported JSON escape");
            escape = false;
        }
        else if (c == '\\') escape = true;
        else if (c == '"') return out;
        else out.push_back(c);
    }
    throw std::runtime_error("unterminated JSON string");
}

std::uint64_t json_u64(std::string_view object, std::string_view key) {
    const std::string marker = "\"" + std::string(key) + "\"";
    auto              p      = object.find(marker);
    if (p == std::string_view::npos) throw std::runtime_error("missing JSON integer field: " + std::string(key));
    p = object.find(':', p + marker.size());
    if (p == std::string_view::npos) throw std::runtime_error("invalid JSON field");
    p = skip_ws(object, p + 1);
    std::uint64_t value = 0;
    const auto    start = p;
    while (p < object.size() && object[p] >= '0' && object[p] <= '9')
    {
        const unsigned digit = unsigned(object[p++] - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
            throw std::runtime_error("JSON integer overflow");
        value = value * 10 + digit;
    }
    if (p == start) throw std::runtime_error("JSON field is not an unsigned integer");
    return value;
}

std::uint64_t json_top_u64(std::string_view json, std::string_view key) {
    return json_u64(json, key);
}

std::string json_top_string(std::string_view json, std::string_view key) {
    return json_string(json, key);
}

std::vector<std::string_view> chunk_objects(std::string_view json) {
    auto p = json.find("\"chunks\"");
    if (p == std::string_view::npos) throw std::runtime_error("metadata has no chunks array");
    p = json.find('[', p);
    if (p == std::string_view::npos) throw std::runtime_error("invalid chunks array");
    std::vector<std::string_view> out;
    bool inString = false, escape = false;
    int  depth = 0;
    std::size_t objectStart = 0;
    for (++p; p < json.size(); ++p)
    {
        const char c = json[p];
        if (inString)
        {
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') inString = false;
            continue;
        }
        if (c == '"') inString = true;
        else if (c == '{')
        {
            if (depth++ == 0) objectStart = p;
        }
        else if (c == '}')
        {
            if (--depth < 0) throw std::runtime_error("invalid JSON object nesting");
            if (depth == 0) out.push_back(json.substr(objectStart, p - objectStart + 1));
        }
        else if (c == ']' && depth == 0) return out;
    }
    throw std::runtime_error("unterminated chunks array");
}

}  // namespace

std::string sha256_hex(ByteView data) {
    Sha256 hash;
    hash.update(data.data, data.size);
    const auto bytes = hash.finish();
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto byte : bytes) out << std::setw(2) << unsigned(byte);
    return out.str();
}

Package Package::load(const std::filesystem::path& path) {
    Package package;
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("open failed: " + path.string());
    const auto length = stream.tellg();
    if (length == std::ifstream::pos_type(-1))
        throw std::runtime_error("package length query failed");
    const auto fileSize = static_cast<std::size_t>(length);
    if (fileSize < HeaderSize)
        throw std::runtime_error("ABJCHESSV82 package is shorter than its header");
    package.bytes_.resize(fileSize);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(package.bytes_.data()), package.bytes_.size());
    if (!stream) throw std::runtime_error("short package read");

    if (!std::equal(Magic.begin(), Magic.end(), package.bytes_.begin()))
        throw std::runtime_error("bad ABJCHESSV82 magic");
    package.version_ = read_u32_le(package.bytes_.data() + 16);
    if (package.version_ != Version) throw std::runtime_error("unsupported ABJCHESSV82 package version");
    const std::uint32_t metadataLength = read_u32_le(package.bytes_.data() + 20);
    if (metadataLength == 0 || metadataLength > fileSize - HeaderSize)
        throw std::runtime_error("ABJCHESSV82 metadata length is invalid");
    package.payloadOffset_ = HeaderSize + metadataLength;
    package.payloadSize_ = package.bytes_.size() - package.payloadOffset_;
    package.metadataJson_.assign(reinterpret_cast<const char*>(package.bytes_.data() + HeaderSize),
                                 metadataLength);
    if (json_top_u64(package.metadataJson_, "format_version") != Version)
        throw std::runtime_error("metadata format_version mismatch");
    if (json_top_string(package.metadataJson_, "schema") != Schema)
        throw std::runtime_error("ABJCHESSV82 schema identity mismatch");
    if (json_top_string(package.metadataJson_, "feature_identity") != FeatureIdentity)
        throw std::runtime_error("ABJCHESSV82 feature identity mismatch");
    if (json_top_string(package.metadataJson_, "interpolation_format") != "Q0.8"
        || json_top_string(package.metadataJson_, "interpolation_formula") != "continuous-q0.8-v1")
        throw std::runtime_error("ABJCHESSV82 interpolation marker mismatch");
    if (json_top_u64(package.metadataJson_, "feature_dimensions") != 31776
        || json_top_u64(package.metadataJson_, "accumulator_width") != 2048
        || json_top_u64(package.metadataJson_, "psqt_buckets") != 16
        || json_top_u64(package.metadataJson_, "layer_stacks") != 16)
        throw std::runtime_error("ABJCHESSV82 tensor dimensions mismatch");

    for (const auto object : chunk_objects(package.metadataJson_))
    {
        Chunk chunk;
        chunk.name       = json_string(object, "name");
        chunk.role       = json_string(object, "role", false);
        chunk.dtype      = json_string(object, "dtype", false);
        chunk.dataOffset = json_u64(object, "data_offset");
        chunk.size       = json_u64(object, "size");
        chunk.sha256     = json_string(object, "sha256");
        if (chunk.size == 0 || chunk.dataOffset > package.payloadSize_
            || chunk.size > package.payloadSize_ - chunk.dataOffset)
            throw std::runtime_error("chunk outside payload: " + chunk.name);
        if (std::any_of(package.chunks_.begin(), package.chunks_.end(),
                        [&](const Chunk& c) { return c.name == chunk.name; }))
            throw std::runtime_error("duplicate chunk: " + chunk.name);
        package.chunks_.push_back(std::move(chunk));
    }
    if (package.chunks_.size() != RequiredChunks.size())
        throw std::runtime_error("ABJCHESSV82 package must contain exactly four chunks");
    std::uint64_t expectedOffset = 0;
    for (const auto& required : RequiredChunks)
    {
        if (!package.has_chunk(required.name))
            throw std::runtime_error("missing required chunk: " + std::string(required.name));
        const auto& chunk = package.chunk(required.name);
        if (chunk.size != required.size)
            throw std::runtime_error("ABJCHESSV82 chunk size mismatch: " + chunk.name);
        if (chunk.dataOffset != expectedOffset)
            throw std::runtime_error("ABJCHESSV82 chunk layout mismatch: " + chunk.name);
        expectedOffset += required.size;
    }
    if (expectedOffset != RequiredPayloadSize || package.payloadSize_ != RequiredPayloadSize)
        throw std::runtime_error("ABJCHESSV82 payload size mismatch");

    for (std::size_t i = 0; i < package.chunks_.size(); ++i)
        for (std::size_t j = i + 1; j < package.chunks_.size(); ++j)
        {
            const auto& a = package.chunks_[i];
            const auto& b = package.chunks_[j];
            if (a.dataOffset < b.dataOffset + b.size && b.dataOffset < a.dataOffset + a.size)
                throw std::runtime_error("overlapping ABJCHESSV82 chunks");
        }
    for (const auto& chunk : package.chunks_)
    {
        auto digest = sha256_hex(package.view(chunk));
        std::transform(digest.begin(), digest.end(), digest.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        auto expectedHash = chunk.sha256;
        std::transform(expectedHash.begin(), expectedHash.end(), expectedHash.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (digest != expectedHash) throw std::runtime_error("ABJCHESSV82 chunk SHA-256 mismatch: " + chunk.name);
    }
    return package;
}

const Chunk& Package::chunk(const std::string& name) const {
    const auto it = std::find_if(chunks_.begin(), chunks_.end(),
                                 [&](const Chunk& chunk) { return chunk.name == name; });
    if (it == chunks_.end()) throw std::out_of_range("chunk not found: " + name);
    return *it;
}

bool Package::has_chunk(const std::string& name) const noexcept {
    return std::any_of(chunks_.begin(), chunks_.end(),
                       [&](const Chunk& chunk) { return chunk.name == name; });
}

ByteView Package::view(const Chunk& chunk) const {
    if (chunk.dataOffset > payloadSize_
        || chunk.size > payloadSize_ - chunk.dataOffset)
        throw std::out_of_range("chunk view outside package");
    return {bytes_.data() + payloadOffset_ + chunk.dataOffset,
            static_cast<std::size_t>(chunk.size)};
}

ByteView Package::view(const std::string& name) const { return view(chunk(name)); }

}  // namespace ABJNNUE
