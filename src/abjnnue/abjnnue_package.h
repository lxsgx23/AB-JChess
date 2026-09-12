#ifndef ABJNNUE_PACKAGE_H_INCLUDED
#define ABJNNUE_PACKAGE_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ABJNNUE {

struct ByteView {
    const std::uint8_t* data = nullptr;
    std::size_t         size = 0;
};

struct Chunk {
    std::string   name;
    std::string   role;
    std::string   dtype;
    std::uint64_t dataOffset = 0;
    std::uint64_t size       = 0;
    std::string   sha256;
};

// V8.2 is deliberately a closed format.  The magic, version and schema string
// are part of the weight contract; older package identities must never be
// accepted here.
class Package {
   public:
    static constexpr std::uint32_t Version = 82;
    static constexpr std::size_t HeaderSize = 24;
    static constexpr const char* Schema = "abjchess-v8.2-full4way-noaux-v1";
    static constexpr const char* FeatureIdentity =
      "HalfKAv2_hm_jieqi_v8.1_full4way_meta_midmirror_threat_interp";

    static Package load(const std::filesystem::path& path);

    std::uint32_t            version() const noexcept { return version_; }
    std::size_t              payload_size() const noexcept { return payloadSize_; }
    const std::string&       metadata_json() const noexcept { return metadataJson_; }
    const std::vector<Chunk>& chunks() const noexcept { return chunks_; }
    bool                     has_chunk(const std::string& name) const noexcept;
    const Chunk&             chunk(const std::string& name) const;
    ByteView                 view(const Chunk& chunk) const;
    ByteView                 view(const std::string& name) const;

   private:
    std::uint32_t             version_       = 0;
    std::size_t               payloadOffset_ = 0;
    std::size_t               payloadSize_   = 0;
    std::string               metadataJson_;
    std::vector<std::uint8_t> bytes_;
    std::vector<Chunk>         chunks_;
};

std::string sha256_hex(ByteView data);

}  // namespace ABJNNUE

#endif
