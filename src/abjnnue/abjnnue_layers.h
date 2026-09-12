#ifndef ABJNNUE_LAYERS_H_INCLUDED
#define ABJNNUE_LAYERS_H_INCLUDED

#include <cstdint>

namespace ABJNNUE::Layers {

// Evaluates one V8 2048 -> pairwise 1024 -> 15 -> 32 -> 1 runtime head.
std::int32_t propagate(const std::uint8_t* head, const std::uint8_t* input);

// Bit-exact reference used to validate every architecture-specific path.
std::int32_t propagate_scalar(const std::uint8_t* head, const std::uint8_t* input);

const char* backend_name() noexcept;

}  // namespace ABJNNUE::Layers

#endif
