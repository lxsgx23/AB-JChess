#ifndef ABJNNUE_NETWORK_H_INCLUDED
#define ABJNNUE_NETWORK_H_INCLUDED

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "abjnnue_accumulator.h"
#include "abjnnue_inference.h"
#include "abjnnue_model.h"

namespace Stockfish::Eval::NNUE {

using AccumulatorStack = ::ABJNNUE::AccumulatorStack;

class NetworkBig {
   public:
    NetworkBig() = default;

    void load(const std::string& rootDirectory, const std::string& evalFile);
    void verify(const std::string& requested,
                const std::function<void(std::string_view)>& reporter) const;

    const ::ABJNNUE::Model& model() const;
    const std::string&      current_file() const noexcept { return currentFile_; }

   private:
    std::shared_ptr<const ::ABJNNUE::Model> model_;
    std::string                             currentFile_;
};

struct Networks {
    Networks() = default;
    explicit Networks(NetworkBig network) : big(std::move(network)) {}

    NetworkBig big;
};

// Refresh state is local to each search worker and is cleared at search start
// and whenever the replicated network is reloaded.
struct AccumulatorCaches {
    AccumulatorCaches() = default;
    explicit AccumulatorCaches(const Networks&) {}

    void clear(const Networks&) noexcept { refresh.clear(); }

    ::ABJNNUE::RefreshCache refresh;
};

}  // namespace Stockfish::Eval::NNUE

#endif
