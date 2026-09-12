#include "abjnnue_network.h"

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace Stockfish::Eval::NNUE {

namespace {

std::vector<std::filesystem::path> candidate_paths(const std::string& rootDirectory,
                                                   const std::string& evalFile) {
    const std::filesystem::path requested(evalFile);
    if (requested.is_absolute())
        return {requested};

    std::vector<std::filesystem::path> paths;
    if (!rootDirectory.empty())
        paths.emplace_back(std::filesystem::path(rootDirectory) / requested);
    paths.emplace_back(requested);
    return paths;
}

}  // namespace

void NetworkBig::load(const std::string& rootDirectory, const std::string& evalFile) {
    if (evalFile.empty())
        throw std::invalid_argument("EvalFile must be supplied through UCI before loading a V8.2 .nnue package");

    std::ostringstream missing;
    bool first = true;
    for (const auto& candidate : candidate_paths(rootDirectory, evalFile))
    {
        if (!std::filesystem::exists(candidate))
        {
            if (!first)
                missing << ", ";
            missing << candidate.string();
            first = false;
            continue;
        }

        auto loaded = ::ABJNNUE::Model::load(candidate);
        model_ = std::move(loaded);
        currentFile_ = evalFile;
        return;
    }

    throw std::runtime_error("ABJCHESSV82 file not found; tried: " + missing.str());
}

void NetworkBig::verify(const std::string& requested,
                        const std::function<void(std::string_view)>& reporter) const {
    if (!model_ || currentFile_ != requested)
        throw std::runtime_error("ABJCHESSV82 package was not loaded for EvalFile=" + requested);
    if (reporter)
        reporter(model_->summary());
}

const ::ABJNNUE::Model& NetworkBig::model() const {
    if (!model_)
        throw std::logic_error("ABJNNUE model is unavailable");
    return *model_;
}

}  // namespace Stockfish::Eval::NNUE
