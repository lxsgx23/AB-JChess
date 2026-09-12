/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "engine.h"

#include <algorithm>
#include <cassert>
#include <deque>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "evaluate.h"
#include "misc.h"
#include "abjnnue/abjnnue_network.h"
#include "numa.h"
#include "perft.h"
#include "position.h"
#include "search.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"

namespace Stockfish {

namespace NN = Eval::NNUE;

constexpr auto StartFEN =
  "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1";
constexpr int MaxHashMB  = Is64Bit ? 33554432 : 2048;
int           MaxThreads = std::max(1024, 4 * int(get_hardware_concurrency()));

namespace {

constexpr std::string_view PieceToChar(" RACPNBK racpnbkXx");

struct ParsedHistoryMove {
    Move                 move = Move::none();
    std::optional<Piece> movedIdentity;
    std::optional<Piece> capturedIdentity;
};

Piece parse_history_identity(const Position& position,
                             char            token,
                             Color           expectedColor,
                             const char*     role) {
    const std::size_t idx = PieceToChar.find(token);
    if (idx == std::string_view::npos || idx == NO_PIECE || idx >= PIECE_NB)
        throw std::invalid_argument(std::string("invalid ") + role + " identity suffix");

    const Piece pc = Piece(idx);
    if (type_of(pc) < ROOK || type_of(pc) > BISHOP)
        throw std::invalid_argument(std::string("invalid ") + role + " identity suffix");
    if (color_of(pc) != expectedColor)
        throw std::invalid_argument(std::string("wrong color for ") + role + " identity suffix");
    if (position.rest_piece(pc) <= 0)
        throw std::invalid_argument(std::string(role) + " identity is absent from the pool");
    return pc;
}

ParsedHistoryMove parse_history_move(const Position& position, const std::string& token) {
    if (token.size() < 4 || token.size() > 6)
        throw std::invalid_argument(
          "history move must contain four coordinates and valid identities");

    ParsedHistoryMove parsed;
    parsed.move = UCIEngine::to_move(position, token.substr(0, 4));
    if (parsed.move == Move::none())
        throw std::invalid_argument("illegal or malformed move in position history");

    const bool movedDark = position.move_dark(parsed.move);
    const bool capturedDark =
      position.capture(parsed.move) && position.is_dark(parsed.move.to_sq());
    const Color mover = position.side_to_move();

    const std::size_t minimumSize = 4 + std::size_t(movedDark);
    const std::size_t maximumSize = minimumSize + std::size_t(capturedDark);
    if (token.size() < minimumSize || token.size() > maximumSize)
        throw std::invalid_argument("identity suffix does not match the hidden move/capture");

    std::size_t suffix = 4;
    if (movedDark)
        parsed.movedIdentity =
          parse_history_identity(position, token[suffix++], mover, "moved-dark");
    if (capturedDark && suffix < token.size())
        parsed.capturedIdentity =
          parse_history_identity(position, token[suffix++], ~mover, "captured-dark");

    return parsed;
}

void apply_history_move(Position&                position,
                        std::deque<StateInfo>&   states,
                        const ParsedHistoryMove& parsed) {
    states.emplace_back();
    position.do_move(parsed.move, states.back());
    if (parsed.movedIdentity)
        position.do_flip(parsed.move.to_sq(), *parsed.movedIdentity);
    if (parsed.capturedIdentity)
        position.remove_rest_piece(*parsed.capturedIdentity);
}

}  // namespace

Engine::Engine(std::optional<std::string> path) :
    binaryDirectory(path ? CommandLine::get_binary_directory(*path) : ""),
    numaContext(NumaConfig::from_system()),
    states(new std::deque<StateInfo>(1)),
    threads(),
    networks(numaContext, NN::Networks(NN::NetworkBig())) {
    pos.set(StartFEN, &states->back());


    options.add(  //
      "Debug Log File", Option("", [](const Option& o) {
          start_logger(o);
          return std::nullopt;
      }));

    options.add(  //
      "NumaPolicy", Option("auto", [this](const Option& o) {
          set_numa_config_from_option(o);
          return numa_config_information_as_string() + "\n"
               + thread_allocation_information_as_string();
      }));

    options.add(  //
      "Threads", Option(1, 1, MaxThreads, [this](const Option&) {
          resize_threads();
          return thread_allocation_information_as_string();
      }));

    options.add(  //
      "Hash", Option(16, 1, MaxHashMB, [this](const Option& o) {
          set_tt_size(o);
          return std::nullopt;
      }));

    options.add(  //
      "Clear Hash", Option([this](const Option&) {
          search_clear();
          return std::nullopt;
      }));

    options.add(  //
      "Ponder", Option(false));

    options.add(  //
      "MultiPV", Option(1, 1, MAX_MOVES));

    options.add(  //
      "ChasingRule", Option("github skyrule_Jieqi", "skyrule_Jieqi", [this](const Option&) {
          search_clear();
          return std::nullopt;
      }));

    options.add("Move Overhead", Option(10, 0, 5000));

#if !defined(ABJCHESS_RELEASE)
    options.add("AggressiveLevel", Option(3, 0, 10));

    const Option::OnChange clearSearch = [this](const Option&) {
        search_clear();
        return std::nullopt;
    };

    options.add("RevealBonusBase", Option(0, -256, 256, clearSearch));
    options.add("RevealBonusPhase", Option(0, -256, 256, clearSearch));
    options.add("RevealBonusPool", Option(0, -256, 256, clearSearch));
    options.add("RevealBonusUnknown", Option(0, -256, 256, clearSearch));
    options.add("RevealBonusComeback", Option(0, -256, 256, clearSearch));
    options.add("RevealMoveOrder", Option(0, -16384, 16384, clearSearch));
    options.add("RevealReduction", Option(0, -2180, 2180, clearSearch));
    options.add("RevealPruningMargin", Option(0, -512, 512, clearSearch));
    options.add("RevealQuietBase", Option(-12, -96, 96, clearSearch));
    options.add("RevealQuietPhase", Option(0, -96, 96, clearSearch));
    options.add("RevealQuietSafety", Option(24, -192, 192, clearSearch));
    options.add("RevealQuietHighValue", Option(-12, -96, 96, clearSearch));
    options.add("RevealQuietDiversity", Option(0, -96, 96, clearSearch));
    options.add("RevealQuietComeback", Option(0, -96, 96, clearSearch));
    options.add("RevealQuietMoveOrder", Option(0, -2048, 2048, clearSearch));
    options.add("RevealQuietReduction", Option(0, -545, 545, clearSearch));
#endif

    options.add("nodestime", Option(0, 0, 10000));

    options.add("UCI_ShowWDL", Option(false));

    options.add(  //
      // NNUE is deliberately not selected at process startup.  The caller
      // must provide the package through the UCI option so the executable has
      // no embedded filename or sibling-path requirement.
      "EvalFile", Option("", [this](const Option& o) {
          load_big_network(o);
          return std::nullopt;
      }));

    resize_threads();
}

std::uint64_t Engine::perft(const std::string& fen, Depth depth) {
    verify_networks();

    return Benchmark::perft(fen, depth);
}

void Engine::go(Search::LimitsType& limits) {
    assert(limits.perft == 0);
    verify_networks();

    threads.start_thinking(pos, states, limits);
}
void Engine::stop() { threads.stop = true; }

void Engine::search_clear() {
    wait_for_search_finished();

    tt.clear(threads);
    threads.clear();
}

void Engine::set_on_update_no_moves(std::function<void(const Engine::InfoShort&)>&& f) {
    updateContext.onUpdateNoMoves = std::move(f);
}

void Engine::set_on_update_full(std::function<void(const Engine::InfoFull&)>&& f) {
    updateContext.onUpdateFull = std::move(f);
}

void Engine::set_on_iter(std::function<void(const Engine::InfoIter&)>&& f) {
    updateContext.onIter = std::move(f);
}

void Engine::set_on_bestmove(std::function<void(std::string_view, std::string_view)>&& f) {
    updateContext.onBestmove = std::move(f);
}

void Engine::set_on_verify_networks(std::function<void(std::string_view)>&& f) {
    onVerifyNetworks = std::move(f);
}

void Engine::wait_for_search_finished() { threads.main_thread()->wait_for_search_finished(); }

void Engine::set_position(const std::string& fen, const std::vector<std::string>& moves) {
    stop();
    wait_for_search_finished();

    auto     stagedStates = StateListPtr(new std::deque<StateInfo>(1));
    Position stagedPosition;
    stagedPosition.set(fen, &stagedStates->back());

    for (const auto& token : moves)
    {
        const ParsedHistoryMove parsed = parse_history_move(stagedPosition, token);
        apply_history_move(stagedPosition, *stagedStates, parsed);
    }

    pos.swap(stagedPosition);
    states.swap(stagedStates);
}

// modifiers

void Engine::set_numa_config_from_option(const std::string& o) {
    if (o == "auto" || o == "system")
    {
        numaContext.set_numa_config(NumaConfig::from_system());
    }
    else if (o == "hardware")
    {
        // Don't respect affinity set in the system.
        numaContext.set_numa_config(NumaConfig::from_system(false));
    }
    else if (o == "none")
    {
        numaContext.set_numa_config(NumaConfig{});
    }
    else
    {
        numaContext.set_numa_config(NumaConfig::from_string(o));
    }

    // Force reallocation of threads in case affinities need to change.
    resize_threads();
    threads.ensure_network_replicated();
}

void Engine::resize_threads() {
    threads.wait_for_search_finished();
    threads.set(numaContext.get_numa_config(), {options, threads, tt, networks}, updateContext);

    // Reallocate the hash with the new threadpool size
    set_tt_size(options["Hash"]);
    threads.ensure_network_replicated();
}

void Engine::set_tt_size(size_t mb) {
    wait_for_search_finished();
    tt.resize(mb, threads);
}

void Engine::set_ponderhit(bool b) { threads.main_manager()->ponder = b; }

// network related

void Engine::verify_networks() const {
    if (std::string(options["EvalFile"]).empty())
        throw std::runtime_error("ABJNNUE EvalFile is not set; send 'setoption name EvalFile value <path>' before searching");
    networks->big.verify(options["EvalFile"], onVerifyNetworks);
}

void Engine::load_networks() {
    if (std::string(options["EvalFile"]).empty())
        return;
    networks.modify_and_replicate([this](NN::Networks& networks_) {
        networks_.big.load(binaryDirectory, options["EvalFile"]);
    });
    threads.clear();
    threads.ensure_network_replicated();
}

void Engine::load_big_network(const std::string& file) {
    NN::NetworkBig loaded;
    try
    {
        loaded.load(binaryDirectory, file);
        const auto loadedFile = loaded.current_file();

        networks.modify_and_replicate(
          [&loaded](NN::Networks& networks_) { networks_.big = std::move(loaded); });
        threads.clear();
        threads.ensure_network_replicated();

        // The public API is also used directly by tests and embedding code,
        // bypassing Option::operator=. Keep the option and live network bound
        // to the same requested filename on every successful load path.
        options.options_map.at("EvalFile").currentValue = loadedFile;
        return;
    }
    catch (const std::exception& error)
    {
        // OptionsMap commits the requested value before invoking this callback.
        // Restore the live network's spelling before notifying an embedding
        // callback, since that callback is user code and may throw.
        options.options_map.at("EvalFile").currentValue = networks->big.current_file();

        if (onVerifyNetworks)
            onVerifyNetworks(std::string("ABJNNUE load failed: ") + error.what());
        return;
    }
}

// utility functions

void Engine::trace_eval() const {
    StateListPtr trace_states(new std::deque<StateInfo>(1));
    Position     p;
    p.set(pos.fen(), &trace_states->back());

    verify_networks();

    sync_cout << "\n" << Eval::trace(p, *networks) << sync_endl;
}

const OptionsMap& Engine::get_options() const { return options; }
OptionsMap&       Engine::get_options() { return options; }

std::string Engine::fen() const { return pos.fen(); }

void Engine::flip() { pos.flip(); }

std::string Engine::visualize() const {
    std::stringstream ss;
    ss << pos;
    return ss.str();
}

int Engine::get_hashfull(int maxAge) const { return tt.hashfull(maxAge); }

std::vector<std::pair<size_t, size_t>> Engine::get_bound_thread_count_by_numa_node() const {
    auto                                   counts = threads.get_bound_thread_count_by_numa_node();
    const NumaConfig&                      cfg    = numaContext.get_numa_config();
    std::vector<std::pair<size_t, size_t>> ratios;
    NumaIndex                              n = 0;
    for (; n < counts.size(); ++n)
        ratios.emplace_back(counts[n], cfg.num_cpus_in_numa_node(n));
    if (!counts.empty())
        for (; n < cfg.num_numa_nodes(); ++n)
            ratios.emplace_back(0, cfg.num_cpus_in_numa_node(n));
    return ratios;
}

std::string Engine::get_numa_config_as_string() const {
    return numaContext.get_numa_config().to_string();
}

std::string Engine::numa_config_information_as_string() const {
    auto cfgStr = get_numa_config_as_string();
    return "Available processors: " + cfgStr;
}

std::string Engine::thread_binding_information_as_string() const {
    auto              boundThreadsByNode = get_bound_thread_count_by_numa_node();
    std::stringstream ss;
    if (boundThreadsByNode.empty())
        return ss.str();

    bool isFirst = true;

    for (auto&& [current, total] : boundThreadsByNode)
    {
        if (!isFirst)
            ss << ":";
        ss << current << "/" << total;
        isFirst = false;
    }

    return ss.str();
}

std::string Engine::thread_allocation_information_as_string() const {
    std::stringstream ss;

    size_t threadsSize = threads.size();
    ss << "Using " << threadsSize << (threadsSize > 1 ? " threads" : " thread");

    auto boundThreadsByNodeStr = thread_binding_information_as_string();
    if (boundThreadsByNodeStr.empty())
        return ss.str();

    ss << " with NUMA node thread binding: ";
    ss << boundThreadsByNodeStr;

    return ss.str();
}
}
