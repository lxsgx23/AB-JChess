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

#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "history.h"
#include "misc.h"
#include "abjnnue/abjnnue_network.h"
#include "numa.h"
#include "position.h"
#include "reveal.h"
#include "score.h"
#include "timeman.h"
#include "types.h"

namespace Stockfish {

// Different node types, used as a template parameter
enum NodeType {
    NonPV,
    PV,
    Root
};

class TranspositionTable;
class ThreadPool;
class OptionsMap;

namespace Search {

// Read all reveal UCI options once at search start so node hot paths only use POD data.
Reveal::Parameters snapshot_reveal_parameters(const OptionsMap& options);

// Stack struct keeps track of the information we need to remember from nodes
// shallower and deeper in the tree during the search. Each search thread has
// its own array of Stack objects, indexed by the current ply.
struct Stack {
    Move*                       pv;
    PieceToHistory*             continuationHistory;
    CorrectionHistory<PieceTo>* continuationCorrectionHistory;
    int                         ply;
    Move                        currentMove;
    bool                        capturedDark;
    Move                        excludedMove;
    Value                       staticEval;
    int                         statScore;
    int                         moveCount;
    bool                        inCheck;
    bool                        ttPv;
    bool                        ttHit;
    int                         cutoffCnt;
    int                         reduction;
    bool                        isPvNode;
    int                         quietMoveStreak;
};


// RootMove struct is used for moves at the root of the tree. For each root move
// we store a score and a PV (really a refutation in the case of moves which
// fail low). Score is normally set at -VALUE_INFINITE for all non-pv moves.
struct RootMove {

    explicit RootMove(Move m) :
        pv(1, m) { }
    bool extract_ponder_from_tt(const TranspositionTable& tt, Position& pos, Color rootObserver);
    bool operator==(const Move& m) const { return pv[0] == m; }
    // Sort in descending order
    bool operator<(const RootMove& m) const {
        return m.score != score ? m.score < score : m.previousScore < previousScore;
    }

    uint64_t          effort           = 0;
    Value             score            = -VALUE_INFINITE;
    Value             previousScore    = -VALUE_INFINITE;
    Value             averageScore     = -VALUE_INFINITE;
    Value             meanSquaredScore = -VALUE_INFINITE * VALUE_INFINITE;
    Value             uciScore         = -VALUE_INFINITE;
    bool              scoreLowerbound  = false;
    bool              scoreUpperbound  = false;
    int               selDepth         = 0;
    std::vector<Move> pv;
};

using RootMoves = std::vector<RootMove>;

Key observer_tt_key(Key positionKey, Color rootObserver);

int hidden_capture_candidates(const Position&,
                              Color rootObserver,
                              bool  capturedDark,
                              Position::RestPieceList&);

struct ProbCutContext {
    Value beta;
    Depth depth;
};

Depth          chance_branch_depth(Depth depth, int branchIndex);
ProbCutContext probcut_context(Value probCutBeta, Depth nodeDepth, bool capturedDark);

class ChanceBranchReduction {
   public:
    explicit ChanceBranchReduction(int& reduction) :
        reduction(reduction), parentReduction(reduction) { }

    ~ChanceBranchReduction() { reduction = 0; }

    void restore() const { reduction = parentReduction; }

    ChanceBranchReduction(const ChanceBranchReduction&)            = delete;
    ChanceBranchReduction(ChanceBranchReduction&&)                 = delete;
    ChanceBranchReduction& operator=(const ChanceBranchReduction&) = delete;
    ChanceBranchReduction& operator=(ChanceBranchReduction&&)      = delete;

   private:
    int&      reduction;
    const int parentReduction;
};


// LimitsType struct stores information sent by the caller about the analysis required.
struct LimitsType {

    // Init explicitly due to broken value-initialization of non POD in MSVC
    LimitsType() {
        time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] = npmsec = movetime = TimePoint(0);
        movestogo = depth = mate = perft = infinite = 0;
        nodes                                       = 0;
        ponderMode                                  = false;
    }

    bool use_time_management() const { return time[WHITE] || time[BLACK]; }

    std::vector<std::string> searchmoves;
    TimePoint                time[COLOR_NB], inc[COLOR_NB], npmsec, movetime, startTime;
    int                      movestogo, depth, mate, perft, infinite;
    uint64_t                 nodes;
    bool                     ponderMode;
};


// The UCI stores the uci options, thread pool, and transposition table.
// This struct is used to easily forward data to the Search::Worker class.
struct SharedState {
    SharedState(const OptionsMap&                               optionsMap,
                ThreadPool&                                     threadPool,
                TranspositionTable&                             transpositionTable,
                const LazyNumaReplicated<Eval::NNUE::Networks>& nets) :
        options(optionsMap),
        threads(threadPool),
        tt(transpositionTable),
        networks(nets) { }

    const OptionsMap&                               options;
    ThreadPool&                                     threads;
    TranspositionTable&                             tt;
    const LazyNumaReplicated<Eval::NNUE::Networks>& networks;
};

class Worker;

// Null Object Pattern, implement a common interface for the SearchManagers.
// A Null Object will be given to non-mainthread workers.
class ISearchManager {
   public:
    virtual ~ISearchManager() { }
    virtual void check_time(Search::Worker&) = 0;
};

struct InfoShort {
    int   depth;
    Score score;
};

struct InfoFull: InfoShort {
    int              selDepth;
    size_t           multiPV;
    std::string_view wdl;
    std::string_view bound;
    size_t           timeMs;
    size_t           nodes;
    size_t           nps;
    size_t           tbHits;
    std::string_view pv;
    int              hashfull;
};

struct InfoIteration {
    int              depth;
    std::string_view currmove;
    size_t           currmovenumber;
};

// SearchManager manages the search from the main thread. It is responsible for
// keeping track of the time, and storing data strictly related to the main thread.
class SearchManager: public ISearchManager {
   public:
    using UpdateShort    = std::function<void(const InfoShort&)>;
    using UpdateFull     = std::function<void(const InfoFull&)>;
    using UpdateIter     = std::function<void(const InfoIteration&)>;
    using UpdateBestmove = std::function<void(std::string_view, std::string_view)>;

    struct UpdateContext {
        UpdateShort    onUpdateNoMoves;
        UpdateFull     onUpdateFull;
        UpdateIter     onIter;
        UpdateBestmove onBestmove;
    };


    SearchManager(const UpdateContext& updateContext) :
        updates(updateContext) { }

    void check_time(Search::Worker& worker) override;

    void pv(const Search::Worker&     worker,
            const ThreadPool&         threads,
            const TranspositionTable& tt,
            Depth                     depth) const;

    Stockfish::TimeManagement tm;
    double                    originalTimeAdjust;
    int                       callsCnt;
    std::atomic_bool          ponder;

    std::array<Value, 4> iterValue;
    double               previousTimeReduction;
    Value                bestPreviousScore;
    Value                bestPreviousAverageScore;
    bool                 stopOnPonderhit;

    size_t id;

    const UpdateContext& updates;
};

class NullSearchManager: public ISearchManager {
   public:
    void check_time(Search::Worker&) override { }
};


// Search::Worker is the class that does the actual search.
// It is instantiated once per thread, and it is responsible for keeping track
// of the search history, and storing data required for the search.
class Worker {
   public:
    Worker(SharedState&, std::unique_ptr<ISearchManager>, size_t, NumaReplicatedAccessToken);

    // Called at instantiation to initialize reductions tables.
    // Reset histories, usually before a new game.
    void clear();

    // Called when the program receives the UCI 'go' command.
    // It searches from the root position and outputs the "bestmove".
    void start_searching();

    bool is_mainthread() const { return threadIdx == 0; }

    void ensure_network_replicated();

    // Public because they need to be updatable by the stats
    ButterflyHistory mainHistory;
    LowPlyHistory    lowPlyHistory;

    CapturePieceToHistory captureHistory;
    ContinuationHistory   continuationHistory[2][2];
    PawnHistory           pawnHistory;

    CorrectionHistory<Pawn>         pawnCorrectionHistory;
    CorrectionHistory<Minor>        minorPieceCorrectionHistory;
    CorrectionHistory<NonPawn>      nonPawnCorrectionHistory;
    CorrectionHistory<Continuation> continuationCorrectionHistory;

    TTMoveHistory ttMoveHistory;

   private:
    void iterative_deepening();
    void refresh_chasing_rule();
    void reset_correction_histories();

    void do_move(Position& pos, const Move move, StateInfo& st);
    void do_move(Position& pos, const Move move, StateInfo& st, const bool givesCheck);
    void do_null_move(Position& pos, StateInfo& st);
    void undo_move(Position& pos, const Move move);
    void undo_null_move(Position& pos);

    // This is the main search function, for both PV and non-PV nodes
    template<NodeType nodeType, bool UseReveal>
    Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode);

    // Quiescence search function, which is called by the main search
    template<NodeType nodeType, bool UseReveal>
    Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta);

    // Resolve the captured identity when the root observer captured a dark piece.
    template<NodeType nodeType, bool UseReveal>
    Value hidden_capture_search(Position&              pos,
                                Stack*                 ss,
                                Value                  alpha,
                                Value                  beta,
                                bool                   isQsearch = true,
                                [[maybe_unused]] Depth depth     = -1,
                                [[maybe_unused]] bool  cutNode   = false);

    // Search a parent move through the complete hidden-capture identity aggregate,
    // then apply its reveal reward exactly once.
    template<NodeType nodeType, bool UseReveal>
    Value reveal_search(Position&              pos,
                        Stack*                 ss,
                        Value                  alpha,
                        Value                  beta,
                        int                    revealBonus,
                        bool                   isQsearch = true,
                        [[maybe_unused]] Depth depth     = -1,
                        [[maybe_unused]] bool  cutNode   = false);

    // This is the flip search function for any nodes
    template<NodeType nodeType, bool UseReveal>
    Value flip_search(Position&              pos,
                      Stack*                 ss,
                      Value                  alpha,
                      Value                  beta,
                      bool                   isQsearch = true,
                      [[maybe_unused]] Depth depth     = -1,
                      [[maybe_unused]] bool  cutNode   = false);

    Depth reduction(bool i, Depth d, int mn, int delta) const;

    // Pointer to the search manager, only allowed to be called by the main thread
    SearchManager* main_manager() const {
        assert(threadIdx == 0);
        return static_cast<SearchManager*>(manager.get());
    }

    TimePoint elapsed() const;
    TimePoint elapsed_time() const;

    Value evaluate(const Position&);
    Value evaluate_for_reveal(const Position&);
    int   reveal_bonus(const Position&,
                       const Reveal::QuietMoveContext& quietContext,
                       Value&                          rawParentEval);

    LimitsType limits;

    size_t                pvIdx, pvLast;
    std::atomic<uint64_t> nodes, bestMoveChanges;
    int                   selDepth, nmpMinPly;

    Value optimism[COLOR_NB];

    Position    rootPos;
    Color       rootObserver = WHITE;
    StateInfo   rootState;
    RootMoves   rootMoves;
    Depth       rootDepth, completedDepth;
    Value       rootDelta;
    ChasingRule searchRule = CHASING_RULE_GITHUB;
    bool        useSkyrule = false;

    Reveal::Parameters revealParameters;
    int                revealPruningSlots = 0;
    bool               quietRewardEnabled = false;
    bool               quietSafetyNeeded  = false;
    bool               revealSearchTuning = false;

    size_t                    threadIdx;
    NumaReplicatedAccessToken numaAccessToken;

    // Reductions lookup table initialized at startup
    std::array<int, MAX_PLY + 10> reductions;  // [depth or moveNumber]

    // The main thread has a SearchManager, the others have a NullSearchManager
    std::unique_ptr<ISearchManager> manager;

    const OptionsMap&                               options;
    ThreadPool&                                     threads;
    TranspositionTable&                             tt;
    const LazyNumaReplicated<Eval::NNUE::Networks>& networks;

    // Used by NNUE
    Eval::NNUE::AccumulatorStack  accumulatorStack;
    Eval::NNUE::AccumulatorCaches refreshTable;

    friend class Stockfish::ThreadPool;
    friend class SearchManager;
};

struct ConthistBonus {
    int index;
    int weight;
};


}  // namespace Search

}  // namespace Stockfish

#endif  // #ifndef SEARCH_H_INCLUDED
