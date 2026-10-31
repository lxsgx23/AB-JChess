/*
  ABJchess (AB JChess), a UCI Jieqi engine.
  Contains code derived from AB-JChess / Pikafish / Stockfish:
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)
  Based on jqe, Copyright (c) 2026 Devin O'Keefe (MIT licence, see abjchess/LICENSE-jqe-MIT.txt).
  See THIRD_PARTY_NOTICES.md at the repository root.

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "eval.h"
#include "nnue.h"
#include "search.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include "hnnue.h"
#include "jqthread.h"
#if defined(_WIN32)
extern "C" __declspec(dllimport) unsigned long __stdcall GetModuleFileNameW(void*, wchar_t*, unsigned long);
extern "C" __declspec(dllimport) int __stdcall WideCharToMultiByte(unsigned int, unsigned long, const wchar_t*, int, char*, int, const char*, int*);
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <cstdint>
#include <climits>
#include <cstdlib>
#else
#include <unistd.h>
#include <climits>
#endif
// Default evaluation net: a separate (losslessly compressed) file next to the engine, no net inside the binary.
#ifndef ABJ_NET_FILE
#define ABJ_NET_FILE "abjchess-20261030.nnue"  // compressed HNN small net (JQHNNZ01; format detected by magic)
#endif
#ifndef ABJ_TIER_NAME
#define ABJ_TIER_NAME "native"
#endif
#if defined(_WIN32)
#define ABJ_OS "windows"
#elif defined(__ANDROID__)
#define ABJ_OS "android"
#elif defined(__APPLE__)
#define ABJ_OS "macos"
#else
#define ABJ_OS "linux"
#endif
#if defined(__x86_64__) || defined(_M_X64)
#define ABJ_ARCH "x86_64"
#elif defined(__aarch64__)
#define ABJ_ARCH "arm64"
#elif defined(__arm__)
#define ABJ_ARCH "armv7"
#else
#define ABJ_ARCH "unknown"
#endif

#define JQ_ACCV(a) (a).hnn.v

namespace jq {

void selfplay(std::istringstream& is);

namespace {

std::mutex outMutex;
void out(const std::string& s) {
    std::lock_guard<std::mutex> lock(outMutex);
    std::cout << s << std::endl;
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return char(std::tolower(ch));
    });
    return value;
}

const char* BENCH_FENS[] = {
    "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1",
    "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/4P4/X1X3X1X/1X5X1/9/XXXXKXXXX b R2A2C2P4N2B2r2a2c2p5n2b2 1 1",
    "r1xxkxxxx/9/1x5c1/x1x1x1x1x/9/9/X1X1X1X1X/1C5X1/9/XXXXKXXXX w R2A2C1P5N2B2r1a2c1p5n2b2 2 2",
    "2xakxxxx/4r4/1x5c1/x1x1x1x1p/9/2P6/X3X1X1X/1C5X1/9/XXXXKXXXX w R2A2C1P4N2B2r1a1c1p4n2b2 0 4",  // 2026-10-30: was X1X1X1X1X (17 red pieces, rejected by set_fen and then searched half-parsed)
    "4k4/9/9/9/9/9/9/9/4A4/3RK4 w - 0 1",
    "3ak4/4a4/9/9/9/9/9/9/9/3RK4 w - 0 1",
};

constexpr const char* EMBEDDED = "<embedded>";

// Directory of the running executable with a trailing separator (UTF-8 on Windows), "" if unknown.
std::string exe_dir() {
    std::string p;
#if defined(_WIN32)
    std::wstring w(32768, L'\0');
    const unsigned long n = GetModuleFileNameW(nullptr, &w[0], static_cast<unsigned long>(w.size()));
    if (n == 0 || n >= w.size()) return "";
    const int m = WideCharToMultiByte(65001u, 0, w.c_str(), int(n), nullptr, 0, nullptr, nullptr);
    if (m <= 0) return "";
    p.assign(std::size_t(m), '\0');
    WideCharToMultiByte(65001u, 0, w.c_str(), int(n), &p[0], m, nullptr, nullptr);
#elif defined(__APPLE__)
    char buf[PATH_MAX * 4];
    uint32_t sz = sizeof buf;
    if (_NSGetExecutablePath(buf, &sz) != 0) return "";
    char real[PATH_MAX];
    p = realpath(buf, real) ? std::string(real) : std::string(buf);
#else
    char buf[PATH_MAX * 4];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n <= 0) return "";
    p.assign(buf, std::size_t(n));
#endif
    const std::size_t k = p.find_last_of("\\/");
    return k == std::string::npos ? std::string() : p.substr(0, k + 1);
}
bool file_exists(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}
bool is_absolute(const std::string& p) {
    return !p.empty() && (p[0] == '/' || p[0] == '\\' || (p.size() > 1 && p[1] == ':'));
}
// A net path as given: absolute paths as is; relative ones in the engine's directory first, then the current one.
std::string resolve_net(const std::string& name, std::string* tried = nullptr) {
    if (is_absolute(name)) { if (tried) *tried = name; return file_exists(name) ? name : std::string(); }
    const std::string d = exe_dir();
    if (tried) *tried = (d.empty() ? std::string() : d + name + ", ") + "./" + name;
    if (!d.empty() && file_exists(d + name)) return d + name;
    if (file_exists(name)) return name;
    return "";
}
std::string net_load_desc() {
    const hnn::LoadInfo& li = hnn::last_load();
    if (li.fmt == "raw") return "raw " + std::to_string(li.fileBytes) + " bytes";
    char ms[32];
    std::snprintf(ms, sizeof ms, "%.1f", li.decodeMs);
    return "compressed " + li.fmt + " " + std::to_string(li.fileBytes) + " -> " + std::to_string(li.rawBytes) +
           " bytes, CRC-32 ok, decoded in " + ms + " ms";
}

struct Engine {
    Position pos;
    Searcher searcher{1, 64};
    Thread thread;
    std::string evalFile;
    std::string netType = "auto";  // isa: NetType override (auto / small / big)

    // Kind of the active evaluation net, auto-detected by the loader from the file magic / layout / JQT1 trailer:
    // "big" = HNN net with JQT1 threat features, "small" = HNN net without threats (embedded default),
    // "abj" = AB-JChess ABJNNUE package, "legacy" = JQN1 / JQSNNUE1, "none".
    static std::string net_kind() {
        if (!nnue::loaded()) return "none";
        return "small";  // small-net-only build: the loader accepts nothing else
        if (nnue::is_abj()) return "abj";
        if (hnn::loaded()) return hnn::threats_on() ? "big" : "small";
        return "legacy";
    }
    static std::string net_desc() {
        const std::string k = net_kind();
        if (k == "big") return "big (HNN W " + std::to_string(hnn::width()) + ", JQT1 threat features width " + std::to_string(hnn::threat_width()) + ")";
        if (k == "small") return "small (HNN W " + std::to_string(hnn::width()) + ", md5 " + hnn::last_load().rawMd5 + ")";
        return k;
    }
    // Restore a previously active net after a refused EvalFile (the loader may have been part-way through).
    void restore_net(const std::string& prev) {
        std::string err;
        bool ok = true;
        if (prev.empty() || prev == "<empty>") nnue::unload();
        else if (prev == EMBEDDED) ok = nnue::load_embedded(&err);
        else {
            const std::string path = resolve_net(prev);
            ok = !path.empty() && nnue::load(path, &err);
        }
        evalFile = ok ? prev : std::string();
        if (!ok) { nnue::unload(); out("info string ERROR could not restore the previous net " + prev + ": " + err + " (no net loaded)"); }
    }

    Engine() {
        out("AB JChess by Huorongrong,Laoxu(Kouza)");
        out("AB JChess is free of charge. Resale is prohibited.");
        pos.set_fen(START_FEN);
        // Default net: the compressed small net file ABJ_NET_FILE next to the engine (or in the current directory).
        // No net is compiled into the binary; a missing / bad default net is reported loudly and searching is refused.
        evalFile = ABJ_NET_FILE;
        std::string tried, err;
        const std::string path = resolve_net(ABJ_NET_FILE, &tried);
        if (path.empty()) netError = "default net " + std::string(ABJ_NET_FILE) + " not found (looked in " + tried + ")";
        else if (!nnue::load(path, &err)) netError = "default net " + path + " failed to load: " + err;
        else netInfo = "net type " + net_desc() + " (default " + std::string(ABJ_NET_FILE) + ", " + net_load_desc() + ")";
        if (!netError.empty()) { evalFile.clear(); out("info string ERROR " + netError + "; set EvalFile to a net file; search refused until a net is loaded"); }
    }
    std::string netError, netInfo;
    bool firstReady = true;
    // isready: the first one reports the dispatched ISA tier and the active net (info strings before readyok).
    void cmd_isready() {
        if (firstReady) {
            firstReady = false;
            const char* forced = std::getenv("ABJ_TIER");
            out(std::string("info string ISA tier ") + ABJ_TIER_NAME + " (" + ABJ_OS + " " + ABJ_ARCH + (forced && *forced ? ", ABJ_TIER set" : "") + ")");
            if (nnue::loaded())
                out("info string net type " + net_desc() + " (" + (evalFile == ABJ_NET_FILE ? "default " : "") + evalFile +
                    (hnn::loaded() ? ", " + net_load_desc() : std::string()) + ")");
        }
        if (!nnue::loaded()) out("info string ERROR no evaluation net loaded" + (netError.empty() ? std::string() : " (" + netError + ")") + "; set EvalFile; search refused");
        out("readyok");
    }
    bool refuse_without_net() {
        if (nnue::loaded()) return false;
        out("info string ERROR no evaluation net loaded" + (netError.empty() ? std::string() : " (" + netError + ")") + "; set EvalFile; search refused");
        return true;
    }
    ~Engine() { wait(); }
    void wait() {
        if (thread.joinable()) {
            searcher.stop();
            thread.join();
        }
    }

    void cmd_uci() {
        out("id name AB JChess 1 RB");
        out("id author Huorongrong,Laoxu(Kouza)");
        out("option name Hash type spin default 64 min 1 max 16384");
        out("option name Threads type spin default 1 min 1 max 256");
        out(std::string("option name EvalFile type string default ") + ABJ_NET_FILE);
        out("option name Clear Hash type button");
        out("option name ChaseRule type combo default skyrule var none var default var skyrule");
        out("uciok");
    }

    void cmd_setoption(std::istringstream& is) {
        std::string tok, name, value;
        if (!(is >> tok) || lower_ascii(tok) != "name") {
            out("info string invalid setoption command");
            return;
        }
        bool hasValue = false;
        while (is >> tok) {
            if (tok == "value") {
                hasValue = true;
                break;
            }
            name += (name.empty() ? "" : " ") + tok;
        }
        if (hasValue)
            while (is >> tok) value += (value.empty() ? "" : " ") + tok;

        const std::string option = lower_ascii(name);
        wait();
        try {
            if (option == "hash") searcher.resize_hash(std::size_t(std::clamp(std::stoi(value), 1, 16384)));
            else if (option == "threads") searcher.set_threads(std::clamp(std::stoi(value), 1, 256));
            else if (option == "clear hash") searcher.clear();
            else if (option == "chasingrule" || option == "chaserule") {
                const std::string rule = lower_ascii(value);
                if (rule == "0" || rule == "none") g_chaseRule = CHASE_RULE_NONE;
                else if (rule == "1" || rule == "default" || rule == "github") g_chaseRule = CHASE_RULE_DEFAULT;
                else if (rule == "2" || rule == "skyrule" || rule == "skyrule_jieqi") g_chaseRule = CHASE_RULE_SKYRULE;
                else {
                    out("info string invalid ChaseRule " + value + " (use none, default, or skyrule)");
                    return;
                }
                searcher.clear();
            }
            else if (option == "hnnscale") { hnn::g_scale = std::stoi(value); searcher.clear(); }
            else if (option == "evalfile") {
                const std::string prevEval = evalFile;
                evalFile = value;
                std::string err;
                if (value.empty() || value == "<empty>") nnue::unload();
                else if (value == EMBEDDED) {
                    if (!nnue::load_embedded(&err)) { out("info string EvalFile load failed: " + err); restore_net(prevEval); }
                    else if (netType != "auto" && net_kind() != netType) {
                        out("info string ERROR embedded net is " + net_desc() + " but NetType is " + netType + "; refused, active net stays " + (prevEval.empty() ? std::string("<none>") : prevEval));
                        restore_net(prevEval);
                    }
                    else out("info string EvalFile loaded <embedded> (" + std::to_string(nnue::embedded_size()) + " bytes compressed)");
                }
                else {
                    std::string tried;
                    const std::string path = resolve_net(value, &tried);
                    const std::string prev = prevEval;
                    if (path.empty()) {
                        out("info string ERROR EvalFile " + value + " not found (looked in " + tried + "); active net stays " + (prev.empty() ? std::string("<none>") : prev));
                        evalFile = prev;
                    } else if (!nnue::load(path, &err)) {
                        out("info string EvalFile load failed: " + err);
                        out("info string ERROR EvalFile " + value + " refused (unknown, damaged or unsupported net); active net stays " + (prev.empty() ? std::string("<none>") : prev));
                        restore_net(prev);
                    } else if (netType != "auto" && net_kind() != netType) {
                        out("info string ERROR EvalFile " + value + " is a " + net_desc() + " net but NetType is " + netType + "; refused, active net stays " + (prev.empty() ? std::string("<none>") : prev));
                        restore_net(prev);
                    } else {
                        out("info string EvalFile loaded " + value + (path != value ? " (" + path + ")" : std::string()));
                        netError.clear();
                        if (hnn::loaded()) out("info string net file " + net_load_desc());
                        out("info string net type " + net_desc());
                    }
                }
                searcher.clear();
            }
            else {
                for (Param& p : params())
                    if (option == lower_ascii(p.name)
                        || (option == "nodestime" && std::string(p.name) == "NodesTime")) {
                        *p.value = std::clamp(std::stoi(value), p.lo, p.hi);
                        search::init();
                        searcher.clear();
                        return;
                    }
                out("info string unknown option " + name);
            }
        } catch (const std::exception& error) {
            out("info string invalid value for option " + name + ": " + error.what());
        }
    }

    void cmd_position(std::istringstream& is) {
        std::string tok, fen;
        bool hasMoves = false;
        is >> tok;
        if (tok == "startpos") {
            fen = START_FEN;
            if (is >> tok) {
                if (tok != "moves") {
                    out("info string invalid position command");
                    return;
                }
                hasMoves = true;
            }
        } else if (tok == "fen") {
            while (is >> tok) {
                if (tok == "moves") {
                    hasMoves = true;
                    break;
                }
                fen += (fen.empty() ? "" : " ") + tok;
            }
        } else
            return;
        std::string err;
        Position next;
        if (!next.set_fen(fen, &err)) {
            out("info string bad fen: " + err);
            return;
        }
        if (hasMoves)
            while (is >> tok)
                if (!next.apply_token(tok, &err)) {
                    out("info string bad move " + tok + ": " + err);
                    return;
                }
        pos = next;
    }

    void cmd_go(std::istringstream& is) {
        Limits lim;
        std::string tok;
        int perftDepth = 0;
        bool searchMovesSpecified = false;
        std::vector<std::string> searchMoveTokens;
        while (is >> tok) {
            if (tok == "depth") is >> lim.depth;
            else if (tok == "nodes") is >> lim.nodes;
            else if (tok == "movetime") is >> lim.movetime;
            else if (tok == "wtime") is >> lim.time[RED];
            else if (tok == "btime") is >> lim.time[BLACK];
            else if (tok == "winc") is >> lim.inc[RED];
            else if (tok == "binc") is >> lim.inc[BLACK];
            else if (tok == "movestogo") is >> lim.movestogo;
            else if (tok == "infinite") lim.infinite = true;
            else if (tok == "perft") is >> perftDepth;
            else if (tok == "mate") is >> lim.mate;
            else if (tok == "ponder") lim.ponder = true;
            else if (tok == "searchmoves") {
                searchMovesSpecified = true;
                while (is >> tok) searchMoveTokens.push_back(tok);
            }
        }
        wait();
        Position root = pos;
        if (perftDepth > 0) {
            out("Nodes searched: " + std::to_string(search::perft(root, perftDepth)));
            return;
        }
        if (refuse_without_net()) { out("bestmove (none)"); return; }  // no silent fallback to another evaluation
        if (searchMovesSpecified) {
            Move legal[MAX_MOVES];
            const int count = root.gen_legal(legal);
            for (const std::string& move : searchMoveTokens) {
                if (move.size() != 4) continue;
                const Move candidate = parse_move_coords(move);
                for (int i = 0; i < count; ++i)
                    if (base_move(candidate) == legal[i]) {
                        if (std::find(lim.searchMoves.begin(), lim.searchMoves.end(), legal[i]) == lim.searchMoves.end())
                            lim.searchMoves.push_back(legal[i]);
                        break;
                    }
            }
            if (lim.searchMoves.empty()) {
                out("info string no legal moves in searchmoves");
                out("bestmove 0000");
                return;
            }
        }
        searcher.prepare(lim.ponder);  // preserve an immediate stop or ponderhit before the worker starts
        thread = Thread([this, root, lim] {
            static const std::function<void(const std::string&)> info = [](const std::string& s) { out(s); };
            try {
                const SearchResult r = searcher.go(root, lim, info, false);
                std::string best = r.best == MOVE_NONE ? "0000" : move_str(r.best);
                std::string response = "bestmove " + best;
                if (r.best != MOVE_NONE && r.pv.size() > 1)
                    response += " ponder " + move_str(r.pv[1]);
                out(response);
            } catch (const std::exception& error) {
                out(std::string("info string search failed: ") + error.what());
                out("bestmove 0000");
            }
        });
    }

    void cmd_bench(std::istringstream& is) {
        int depth = 7;
        is >> depth;
        wait();
        if (refuse_without_net()) return;
        uint64_t nodes = 0;
        const auto t0 = std::chrono::steady_clock::now();
        searcher.clear();
        for (const char* f : BENCH_FENS) {
            Position p;
            std::string err;
            if (!p.set_fen(f, &err)) {  // never search a half-parsed position
                out(std::string("info string ERROR bench: invalid FEN skipped (") + err + "): " + f);
                continue;
            }
            Limits lim;
            lim.depth = depth;
            const SearchResult r = searcher.go(p, lim, [](const std::string&) {});
            nodes += r.nodes;
            out("bench " + std::string(f) + " -> " + move_str(r.best) + " score " + std::to_string(r.score) +
                " nodes " + std::to_string(r.nodes));
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        out("Nodes searched: " + std::to_string(nodes));
        out("Nodes/second  : " + std::to_string(nodes * 1000 / uint64_t(std::max<int64_t>(1, ms))));
        out("Time (ms)     : " + std::to_string(ms));
    }
};

}  // namespace

int uci_loop(int argc, char** argv) {
    Engine engine;
    std::string line;
    // Command-line arguments are executed as one command (e.g. `abjchess bench 8`), then exit.
    const bool once = argc > 1;
    if (once) {
        for (int i = 1; i < argc; ++i) line += (i > 1 ? " " : "") + std::string(argv[i]);
    }
    while (once || std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd;
        is >> cmd;
        if (cmd == "uci") engine.cmd_uci();
        else if (cmd == "isready") engine.cmd_isready();
        else if (cmd == "ponderhit") engine.searcher.ponderhit();
        else if (cmd == "debug" || cmd == "register") { }
        else if (cmd == "ucinewgame") { engine.wait(); engine.searcher.clear(); }
        else if (cmd == "setoption") engine.cmd_setoption(is);
        else if (cmd == "position") { engine.wait(); engine.cmd_position(is); }
        else if (cmd == "go") engine.cmd_go(is);
        else if (cmd == "stop") engine.wait();
        else if (cmd == "quit") break;
        else if (cmd == "help" || cmd == "--help" || cmd == "license" || cmd == "--license")
            out("AB JChess is a jieqi engine communicating through the UCI protocol.");
        else if (cmd == "d") out(engine.pos.pretty());
        else if (cmd == "moves") {  // legal moves of the current position (diagnostics)
            Move ms[MAX_MOVES];
            const int n = engine.pos.gen_legal(ms);
            std::string line2 = "moves";
            for (int i = 0; i < n; ++i) line2 += " " + move_str(ms[i]);
            out(line2);
        }
        else if (cmd == "hnneval") {
            if (!nnue::is_hnn()) out("info string no HNN net");
            else {
                static hnn::Acc a;
                hnn::refresh(engine.pos, a);
                int ps = 0, po = 0;
                const int rv = hnn::runtime_value(a, engine.pos.stm, &ps, &po);
                out("hnn psqt " + std::to_string(ps) + " positional " + std::to_string(po) + " runtime " +
                    std::to_string(rv) + " abjchess " + std::to_string(hnn::evaluate(a, engine.pos.stm)) + " material " +
                    std::to_string(a.m.material));
            }
        }
        else if (cmd == "dualstats") {
            using nnue::g_dualEv;
            const uint64_t t = g_dualEv[0] + g_dualEv[1];
            std::printf("dualstats big %llu small %llu smallRejected %llu smallFrac %.4f\n", (unsigned long long)g_dualEv[0], (unsigned long long)g_dualEv[1],
                        (unsigned long long)g_dualEv[2], t ? double(g_dualEv[1]) / double(t) : 0.0);
            std::fflush(stdout);
            g_dualEv[0] = g_dualEv[1] = g_dualEv[2] = 0;
        }
        else if (cmd == "dualeval" && nnue::hsm_loaded() && !nnue::is_abj()) {  // dual HNN: big (EvalFile) and small (EvalFileSmall) values
            static nnue::Accumulator a;
            nnue::refresh(engine.pos, a);
            out("dualhnn big " + std::to_string(nnue::evaluate(a, engine.pos.stm)) + " small " + std::to_string(nnue::evaluate_hsm(a, engine.pos.stm)) +
                " mat " + std::to_string(nnue::simple_material(engine.pos)) + " hidden " + std::to_string(nnue::hidden_count(engine.pos)));
        }
        else if (cmd == "dualeval") {  // side-to-move values: big net, small net raw and scaled (dual mode)
            if (!nnue::dual()) out("info string dualeval needs EvalFile (JQN1) and EvalFileBig");
            else {
                static nnue::Accumulator a;
                nnue::refresh(engine.pos, a);
                const int sc = nnue::g_smallScale;
                nnue::g_smallScale = 1000;
                const int raw = nnue::evaluate_small(a, engine.pos.stm);
                nnue::g_smallScale = sc;
                // quiet: side to move not in check and no capture with a positive exchange value
                bool quiet = !engine.pos.in_check();
                Move ms[MAX_MOVES];
                const int nm = quiet ? engine.pos.gen_legal(ms) : 0;
                for (int i = 0; i < nm && quiet; ++i)
                    if (engine.pos.board[to_sq(ms[i])] != NO_PIECE && search::see_value(engine.pos, ms[i]) > 0) quiet = false;
                out("dual big " + std::to_string(nnue::evaluate(a, engine.pos.stm)) + " small " + std::to_string(raw) +
                    " scaled " + std::to_string(nnue::evaluate_small(a, engine.pos.stm)) + " quiet " + std::to_string(int(quiet)));
            }
        }
        else if (cmd == "eval") out("classical " + std::to_string(classical_eval(engine.pos)) + " eval " + std::to_string(evaluate(engine.pos)));
        else if (cmd == "perft") {
            int d = 1;
            is >> d;
            Position p = engine.pos;
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t n = search::perft(p, d);
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
            out("perft " + std::to_string(d) + " " + std::to_string(n) + " (" + std::to_string(ms) + " ms)");
        } else if (cmd == "see") {
            std::string mv;
            is >> mv;
            const Move m = parse_move_coords(mv);
            out("see " + mv + " " + std::to_string(search::see_value(engine.pos, m)));
        } else if (cmd == "perftc") {
            int d = 1;
            is >> d;
            Position p = engine.pos;
            uint64_t leaves = 0;
            const double mass = search::perft_chance(p, d, leaves);
            char buf[128];
            std::snprintf(buf, sizeof buf, "perftc %d leaves %llu expected %.6f", d, (unsigned long long)leaves, mass);
            out(buf);
        } else if (cmd == "nnbench") {
            if (!nnue::loaded()) { out("info string no net"); continue; }
            Position p = engine.pos;
            Move ml[MAX_MOVES];
            const int n = p.gen_legal(ml);
            static nnue::Accumulator a0, a1;
            if (is.str().find("ftz") != std::string::npos) nnue::set_flush_denormals();
            nnue::Dirty d;
            nnue::refresh(p, a0);
            const int iters = 1000000;
            auto t0 = std::chrono::steady_clock::now();
            int64_t sink = 0;
            for (int i = 0; i < iters; ++i) sink += nnue::evaluate(a0, p.stm);
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                Move m = ml[i % n];
                if (p.is_dark_move(m)) m = make_move(from_sq(m), to_sq(m), SOLDIER);
                nnue::make_dirty(p, m, d);
                nnue::update(a0, a1, d);
                sink += int64_t(JQ_ACCV(a1)[0][0]);
            }
            auto t2 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters / 10; ++i) { nnue::refresh(p, a1); sink += int64_t(JQ_ACCV(a1)[1][1]); }
            auto t3 = std::chrono::steady_clock::now();
            auto ns = [&](auto x, auto y, int k) { return std::chrono::duration<double, std::nano>(y - x).count() / k; };
            out("nnbench evaluate " + std::to_string(ns(t0, t1, iters)) + " ns, update " + std::to_string(ns(t1, t2, iters)) +
                " ns, refresh " + std::to_string(ns(t2, t3, iters / 10)) + " ns (sink " + std::to_string(sink & 1) + ")");
        } else if (cmd == "exportnet" || cmd == "decodenet") {
            // exportnet OUT: the active EvalFile decoded to the raw .hnn bytes; decodenet IN OUT: any net file (raw or
            // JQHNNL01 / JQHNNZ01 compressed) to the raw .hnn bytes (CRC-32 / size checked: md5 equals the original .hnn)
            std::string in, path, err;
            if (cmd == "decodenet") is >> in; else in = engine.evalFile;
            is >> path;
            const std::string src = in.empty() ? std::string() : resolve_net(in);
            if (path.empty()) out("info string usage: decodenet <in> <out> | exportnet <out>");
            else if (src.empty()) out("info string " + cmd + " failed: net " + (in.empty() ? std::string("<none>") : in) + " not found");
            else if (!nnue::decode_net_file(src, path, &err)) out("info string " + cmd + " failed: " + err);
            else out("info string " + cmd + " wrote " + path);
        } else if (cmd == "bench") engine.cmd_bench(is);
        else if (cmd == "selfplay") { engine.wait(); selfplay(is); }
        else if (!cmd.empty() && cmd[0] != '#')
            out("Unknown command: '" + line + "'. Type help for more information.");
        if (once) {
            engine.wait();
            break;
        }
    }
    engine.wait();
    if (const char* fn = std::getenv("ABJ_DUALSTATS"); fn && (nnue::g_dualEv[0] | nnue::g_dualEv[1])) {   // in-game dual stats
        if (FILE* f = std::fopen(fn, "a")) {
            std::fprintf(f, "%llu %llu %llu\n", (unsigned long long)nnue::g_dualEv[0], (unsigned long long)nnue::g_dualEv[1], (unsigned long long)nnue::g_dualEv[2]);
            std::fclose(f);
        }
    }
    return 0;
}

}  // namespace jq
