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
// jq::Thread: std::thread everywhere except macOS, whose secondary threads get only 512 KB of stack by default --
// too little for a deep search (a search + child frame is about 5.5 KB, MAX_PLY 120, plus qsearch). There a pthread
// with an 8 MB stack is used instead (same interface as far as the engine uses it: construct from a callable,
// move, joinable, join).
#pragma once
#include <thread>
#if defined(__APPLE__)
#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <utility>
#endif

namespace jq {
#if !defined(__APPLE__)
using Thread = std::thread;
#else
class Thread {
public:
    Thread() = default;
    template <class F> explicit Thread(F&& f) {
        auto* fn = new std::function<void()>(std::forward<F>(f));
        pthread_attr_t a;
        pthread_attr_init(&a);
        pthread_attr_setstacksize(&a, std::size_t(8) << 20);
        if (pthread_create(&t_, &a, &Thread::run, fn) != 0) { std::fprintf(stderr, "pthread_create failed\n"); std::abort(); }
        pthread_attr_destroy(&a);
        live_ = true;
    }
    Thread(Thread&& o) noexcept : t_(o.t_), live_(o.live_) { o.live_ = false; }
    Thread& operator=(Thread&& o) noexcept {
        if (live_) std::terminate();  // as std::thread
        t_ = o.t_; live_ = o.live_; o.live_ = false;
        return *this;
    }
    Thread(const Thread&) = delete;
    Thread& operator=(const Thread&) = delete;
    ~Thread() { if (live_) std::terminate(); }
    bool joinable() const { return live_; }
    void join() { pthread_join(t_, nullptr); live_ = false; }
private:
    static void* run(void* p) {
        auto* fn = static_cast<std::function<void()>*>(p);
        (*fn)();
        delete fn;
        return nullptr;
    }
    pthread_t t_{};
    bool live_ = false;
};
#endif
}  // namespace jq
