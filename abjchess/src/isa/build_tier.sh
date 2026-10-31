#!/bin/bash
# build_tier.sh TIER OUTDIR [extra flags]: Linux x86-64 static per-tier executable OUTDIR/abjchess-TIER (LTO, one partition);
# keeps OUTDIR/TIER/abjchess.ltrans0.ltrans.o (the whole tier as one relocatable object, used by build_universal.sh).
# Env: CXX (g++), PGO=gen|use:<dir>, NOLTO=1, ENTRY=1 (tier main renamed abj_main_<id>, for the universal binary).
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
. "$HERE/isa/tiers.sh"
TIER=$1; OUT=$(realpath -m "$2"); shift 2
CXX=${CXX:-g++}
ID=$(tier_id "$TIER")
B="$OUT/$TIER"; rm -rf "$B"; mkdir -p "$B"
TF=$(tier_flags "$TIER")
COMMON="-std=c++20 -O3 -DNDEBUG -pthread -I$HERE -ffp-contract=off -DABJ_TIER_NAME=\"$TIER\" $SMALL_FLAG $EXTRA_CXXFLAGS"
if $CXX --version | grep -q clang; then LTO="-flto=full"; LTOSAVE="${LTOSAVE:--Wl,--save-temps}"; else LTO="-flto=1 -flto-partition=one"; LTOSAVE="-save-temps"; fi
[ -n "$NOLTO" ] && LTO="" && LTOSAVE=""
LINK=${LINK:--static}
PG=""
case "$PGO" in gen) PG="-fprofile-generate=$B/pgo -fprofile-update=single";; use:*) PG="-fprofile-use=${PGO#use:} -fprofile-partial-training -Wno-missing-profile";; esac
OBJS=""
for f in $ENGINE_FILES main; do
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS:-2}" ]; do wait -n; done
    nice -n 19 $CXX $COMMON $TF $LTO $PG "$@" -c "$HERE/$f.cpp" -o "$B/$f.o" &
    OBJS="$OBJS $B/$f.o"
done
wait
DEFS=""
[ -n "$ENTRY" ] && DEFS="-Wl,--defsym=main=abj_main_$ID"
( cd "$B" && nice -n 19 $CXX $COMMON $TF $LTO $PG "$@" $LTOSAVE $LINK $OBJS -o "$B/abjchess" $DEFS )
if [ -f "$B/abjchess.exe" ]; then cp "$B/abjchess.exe" "$OUT/abjchess-$TIER.exe"; else cp "$B/abjchess" "$OUT/abjchess-$TIER"; fi
ls -l "$OUT"/abjchess-$TIER*
