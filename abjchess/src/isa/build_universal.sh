#!/bin/bash
# build_universal.sh linux|windows|android-arm64|android-armv7|android-x86_64 OUTDIR: all x86-64 tiers (LTO, one partition each, everything but main internalised)
# + isa/dispatch_x86.cpp -> one static executable (abjchess-linux-x86-64 / abjchess-windows-x86-64.exe). REUSE=1 keeps tier objects.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd); . "$HERE/isa/tiers.sh"
PLAT=$1; OUT=$(realpath -m "$2"); mkdir -p "$OUT"
NDKB=${NDKB:-${ANDROID_NDK_HOME:-$HOME/android-ndk-r27c}/toolchains/llvm/prebuilt/linux-x86_64/bin}
DISP=dispatch_x86.cpp; DFLAGS="-march=x86-64"; LTOOBJ=abjchess.ltrans0.ltrans.o; ALL=$X86_TIERS
if [ "$PLAT" = android-arm64 ]; then
    export CXX=$NDKB/aarch64-linux-android29-clang++; OBJCOPY=$NDKB/llvm-objcopy; STRIP=$NDKB/llvm-strip; INIT=.init_array
    EXE="$OUT/abjchess-android-arm64-v8a"; DISP=dispatch_arm.cpp; DFLAGS="-march=armv8-a"; LTOOBJ=abjchess.lto.o; ALL="armv8-dotprod armv8"
elif [ "$PLAT" = android-armv7 ]; then
    export CXX=$NDKB/armv7a-linux-androideabi29-clang++; OBJCOPY=$NDKB/llvm-objcopy; STRIP=$NDKB/llvm-strip; INIT=.init_array
    EXE="$OUT/abjchess-android-armeabi-v7a"; DISP=dispatch_arm.cpp; DFLAGS="-march=armv7-a -mthumb"; LTOOBJ=abjchess.lto.o; ALL="armv7-neon armv7"
elif [ "$PLAT" = android-x86_64 ]; then
    export CXX=$NDKB/x86_64-linux-android29-clang++; OBJCOPY=$NDKB/llvm-objcopy; STRIP=$NDKB/llvm-strip; INIT=.init_array
    EXE="$OUT/abjchess-android-x86_64"; LTOOBJ=abjchess.lto.o
elif [ "$PLAT" = windows ]; then
    export CXX=x86_64-w64-mingw32-g++-posix EXTRA_CXXFLAGS="-include wincompat.h"; OBJCOPY=x86_64-w64-mingw32-objcopy
    STRIP=x86_64-w64-mingw32-strip; INIT=.ctors; EXE="$OUT/abjchess-windows-x86-64.exe"
elif [ "$PLAT" = windows-llvm ]; then
    # LLVM-MinGW (clang + lld, libc++; x86_64-w64-mingw32-clang++ on PATH): full LTO, lld saves the post-LTO object
    # (abjchess.exe.lto.obj); emulated TLS like GCC (no .tls sections); GNU mingw objcopy for the section renames.
    export CXX=x86_64-w64-mingw32-clang++ EXTRA_CXXFLAGS="-include wincompat.h -femulated-tls" LTOSAVE="-Wl,-Xlink=-lldsavetemps"
    OBJCOPY=/usr/bin/x86_64-w64-mingw32-objcopy; STRIP=/usr/bin/x86_64-w64-mingw32-strip; INIT=.ctors
    EXE="$OUT/abjchess-windows-x86-64.exe"; LTOOBJ=abjchess.exe.lto.obj
else
    export CXX=g++; OBJCOPY=objcopy; STRIP=strip; INIT=.init_array; EXE="$OUT/abjchess-linux-x86-64"
fi
OBJS=""
for t in ${TIERS:-$ALL}; do
    id=$(tier_id $t)
    [ -n "$REUSE" ] && [ -f "$OUT/$t/$LTOOBJ" ] || "$HERE/isa/build_tier.sh" $t "$OUT" > "$OUT/build_$t.log" 2>&1 || { echo "build $t failed"; tail "$OUT/build_$t.log"; exit 1; }
    L="$OUT/$t/$LTOOBJ"
    [ "$LTOOBJ" != abjchess.ltrans0.ltrans.o ] || [ $(ls "$OUT/$t"/abjchess.ltrans*.ltrans.o | wc -l) = 1 ] || { echo "more than one ltrans partition for $t"; exit 1; }
    # keep main (renamed below) and, for PE, the .refptr.* indirection cells libstdc++.a also references (identical
    # contents in every tier: one pointer to the same symbol; merged with --allow-multiple-definition)
    $OBJCOPY --wildcard --keep-global-symbol=main --keep-global-symbol='.refptr.*' -R .group "$L" "$OUT/t_$id.tmp.o"
    if [ "${PLAT%-llvm}" = windows ]; then SEC=".data\$jqi_${id}_m"; else SEC="jqinit_$id"; fi
    $OBJCOPY --redefine-sym main=abj_main_$id --rename-section $INIT=$SEC,alloc,load,data,contents "$OUT/t_$id.tmp.o" "$OUT/t_$id.o"
    # nothing else may run implicitly: no other constructor / destructor / TLS-callback sections
    if readelf -SW "$OUT/t_$id.o" >/dev/null 2>&1 && readelf -SW "$OUT/t_$id.o" | grep -qE ' (\.ctors|\.dtors|\.init_array|\.fini_array|\.preinit_array) '; then
        echo "$t: unexpected init/TLS section (ELF)"; readelf -SW "$OUT/t_$id.o" | grep -E "ctors|init|fini"; exit 1; fi
    if [ "${PLAT%-llvm}" = windows ] && objdump -h "$OUT/t_$id.o" 2>/dev/null | grep -qE ' (\.ctors|\.dtors|\.init_array|\.fini_array|\.preinit_array|\.CRT\$|\.tls)'; then
        echo "$t: unexpected init/TLS section"; objdump -h "$OUT/t_$id.o" | grep -E 'ctors|init|fini|CRT|tls'; exit 1; fi
    OBJS="$OBJS $OUT/t_$id.o"
done
$CXX -std=c++20 -O2 $DFLAGS -c "$HERE/isa/$DISP" -o "$OUT/dispatch.o"
LDX=""; [ "${PLAT%-llvm}" = windows ] && LDX="-Wl,--allow-multiple-definition"
$CXX -static -pthread $LDX "$OUT/dispatch.o" $OBJS -o "$EXE"
$STRIP "$EXE"
ls -l "$EXE"
