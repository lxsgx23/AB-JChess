#!/bin/bash
# build_macos.sh OUTDIR: the macOS universal engine OUTDIR/abjchess-macos -- ONE fat executable (lipo arm64 + x86_64), each
# slice containing every tier of its architecture with runtime dispatch (isa/dispatch_macos.cpp):
#   arm64: armv8-dotprod armv8;  x86_64: x86-64-bmi2 x86-64-avx2 (Intel Macs without AVX2: clear error, exit 2).
# Per tier: compile (Apple clang, -ffp-contract=off as every tier, main() renamed abj_main_<id>), 'ld -r' with only
# abj_main_<id> exported (all other symbols local: no cross-tier sharing of inline / template code), static
# initializers moved out of __mod_init_func (isa/macho_initsec.py, run by the dispatcher for the selected tier only).
# Checks: each tier object defines exactly one external symbol. No net inside: put abjchess-20261030.nnue next to it.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd); . "$HERE/isa/tiers.sh"
mkdir -p "$1"; OUT=$(cd "$1" && pwd)
CXX=${CXX:-clang++}; MIN=-mmacosx-version-min=11.0; J=${JOBS:-3}
ARM_TIERS="armv8-dotprod armv8"; X86_MAC="x86-64-bmi2 x86-64-avx2"  # no baseline x86-64 tier on macOS (Intel Macs need AVX2)
tier() {  # arch tier index
    local A=$1 T=$2 K=$3 ID; ID=$(tier_id $T); local B="$OUT/obj-$A-$T"; mkdir -p "$B"
    local F="-std=c++20 -O3 -DNDEBUG -pthread -I$HERE -ffp-contract=off -DABJ_TIER_NAME=\"$T\" $SMALL_FLAG -arch $A $MIN $(tier_flags $T)"
    local O="" n=0 f
    for f in $ENGINE_FILES main; do  # batches of J (macOS bash 3.2 has no wait -n)
        local S="$HERE/$f.cpp"
        if [ $f = main ]; then S="$B/main_$ID.cpp"; sed "s/^int main(/int abj_main_$ID(/" "$HERE/main.cpp" > "$S"
            grep -q "^int abj_main_$ID(int argc, char\*\* argv)" "$S" || { echo "main.cpp: main() not found"; exit 1; }; fi
        $CXX $F -c "$S" -o "$B/$f.o" & O="$O $B/$f.o"; n=$((n + 1))
        [ $((n % J)) = 0 ] && wait
    done
    wait
    for f in $O; do [ -f "$f" ] || { echo "compile failed: $f"; exit 1; }; done
    local SYM="__Z$(( ${#ID} + 9 ))abj_main_${ID}iPPc"
    ld -r -arch $A -exported_symbol "$SYM" $O -o "$OUT/t_${A}_$ID.o"
    python3 "$HERE/isa/macho_initsec.py" "$OUT/t_${A}_$ID.o" "__jqi$K"
    local G; G=$(nm -gU "$OUT/t_${A}_$ID.o" | awk '{print $NF}')
    [ "$G" = "$SYM" ] || { echo "$T: external symbols other than abj_main_$ID:"; echo "$G" | head; exit 1; }
    TOBJ="$TOBJ $OUT/t_${A}_$ID.o"
}
slice() {  # arch tiers...
    local A=$1; shift; TOBJ=""; local k=0 t
    for t in "$@"; do tier $A $t $k; k=$((k + 1)); done
    local DF="-march=x86-64"; [ $A = arm64 ] && DF=""
    $CXX -std=c++20 -O2 -arch $A $MIN $DF -c "$HERE/isa/dispatch_macos.cpp" -o "$OUT/dispatch_$A.o"
    $CXX -arch $A $MIN -pthread "$OUT/dispatch_$A.o" $TOBJ -o "$OUT/abjchess-macos-$A"
    strip "$OUT/abjchess-macos-$A"
}
slice arm64 $ARM_TIERS
slice x86_64 $X86_MAC
lipo -create "$OUT/abjchess-macos-arm64" "$OUT/abjchess-macos-x86_64" -output "$OUT/abjchess-macos"
rm -rf "$OUT"/obj-* "$OUT"/t_*.o "$OUT"/dispatch_*.o "$OUT/abjchess-macos-arm64" "$OUT/abjchess-macos-x86_64"
lipo -info "$OUT/abjchess-macos"; ls -l "$OUT/abjchess-macos"
