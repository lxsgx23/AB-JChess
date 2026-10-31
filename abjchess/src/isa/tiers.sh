# Tier table shared by the ISA build scripts (sourced). Names follow Stockfish's ARCH names.
X86_TIERS="x86-64 x86-64-sse3-popcnt x86-64-ssse3 x86-64-sse41-popcnt x86-64-avx2 x86-64-bmi2 x86-64-avxvnni x86-64-avx512 x86-64-vnni512 x86-64-avx512icl"
ARM64_TIERS="armv8 armv8-dotprod"
tier_flags() {  # $1 = tier -> compiler flags (GCC / Clang)
    local NOBMI="-include nobmi.h"
    local AVX2="-mavx2 -mfma -mbmi -mlzcnt -mpopcnt"
    local BMI2="$AVX2 -mbmi2 -DJQ_USE_PEXT"
    local A512="$BMI2 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512cd"
    case "$1" in
        x86-64)              echo "-march=x86-64 -mtune=generic $NOBMI" ;;
        x86-64-sse3-popcnt)  echo "-march=x86-64 -mtune=generic -msse3 -mpopcnt $NOBMI" ;;
        x86-64-ssse3)        echo "-march=x86-64 -mtune=generic -mssse3 $NOBMI" ;;
        x86-64-sse41-popcnt) echo "-march=x86-64 -mtune=generic -msse4.1 -mpopcnt $NOBMI" ;;
        x86-64-avx2)         echo "-march=x86-64 -mtune=generic $AVX2 -include nobmi.h -DJQ_NOBMI2_ONLY" ;;
        x86-64-bmi2)         echo "-march=x86-64 -mtune=generic $BMI2" ;;
        x86-64-avxvnni)      echo "-march=x86-64 -mtune=generic $BMI2 -mavxvnni" ;;
        x86-64-avx512)       echo "-march=x86-64 -mtune=generic $A512" ;;
        x86-64-vnni512)      echo "-march=x86-64 -mtune=generic $A512 -mavx512vnni" ;;
        x86-64-avx512icl)    echo "-march=x86-64 -mtune=generic $A512 -mavx512vnni -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vpopcntdq -mavx512bitalg -mvpclmulqdq -mgfni -mvaes" ;;
        armv8)               echo "-march=armv8-a -include nobmi.h" ;;
        armv8-dotprod)       echo "-march=armv8.2-a+dotprod -include nobmi.h" ;;
        armv7-neon)          echo "-march=armv7-a -mfloat-abi=softfp -mfpu=neon -mthumb -include nobmi.h" ;;
        armv7)               echo "-march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -mthumb -include nobmi.h" ;;
        *) echo "unknown tier $1" >&2; return 1 ;;
    esac
}
tier_id() { echo "$1" | tr -- '-' '_'; }
# Engine source files (small-net release engine: the HNN small net is the only evaluator).
ENGINE_FILES="tables position eval nnue_small hnnue search uci selfplay bitboard position_ref"; SMALL_FLAG=""
