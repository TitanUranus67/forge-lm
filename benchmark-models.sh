#!/usr/bin/env bash
set -uo pipefail

app="${APP:-./app/LLM.Cli}"
steps="${STEPS:-3}"
backend="${BACKEND:-cuda}"
results="${RESULTS:-out/model-benchmarks.log}"
presets="${PRESETS:-forge-220m forge-320m}"
matmul_modes="${MATMUL_MODES:-custom fp32}"
geometries="${GEOMETRIES:-1:32 2:16 4:8}"

if [[ ! -x "$app" ]]; then
    printf 'error: executable not found: %s (set APP to override)\n' "$app" >&2
    exit 1
fi

mkdir -p "$(dirname "$results")"
: > "$results"

printf 'Forge candidate benchmark\n' | tee -a "$results"
printf 'GPU: ' | tee -a "$results"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | tee -a "$results" || true
printf 'Steps per measurement: %s\n\n' "$steps" | tee -a "$results"

for preset in $presets; do
    for mode in $matmul_modes; do
        for geometry in $geometries; do
            batch=${geometry%%:*}
            accum=${geometry##*:}
            printf '\n=== %s | %s | batch %s | accum %s ===\n' \
                "$preset" "$mode" "$batch" "$accum" | tee -a "$results"
            set +e
            "$app" benchmark \
                --preset "$preset" \
                --backend "$backend" \
                --matmul-precision "$mode" \
                --batch "$batch" \
                --accum "$accum" \
                --steps "$steps" 2>&1 | tee -a "$results"
            result=${PIPESTATUS[0]}
            set -e
            if ((result != 0)); then
                printf 'RESULT: failed (exit %d)\n' "$result" | tee -a "$results"
            fi
        done
    done
done

printf '\nSaved complete results to %s\n' "$results"
