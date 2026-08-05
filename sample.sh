#!/usr/bin/env bash
set -euo pipefail

# Generate samples from the latest Forge checkpoint.
# Generation stops when EOS is sampled or when TOKENS is reached.
model="${MODEL:-out/forge-220m.bin}"
tokenizer="${TOKENIZER:-data/tokenizer.json}"
tokens="${TOKENS:-500}"
temperature="${TEMPERATURE:-0.7}"
top_k="${TOPK:-30}"
repetition_penalty="${REPETITION_PENALTY:-1.1}"
no_repeat_ngram="${NO_REPEAT_NGRAM:-4}"
seed="${SEED:-1}"

if [[ ! -f "$model" ]]; then
    echo "checkpoint not found: $model" >&2
    exit 1
fi

echo "checkpoint: $model (last written $(date -r "$model" '+%Y-%m-%d %H:%M:%S'))"

names=(
    "Story"
    "Abstract"
    "Science"
    "Procedure"
    "Technical"
)

prompts=(
    "Once upon a time "
    "The meaning of life is "
    "Photosynthesis is the process by which "
    "To bake a loaf of bread, first "
    "In computer science, an algorithm is "
)

for i in "${!prompts[@]}"; do
    prompt="${prompts[$i]}"
    printf '\n=== %s: %s===\n' "${names[$i]}" "$prompt"
    ./LLM.Cli generate \
        --backend cuda \
        --model "$model" \
        --tokenizer "$tokenizer" \
        --prompt "$prompt" \
        --tokens "$tokens" \
        --temperature "$temperature" \
        --topk "$top_k" \
        --repetition-penalty "$repetition_penalty" \
        --no-repeat-ngram "$no_repeat_ngram" \
        --seed "$seed"
done
