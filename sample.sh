#!/usr/bin/env bash
set -euo pipefail

# Generate samples from the latest Forge checkpoint.
# Generation stops when EOS is sampled or when TOKENS is reached.
model="${MODEL:-out/forge-98m.bin}"
tokenizer="${TOKENIZER:-data/forge}"
tokens="${TOKENS:-500}"
temperature="${TEMPERATURE:-0.7}"
top_k="${TOPK:-30}"
repetition_penalty="${REPETITION_PENALTY:-1.1}"
no_repeat_ngram="${NO_REPEAT_NGRAM:-4}"

if [[ ! -f "$model" ]]; then
    echo "checkpoint not found: $model" >&2
    exit 1
fi

echo "checkpoint: $model (last written $(date -r "$model" '+%Y-%m-%d %H:%M:%S'))"

prompts=(
    "Once upon a time "
    "The meaning of life is "
    "The history of the world began "
)

for prompt in "${prompts[@]}"; do
    printf '\n=== %s===\n' "$prompt"
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
        --seed "$RANDOM"
done
