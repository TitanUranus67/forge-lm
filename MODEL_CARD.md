# Forge-98M model card

Forge-98M is the first model released with ForgeLM. It is a small, experimental
base language model intended to make an end-to-end C# training stack inspectable
and reproducible.

## Model details

| Property | Value |
|---|---:|
| Parameters | 97,934,592 |
| Architecture | Decoder-only, GPT-2 style, pre-LayerNorm |
| Layers | 12 |
| Model width | 768 |
| Attention heads | 12 |
| Context length | 512 tokens |
| Vocabulary | 16,257 byte-level BPE tokens, including EOS |
| Embeddings | Token embedding tied to output projection |
| Checkpoint format | ForgeLM `FORGEMODEL1` |
| License | Apache-2.0 |

The checkpoint is backend-independent and can be loaded by the CPU, CUDA, or
D3D12 implementation in this repository.

## Training

Forge-98M was trained from scratch on an NVIDIA RTX 4090 using the first three
FineWeb-Edu `sample/10BT` parquet shards. Documents were split deterministically
by URL, BPE merges were prevented from crossing document boundaries, and an EOS
token was appended to every document.

| Property | Value |
|---|---:|
| Training-token budget | 1,024,000,000 |
| Available training corpus | 1,905,884,230 tokens |
| Available validation corpus | 212,391,746 tokens |
| Optimizer updates | 31,250 |
| Tokens per update | 32,768 |
| Physical batch / accumulation | 16 / 4 |
| Peak / minimum learning rate | 3e-4 / 3e-5 |
| Warmup | 4,096,000 tokens (125 updates) |
| Schedule | Linear warmup, then cosine decay |
| Optimizer | AdamW with global gradient clipping |
| Final training loss | 3.7477 |
| Final validation loss | 3.7688 |
| Validation perplexity | 43.3 |

Validation loss is the mean of 50 deterministic held-out batches. The final
training token budget is about 54% of one pass through the available training
tokens.

Exact dataset URLs, preparation counts, and tokenizer/data hashes are in
[data/forge/SOURCE.md](data/forge/SOURCE.md).

## Intended use

Forge-98M is intended for:

- studying a complete language-model implementation written in C#;
- testing ForgeLM checkpoint, backend, and inference behavior;
- experimentation with base-model text completion;
- serving as a reproducible baseline for future Forge models.

It is not intended to act as a factual authority, autonomous agent, safety
classifier, or production assistant.

## Limitations

Forge-98M is small, has a 512-token context window, and was trained on a limited
slice of web data. It can repeat itself, drift off topic, produce malformed or
factually incorrect text, and reproduce undesirable patterns found in its
training corpus. It has not been instruction-tuned, aligned to human preferences,
or comprehensively evaluated for bias, toxicity, privacy leakage, or downstream
safety. Generated text should be treated as untrusted.

The reported validation result measures next-token prediction on the held-out
split created during preparation. It is not a benchmark of reasoning, factuality,
coding, or chat quality.

## Artifacts

Download `forge-98m.bin` and `tokenizer.json` from the
[latest GitHub release](https://github.com/TitanUranus67/forge-lm/releases/latest).

| Artifact | SHA-256 |
|---|---|
| `forge-98m.bin` | `563a112a7b8ab61e95e5ffe47968896362d59831532d7369ed008be015828481` |
| `tokenizer.json` | `a596ca7d27d2b5d61919db7e7eb2001e20bb38c224fb8936386a68833a0512ca` |

Generate a completion with:

```sh
dotnet run -c Release --project src/LLM.Cli -- generate \
  --backend auto \
  --model forge-98m.bin \
  --tokenizer tokenizer.json \
  --prompt "Once upon a time " \
  --tokens 200 --temperature 0.8 --topk 40
```

Generation stops at EOS or at the requested token safety limit.

## Data attribution

FineWeb-Edu is published by Hugging Face under ODC-By 1.0 and is derived from
Common Crawl. The model may reflect the limitations and content of those
sources. See the
[FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
and [Common Crawl terms of use](https://commoncrawl.org/terms-of-use) before
redistributing or deploying derived artifacts.
