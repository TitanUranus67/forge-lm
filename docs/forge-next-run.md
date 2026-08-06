# Forge next-run plan

No production run should start from this document alone. The model choice, GPU,
measured throughput, total token budget, and projected cost must be reviewed first.

## Candidate models

| Preset | Shape | Parameters at vocab 16,257 | Context | Planning budget |
|---|---:|---:|---:|---:|
| `forge-220m` | 16 layers, width 1024, 16 heads | 219,237,376 | 1024 | 4.4B tokens |
| `forge-320m` | 24 layers, width 1024, 16 heads | 320,007,168 | 1024 | 6.4B tokens |

The planning budgets use 20 tokens per parameter. They are cost-estimation targets,
not CLI defaults. `train --preset ...` still requires an explicit `--tokens`,
`--epochs`, or `--steps` decision for a production run.

Start with Forge-220M unless Forge-320M's measured total cost and wall time are
comfortably acceptable. The 320M candidate is not automatically better if its larger
activation footprint forces an inefficient microbatch.

## Benchmark gate

Publish the current Release build to the candidate GPU, then benchmark each
preset and geometry directly. For example:

```bash
./app/forge benchmark --backend cuda --preset forge-220m \
  --matmul-precision custom --batch 1 --accum 32 --steps 3

./app/forge benchmark --backend cuda --preset forge-320m \
  --matmul-precision custom --batch 1 --accum 32 --steps 3
```

Repeat with `--matmul-precision fp32` and every batch/accumulation geometry that
retains 32,768 tokens per optimizer update. Use at least three measured updates
after the warmup; raise `--steps` to 5 if results are close.

For every surviving row record:

- steady tokens/second, GPU model, VRAM, and hourly price;
- projected hours = planning token budget / tokens per second / 3600;
- projected cost = projected hours times hourly price;
- whether loss remains finite and repeatable over a short training smoke test.

Choose the fastest stable geometry for each model, then choose the model only after
comparing complete-run time and cost. Do not select from dashboard utilization alone.

## Curated web mixture

The first supported mixture is intentionally conservative: 80% FineWeb-Edu for
educational quality and 20% unfiltered FineWeb for broader styles and topics. Both
sources must use the exact tokenizer trained on the FineWeb-Edu source.

Example preparation sequence (adjust shard counts only after checking available token
counts against the requested shares):

```bash
./app/forge prepare-fineweb \
  --out data/forge-next/fineweb-edu --dataset fineweb-edu \
  --shards 10 --merges 16000 --encode-workers 8

./app/forge prepare-fineweb \
  --out data/forge-next/fineweb --dataset fineweb \
  --shards 3 --tokenizer data/forge-next/fineweb-edu/tokenizer.json \
  --exclude-index data/forge-next/fineweb-edu/corpus.idx \
  --encode-workers 8

./app/forge prepare-mixture \
  --manifest docs/forge-next-mixture.example.json \
  --out data/forge-next/mixed
```

The breadth preparation excludes every stable document identity already present in
the Edu corpus; this matters because FineWeb-Edu is derived from FineWeb. The exclusion
index identity is part of the corpus manifest, so a changed or omitted exclusion list
cannot silently reuse stale artifacts.

`prepare-mixture` verifies byte-identical tokenizers, checks that each source contains
enough tokens for its weighted share, interleaves complete EOS-terminated documents,
publishes outputs transactionally, and records source sizes and actual token totals in
`.forge-mixture.json`. The final document may take an output slightly above its target.

Before paying for training, inspect random decoded documents from both source datasets.
The current mixer makes source balance reproducible; it does not make an unreviewed
source trustworthy. Wikipedia, books, code, or conversation data should be added only
through separately reviewed ingestion adapters and licensing checks, not silently
folded into this first mixture.

## Final preflight

1. Run the complete CPU, CUDA, and D3D12 test suite where those devices are available.
2. Verify mixture manifests and record tokenizer/train/validation SHA-256 identities.
3. Run a short fresh-model smoke test through warmup, validation, save, and exact resume.
4. Generate a fixed prompt-and-seed evaluation with raw decoding controls disabled,
   then repeat with documented repetition controls for a separate user-facing check.
5. Review the benchmark table and approve the model, GPU, token budget, projected time,
   and projected cost before launching the full run.
