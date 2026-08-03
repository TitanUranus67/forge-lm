# Forge training data

Prepared on 2026-08-03 from Hugging Face FineWeb-Edu, configuration
`sample/10BT`:

- Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Shard listing API: https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT
- Shard 1: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet
- Shard 2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/001_00000.parquet
- Shard 3: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/002_00000.parquet

Preparation command:

```powershell
dotnet run -c Release --no-build --project src/LLM.Cli -- prepare-fineweb `
    --out data/forge `
    --shards 3 `
    --merges 16000
```

Preparation details:

- Dataset selection: `fineweb-edu` (the command default)
- Source documents: 2,182,000
- Extracted corpus: 10,424,324,787 bytes
- Tokenizer training sample: 43,643 training documents, approximately 197.1 MB
- Vocabulary: byte-level BPE with 16,000 learned merges plus EOS
- Validation split: stable document-level split keyed by source URL
- Encoding: EOS is appended to every document; merges never cross document boundaries
- Intended Forge-98M training budget: 1,024,000,000 tokens

The generated `.fineweb-*.json` manifests record exact shard identities,
preparation settings, tokenizer identity, and encoded output sizes. Preserve those
manifests with `tokenizer.json`, `train.bin`, and `val.bin` when archiving or moving
this dataset.
