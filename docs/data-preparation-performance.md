# Data preparation performance

## Status

Implemented after measuring the first Forge preparation. Encoding now processes a
bounded batch of documents concurrently and writes encoded results in original
document order. `--encode-workers` controls parallelism; the automatic default is
the smaller of eight and the available logical-processor count.

## Measured baseline - 2026-08-03

FineWeb-Edu preparation used three `sample/10BT` shards:

- Corpus: 2,182,000 documents and 10,424,324,787 bytes.
- Host: AMD Ryzen 7 5800XT, 8 physical cores / 16 logical processors.
- Encoding throughput: approximately 3.5-3.7 MB/s.
- Live process use: approximately 99% of one logical processor and 609 MB RAM.
- Disk read rate: approximately 3.4 MB/s, showing that storage was waiting on the
  encoder rather than limiting it.
- Estimated sequential encoding time: approximately 48-52 minutes.
- Tokenizer training is a separate single-threaded phase; its 16,000 merges over a
  197.1 MB training-only sample took 55.5 minutes.

The original `StreamEncode` loop read and tokenized one document at a time. The
byte-level BPE call was the encoding bottleneck.

## Deterministic parallel document encoding

The implementation parallelizes document tokenization while preserving byte-for-byte output:

1. Read a bounded batch of index entries and document bytes in their existing order.
2. Run `BpeTokenizer.Encode` over the batch with configurable parallelism; documents
   are independent because merges cannot cross boundaries.
3. Write the completed batch sequentially to `train.bin` or `val.bin`, preserving
   exact original order within both outputs, and append EOS to every document.
4. Return pooled input buffers after each batch so memory use cannot scale with
   corpus size.
5. Propagate failures and cancellation, close temporary outputs, and retain the
   existing transactional publication behavior.

## Correctness and performance gates

- Sequential and parallel preparation must produce byte-identical `train.bin` and
  `val.bin`, identical token/document counts, and identical manifests.
- Test out-of-order worker completion, validation interleaving, EOS placement,
  very small and unusually large documents, cancellation, and worker exceptions.
- Demonstrate bounded memory use over a sustained corpus sample.
- Benchmark multiple worker counts rather than assuming every logical processor is
  beneficial.
- Real-corpus benchmark results on the Ryzen 7 5800XT were 10.0 MB/s with four
  workers, 12.9-13.3 MB/s with eight, and 11.9-12.0 MB/s with sixteen, versus the
  3.5-3.7 MB/s sequential baseline. Eight workers won at about 3.6x baseline and
  should encode this corpus in roughly 13-14 minutes.

Parallelizing tokenizer merge training is a distinct, more difficult optimization
because merge selection depends on global pair counts after every iteration. Treat
that as a separate project only if datasets will be rebuilt often enough to justify
the complexity.
