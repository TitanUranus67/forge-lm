# Data preparation performance

## Status

Deferred until after the first Forge dataset is complete. The current preparation
was already well underway when the bottleneck was measured, so restarting it would
cost more time than finishing the one-time job.

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

The current `StreamEncode` loop reads and tokenizes one document at a time. The
byte-level BPE call is the encoding bottleneck.

## Future improvement: deterministic parallel document encoding

Parallelize document tokenization while preserving byte-for-byte output:

1. Read document-index entries in their existing order and assign a monotonically
   increasing sequence number to each document.
2. Feed bounded work items to a configurable number of encoding workers. Each item
   contains the document bytes, validation flag, and sequence number.
3. Have workers run `BpeTokenizer.Encode` independently and append EOS to their
   result; documents are independent because merges cannot cross boundaries.
4. Feed results to one ordered writer that waits for the next sequence number and
   writes to `train.bin` or `val.bin` exactly as the sequential implementation does.
5. Bound both queues and returned results so memory use cannot scale with corpus
   size or a single slow document.
6. Propagate worker failures and cancellation promptly, close temporary outputs,
   and retain the existing transactional publication behavior.
7. Add an `--encode-workers` option, with `1` retaining the reference sequential
   path and an automatic default based on available processors.

## Correctness and performance gates

- Sequential and parallel preparation must produce byte-identical `train.bin` and
  `val.bin`, identical token/document counts, and identical manifests.
- Test out-of-order worker completion, validation interleaving, EOS placement,
  very small and unusually large documents, cancellation, and worker exceptions.
- Demonstrate bounded memory use over a sustained corpus sample.
- Benchmark multiple worker counts rather than assuming every logical processor is
  beneficial.
- Target at least 4x throughput on the Ryzen 7 5800XT. A practical 15-25 MB/s would
  reduce this corpus's encoding phase to roughly 7-12 minutes.

Parallelizing tokenizer merge training is a distinct, more difficult optimization
because merge selection depends on global pair counts after every iteration. Treat
that as a separate project only if datasets will be rebuilt often enough to justify
the complexity.
