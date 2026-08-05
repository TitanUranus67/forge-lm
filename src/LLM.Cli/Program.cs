using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using LLM.Core.Checkpoint;
using LLM.Core.Inference;
using LLM.Core.Model;
using LLM.Core.Tensor;
using LLM.Core.Tensor.Cuda;
using LLM.Core.Tensor.Gpu;
using LLM.Core.Tokenizer;
using LLM.Core.Training;

// LLM.Cli — Forge command line: prepare / train / generate.
// (train-tokenizer is folded into `prepare`.)

if (args.Length == 0 || args[0] is "--help" or "-h" or "help")
{
    Cli.PrintUsage();
    return args.Length == 0 ? 1 : 0;
}

try
{
    return args[0] switch
    {
        "prepare" => Cli.Prepare(args[1..]),
        "prepare-fineweb" => Cli.PrepareFineWeb(args[1..]),
        "train" => Cli.Train(args[1..]),
        "benchmark" => Cli.Benchmark(args[1..]),
        "generate" => Cli.Generate(args[1..]),
        "chat" => Cli.Chat(args[1..]),
        _ => Cli.Unknown(args[0]),
    };
}
catch (Exception ex) when (ex is not OperationCanceledException)
{
    Console.Error.WriteLine($"error: {ex.Message}\n{ex.StackTrace}");
    return 1;
}

internal static partial class Cli
{
    private const string DefaultCorpusUrl =
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

    internal static int Unknown(string cmd)
    {
        Console.Error.WriteLine($"unknown command '{cmd}'.");
        PrintUsage();
        return 1;
    }

    internal static void PrintUsage() => Console.WriteLine("""
        LLM.Cli — Forge from scratch in C#

        Usage:
          llm prepare   [--corpus <path-or-url>] --out <dir> [--merges 2000] [--tokenizer <path>]
          llm prepare-fineweb --out <dir> [--dataset fineweb-edu|fineweb] [--shards 10]
                              [--merges 16000] [--toktrainmb 200] [--rebuild true]
          llm train     --data <dir> [--preset forge-98m] [--steps 5000 | --tokens N | --epochs 1]
                        [--name Forge-98M] [--dmodel 128] [--layers 4] [--heads 4]
                        [--ctx 128] [--batch 8] [--accum 16] [--lr 6e-4] [--minlr 6e-5]
                        [--warmup 100 | --warmup-tokens N] [--wd 0.1]
                        [--gradclip 1.0] [--seed 42] [--logevery 10] [--valevery 250]
                        [--valbatches 50] [--valseed 424242]
                        [--saveevery 0] [--out out/model.bin] [--init <checkpoint>]
                        [--backend auto|cpu|gpu|cuda]
                        [--matmul-precision custom|fp32|tf32]
          llm benchmark [--backend cuda] [--batch 4] [--accum 16] [--steps 3]
                        [--vocab 16257] [--ctx 512] [--dmodel 768] [--layers 12] [--heads 12]
          llm generate  --model <checkpoint> --tokenizer <dir-or-path> [--prompt "Once upon a time"]
                        [--tokens 200] [--temperature 0.8] [--topk 40] [--seed 1] [--backend auto|cpu|gpu|cuda]
                        [--repetition-penalty 1.0] [--no-repeat-ngram 0]
                        [--matmul-precision custom|fp32|tf32]
          llm chat      --model <checkpoint> --tokenizer <dir-or-path>
                        [--tokens 100] [--temperature 0.8] [--topk 40] [--seed 1] [--backend auto|cpu|gpu|cuda]
                        [--repetition-penalty 1.0] [--no-repeat-ngram 0]
                        [--matmul-precision custom|fp32|tf32]

        There is no separate train-tokenizer command: tokenizer training is folded
        into `prepare` (use --tokenizer to supply a pre-trained one).
        --backend cuda runs on an NVIDIA GPU through ILGPU (Linux or Windows);
        --backend gpu selects Windows D3D12. The default auto mode prefers CUDA,
        then D3D12 on Windows, and always falls back to CPU.
        """);

    /// <summary>Creates the requested tensor backend and prints the actual selection.</summary>
    private static ITensorBackend CreateBackend(string name, CudaMatMulMode cudaMatMulMode = CudaMatMulMode.Custom)
    {
        BackendSelection.Choice<ITensorBackend> choice;
        try
        {
            choice = BackendSelection.Create<ITensorBackend>(name, OperatingSystem.IsWindows(), candidate => candidate switch
            {
                "cpu" => new CpuBackend(),
                "gpu" => new GpuBackend(),
                "cuda" => new CudaBackend(matMulMode: cudaMatMulMode),
                _ => throw new InvalidOperationException($"unexpected backend candidate '{candidate}'"),
            }, (candidate, ex) =>
            {
                if (candidate == "cuda" && CudaBackend.IsAvailable)
                    Console.Error.WriteLine($"warning: CUDA was detected but initialization failed; auto is falling back: {ex.Message}");
            });
        }
        catch (Exception ex) when (name.Equals("gpu", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"--backend gpu requested, but no D3D12 device is available: {ex.Message}", ex);
        }
        catch (Exception ex) when (name.Equals("cuda", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"--backend cuda requested, but no CUDA device is available: {ex.Message}", ex);
        }

        if (cudaMatMulMode is not CudaMatMulMode.Custom && choice.Value is not CudaBackend)
        {
            if (choice.Value is IDisposable disposable) disposable.Dispose();
            throw new ArgumentException("--matmul-precision fp32 or tf32 requires the CUDA backend.");
        }

        switch (choice.Value)
        {
            case CudaBackend cuda:
                Console.WriteLine($"backend: cuda ({cuda.DeviceName}, {cuda.DeviceMemoryBytes / 1e9:F1} GB; " +
                                  $"matmul {cuda.MatMulDescription})");
                break;
            case GpuBackend gpu:
                Console.WriteLine($"backend: gpu ({gpu.DeviceName}, {gpu.DeviceMemoryBytes / 1e9:F1} GB)");
                break;
            default:
                Console.WriteLine(choice.Name == "cpu" && name.Equals("auto", StringComparison.OrdinalIgnoreCase)
                    ? "backend: cpu (auto fallback)" : "backend: cpu");
                break;
        }
        return choice.Value;
    }

    private static CudaMatMulMode ParseCudaMatMulMode(string value) => value.ToLowerInvariant() switch
    {
        "custom" => CudaMatMulMode.Custom,
        "fp32" => CudaMatMulMode.CuBlasFp32,
        "tf32" => CudaMatMulMode.CuBlasTf32,
        _ => throw new ArgumentException(
            $"Invalid --matmul-precision '{value}'. Expected custom, fp32, or tf32."),
    };

    // ---- prepare -------------------------------------------------------------

    internal static int Prepare(string[] args)
    {
        var p = new Args(args);
        if (p.Help)
        {
            Console.WriteLine("""
                llm prepare [--corpus <path-or-url>] --out <dir> [--merges 2000] [--tokenizer <path>]

                  Downloads the corpus when --corpus is an http(s) URL (default:
                  tiny-shakespeare), trains (or loads) a byte-level BPE tokenizer,
                  encodes the corpus, and writes tokenizer.json, train.bin and
                  val.bin (90/10 split, raw little-endian uint16) into --out.
                  Tokenizer selection: --tokenizer path first, then an existing
                  tokenizer.json in --out, otherwise a fresh one is trained.
                """);
            return 0;
        }

        string corpus = p.Get("corpus", DefaultCorpusUrl);
        string outDir = p.Require("out");
        int merges = p.GetInt("merges", 2000);
        string? tokPath = p.Get("tokenizer");
        p.Done();

        const int maxMerges = 65536 - 256 - 1; // reserve one uint16 id for EOS
        if (merges < 0 || merges > maxMerges)
            throw new ArgumentException($"--merges must be in [0, {maxMerges}] (uint16 token ids cap vocab at 65536).");

        Directory.CreateDirectory(outDir);

        // 1. corpus bytes (download if URL)
        byte[] bytes;
        if (corpus.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            corpus.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            string local = Path.Combine(outDir, "corpus.txt");
            if (File.Exists(local))
            {
                Console.WriteLine($"corpus: reusing {local}");
            }
            else
            {
                Console.WriteLine($"corpus: downloading {corpus}");
                using var http = new HttpClient();
                bytes = http.GetByteArrayAsync(corpus).GetAwaiter().GetResult();
                File.WriteAllBytes(local, bytes);
                Console.WriteLine($"corpus: saved to {local}");
            }
            bytes = File.ReadAllBytes(local);
        }
        else
        {
            bytes = File.ReadAllBytes(corpus);
        }
        Console.WriteLine($"corpus: {bytes.Length:N0} bytes");

        // 2. tokenizer: explicit > reuse > train
        string tokOut = Path.Combine(outDir, "tokenizer.json");
        BpeTokenizer tok;
        if (tokPath is not null)
        {
            tok = BpeTokenizer.Load(tokPath);
            Console.WriteLine($"tokenizer: loaded {tokPath} (vocab {tok.VocabSize})");
            tok.Save(tokOut);
        }
        else if (File.Exists(tokOut))
        {
            tok = BpeTokenizer.Load(tokOut);
            Console.WriteLine($"tokenizer: reusing {tokOut} (vocab {tok.VocabSize})");
        }
        else
        {
            Console.WriteLine($"tokenizer: training {merges} merges...");
            var sw = Stopwatch.StartNew();
            tok = BpeTokenizer.Train(bytes, merges, (done, total) =>
            {
                if (done % 100 == 0 || done == total)
                    Console.WriteLine($"tokenizer: {done}/{total} merges ({sw.Elapsed.TotalSeconds:F1}s)");
            });
            tok.Save(tokOut);
            Console.WriteLine($"tokenizer: saved {tokOut} (vocab {tok.VocabSize})");
        }

        // 3. encode + split + write
        Console.WriteLine("encoding corpus...");
        var swEnc = Stopwatch.StartNew();
        int[] ids = tok.Encode(bytes);
        Console.WriteLine($"encoded {ids.Length:N0} text tokens in {swEnc.Elapsed.TotalSeconds:F1}s");

        int cut = (int)(ids.Length * 0.9);
        int trainTokens = cut;
        int valTokens = ids.Length - cut;
        int eos = tok.EosTokenId;
        var trainIds = new int[trainTokens + 1];
        ids.AsSpan(0, trainTokens).CopyTo(trainIds);
        trainIds[^1] = eos;
        var valIds = new int[valTokens + 1];
        ids.AsSpan(cut).CopyTo(valIds);
        valIds[^1] = eos;
        WriteIds(Path.Combine(outDir, "train.bin"), trainIds);
        WriteIds(Path.Combine(outDir, "val.bin"), valIds);
        trainTokens++;
        valTokens++;

        int totalTokens = trainTokens + valTokens;
        Console.WriteLine($"train.bin: {trainTokens:N0} tokens, val.bin: {valTokens:N0} tokens");
        Console.WriteLine($"stats: {bytes.Length:N0} bytes -> {totalTokens:N0} tokens, vocab {tok.VocabSize}, " +
                          $"compression {bytes.Length / (double)totalTokens:F2}x");
        return 0;
    }

    // ---- benchmark -----------------------------------------------------------

    internal static int Benchmark(string[] args)
    {
        var p = new Args(args);
        if (p.Help)
        {
            Console.WriteLine("""
                llm benchmark [options]

                  Runs one unmeasured warmup optimizer update followed by a short,
                  synthetic GPT training benchmark. No dataset or checkpoint is read
                  or written. Defaults model the full fresh-run architecture.

                  --backend cuda|gpu|cpu       backend (default cuda)
                  --matmul-precision custom|fp32|tf32
                  --vocab 16257 --ctx 512 --dmodel 768 --layers 12 --heads 12
                  --batch 4 --accum 16 --steps 3
                """);
            return 0;
        }

        int vocab = p.GetInt("vocab", 16257);
        int ctx = p.GetInt("ctx", 512);
        int dmodel = p.GetInt("dmodel", 768);
        int layers = p.GetInt("layers", 12);
        int heads = p.GetInt("heads", 12);
        int batch = p.GetInt("batch", 4);
        int accumulation = p.GetInt("accum", 16);
        int steps = p.GetInt("steps", 3);
        int seed = p.GetInt("seed", 42);
        string backendName = p.Get("backend", "cuda");
        CudaMatMulMode cudaMatMulMode = ParseCudaMatMulMode(p.Get("matmul-precision", "custom"));
        p.Done();

        if (vocab < 257 || vocab > ushort.MaxValue + 1) throw new ArgumentException("--vocab must be in [257, 65536].");
        if (batch < 1 || accumulation < 1 || steps < 1)
            throw new ArgumentException("--batch, --accum, and --steps must be >= 1.");
        var config = new ModelConfig(vocab, ctx, dmodel, layers, heads);
        ITensorBackend backend = CreateBackend(backendName, cudaMatMulMode);
        try
        {
            var model = new GptModel(config, backend, new Random(seed), "Forge-98M");
            long tokensPerUpdate = checked((long)batch * ctx * accumulation);
            long dataTokens = checked(tokensPerUpdate * (steps + 2) + 1);
            if (dataTokens > int.MaxValue)
                throw new ArgumentException("Benchmark data would exceed the in-memory limit.");
            var ids = new int[(int)dataTokens];
            for (int i = 0; i < ids.Length; i++)
                ids[i] = (int)((uint)(i * 1103515245 + 12345) % (uint)vocab);
            using var data = new DataLoader(ids);

            TrainOptions Options(int measuredSteps) => new()
            {
                Steps = measuredSteps,
                MaxLr = 1e-4f,
                MinLr = 1e-4f,
                WarmupSteps = 0,
                WeightDecay = 0.1f,
                GradClip = 1f,
                ContextLength = ctx,
                BatchSize = batch,
                AccumulationSteps = accumulation,
                Seed = seed,
                LogEvery = measuredSteps,
                ValEvery = 0,
            };

            Console.WriteLine($"benchmark: warmup 1 update; model {model.Params.Count:N0} params; " +
                              $"microbatch {batch} x {ctx}; accum {accumulation}; {tokensPerUpdate:N0} tok/update");
            Trainer.Train(model, data, val: null, Options(1));
            TrainSummary summary = Trainer.Train(model, data, val: null, Options(steps));
            long measuredTokens = checked(tokensPerUpdate * steps);
            Console.WriteLine($"benchmark: {measuredTokens:N0} tokens in {summary.Elapsed.TotalSeconds:F2}s = " +
                              $"{measuredTokens / summary.Elapsed.TotalSeconds:N0} tok/s; loss {summary.FinalTrainLoss:F4}");
            return 0;
        }
        finally
        {
            (backend as IDisposable)?.Dispose();
        }
    }

    // ---- train ---------------------------------------------------------------

    internal static int Train(string[] args)
    {
        var p = new Args(args);
        if (p.Help)
        {
            Console.WriteLine("""
                llm train --data <dir> [options]

                  Trains Forge on tokenizer.json + train.bin/val.bin produced by
                  `prepare`. --preset forge-98m selects 768/12/12/512, a safe
                  batch 4 / accum 16 starting point, the Forge-98M name, and
                  out/forge-98m.bin. Explicit flags can override preset defaults.
                  --epochs derives the token budget from train.bin.
                  Each physical pass processes --batch sequences (custom default 8,
                  Forge-98M preset 4); --accum physical passes are averaged into each
                  optimizer update (default 16).
                  --init resumes a current training checkpoint exactly (model, Adam,
                  global step, LR schedule, sampler, and input identities). Checkpoints
                  are SHA-256 verified and the previous save rotates to .bak.
                  Architecture flags are ignored
                  when loading. --saveevery N writes the checkpoint every N global
                  steps. --backend cuda runs on NVIDIA CUDA (including Linux);
                  --backend gpu runs on Windows D3D12, and auto chooses safely.
                  --matmul-precision custom keeps the original CUDA kernel; fp32
                  selects strict cuBLAS SGEMM; tf32 explicitly enables NVIDIA TF32
                  math and requires compute capability 8.0 or newer.
                  Ctrl+C stops after the current step and still saves.
                  --accum N averages N physical batches before each Adam/LR step.
                  --tokens, --epochs, and --warmup-tokens convert budgets into optimizer
                  update counts using batch * context * accumulation.
                """);
            return 0;
        }

        string dataDir = p.Require("data");
        string preset = p.Get("preset", "custom");
        bool forge98M = preset switch
        {
            "custom" => false,
            "forge-98m" => true,
            _ => throw new ArgumentException("--preset must be forge-98m or omitted."),
        };
        int? stepsArg = p.GetInt("steps");
        long? tokensArg = p.GetLong("tokens");
        float? epochsArg = p.GetFloat("epochs");
        int dmodel = p.GetInt("dmodel", forge98M ? 768 : 128);
        int layers = p.GetInt("layers", forge98M ? 12 : 4);
        int heads = p.GetInt("heads", forge98M ? 12 : 4);
        int ctx = p.GetInt("ctx", forge98M ? 512 : 128);
        int? batchArg = p.GetInt("batch");
        int? accumulationArg = p.GetInt("accum");
        float? lrArg = p.GetFloat("lr");
        float? minlrArg = p.GetFloat("minlr");
        int? warmupArg = p.GetInt("warmup");
        long? warmupTokensArg = p.GetLong("warmup-tokens");
        float? wdArg = p.GetFloat("wd");
        float? gradclipArg = p.GetFloat("gradclip");
        int seed = p.GetInt("seed", 42);
        int logevery = p.GetInt("logevery", 10);
        int valevery = p.GetInt("valevery", 250);
        int valbatches = p.GetInt("valbatches", 50);
        int valseed = p.GetInt("valseed", 424242);
        string modelName = p.Get("name", forge98M ? "Forge-98M" : "Forge");
        string outPath = p.Get("out", Path.Combine("out", forge98M ? "forge-98m.bin" : "model.bin"));
        string? init = p.Get("init");
        int saveevery = p.GetInt("saveevery", 0);
        string backendName = p.Get("backend", "auto");
        CudaMatMulMode cudaMatMulMode = ParseCudaMatMulMode(p.Get("matmul-precision", "custom"));
        p.Done();

        int budgetFlags = (stepsArg is not null ? 1 : 0) + (tokensArg is not null ? 1 : 0) +
                          (epochsArg is not null ? 1 : 0);
        if (budgetFlags > 1)
            throw new ArgumentException("Use only one of --steps, --tokens, or --epochs.");
        if (epochsArg is float epochs && (!float.IsFinite(epochs) || epochs <= 0f))
            throw new ArgumentException("--epochs must be finite and > 0.");
        if (warmupArg is not null && warmupTokensArg is not null)
            throw new ArgumentException("Use either --warmup or --warmup-tokens, not both.");

        ITensorBackend backend = CreateBackend(backendName, cudaMatMulMode);
        GptModel model;
        TrainingState? trainState = null;
        if (init is not null)
        {
            Checkpoint.LoadedTrainingCheckpoint loaded = Checkpoint.LoadTraining(init, backend);
            model = loaded.Model;
            trainState = loaded.TrainingState
                ?? throw new InvalidDataException("Training checkpoint did not contain resumable state.");
            string resumeDescription =
                $"training state at global step {trainState.GlobalStep:N0} (Adam step {trainState.Optimizer.StepCount:N0})";
            Console.WriteLine($"model: {model.Name} loaded from {init} ({resumeDescription}; " +
                              $"vocab {model.Config.VocabSize}, ctx {model.Config.ContextLength}, " +
                              $"dmodel {model.Config.DModel}, layers {model.Config.NLayers}, heads {model.Config.NHeads})");
        }
        else
        {
            var tok = BpeTokenizer.Load(Path.Combine(dataDir, "tokenizer.json"));
            var config = new ModelConfig(tok.VocabSize, ctx, dmodel, layers, heads);
            model = new GptModel(config, backend, new Random(seed), modelName);
            Console.WriteLine($"model: {model.Name} — vocab {config.VocabSize}, ctx {ctx}, dmodel {dmodel}, " +
                              $"layers {layers}, heads {heads}, params {model.Params.Count:N0}");
        }

        TrainingConfiguration? stored = trainState?.Configuration;
        int batch = batchArg ?? stored?.BatchSize ?? (forge98M ? 4 : 8);
        int accumulation = accumulationArg ?? stored?.AccumulationSteps ?? 16;
        if (batch < 1) throw new ArgumentException("--batch must be >= 1.");
        if (accumulation < 1) throw new ArgumentException("--accum must be >= 1.");
        long tokensPerUpdate = checked((long)batch * model.Config.ContextLength * accumulation);
        using var trainLoader = new DataLoader(Path.Combine(dataDir, "train.bin"));
        using var valLoader = new DataLoader(Path.Combine(dataDir, "val.bin"));
        Console.WriteLine($"data: {trainLoader.Length:N0} train tokens, {valLoader.Length:N0} val tokens");
        long? epochTokenBudget = epochsArg is float epochCount
            ? checked((long)Math.Ceiling((trainLoader.Length - 1) * (double)epochCount))
            : null;
        int steps = tokensArg is long tokenBudget
            ? UpdatesForTokens(tokenBudget, tokensPerUpdate, "--tokens")
            : epochTokenBudget is long epochTokens
                ? UpdatesForTokens(epochTokens, tokensPerUpdate, "--epochs")
            : stepsArg ?? stored?.TotalSteps ?? 5000;
        float lr = lrArg ?? stored?.MaxLr ?? 6e-4f;
        float minlr = minlrArg ?? stored?.MinLr ?? 6e-5f;
        int warmup = warmupTokensArg is long warmupTokenBudget
            ? UpdatesForTokens(warmupTokenBudget, tokensPerUpdate, "--warmup-tokens")
            : warmupArg ?? stored?.WarmupSteps ?? 100;
        float wd = wdArg ?? stored?.WeightDecay ?? 0.1f;
        float gradclip = gradclipArg ?? stored?.GradClip ?? 1.0f;

        long physicalBatchTokens = (long)batch * model.Config.ContextLength;
        Console.WriteLine($"training: microbatch {batch} x {model.Config.ContextLength} ctx " +
                          $"({physicalBatchTokens:N0} tokens), accumulation {accumulation} " +
                          $"({tokensPerUpdate:N0} tokens/optimizer update)");

        var opts = new TrainOptions
        {
            Steps = steps,
            MaxLr = lr,
            MinLr = minlr,
            WarmupSteps = warmup,
            WeightDecay = wd,
            GradClip = gradclip,
            ContextLength = model.Config.ContextLength,
            BatchSize = batch,
            AccumulationSteps = accumulation,
            Seed = seed,
            LogEvery = logevery,
            ValEvery = valevery,
            ValBatches = valbatches,
            ValSeed = valseed,
            SaveEvery = saveevery,
        };

        Console.WriteLine("data identity: hashing tokenizer.json, train.bin, and val.bin...");
        string tokenizerIdentity = ContentSha256(Path.Combine(dataDir, "tokenizer.json"));
        string dataIdentity = CombinedIdentity(tokenizerIdentity,
            ContentSha256(Path.Combine(dataDir, "train.bin")), ContentSha256(Path.Combine(dataDir, "val.bin")));
        trainState ??= TrainingState.CreateNew(backend, opts, 0, dataIdentity, tokenizerIdentity);
        trainState.RequireDataIdentity(dataIdentity, tokenizerIdentity);

        var display = new TrainDisplay(steps, checked((int)tokensPerUpdate), trainState.GlobalStep);

        string? outDir = Path.GetDirectoryName(Path.GetFullPath(outPath));
        if (outDir is not null) Directory.CreateDirectory(outDir);

        // write to a temp file then move, so an interrupt mid-save can't corrupt the last good checkpoint
        void SaveCheckpoint(GptModel m, TrainingState state, string tag)
        {
            string tmp = outPath + ".tmp";
            Checkpoint.SaveTraining(m, state, tmp);
            Checkpoint.PublishAtomically(tmp, outPath, outPath + ".bak");
            display.PrintLine($"checkpoint: saved {outPath} ({tag})");
        }

        // Ctrl+C cancels after the current step; the checkpoint below is still written
        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (s, e) =>
        {
            e.Cancel = true;
            if (!cts.IsCancellationRequested)
            {
                Console.WriteLine("\ninterrupt: finishing current step, then saving...");
                cts.Cancel();
            }
        };

        // polled once per step by the trainer; 'p' pauses, then r/s/q resumes/saves/quits.
        // KeyAvailable throws when stdin is redirected, so guard with IsInputRedirected.
        bool stoppedByUser = false;
        TrainCommand ControlHook(int step)
        {
            if (Console.IsInputRedirected || !Console.KeyAvailable) return TrainCommand.Continue;
            if (char.ToLowerInvariant(Console.ReadKey(intercept: true).KeyChar) != 'p')
                return TrainCommand.Continue;

            display.PrintLine("paused — [r]esume, [s]ave+resume, [q]save+quit");
            while (true)
            {
                switch (char.ToLowerInvariant(Console.ReadKey(intercept: true).KeyChar))
                {
                    case 'r': return TrainCommand.Continue;
                    case 's':
                        SaveCheckpoint(model, trainState, "manual save");
                        return TrainCommand.Continue;
                    case 'q':
                        stoppedByUser = true;
                        return TrainCommand.SaveAndQuit;
                }
            }
        }

        Console.WriteLine("controls: p = pause, Ctrl+C = save and quit");
        TrainSummary summary = Trainer.Train(model, trainLoader, valLoader, opts, display.OnLog,
            onSave: saveevery > 0 ? (m, s) => SaveCheckpoint(m, s, $"step {s.GlobalStep}") : null,
            cancel: cts.Token, controlHook: ControlHook, state: trainState);

        display.Complete();
        if (stoppedByUser)
            Console.WriteLine($"stopped by user at step {summary.Steps}");
        string interrupted = !stoppedByUser && summary.Steps < steps
            ? $" (interrupted at step {summary.Steps}/{steps})" : "";
        Console.WriteLine($"done: global step {summary.Steps} after {summary.Elapsed:h\\:mm\\:ss} this run{interrupted}, " +
                          $"final train loss {summary.FinalTrainLoss:F4}" +
                          (summary.FinalValLoss.HasValue ? $", final val loss {summary.FinalValLoss.Value:F4}" : ""));
        SaveCheckpoint(model, trainState, "final");
        return 0;
    }

    // ---- generate ------------------------------------------------------------

    internal static int Generate(string[] args)
    {
        var p = new Args(args);
        if (p.Help)
        {
            Console.WriteLine("""
                llm generate --model <checkpoint> --tokenizer <dir-or-path> [options]

                  Loads a checkpoint and tokenizer and samples text autoregressively.
                  --tokenizer may be a tokenizer.json file or the directory holding it.
                  An empty --prompt starts from a single newline token.
                  --repetition-penalty values above 1 demote tokens already in context.
                  --no-repeat-ngram N prevents repeating an N-token sequence; 0 disables it.
                """);
            return 0;
        }

        string modelPath = p.Require("model");
        string tokArg = p.Require("tokenizer");
        string prompt = p.Get("prompt", "Once upon a time");
        int tokens = p.GetInt("tokens", 200);
        float temperature = p.GetFloat("temperature", 0.8f);
        int topk = p.GetInt("topk", 40);
        float repetitionPenalty = p.GetFloat("repetition-penalty", 1f);
        int noRepeatNgram = p.GetInt("no-repeat-ngram", 0);
        int seed = p.GetInt("seed", 1);
        string backendName = p.Get("backend", "auto");
        CudaMatMulMode cudaMatMulMode = ParseCudaMatMulMode(p.Get("matmul-precision", "custom"));
        p.Done();

        string tokPath = Directory.Exists(tokArg) ? Path.Combine(tokArg, "tokenizer.json") : tokArg;
        var tok = BpeTokenizer.Load(tokPath);
        Checkpoint.LoadedTrainingCheckpoint loaded = Checkpoint.LoadWithMetadata(
            modelPath, CreateBackend(backendName, cudaMatMulMode));
        string tokenizerIdentity = ContentSha256(tokPath);
        if (loaded.TokenizerIdentity != tokenizerIdentity)
            throw new InvalidDataException("Tokenizer does not match the tokenizer recorded in the checkpoint.");
        GptModel model = loaded.Model;
        Console.WriteLine($"model: {model.Name} — {modelPath} (vocab {model.Config.VocabSize}, ctx {model.Config.ContextLength}, " +
                          $"params {model.Params.Count:N0})");

        int[] promptIds = prompt.Length > 0 ? tok.Encode(prompt) : [tok.EosTokenId];
        var rng = new Random(seed);

        Console.WriteLine("---");
        Console.Write(prompt);
        var sw = Stopwatch.StartNew();
        int count = 0;
        BpeTokenizer.Utf8StreamDecoder decoder = tok.CreateUtf8StreamDecoder();
        foreach (int id in Sampler.Generate(model, promptIds, tokens, temperature, topk, rng,
                     eosId: tok.EosTokenId, repetitionPenalty: repetitionPenalty,
                     noRepeatNgramSize: noRepeatNgram))
        {
            Console.Write(decoder.DecodeToken(id));
            count++;
        }
        Console.Write(decoder.Flush());
        sw.Stop();
        Console.WriteLine();
        Console.WriteLine("---");
        Console.WriteLine($"{count} tokens in {sw.Elapsed.TotalSeconds:F1}s " +
                          $"({count / Math.Max(sw.Elapsed.TotalSeconds, 1e-9):F1} tok/s)");
        return 0;
    }

    // ---- chat ----------------------------------------------------------------

    internal static int Chat(string[] args)
    {
        var p = new Args(args);
        if (p.Help)
        {
            Console.WriteLine("""
                llm chat --model <checkpoint> --tokenizer <dir-or-path> [options]

                  Interactive REPL over a checkpoint: each line you type is appended to
                  the rolling context and the model continues it. This is a base
                  language model (not instruction-tuned) — it continues text in its
                  trained style rather than answering questions.

                  In-chat commands: /reset (clear context), /quit (exit).
                  An empty line just lets the model continue from the current context.
                  --repetition-penalty values above 1 demote tokens already in context.
                  --no-repeat-ngram N prevents repeating an N-token sequence; 0 disables it.
                """);
            return 0;
        }

        string modelPath = p.Require("model");
        string tokArg = p.Require("tokenizer");
        int tokensPerTurn = p.GetInt("tokens", 100);
        float temperature = p.GetFloat("temperature", 0.8f);
        int topk = p.GetInt("topk", 40);
        float repetitionPenalty = p.GetFloat("repetition-penalty", 1f);
        int noRepeatNgram = p.GetInt("no-repeat-ngram", 0);
        int seed = p.GetInt("seed", 1);
        string backendName = p.Get("backend", "auto");
        CudaMatMulMode cudaMatMulMode = ParseCudaMatMulMode(p.Get("matmul-precision", "custom"));
        p.Done();

        string tokPath = Directory.Exists(tokArg) ? Path.Combine(tokArg, "tokenizer.json") : tokArg;
        var tok = BpeTokenizer.Load(tokPath);
        Checkpoint.LoadedTrainingCheckpoint loaded = Checkpoint.LoadWithMetadata(
            modelPath, CreateBackend(backendName, cudaMatMulMode));
        string tokenizerIdentity = ContentSha256(tokPath);
        if (loaded.TokenizerIdentity != tokenizerIdentity)
            throw new InvalidDataException("Tokenizer does not match the tokenizer recorded in the checkpoint.");
        GptModel model = loaded.Model;
        var rng = new Random(seed);

        Console.WriteLine($"model: {model.Name} — {modelPath} (vocab {model.Config.VocabSize}, ctx {model.Config.ContextLength}, " +
                          $"params {model.Params.Count:N0})");
        Console.WriteLine("chat — type a line and the model continues it. /reset clears context, /quit exits.");

        var history = new List<int> { tok.EosTokenId };
        while (true)
        {
            Console.Write("\nyou> ");
            string? line = Console.ReadLine();
            if (line is null or "/quit") break;
            if (line is "/reset")
            {
                history.Clear();
                history.Add(tok.EosTokenId);
                Console.WriteLine("(context cleared)");
                continue;
            }
            if (line.Length > 0)
                history.AddRange(tok.Encode(line + "\n"));

            // keep the prompt bounded; Generate truncates to the context window anyway
            int keep = model.Config.ContextLength * 2;
            if (history.Count > keep)
                history.RemoveRange(0, history.Count - keep);

            Console.Write("llm> ");
            BpeTokenizer.Utf8StreamDecoder decoder = tok.CreateUtf8StreamDecoder();
            foreach (int id in Sampler.Generate(model, history, tokensPerTurn, temperature, topk, rng,
                         eosId: tok.EosTokenId, repetitionPenalty: repetitionPenalty,
                         noRepeatNgramSize: noRepeatNgram))
            {
                Console.Write(decoder.DecodeToken(id));
                history.Add(id);
            }
            Console.Write(decoder.Flush());
            Console.WriteLine();
        }
        return 0;
    }

    // ---- helpers -------------------------------------------------------------

    private static int UpdatesForTokens(long tokens, long tokensPerUpdate, string flag)
    {
        if (tokens <= 0) throw new ArgumentException($"{flag} must be positive.");
        long updates = checked((tokens + tokensPerUpdate - 1) / tokensPerUpdate);
        if (updates > int.MaxValue)
            throw new ArgumentException($"{flag} requires {updates:N0} optimizer updates, above the supported limit.");
        return (int)updates;
    }

    private static string ContentSha256(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            8 << 20, FileOptions.SequentialScan);
        return Convert.ToHexString(SHA256.HashData(stream));
    }

    private static string CombinedIdentity(params string[] components) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(string.Join('|', components))));

    private static void WriteIds(string path, ReadOnlySpan<int> ids)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);
        foreach (int id in ids)
            bw.Write((ushort)id);
    }

    /// <summary>Minimal --flag value parser (also accepts --flag=value).</summary>
    private sealed class Args
    {
        private readonly Dictionary<string, string> _flags = new(StringComparer.OrdinalIgnoreCase);

        public Args(string[] argv)
        {
            for (int i = 0; i < argv.Length; i++)
            {
                string a = argv[i];
                if (a is "--help" or "-h") { Help = true; continue; }
                if (!a.StartsWith("--", StringComparison.Ordinal))
                    throw new ArgumentException($"unexpected argument '{a}' (expected --flag value)");
                string key, value;
                int eq = a.IndexOf('=', StringComparison.Ordinal);
                if (eq >= 0) { key = a[2..eq]; value = a[(eq + 1)..]; }
                else
                {
                    key = a[2..];
                    if (i + 1 >= argv.Length) throw new ArgumentException($"flag --{key} needs a value");
                    value = argv[++i];
                }
                _flags[key] = value;
            }
        }

        public bool Help { get; }

        private string? Take(string key) =>
            _flags.Remove(key, out string? v) ? v : null;

        public string Get(string key, string fallback) => Take(key) ?? fallback;

        public string? Get(string key) => Take(key);

        public string Require(string key) =>
            Take(key) ?? throw new ArgumentException($"missing required flag --{key}");

        public int GetInt(string key, int fallback) =>
            Take(key) is string v ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;

        public int? GetInt(string key) =>
            Take(key) is string v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

        public long? GetLong(string key) =>
            Take(key) is string v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

        public float GetFloat(string key, float fallback) =>
            Take(key) is string v ? float.Parse(v, CultureInfo.InvariantCulture) : fallback;

        public float? GetFloat(string key) =>
            Take(key) is string v ? float.Parse(v, CultureInfo.InvariantCulture) : null;

        public bool GetBool(string key, bool fallback) =>
            Take(key) is string v ? bool.Parse(v) : fallback;

        public void Done()
        {
            if (Help) return;
            if (_flags.Count > 0)
                throw new ArgumentException($"unknown flag(s): {string.Join(", ", _flags.Keys.Select(k => "--" + k))}");
        }
    }
}
