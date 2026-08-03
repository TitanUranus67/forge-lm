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

// LLM.Cli — mini-GPT command line: prepare / train / generate.
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
        LLM.Cli — mini-GPT from scratch in C#

        Usage:
          llm prepare   [--corpus <path-or-url>] --out <dir> [--merges 2000] [--tokenizer <path>]
          llm prepare-fineweb --out <dir> [--shards 10] [--merges 16000] [--toktrainmb 200] [--rebuild true]
          llm train     --data <dir> [--steps 5000 | --tokens N] [--dmodel 128] [--layers 4] [--heads 4]
                        [--ctx 128] [--batch 8] [--accum 16] [--lr 6e-4] [--minlr 6e-5]
                        [--warmup 100 | --warmup-tokens N] [--wd 0.1]
                        [--gradclip 1.0] [--seed 42] [--logevery 10] [--valevery 250]
                        [--valbatches 50] [--valseed 424242]
                        [--saveevery 0] [--out out/model.bin] [--init <checkpoint>]
                        [--resume-step N] [--backend auto|cpu|gpu|cuda]
                        [--matmul-precision custom|fp32|tf32]
          llm generate  --model <checkpoint> --tokenizer <dir-or-path> [--prompt "Once upon a time"]
                        [--tokens 200] [--temperature 0.8] [--topk 40] [--seed 1] [--backend auto|cpu|gpu|cuda]
                        [--matmul-precision custom|fp32|tf32]
          llm chat      --model <checkpoint> --tokenizer <dir-or-path>
                        [--tokens 100] [--temperature 0.8] [--topk 40] [--seed 1] [--backend auto|cpu|gpu|cuda]
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

        const int maxMerges = 65536 - 256; // uint16 token files cap the vocab at 65536
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
        Console.WriteLine($"encoded {ids.Length:N0} tokens in {swEnc.Elapsed.TotalSeconds:F1}s");

        int cut = (int)(ids.Length * 0.9);
        WriteIds(Path.Combine(outDir, "train.bin"), ids.AsSpan(0, cut));
        WriteIds(Path.Combine(outDir, "val.bin"), ids.AsSpan(cut));

        Console.WriteLine($"train.bin: {cut:N0} tokens, val.bin: {ids.Length - cut:N0} tokens");
        Console.WriteLine($"stats: {bytes.Length:N0} bytes -> {ids.Length:N0} tokens, vocab {tok.VocabSize}, " +
                          $"compression {bytes.Length / (double)ids.Length:F2}x");
        return 0;
    }

    // ---- train ---------------------------------------------------------------

    internal static int Train(string[] args)
    {
        var p = new Args(args);
        if (p.Help)
        {
            Console.WriteLine("""
                llm train --data <dir> [options]

                  Trains a GPT on the tokenizer.json + train.bin/val.bin produced by
                  `prepare`, and writes a checkpoint to --out (default out/model.bin).
                  Each physical pass processes --batch sequences (default 8); --accum
                  physical passes are averaged into each optimizer update (default 16).
                  --init resumes a V2/V3 training checkpoint exactly (model, Adam,
                  global step, LR schedule, and sampler RNG). V3 also verifies a
                  SHA-256 checksum plus tokenizer/training-data identity and rotates
                  the previous save to .bak. V1 checkpoints contain weights
                  only; --resume-step N can supply their known cumulative scheduler
                  position during the one-time upgrade. Architecture flags are ignored
                  when loading. --saveevery N writes the checkpoint every N global
                  steps. --backend cuda runs on NVIDIA CUDA (including Linux);
                  --backend gpu runs on Windows D3D12, and auto chooses safely.
                  --matmul-precision custom keeps the original CUDA kernel; fp32
                  selects strict cuBLAS SGEMM; tf32 explicitly enables NVIDIA TF32
                  math and requires compute capability 8.0 or newer.
                  Ctrl+C stops after the current step and still saves.
                  --accum N averages N physical batches before each Adam/LR step.
                  --tokens and --warmup-tokens convert token budgets into optimizer
                  update counts using batch * context * accumulation.
                """);
            return 0;
        }

        string dataDir = p.Require("data");
        int? stepsArg = p.GetInt("steps");
        long? tokensArg = p.GetLong("tokens");
        int dmodel = p.GetInt("dmodel", 128);
        int layers = p.GetInt("layers", 4);
        int heads = p.GetInt("heads", 4);
        int ctx = p.GetInt("ctx", 128);
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
        string outPath = p.Get("out", Path.Combine("out", "model.bin"));
        string? init = p.Get("init");
        int resumeStep = p.GetInt("resume-step", 0);
        int saveevery = p.GetInt("saveevery", 0);
        string backendName = p.Get("backend", "auto");
        CudaMatMulMode cudaMatMulMode = ParseCudaMatMulMode(p.Get("matmul-precision", "custom"));
        p.Done();

        if (stepsArg is not null && tokensArg is not null)
            throw new ArgumentException("Use either --steps or --tokens, not both.");
        if (warmupArg is not null && warmupTokensArg is not null)
            throw new ArgumentException("Use either --warmup or --warmup-tokens, not both.");

        ITensorBackend backend = CreateBackend(backendName, cudaMatMulMode);
        GptModel model;
        TrainingState? trainState = null;
        if (init is not null)
        {
            Checkpoint.LoadedTrainingCheckpoint loaded = Checkpoint.LoadTraining(init, backend);
            model = loaded.Model;
            trainState = loaded.TrainingState;
            string resumeDescription = trainState is null
                ? "legacy weights-only checkpoint"
                : $"training state at global step {trainState.GlobalStep:N0} (Adam step {trainState.Optimizer.StepCount:N0})";
            Console.WriteLine($"model: loaded {init} ({resumeDescription}; " +
                              $"vocab {model.Config.VocabSize}, ctx {model.Config.ContextLength}, " +
                              $"dmodel {model.Config.DModel}, layers {model.Config.NLayers}, heads {model.Config.NHeads})");
            if (trainState is null)
                Console.WriteLine("warning: this V1 checkpoint has no Adam/RNG/scheduler state; " +
                                  "this restart must initialize those once before future resumes become exact.");
        }
        else
        {
            var tok = BpeTokenizer.Load(Path.Combine(dataDir, "tokenizer.json"));
            var config = new ModelConfig(tok.VocabSize, ctx, dmodel, layers, heads);
            model = new GptModel(config, backend, new Random(seed));
            Console.WriteLine($"model: vocab {config.VocabSize}, ctx {ctx}, dmodel {dmodel}, " +
                              $"layers {layers}, heads {heads}, params {model.Params.Count:N0}");
        }

        if (trainState is not null && resumeStep != 0)
            throw new ArgumentException("--resume-step is only valid when upgrading a legacy V1 weights-only checkpoint.");
        if (init is null && resumeStep != 0)
            throw new ArgumentException("--resume-step requires --init with a legacy V1 weights-only checkpoint.");

        TrainingConfiguration? stored = trainState?.Configuration;
        int batch = batchArg ?? stored?.BatchSize ?? 8;
        int accumulation = accumulationArg ?? stored?.AccumulationSteps ?? 16;
        if (batch < 1) throw new ArgumentException("--batch must be >= 1.");
        if (accumulation < 1) throw new ArgumentException("--accum must be >= 1.");
        long tokensPerUpdate = checked((long)batch * model.Config.ContextLength * accumulation);
        int steps = tokensArg is long tokenBudget
            ? UpdatesForTokens(tokenBudget, tokensPerUpdate, "--tokens")
            : stepsArg ?? stored?.TotalSteps ?? 5000;
        float lr = lrArg ?? stored?.MaxLr ?? 6e-4f;
        float minlr = minlrArg ?? stored?.MinLr ?? 6e-5f;
        int warmup = warmupTokensArg is long warmupTokenBudget
            ? UpdatesForTokens(warmupTokenBudget, tokensPerUpdate, "--warmup-tokens")
            : warmupArg ?? stored?.WarmupSteps ?? 100;
        float wd = wdArg ?? stored?.WeightDecay ?? 0.1f;
        float gradclip = gradclipArg ?? stored?.GradClip ?? 1.0f;

        var trainLoader = new DataLoader(Path.Combine(dataDir, "train.bin"));
        var valLoader = new DataLoader(Path.Combine(dataDir, "val.bin"));
        Console.WriteLine($"data: {trainLoader.Length:N0} train tokens, {valLoader.Length:N0} val tokens");
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
        trainState ??= TrainingState.CreateNew(backend, opts, resumeStep, dataIdentity, tokenizerIdentity);
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
                """);
            return 0;
        }

        string modelPath = p.Require("model");
        string tokArg = p.Require("tokenizer");
        string prompt = p.Get("prompt", "Once upon a time");
        int tokens = p.GetInt("tokens", 200);
        float temperature = p.GetFloat("temperature", 0.8f);
        int topk = p.GetInt("topk", 40);
        int seed = p.GetInt("seed", 1);
        string backendName = p.Get("backend", "auto");
        CudaMatMulMode cudaMatMulMode = ParseCudaMatMulMode(p.Get("matmul-precision", "custom"));
        p.Done();

        string tokPath = Directory.Exists(tokArg) ? Path.Combine(tokArg, "tokenizer.json") : tokArg;
        var tok = BpeTokenizer.Load(tokPath);
        Checkpoint.LoadedTrainingCheckpoint loaded = Checkpoint.LoadWithMetadata(
            modelPath, CreateBackend(backendName, cudaMatMulMode));
        string tokenizerIdentity = ContentSha256(tokPath);
        if (loaded.TokenizerIdentity is not null && loaded.TokenizerIdentity != tokenizerIdentity)
            throw new InvalidDataException("Tokenizer does not match the tokenizer recorded in the checkpoint.");
        if (loaded.TokenizerIdentity is null)
            Console.WriteLine("warning: legacy checkpoint has no tokenizer identity to verify.");
        GptModel model = loaded.Model;
        Console.WriteLine($"model: {modelPath} (vocab {model.Config.VocabSize}, ctx {model.Config.ContextLength}, " +
                          $"params {model.Params.Count:N0})");

        int[] promptIds = prompt.Length > 0 ? tok.Encode(prompt) : tok.Encode("\n");
        var rng = new Random(seed);

        Console.WriteLine("---");
        Console.Write(prompt);
        var sw = Stopwatch.StartNew();
        int count = 0;
        BpeTokenizer.Utf8StreamDecoder decoder = tok.CreateUtf8StreamDecoder();
        foreach (int id in Sampler.Generate(model, promptIds, tokens, temperature, topk, rng))
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
                """);
            return 0;
        }

        string modelPath = p.Require("model");
        string tokArg = p.Require("tokenizer");
        int tokensPerTurn = p.GetInt("tokens", 100);
        float temperature = p.GetFloat("temperature", 0.8f);
        int topk = p.GetInt("topk", 40);
        int seed = p.GetInt("seed", 1);
        string backendName = p.Get("backend", "auto");
        CudaMatMulMode cudaMatMulMode = ParseCudaMatMulMode(p.Get("matmul-precision", "custom"));
        p.Done();

        string tokPath = Directory.Exists(tokArg) ? Path.Combine(tokArg, "tokenizer.json") : tokArg;
        var tok = BpeTokenizer.Load(tokPath);
        Checkpoint.LoadedTrainingCheckpoint loaded = Checkpoint.LoadWithMetadata(
            modelPath, CreateBackend(backendName, cudaMatMulMode));
        string tokenizerIdentity = ContentSha256(tokPath);
        if (loaded.TokenizerIdentity is not null && loaded.TokenizerIdentity != tokenizerIdentity)
            throw new InvalidDataException("Tokenizer does not match the tokenizer recorded in the checkpoint.");
        if (loaded.TokenizerIdentity is null)
            Console.WriteLine("warning: legacy checkpoint has no tokenizer identity to verify.");
        GptModel model = loaded.Model;
        var rng = new Random(seed);

        Console.WriteLine($"model: {modelPath} (vocab {model.Config.VocabSize}, ctx {model.Config.ContextLength}, " +
                          $"params {model.Params.Count:N0})");
        Console.WriteLine("chat — type a line and the model continues it. /reset clears context, /quit exits.");

        var history = new List<int>(tok.Encode("\n"));
        while (true)
        {
            Console.Write("\nyou> ");
            string? line = Console.ReadLine();
            if (line is null or "/quit") break;
            if (line is "/reset")
            {
                history.Clear();
                history.AddRange(tok.Encode("\n"));
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
            foreach (int id in Sampler.Generate(model, history, tokensPerTurn, temperature, topk, rng))
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
