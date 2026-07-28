using System.Diagnostics;
using System.Globalization;
using LLM.Core.Checkpoint;
using LLM.Core.Inference;
using LLM.Core.Model;
using LLM.Core.Tensor;
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
        "train" => Cli.Train(args[1..]),
        "generate" => Cli.Generate(args[1..]),
        "chat" => Cli.Chat(args[1..]),
        _ => Cli.Unknown(args[0]),
    };
}
catch (Exception ex) when (ex is not OperationCanceledException)
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}

internal static class Cli
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
        LLM.Cli — mini-GPT from scratch (zero dependencies)

        Usage:
          llm prepare   [--corpus <path-or-url>] --out <dir> [--merges 2000] [--tokenizer <path>]
          llm train     --data <dir> [--steps 5000] [--dmodel 128] [--layers 4] [--heads 4]
                        [--ctx 128] [--lr 6e-4] [--minlr 6e-5] [--warmup 100] [--wd 0.1]
                        [--gradclip 1.0] [--seed 42] [--logevery 10] [--valevery 250]
                        [--saveevery 0] [--out out/model.bin] [--init <checkpoint>]
          llm generate  --model <checkpoint> --tokenizer <dir-or-path> [--prompt "Once upon a time"]
                        [--tokens 200] [--temperature 0.8] [--topk 40] [--seed 1]
          llm chat      --model <checkpoint> --tokenizer <dir-or-path>
                        [--tokens 100] [--temperature 0.8] [--topk 40] [--seed 1]

        There is no separate train-tokenizer command: tokenizer training is folded
        into `prepare` (use --tokenizer to supply a pre-trained one).
        Run any command with --help for details.
        """);

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
                  --init resumes from an existing checkpoint (config comes from it;
                  the architecture flags are then ignored). --saveevery N writes the
                  checkpoint every N steps. Ctrl+C stops after the current step and
                  still saves, so an interrupted run is never lost.
                """);
            return 0;
        }

        string dataDir = p.Require("data");
        int steps = p.GetInt("steps", 5000);
        int dmodel = p.GetInt("dmodel", 128);
        int layers = p.GetInt("layers", 4);
        int heads = p.GetInt("heads", 4);
        int ctx = p.GetInt("ctx", 128);
        float lr = p.GetFloat("lr", 6e-4f);
        float minlr = p.GetFloat("minlr", 6e-5f);
        int warmup = p.GetInt("warmup", 100);
        float wd = p.GetFloat("wd", 0.1f);
        float gradclip = p.GetFloat("gradclip", 1.0f);
        int seed = p.GetInt("seed", 42);
        int logevery = p.GetInt("logevery", 10);
        int valevery = p.GetInt("valevery", 250);
        string outPath = p.Get("out", Path.Combine("out", "model.bin"));
        string? init = p.Get("init");
        int saveevery = p.GetInt("saveevery", 0);
        p.Done();

        var backend = new CpuBackend();
        GptModel model;
        if (init is not null)
        {
            model = Checkpoint.Load(init, backend);
            Console.WriteLine($"model: resumed from {init} " +
                              $"(vocab {model.Config.VocabSize}, ctx {model.Config.ContextLength}, " +
                              $"dmodel {model.Config.DModel}, layers {model.Config.NLayers}, heads {model.Config.NHeads})");
        }
        else
        {
            var tok = BpeTokenizer.Load(Path.Combine(dataDir, "tokenizer.json"));
            var config = new ModelConfig(tok.VocabSize, ctx, dmodel, layers, heads);
            model = new GptModel(config, backend, new Random(seed));
            Console.WriteLine($"model: vocab {config.VocabSize}, ctx {ctx}, dmodel {dmodel}, " +
                              $"layers {layers}, heads {heads}, params {model.Params.Count:N0}");
        }

        var trainLoader = new DataLoader(Path.Combine(dataDir, "train.bin"));
        var valLoader = new DataLoader(Path.Combine(dataDir, "val.bin"));
        Console.WriteLine($"data: {trainLoader.Length:N0} train tokens, {valLoader.Length:N0} val tokens");

        var opts = new TrainOptions
        {
            Steps = steps,
            MaxLr = lr,
            MinLr = minlr,
            WarmupSteps = warmup,
            WeightDecay = wd,
            GradClip = gradclip,
            Seed = seed,
            LogEvery = logevery,
            ValEvery = valevery,
            SaveEvery = saveevery,
        };

        void OnLog(TrainLog l)
        {
            double tokSec = l.Step * (double)model.Config.ContextLength / Math.Max(l.Elapsed.TotalSeconds, 1e-9);
            string val = l.ValLoss.HasValue ? $"  val {l.ValLoss.Value:F4}" : "";
            Console.WriteLine($"step {l.Step,6}/{steps}  lr {l.Lr:E2}  loss {l.TrainLoss:F4}{val}  " +
                              $"{tokSec:N0} tok/s  ({l.Elapsed:mm\\:ss})");
        }

        string? outDir = Path.GetDirectoryName(Path.GetFullPath(outPath));
        if (outDir is not null) Directory.CreateDirectory(outDir);

        // write to a temp file then move, so an interrupt mid-save can't corrupt the last good checkpoint
        void SaveCheckpoint(GptModel m, string tag)
        {
            string tmp = outPath + ".tmp";
            Checkpoint.Save(m, tmp);
            File.Move(tmp, outPath, overwrite: true);
            Console.WriteLine($"checkpoint: saved {outPath} ({tag})");
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

        TrainSummary summary = Trainer.Train(model, trainLoader, valLoader, opts, OnLog,
            onSave: saveevery > 0 ? (m, s) => SaveCheckpoint(m, $"step {s}") : null,
            cancel: cts.Token);

        string interrupted = summary.Steps < steps ? $" (interrupted at step {summary.Steps}/{steps})" : "";
        Console.WriteLine($"done: {summary.Steps} steps in {summary.Elapsed:mm\\:ss}{interrupted}, " +
                          $"final train loss {summary.FinalTrainLoss:F4}" +
                          (summary.FinalValLoss.HasValue ? $", final val loss {summary.FinalValLoss.Value:F4}" : ""));
        SaveCheckpoint(model, "final");
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
        p.Done();

        string tokPath = Directory.Exists(tokArg) ? Path.Combine(tokArg, "tokenizer.json") : tokArg;
        var tok = BpeTokenizer.Load(tokPath);
        var model = Checkpoint.Load(modelPath, new CpuBackend());
        Console.WriteLine($"model: {modelPath} (vocab {model.Config.VocabSize}, ctx {model.Config.ContextLength}, " +
                          $"params {model.Params.Count:N0})");

        int[] promptIds = prompt.Length > 0 ? tok.Encode(prompt) : tok.Encode("\n");
        var rng = new Random(seed);

        Console.WriteLine("---");
        Console.Write(prompt);
        var sw = Stopwatch.StartNew();
        int count = 0;
        foreach (int id in Sampler.Generate(model, promptIds, tokens, temperature, topk, rng))
        {
            Console.Write(tok.Decode(new[] { id }));
            count++;
        }
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
        p.Done();

        string tokPath = Directory.Exists(tokArg) ? Path.Combine(tokArg, "tokenizer.json") : tokArg;
        var tok = BpeTokenizer.Load(tokPath);
        var model = Checkpoint.Load(modelPath, new CpuBackend());
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
            foreach (int id in Sampler.Generate(model, history, tokensPerTurn, temperature, topk, rng))
            {
                Console.Write(tok.Decode(new[] { id }));
                history.Add(id);
            }
            Console.WriteLine();
        }
        return 0;
    }

    // ---- helpers -------------------------------------------------------------

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

        public float GetFloat(string key, float fallback) =>
            Take(key) is string v ? float.Parse(v, CultureInfo.InvariantCulture) : fallback;

        public void Done()
        {
            if (Help) return;
            if (_flags.Count > 0)
                throw new ArgumentException($"unknown flag(s): {string.Join(", ", _flags.Keys.Select(k => "--" + k))}");
        }
    }
}
