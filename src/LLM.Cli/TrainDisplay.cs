using LLM.Core.Training;

/// <summary>
/// Renders the train command's progress. On an interactive console it draws one
/// in-place \r progress line (20-char bar, %, step, loss, rolling tok/s, ETA,
/// elapsed), redrawn at most once per 500 ms; val events and messages are printed
/// on their own lines above the bar. When stdout is redirected the bar is disabled
/// and plain per-event log lines are printed instead, so piping stays parseable.
/// </summary>
internal sealed class TrainDisplay
{
    private const int BarWidth = 20;
    private const int RateWindow = 20; // log events in the rolling tok/s average
    private static readonly TimeSpan MinRedrawInterval = TimeSpan.FromMilliseconds(500);

    private readonly int _totalSteps;
    private readonly int _tokensPerStep;
    private readonly int _startStep;
    private readonly List<(int Step, double Seconds)> _samples = new();

    private int _step;
    private float _loss;
    private double _tokSec;
    private TimeSpan _elapsed;
    private DateTime _lastRedraw = DateTime.MinValue;
    private int _barLength; // chars of the bar line currently on screen (0 = none drawn)

    public TrainDisplay(int totalSteps, int tokensPerStep, int startStep = 0)
    {
        _totalSteps = totalSteps;
        _tokensPerStep = tokensPerStep;
        _startStep = startStep;
        BarEnabled = !Console.IsOutputRedirected;
    }

    /// <summary>False when stdout is redirected: everything falls back to plain log lines.</summary>
    public bool BarEnabled { get; }

    /// <summary>Feeds one log event; redraws the bar (throttled) or prints a plain line.</summary>
    public void OnLog(TrainLog l)
    {
        _step = l.Step;
        _loss = l.TrainLoss;
        _elapsed = l.Elapsed;
        _samples.Add((l.Step, l.Elapsed.TotalSeconds));
        if (_samples.Count > RateWindow + 1) _samples.RemoveAt(0);
        _tokSec = RollingTokSec();

        if (!BarEnabled)
        {
            Console.WriteLine(PlainLine(l));
            return;
        }

        if (l.ValLoss.HasValue)
            PrintLine(PlainLine(l)); // val events get their own line above the bar

        // throttle redraws, but always show the first and final state
        if (DateTime.UtcNow - _lastRedraw >= MinRedrawInterval || _step == 1 || _step == _totalSteps)
            Redraw();
    }

    /// <summary>Prints a message on its own line above the bar, then redraws the bar.</summary>
    public void PrintLine(string message)
    {
        if (!BarEnabled)
        {
            Console.WriteLine(message);
            return;
        }
        if (_barLength > 0) { Console.WriteLine(); _barLength = 0; }
        Console.WriteLine(message);
        Redraw();
    }

    /// <summary>Ends the bar line with a newline if one is on screen (call before final summaries).</summary>
    public void Complete()
    {
        if (BarEnabled && _barLength > 0)
        {
            Console.WriteLine();
            _barLength = 0;
        }
    }

    private string PlainLine(TrainLog l)
    {
        string val = l.ValLoss.HasValue ? $"  val {l.ValLoss.Value:F4}" : "";
        double tokSec = (l.Step - _startStep) * (double)_tokensPerStep / Math.Max(l.Elapsed.TotalSeconds, 1e-9);
        return $"step {l.Step,6}/{_totalSteps}  lr {l.Lr:E2}  loss {l.TrainLoss:F4}{val}  " +
               $"{tokSec:N0} tok/s  ({l.Elapsed:h\\:mm\\:ss})";
    }

    private void Redraw()
    {
        double frac = Math.Clamp(_step / (double)_totalSteps, 0.0, 1.0);
        int filled = (int)(frac * BarWidth);
        string line = $"[{new string('█', filled)}{new string('-', BarWidth - filled)}] {frac * 100,4:F1}%  " +
                      $"step {_step:N0}/{_totalSteps:N0}  loss {_loss:F4}  {FormatRate(_tokSec)} tok/s  " +
                      $"ETA {FormatEta()}  elapsed {_elapsed:h\\:mm\\:ss}";
        if (line.Length < _barLength) line = line.PadRight(_barLength); // wipe stale chars
        Console.Write('\r' + line);
        _barLength = line.Length;
        _lastRedraw = DateTime.UtcNow;
    }

    // rolling tok/s: per-interval rates over the last RateWindow log events,
    // weighted linearly so recent intervals count more
    private double RollingTokSec()
    {
        if (_samples.Count < 2)
            return (_step - _startStep) * _tokensPerStep / Math.Max(_elapsed.TotalSeconds, 1e-9);
        double weighted = 0, weights = 0;
        for (int i = 1; i < _samples.Count; i++)
        {
            double dt = _samples[i].Seconds - _samples[i - 1].Seconds;
            if (dt <= 0) continue;
            double rate = (_samples[i].Step - _samples[i - 1].Step) * (double)_tokensPerStep / dt;
            weighted += rate * i;
            weights += i;
        }
        return weights > 0 ? weighted / weights : 0;
    }

    private string FormatEta()
    {
        if (_tokSec <= 0 || _step <= 0) return "--";
        var eta = TimeSpan.FromSeconds((_totalSteps - _step) * _tokensPerStep / _tokSec);
        if (eta.TotalHours >= 1) return $"{(int)eta.TotalHours}h{eta.Minutes:D2}m";
        if (eta.TotalMinutes >= 1) return $"{eta.Minutes}m{eta.Seconds:D2}s";
        return $"{eta.Seconds}s";
    }

    private static string FormatRate(double tokSec) =>
        tokSec >= 1000 ? $"{tokSec / 1000:F1}k" : $"{tokSec:N0}";
}
