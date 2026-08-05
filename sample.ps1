# sample.ps1 — reproduce the balanced final-model evaluation samples.
# Run from the repo root:  .\sample.ps1
param(
    [string]$Model = "out/forge-98m-final.bin",
    [string]$Tokenizer = "data/forge",
    [string]$Backend = "gpu",
    [int]$Tokens = 500,
    [double]$Temperature = 0.7,
    [int]$TopK = 30,
    [double]$RepetitionPenalty = 1.1,
    [int]$NoRepeatNgram = 4,
    [int]$Seed = 1
)

$prompts = @(
    @{ Name = "Story"; Text = "Once upon a time " },
    @{ Name = "Abstract"; Text = "The meaning of life is " },
    @{ Name = "Science"; Text = "Photosynthesis is the process by which " },
    @{ Name = "Procedure"; Text = "To bake a loaf of bread, first " },
    @{ Name = "Technical"; Text = "In computer science, an algorithm is " }
)

Write-Host "checkpoint: $Model (last written $((Get-Item $Model).LastWriteTime))" -ForegroundColor DarkGray
Write-Host "sampling: backend $Backend, temperature $Temperature, top-k $TopK, repetition penalty $RepetitionPenalty, no-repeat $($NoRepeatNgram)-gram, seed $Seed, ceiling $Tokens tokens" -ForegroundColor DarkGray

foreach ($prompt in $prompts) {
    Write-Host "`n=== $($prompt.Name): $($prompt.Text) ===" -ForegroundColor Cyan
    dotnet run -c Release --no-build --project src/LLM.Cli -- generate `
        --backend $Backend `
        --model $Model --tokenizer $Tokenizer --prompt $prompt.Text `
        --tokens $Tokens --temperature $Temperature --topk $TopK `
        --repetition-penalty $RepetitionPenalty --no-repeat-ngram $NoRepeatNgram `
        --seed $Seed
}
