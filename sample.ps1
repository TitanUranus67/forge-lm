# sample.ps1 — generate text from the current training checkpoint.
# Safe to run while training: uses the CPU backend, so it never touches the GPU/VRAM
# the training run needs. Run from the repo root:  .\sample.ps1
param(
    [string]$Model = "out/forge-98m.bin",
    [string]$Tokenizer = "data/forge",
    [int]$Tokens = 150,
    [double]$Temperature = 0.8,
    [int]$TopK = 40
)

$prompts = @(
    "Once upon a time",
    "The meaning of life is",
    "The history of the world began"
)

Write-Host "checkpoint: $Model (last written $((Get-Item $Model).LastWriteTime))" -ForegroundColor DarkGray

foreach ($p in $prompts) {
    Write-Host "`n=== $p ===" -ForegroundColor Cyan
    dotnet run -c Release --no-build --project src/LLM.Cli -- generate `
        --model $Model --tokenizer $Tokenizer --prompt $p `
        --tokens $Tokens --temperature $Temperature --topk $TopK --seed (Get-Random)
}
