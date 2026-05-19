$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
git -C $repoRoot config core.hooksPath .githooks
Write-Host "Configured git hooks path to .githooks"
Write-Host "The pre-commit hook verifies Mojo kernels through Linux/WSL before each commit."
