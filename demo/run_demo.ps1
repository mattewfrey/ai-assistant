<# 
 Launcher for Gradio Demo.
 Requires: pip install gradio
#>

$ErrorActionPreference = "Stop"

# UTF-8 encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Load .env if exists
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile |
        Where-Object { $_ -and ($_ -notmatch "^\s*#") } |
        ForEach-Object {
            $kv = $_ -split "=", 2
            if ($kv.Count -eq 2) {
                $name = $kv[0].Trim()
                $value = $kv[1]
                [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
            }
        }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Product AI Chat - Gradio Demo" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Make sure backend is running at: http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting Gradio on: http://127.0.0.1:7860" -ForegroundColor Green
Write-Host ""

Set-Location $ProjectRoot
python demo/gradio_app.py
