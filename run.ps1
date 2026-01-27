<# 
 Helper launcher for the backend.
 - Loads env vars from .env (key=value, ignores comments/blank lines).
 - Sets a few sane defaults if they are missing.
#>

$ErrorActionPreference = "Stop"

# Устанавливаем UTF-8 кодировку для корректного отображения русских символов
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

$EnvFile = Join-Path $PSScriptRoot ".env"
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

if (-not $env:APP_ENV) { $env:APP_ENV = "dev" }
if (-not $env:APP_DEBUG) { $env:APP_DEBUG = "true" }
if (-not $env:ENABLE_LOCAL_ROUTER) { $env:ENABLE_LOCAL_ROUTER = "true" }

python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

