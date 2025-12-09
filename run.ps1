# Smart Pharmacy Assistant - Startup Script
# Использование: .\run.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Smart Pharmacy Assistant" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Переходим в директорию скрипта
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Проверяем .env
if (-not (Test-Path ".env")) {
    Write-Host "[ERROR] Файл .env не найден!" -ForegroundColor Red
    exit 1
}

# Проверяем Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "[ERROR] Python не найден" -ForegroundColor Red
    exit 1
}

# Проверяем конфигурацию
Write-Host "[INFO] Проверка настроек..." -ForegroundColor Gray
$envContent = Get-Content ".env" -Raw

if ($envContent -match "OPENAI_API_KEY=sk-") {
    Write-Host "[OK] OPENAI_API_KEY настроен" -ForegroundColor Green
}
else {
    Write-Host "[WARN] OPENAI_API_KEY не настроен" -ForegroundColor Yellow
}

if ($envContent -match "USE_LANGCHAIN=true") {
    Write-Host "[OK] LLM включен" -ForegroundColor Green
}
else {
    Write-Host "[WARN] LLM выключен" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "----------------------------------------" -ForegroundColor Cyan
Write-Host "[START] Запуск сервера..." -ForegroundColor Yellow
Write-Host "[URL] http://localhost:8000" -ForegroundColor Cyan
Write-Host "[DOCS] http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan
Write-Host ""

# Запуск
python -m uvicorn app.main:app --reload
