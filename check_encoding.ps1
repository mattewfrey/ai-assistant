<# 
 Скрипт для проверки кодировки и настройки UTF-8 в PowerShell
#>

Write-Host "=== Проверка кодировки ===" -ForegroundColor Cyan

# Текущие настройки
Write-Host "`nТекущая кодировка вывода: $([Console]::OutputEncoding.EncodingName)" -ForegroundColor Yellow
Write-Host "Текущая кодировка ввода: $([Console]::InputEncoding.EncodingName)" -ForegroundColor Yellow
Write-Host "OutputEncoding: $($OutputEncoding.EncodingName)" -ForegroundColor Yellow

# Устанавливаем UTF-8
Write-Host "`nУстановка UTF-8..." -ForegroundColor Green
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['*:Encoding'] = 'utf8'
chcp 65001 | Out-Null

Write-Host "`nНовая кодировка вывода: $([Console]::OutputEncoding.EncodingName)" -ForegroundColor Green
Write-Host "Новая кодировка ввода: $([Console]::InputEncoding.EncodingName)" -ForegroundColor Green
Write-Host "OutputEncoding: $($OutputEncoding.EncodingName)" -ForegroundColor Green

# Тест отображения русских символов
Write-Host "`n=== Тест отображения русских символов ===" -ForegroundColor Cyan
Write-Host "Привет, мир! Русские символы: а б в г д е ё ж з и й к л м н о п р с т у ф х ц ч ш щ ъ ы ь э ю я" -ForegroundColor White
Write-Host "Заглавные: А Б В Г Д Е Ё Ж З И Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ъ Ы Ь Э Ю Я" -ForegroundColor White
Write-Host "Специальные: № § © ® ™ € £ ¥" -ForegroundColor White

# Проверка Python
Write-Host "`n=== Проверка Python ===" -ForegroundColor Cyan
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonVersion = python --version
    Write-Host "Python найден: $pythonVersion" -ForegroundColor Green
    
    # Проверяем переменную окружения
    if ($env:PYTHONIOENCODING) {
        Write-Host "PYTHONIOENCODING: $env:PYTHONIOENCODING" -ForegroundColor Green
    } else {
        Write-Host "PYTHONIOENCODING не установлена (рекомендуется установить 'utf-8')" -ForegroundColor Yellow
        Write-Host "Установка PYTHONIOENCODING=utf-8..." -ForegroundColor Green
        $env:PYTHONIOENCODING = "utf-8"
    }
} else {
    Write-Host "Python не найден" -ForegroundColor Red
}

Write-Host "`n=== Готово! ===" -ForegroundColor Green
Write-Host "Если русские символы отображаются корректно выше, проблема решена." -ForegroundColor White
Write-Host "Для постоянного применения добавьте эти настройки в профиль PowerShell." -ForegroundColor White
