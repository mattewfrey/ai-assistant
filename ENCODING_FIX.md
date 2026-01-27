# Решение проблемы с кодировкой русских символов в Cursor

## Проблема
Русские символы не отображаются корректно в Cursor редакторе или в терминале.

## Решения

### 1. Настройки Cursor/VSCode (уже применены)
Создан файл `.vscode/settings.json` с настройками:
- Кодировка файлов: UTF-8
- Автоматическое определение кодировки
- Настройка терминала PowerShell для UTF-8

### 2. Настройка PowerShell (уже применена)
Скрипт `run.ps1` обновлён для установки UTF-8 кодировки.

### 3. Дополнительные шаги (если проблема сохраняется)

#### Вариант A: Перезапуск Cursor
1. Закройте Cursor полностью
2. Откройте Cursor заново
3. Откройте файл с русскими символами

#### Вариант B: Ручная настройка PowerShell
Если терминал всё ещё показывает неправильные символы, выполните в PowerShell:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001
```

#### Вариант C: Настройка профиля PowerShell (постоянное решение)
Добавьте в профиль PowerShell (`$PROFILE`):

```powershell
# Проверяем существование профиля
if (!(Test-Path -Path $PROFILE)) {
    New-Item -ItemType File -Path $PROFILE -Force
}

# Добавляем настройки UTF-8
Add-Content -Path $PROFILE -Value "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8"
Add-Content -Path $PROFILE -Value "[Console]::InputEncoding = [System.Text.Encoding]::UTF8"
Add-Content -Path $PROFILE -Value "`$OutputEncoding = [System.Text.Encoding]::UTF8"
Add-Content -Path $PROFILE -Value "`$PSDefaultParameterValues['*:Encoding'] = 'utf8'"
```

#### Вариант D: Настройка Python
Убедитесь, что Python использует UTF-8. Добавьте в начало Python скриптов или в переменные окружения:

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

Или установите переменную окружения:
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### 4. Проверка кодировки файлов
Если конкретный файл отображается неправильно:
1. Откройте файл в Cursor
2. Нажмите на кодировку в правом нижнем углу (например, "UTF-8")
3. Выберите "Reopen with Encoding" → "UTF-8"
4. Если файл был в другой кодировке, выберите "Save with Encoding" → "UTF-8"

### 5. Настройки Windows (если ничего не помогает)
1. Откройте "Региональные параметры" в Windows
2. Перейдите в "Дополнительные параметры"
3. Убедитесь, что "Язык программ, не поддерживающих Юникод" установлен правильно

## Проверка
После применения настроек:
1. Перезапустите Cursor
2. Откройте файл `app/services/langchain_llm.py` или `config/router_config.yaml`
3. Русские символы должны отображаться корректно

## Дополнительная информация
- Все файлы проекта должны быть в кодировке UTF-8
- В файле `config/router_config.yaml` уже указано `Encoding: UTF-8`
- Python файлы используют `# -*- coding: utf-8 -*-` или UTF-8 по умолчанию (Python 3)
