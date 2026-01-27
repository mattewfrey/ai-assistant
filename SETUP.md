# Product AI Chat — Инструкция по запуску

## Требования

- **Python 3.11+** (проверить: `python --version`)
- **pip** для установки зависимостей
- **OpenAI API Key** с доступом к gpt-4o-mini
- **VPN** (если OpenAI недоступен в вашем регионе)

---

## 1. Установка зависимостей

```powershell
cd c:\ai-assistant

# Основные зависимости
pip install -e .

# Для разработки и тестов
pip install -e ".[dev]"

# Для Gradio демо (опционально)
pip install -e ".[demo]"
```

---

## 2. Настройка окружения (.env)

Файл `.env` уже создан. Проверьте/измените следующие параметры:

### Вариант 1: YandexGPT (рекомендуется для РФ)

```env
# LLM провайдер
LLM_PROVIDER=yandex

# Yandex Cloud API Key
YC_API_KEY=ваш-api-key
YC_FOLDER_ID=ваш-folder-id
YANDEX_MODEL=yandexgpt/latest

# URL для получения данных о продуктах
PRODUCT_GATEWAY_BASE_URL=https://flex-stage-product-search.366.ru
```

### Вариант 2: OpenAI (требует VPN в РФ)

```env
# LLM провайдер
LLM_PROVIDER=openai

# OpenAI API Key
OPENAI_API_KEY=sk-proj-ваш-ключ
OPENAI_MODEL=gpt-4o-mini

# URL для получения данных о продуктах
PRODUCT_GATEWAY_BASE_URL=https://flex-stage-product-search.366.ru
```

### Если OpenAI заблокирован в регионе:

**Вариант A: VPN**
- Подключитесь к VPN (США/Европа) перед запуском

**Вариант B: Переключитесь на YandexGPT**
- Установите `LLM_PROVIDER=yandex` и настройте Yandex Cloud ключи

---

## 3. Запуск Backend (основной сервис)

```powershell
cd c:\ai-assistant

# Запуск сервера
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

После запуска:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **API Base**: http://127.0.0.1:8000/api/product-ai/

---

## 4. Тестирование API

### Через Swagger UI (рекомендуется)

1. Откройте http://127.0.0.1:8000/docs
2. Найдите `POST /api/product-ai/chat/message`
3. Нажмите "Try it out"
4. Введите тело запроса:

```json
{
  "product_id": "62eb2515-1608-4812-9caa-12ad48c975c5",
  "message": "Какой состав у этого препарата?"
}
```

5. Нажмите "Execute"

### Через PowerShell

```powershell
$body = @{
    product_id = "62eb2515-1608-4812-9caa-12ad48c975c5"
    message = "Какой состав у этого препарата?"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/product-ai/chat/message" `
    -Method Post `
    -Body $body `
    -ContentType "application/json; charset=utf-8" | ConvertTo-Json -Depth 10
```

### Через curl (Git Bash / WSL)

```bash
curl -X POST "http://127.0.0.1:8000/api/product-ai/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"product_id": "62eb2515-1608-4812-9caa-12ad48c975c5", "message": "Какой состав?"}'
```

---

## 5. Gradio демо (опционально)

Gradio — это веб-интерфейс для демонстрации API без Swagger.

### Запуск (в отдельном терминале, backend должен работать):

```powershell
cd c:\ai-assistant

# Убедитесь что Gradio установлен
pip install gradio

# Запуск демо
python demo/gradio_app.py
```

Откройте в браузере URL, который покажет Gradio (обычно http://127.0.0.1:7860)

---

## 6. Запуск тестов

```powershell
cd c:\ai-assistant

# Все тесты
python -m pytest

# Только Product AI Chat тесты
python -m pytest tests/test_product_chat.py -v

# Все Product AI тесты
python -m pytest tests/test_product_*.py tests/test_drug_*.py tests/test_smart_*.py tests/test_course_*.py tests/test_purchase_*.py tests/test_proactive_*.py -v
```

---

## 7. Доступные API эндпоинты

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/product-ai/chat/message` | POST | Основной чат с AI о продукте |
| `/api/product-ai/faq/{product_id}` | GET | Генерация FAQ для продукта |
| `/api/product-ai/proactive-hints` | POST | Проактивные подсказки |
| `/api/product-ai/drug-interactions` | POST | Проверка взаимодействия лекарств |
| `/api/product-ai/smart-analogs` | POST | Поиск аналогов по МНН |
| `/api/product-ai/course-calculator` | POST | Калькулятор курса лечения |
| `/api/product-ai/personalization` | POST | Персонализация по истории покупок |

---

## 8. Структура проекта

```
c:\ai-assistant\
├── app/
│   ├── main.py              # FastAPI приложение
│   ├── config.py            # Настройки из .env
│   ├── routers/
│   │   └── product_chat.py  # API эндпоинты
│   ├── services/
│   │   ├── product_gateway_client.py   # Клиент к API продуктов
│   │   ├── product_context_builder.py  # Построение контекста для LLM
│   │   ├── product_chat_service.py     # Логика чата
│   │   ├── langchain_llm.py            # Интеграция с OpenAI
│   │   └── ...
│   ├── models/              # Pydantic модели
│   └── prompts/             # Системные промпты для LLM
├── demo/
│   └── gradio_app.py        # Gradio UI
├── tests/                   # Тесты
├── .env                     # Конфигурация (не коммитить!)
└── pyproject.toml           # Зависимости
```

---

## 9. Troubleshooting

### Ошибка 403: unsupported_country_region_territory
OpenAI заблокирован в регионе. Решение: VPN или прокси.

### ModuleNotFoundError
```powershell
pip install -e .
```

### Порт занят
```powershell
# Использовать другой порт
python -m uvicorn app.main:app --port 8001
```

### Кракозябры в консоли (кодировка)
```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8
```

---

## 10. Краткий чеклист запуска

1. ✅ VPN включен (если нужно)
2. ✅ `.env` настроен с OpenAI ключом
3. ✅ `pip install -e .` выполнен
4. ✅ `python -m uvicorn app.main:app --reload --port 8000`
5. ✅ Открыть http://127.0.0.1:8000/docs
6. ✅ Отправить тестовый запрос
