# Руководство администратора: Product AI Chat

## Содержание

1. [Обзор архитектуры](#обзор-архитектуры)
2. [Структура проекта](#структура-проекта)
3. [Конфигурация](#конфигурация)
4. [Запуск сервиса](#запуск-сервиса)
5. [Переключение LLM провайдера](#переключение-llm-провайдера)
6. [Мониторинг и логирование](#мониторинг-и-логирование)
7. [API Endpoints](#api-endpoints)
8. [Кэширование](#кэширование)
9. [Безопасность](#безопасность)
10. [Troubleshooting](#troubleshooting)

---

## Обзор архитектуры

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend /    │────▶│   FastAPI       │────▶│   YandexGPT /   │
│   Gradio Demo   │     │   Backend       │     │   OpenAI        │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   Product API   │
                        │   (366.ru)      │
                        └─────────────────┘
```

**Основные компоненты:**
- **FastAPI Backend** — основной API сервер (порт 8000)
- **Gradio Demo** — интерактивный UI для тестирования (порт 7860)
- **LLM Provider** — YandexGPT или OpenAI
- **Product API** — внешний API для получения данных о товарах

---

## Структура проекта

```
c:\ai-assistant\
├── .env                          # Конфигурация (НЕ коммитить!)
├── app/
│   ├── main.py                   # Точка входа FastAPI
│   ├── config.py                 # Настройки приложения
│   ├── intents.py                # Определения интентов
│   ├── models/                   # Pydantic модели
│   │   ├── assistant.py          # Модели ответов
│   │   └── product_chat.py       # Модели Product Chat
│   ├── routers/
│   │   ├── chat.py               # Основной чат endpoint
│   │   └── product_chat.py       # Product AI Chat endpoints
│   ├── services/
│   │   ├── yandex_gpt.py         # ★ YandexGPT клиент (REST API)
│   │   ├── langchain_llm.py      # LangChain интеграция
│   │   ├── product_chat_service.py    # Логика Product Chat
│   │   ├── product_context_builder.py # Сборка контекста товара
│   │   ├── product_gateway_client.py  # HTTP клиент к Product API
│   │   ├── product_policy_guard.py    # Фильтрация + защита от циклов
│   │   ├── audit_service.py      # ★ Структурированный аудит
│   │   ├── conversation_store.py      # Хранение истории
│   │   ├── metrics.py            # Метрики и rate limiting
│   │   └── cache.py              # Кэширование
│   └── prompts/                  # Системные промпты
├── demo/
│   └── gradio_app.py             # Gradio демо-интерфейс
├── config/
│   └── router_config.yaml        # Конфигурация роутера интентов
└── tests/                        # Тесты
```

---

## Конфигурация

### Файл `.env` — главный конфигурационный файл

**Расположение:** `c:\ai-assistant\.env`

```env
# =============================================================================
# Основные настройки
# =============================================================================
APP_ENV=dev                       # dev | staging | production
APP_DEBUG=true                    # Включить отладку

# =============================================================================
# LLM Provider (выбор: "openai" или "yandex")
# =============================================================================
LLM_PROVIDER=yandex               # ★ Текущий провайдер

# =============================================================================
# YandexGPT Configuration
# =============================================================================
YC_API_KEY=AQVN...                # API ключ Yandex Cloud
YC_FOLDER_ID=b1glp...             # Folder ID каталога
YANDEX_MODEL=yandexgpt/latest     # Модель (yandexgpt/latest, yandexgpt-lite/latest)

# =============================================================================
# OpenAI Configuration (если LLM_PROVIDER=openai)
# =============================================================================
OPENAI_API_KEY=sk-proj-...        # API ключ OpenAI
OPENAI_MODEL=gpt-4o-mini          # Модель
OPENAI_TEMPERATURE=0.2            # Температура (0.0-1.0)

# =============================================================================
# Product API (366.ru)
# =============================================================================
PRODUCT_GATEWAY_BASE_URL=https://flex-stage-product-search.366.ru

# =============================================================================
# Rate Limiting
# =============================================================================
LLM_RATE_LIMIT_WINDOW=60          # Окно в секундах
LLM_RATE_LIMIT_MAX_CALLS=20       # Макс. запросов на окно
```

### Изменение конфигурации

1. Отредактируйте `.env`
2. Перезапустите сервер (или он перезапустится автоматически с `--reload`)

**Важно:** После изменения `.env` кэш настроек сбрасывается при перезапуске.

---

## Запуск сервиса

### Требования

- Python 3.11+ (тестировалось на 3.14)
- Доступ к Product API (VPN для 366.ru)
- Интернет для YandexGPT API

### Установка зависимостей

```powershell
cd c:\ai-assistant
pip install -r requirements.txt  # или pip install -e .
```

### Запуск Backend

```powershell
# Режим разработки (автоперезагрузка)
python -m uvicorn app.main:app --reload --port 8000

# Production режим
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Доступные URL:**
- Swagger UI: http://localhost:8000/docs
- OpenAPI JSON: http://localhost:8000/openapi.json
- Health check: http://localhost:8000/health

### Запуск Gradio Demo

```powershell
# В отдельном терминале
python demo/gradio_app.py
```

**URL:** http://localhost:7860

---

## Переключение LLM провайдера

### С YandexGPT на OpenAI

1. Отредактируйте `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-your-key-here
```

2. **Важно:** Для OpenAI из России нужен VPN или прокси:
```env
OPENAI_BASE_URL=https://your-proxy.com/v1
```

3. Перезапустите сервер

### С OpenAI на YandexGPT

1. Отредактируйте `.env`:
```env
LLM_PROVIDER=yandex
YC_API_KEY=your-api-key
YC_FOLDER_ID=your-folder-id
YANDEX_MODEL=yandexgpt/latest
```

2. Перезапустите сервер

### Получение ключей YandexGPT

1. Зайдите в https://console.yandex.cloud/
2. Выберите каталог (folder)
3. IAM → API ключи → Создать
4. Скопируйте секретный ключ в `YC_API_KEY`
5. `YC_FOLDER_ID` — это ID каталога из URL консоли

---

## Мониторинг и логирование

### Логи сервера

Логи выводятся в stdout терминала:

```
INFO:     127.0.0.1:58365 - "POST /api/product-ai/chat/message HTTP/1.1" 200 OK
INFO:     YandexGPT usage: input=245, output=89, total=334 tokens
```

**Уровни логов:**
- `INFO` — обычные операции, токены LLM
- `WARNING` — некритичные проблемы
- `ERROR` — ошибки обработки

### Токены YandexGPT

Каждый вызов логирует использование токенов:
```
YandexGPT usage: input=245, output=89, total=334 tokens
```

**Для расчёта стоимости:**
- YandexGPT: ~0.4 ₽ / 1000 токенов
- Один запрос: ~300-500 токенов ≈ 0.12-0.20 ₽

### Биллинг Yandex Cloud

Детальная статистика: https://console.yandex.cloud/ → Биллинг → Детализация

### Метрики в API ответе

Каждый ответ содержит debug-информацию:

```json
{
  "meta": {
    "debug": {
      "model": "yandexgpt/latest",
      "llm_used": true,
      "llm_cached": false,
      "token_usage": {
        "prompt_tokens": 245,
        "completion_tokens": 89,
        "total_tokens": 334
      }
    }
  }
}
```

### Аудит событий

Все запросы к Product AI Chat логируются в структурированном формате.

**Типы событий:**
- `chat_request` — входящий запрос
- `chat_response` — успешный ответ
- `policy_block` — заблокировано политикой
- `rate_limit` — превышен лимит
- `error` — ошибка

**Формат аудит-лога:**
```json
{
  "event_type": "chat_response",
  "timestamp": "2026-01-22T10:30:00.000Z",
  "trace_id": "abc123",
  "conversation_id": "conv-456",
  "user_id": "user-789",
  "product_id": "62eb2515-...",
  "llm_provider": "yandex",
  "llm_model": "yandexgpt/latest",
  "llm_tokens_input": 245,
  "llm_tokens_output": 89,
  "success": true,
  "used_fields": ["pharma_info.composition"],
  "confidence": 0.95
}
```

**Расширение:** В production можно отправлять в ELK, ClickHouse и т.д.

---

## API Endpoints

### Product AI Chat

| Метод | Endpoint | Описание |
|-------|----------|----------|
| POST | `/api/product-ai/chat/message` | Отправить сообщение о товаре |
| GET | `/api/product-ai/faq/{product_id}` | Сгенерировать FAQ |
| POST | `/api/product-ai/proactive/hints` | Проактивные подсказки |
| POST | `/api/product-ai/drug-interactions/check` | Проверка взаимодействий |
| POST | `/api/product-ai/analogs/find` | Поиск аналогов |
| POST | `/api/product-ai/course/calculate` | Калькулятор курса |
| POST | `/api/product-ai/personalization/context` | Персонализация |

### Основной чат

| Метод | Endpoint | Описание |
|-------|----------|----------|
| POST | `/api/ai/chat/message` | Общий чат ассистента |

### Пример запроса Product Chat

```bash
curl -X POST "http://localhost:8000/api/product-ai/chat/message" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "62eb2515-1608-4812-9caa-12ad48c975c5",
    "message": "Какой состав?"
  }'
```

---

## Кэширование

### Типы кэша

| Кэш | TTL | Назначение |
|-----|-----|------------|
| Product Context | 180 сек | Данные о товаре из Product API |
| LLM Cache | Session | Повторяющиеся LLM запросы |
| Conversation | 30 мин | История диалога |

### Настройка TTL

В `.env`:
```env
PRODUCT_CONTEXT_TTL_SECONDS=180   # TTL контекста товара
```

### Принудительное обновление

Для FAQ можно сбросить кэш:
```
GET /api/product-ai/faq/{product_id}?force_refresh=true
```

---

## Безопасность

### Policy Guard

`ProductPolicyGuard` фильтрует опасные запросы:
- **Prompt injection** — попытки раскрыть system prompt
- **Out of scope** — вопросы не про товар
- **Фарма-ограничения** — запрет назначать дозировки для рецептурных
- **Циклические вопросы** — защита от спама одинаковых вопросов

**Расположение:** `app/services/product_policy_guard.py`

**Настройки циклической защиты:**
```python
CYCLIC_WINDOW_SECONDS = 300  # 5 минут
CYCLIC_MAX_SIMILAR = 3       # Максимум похожих запросов
```

### Rate Limiting

Защита от злоупотреблений:
```env
LLM_RATE_LIMIT_WINDOW=60      # Окно в секундах
LLM_RATE_LIMIT_MAX_CALLS=20   # Макс. запросов
```

При превышении: HTTP 429 Too Many Requests

### Секреты

**НЕ коммитьте `.env` в git!**

`.gitignore` содержит:
```
.env
*.key
credentials.json
```

---

## Troubleshooting

### Ошибка: "YandexGPT connection error"

**Причина:** Нет доступа к API YandexGPT
**Решение:**
1. Проверьте интернет-соединение
2. Проверьте корректность `YC_API_KEY` и `YC_FOLDER_ID`
3. Убедитесь что API ключ активен в консоли Yandex Cloud

### Ошибка: "Product API unavailable" / "getaddrinfo failed"

**Причина:** Нет доступа к Product API (366.ru)
**Решение:**
1. Включите VPN для доступа к корпоративной сети
2. Проверьте `PRODUCT_GATEWAY_BASE_URL` в `.env`

### Ошибка: "Rate limit exceeded"

**Причина:** Превышен лимит запросов
**Решение:**
1. Подождите 60 секунд
2. Или увеличьте `LLM_RATE_LIMIT_MAX_CALLS` в `.env`

### LangSmith client disabled

**Это не ошибка** — LangSmith трейсинг несовместим с текущей версией. Не влияет на работу.

### Pydantic V1 warning

**Это не ошибка** — предупреждение о совместимости с Python 3.14. Работает корректно.

### Как проверить что YandexGPT работает?

```bash
curl -X POST "http://localhost:8000/api/product-ai/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"product_id": "62eb2515-1608-4812-9caa-12ad48c975c5", "message": "Привет"}'
```

В ответе `meta.debug.model` должно быть `yandexgpt/latest`.

---

## Контакты и поддержка

- **Swagger UI:** http://localhost:8000/docs
- **Yandex Cloud Console:** https://console.yandex.cloud/
- **Документация YandexGPT:** https://yandex.cloud/docs/foundation-models/
