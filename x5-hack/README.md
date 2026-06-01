# NER Service для поисковых запросов Пятёрочка

Веб-сервис для извлечения именованных сущностей (NER) из поисковых запросов клиентов мобильного приложения торговой сети «Пятёрочка».

## Описание задачи

Сервис автоматически извлекает ключевые сущности из пользовательских поисковых запросов с использованием BIO-разметки  . Система распознает четыре типа сущностей:

- **TYPE** — категория товара (молоко, хлеб, вода, чипсы)
- **BRAND** — бренд (Coca-Cola, Простоквашино, Lays)
- **VOLUME** — объём/вес/количество (0.5 л, 1 л, 200 г, 10 шт)
- **PERCENT** — процент (2.5%, 15%)

## Основные требования

- Время отклика на запрос: **не более 1 секунды**  
- Асинхронная обработка параллельных запросов  
- Метрика качества: **macro-averaged F1-score** по BIO-разметке  
- Формат разметки: B-ENTITY (начало), I-ENTITY (продолжение), O (не сущность)  

## Структура проекта

```

.
├── data/                       \# train.csv, test.csv
├── models/
│   └── fine_tuned_rbcc/        \# Обученная модель
├── notebooks/
│   └── train_model.ipynb       \# Ноутбуки и скрипты для обучения и аугментации
├── source_models/              \# Различные версии предобученных Bert-like моделей
├── main.py                     \# FastAPI приложение
├── requirements.txt            \# Зависимости Python
├── Dockerfile                  \# Docker конфигурация
├── ml_service.py               \# Инференс модели
├── docker-compose.yml          \# Docker Compose файл
└── README.md                   \# Документация

```

## Установка и запуск

Веса модели (скачать)
Ссылка на архив с весами модели: \[ссылка на google drive\](https://drive.google.com/drive/folders/1AzFYC2l0rB4FkePIIZIUhXz5E5ieDLp5?usp=sharing).

После скачивания распаковать содержимое в каталог models/fine_tuned_rbcc так, чтобы внутри лежали стандартные файлы Transformers: config.json, файлы токенайзера (tokenizer.json или vocab.txt, а также special_tokens_map.json и tokenizer_config.json) и сами веса pytorch_model.bin или model.safetensors.

Пример запуска с явным путём к весам: 
```
MODEL_DIR=./models/fine_tuned_rbcc uvicorn main:app --host 0.0.0.0 --port 8000;
```

### Предварительные требования

- Python 3.9+
- Docker и Docker Compose
- 4GB+ RAM

### Локальная установка

1. Клонируйте репозиторий:
```

git clone <URL-репозитория>
cd <папка-проекта>

```

2. Создайте виртуальное окружение:
```

python -m venv venv
source venv/bin/activate  \# Linux/macOS
venv\Scripts\activate     \# Windows

```

3. Установите зависимости:
```

pip install -r requirements.txt

```

4. Запустите сервис:
```

uvicorn main:app --host 0.0.0.0 --port 8000

```

### Запуск через Docker (рекомендуется)

```

docker-compose up --build

```

Сервис будет доступен по адресу `http://localhost:8000`.

## API Reference

### POST /api/predict

Извлечение сущностей из поискового запроса.

**Request:**
```

{
"input": "сгущенное молоко"
}

```

**Response:**
```

[
{"start_index": 0, "end_index": 8, "entity": "B-TYPE"},
{"start_index": 9, "end_index": 15, "entity": "I-TYPE"}
]

```

**Примеры запросов:**

```


# Простой запрос

curl -X POST http://localhost:8000/api/predict \
-H "Content-Type: application/json" \
-d '{"input": "молоко простоквашино 2.5%"}'

# Пустой запрос

curl -X POST http://localhost:8000/api/predict \
-H "Content-Type: application/json" \
-d '{"input": ""}'

# Response: []

```

### GET /health

Проверка работоспособности сервиса.

```

curl http://localhost:8000/health

```

**Response:**
```

{"status": "ok"}

```

## Формат данных

### Обучающая выборка (train.csv)

Формат файла с разделителем `;`:

| Колонка | Описание |
|---------|----------|
| sample | Поисковый запрос пользователя |
| annotation | Разметка в формате BIO: `[(start, end, 'B-ENTITY'), ...]` |

**Пример:**
```

id;sample;annotation
1;молоко простоквашино 2.5%;[(0, 6, 'B-TYPE'), (7, 19, 'B-BRAND'), (20, 24, 'B-PERCENT')]

```

### Тестовая выборка (test.csv)

Аналогично train

## Обучение модели

### Подготовка данных

1. Поместите файлы `train.csv` и `test.csv` в папку `data/raw/`  

2. Запустите аугментацию данных:
```

python notebooks/new_data.py

```

### Обучение

Откройте и выполните ноутбук `notebooks/main_pipeline.ipynb`. Модель будет сохранена в `models/fine_tuned_rbcc/`.

## Метрики качества

Основная метрика: **Macro-averaged F1-score**

Расчёт производится следующим образом:

1. Для каждого типа сущности (TYPE, BRAND, VOLUME, PERCENT) вычисляется:
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1 = 2 × (Precision × Recall) / (Precision + Recall)

2. Macro F1 = среднее арифметическое F1 по всем типам сущностей


## Конфигурация

Переменные окружения для настройки производительности:

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| QUEUE_MAXSIZE | Максимальный размер очереди | 2000 |
| QUEUE_WORKERS | Количество обработчиков | 1 |
| PREDICT_TIMEOUT | Таймаут обработки (сек) | 2 |
| TORCH_NUM_THREADS | Потоки PyTorch | 1 |
| OMP_NUM_THREADS | Потоки OpenMP | 1 |

**Пример:**
```

QUEUE_WORKERS=2 PREDICT_TIMEOUT=1 uvicorn main:app --host 0.0.0.0 --port 8000

```

## Технический стек

- **Backend:** FastAPI, Uvicorn
- **ML Framework:** PyTorch, Transformers (HuggingFace)
- **Data Processing:** Pandas, scikit-learn
- **NLP Tools:** spaCy, NLTK (опционально)
- **Deployment:** Docker, Docker Compose

## Зависимости

```

fastapi>=0.104.0
uvicorn[standard]>=0.24.0
torch>=2.0.0
transformers>=4.35.0
pandas>=2.0.0
scikit-learn>=1.3.0

```

Полный список в файле `requirements.txt`.

## Лицензия

MIT License
```
