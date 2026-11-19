# YOLOv8 Detection Service

REST-сервис детекции объектов на изображениях на базе YOLOv8n и FastAPI.

## Возможности

- `GET /health` — проверка состояния сервиса (используется в Docker HEALTHCHECK).
- `POST /predict` — детекция объектов на изображении.
  - Принимает `multipart/form-data` с полем `file`.
  - Query-параметры: `conf` (порог уверенности, 0..1), `iou` (порог IoU для NMS, 0..1).
  - Возвращает JSON: список детекций `{class, confidence, bbox: [x1, y1, x2, y2]}`,
    где `bbox` — в пикселях исходного изображения.
- Ограничение размера файла — 10 МБ (HTTP 413 при превышении).
- Понятные ошибки: 400 (пустой файл/не изображение), 413 (слишком большой), 422 (невалидные параметры).

## Структура проекта

yolo-service/
├── app/
│   ├── config.py        # настройки (пути, пороги, лимиты)
│   ├── inference.py     # preprocess, postprocess, загрузка модели
│   └── main.py          # FastAPI приложение
├── scripts/
│   └── download_model.py # скрипт скачивания весов
├── tests/
│   ├── conftest.py
│   ├── test_postprocess.py   # unit-тесты postprocess
│   ├── test_api.py           # API-тесты с мок-моделью
│   └── test_integration.py   # интеграционный тест (bus.jpg)
├── requirements.txt
├── Dockerfile
├── pytest.ini
└── README.md


## Требования

- Python 3.11+
- Docker (опционально)

## Локальный запуск

```bash
# 1. Создаём виртуальное окружение
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
.venv\Scripts\activate             # Windows

# 2. Ставим зависимости
pip install -r requirements.txt

# 3. Скачиваем веса (либо ultralytics скачает их автоматически при первом запуске)
python scripts/download_model.py

# 4. Запускаем сервис
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

## Запуск тестов

```bash
pytest -v
```

Ожидаемый вывод (пример):

```
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_predict_returns_detections PASSED
tests/test_api.py::test_predict_empty_file_returns_400 PASSED
tests/test_api.py::test_predict_not_an_image_returns_400 PASSED
tests/test_api.py::test_predict_huge_file_returns_413 PASSED
tests/test_api.py::test_predict_invalid_conf_returns_422 PASSED
tests/test_integration.py::test_integration_bus_image PASSED
tests/test_postprocess.py::test_bbox_recalculated_to_original_coordinates PASSED
tests/test_postprocess.py::test_low_confidence_filtered_out PASSED
tests/test_postprocess.py::test_bbox_clipped_to_image_borders PASSED
tests/test_postprocess.py::test_nms_suppresses_overlapping_boxes PASSED
tests/test_postprocess.py::test_letterbox_and_reverse_recalculation PASSED

==================== 12 passed in X.XXs ====================
```

Интеграционный тест автоматически скипается (`SKIPPED`), если нет сети.
Чтобы исключить его из запуска:

```bash
pytest -v -m "not integration"
```

## Docker

### Сборка

```bash
docker build -t yolo-service .
```

### Запуск

```bash
docker run --rm -p 8000:8000 yolo-service
```

Сервис будет доступен на http://localhost:8000.

## Примеры запросов

### GET /health

```bash
curl http://localhost:8000/health
```

Ответ:

```json
{"status": "ok", "model_loaded": true}
```

### POST /predict

```bash
curl -X POST "http://localhost:8000/predict?conf=0.25&iou=0.45" \
     -F "file=@bus.jpg"
```

Ответ:

```json
{
  "image_width": 810,
  "image_height": 1080,
  "detections": [
    {
      "class": "bus",
      "confidence": 0.87,
      "bbox": [12.3, 231.0, 810.0, 1080.0]
    },
    {
      "class": "person",
      "confidence": 0.91,
      "bbox": [48.2, 398.7, 244.5, 902.3]
    }
  ]
}
```

