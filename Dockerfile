# Базовый образ — официальный python 3.11 slim (минимальный, ~130 МБ).
FROM python:3.11-slim

# Системные зависимости:
#   libgl1 и libglib2.0-0 нужны для OpenCV (даже headless-версии).
# --no-install-recommends — не тянем лишнее.
# rm -rf /var/lib/apt/lists/* — чистим кэш apt, чтобы уменьшить образ.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория контейнера.
WORKDIR /app

# Сначала копируем только requirements.txt.
# Если код изменится, а requirements нет — слой pip install возьмётся из кэша.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код.
COPY app/ ./app/
COPY scripts/ ./scripts/

# Создаём non-root пользователя appuser и даём ему права на /app.
# -m — создать домашнюю директорию /home/appuser.
# mkdir -p /app/models — папка для весов (её же создаёт скрипт, но пусть будет).
RUN useradd -m appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app

# Переключаемся на non-root. Дальше всё выполняется от appuser.
USER appuser

# Скачиваем веса при сборке.
RUN python scripts/download_model.py

EXPOSE 8000

# HEALTHCHECK: Docker будет проверять /health каждые 30 секунд.
# --start-period=120s — даём время на загрузку модели.
# --retries=3 — три провала подряд => контейнер unhealthy.
# urllib.request — стандартная библиотека, curl/wget в образ не нужны.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Запуск uvicorn.
# --host 0.0.0.0 — слушать на всех интерфейсах (иначе снаружи контейнера
# не достучаться). --port 8000.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]