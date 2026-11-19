"""
Скрипт скачивания весов yolov8n.pt.

Нужен, чтобы:
    - не коммитить веса в git;
    - явно контролировать, куда их положить (models/yolov8n.pt);
    - в Docker скачать веса на этапе сборки.

Запуск:
    python scripts/download_model.py
"""
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


# Путь, куда кладём веса. Переопределяется через env MODEL_PATH.
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")


def download_if_needed():
    """
    Проверяет наличие весов и скачивает, если их нет.

    Возвращает:
        None. Печатает прогресс и путь к итоговому файлу.
    """
    target_path = Path(MODEL_PATH)

    # Создаём директорию для весов, если её нет.
    # parents=True — создаём все промежуточные папки.
    # exist_ok=True — не падаем, если папка уже есть.
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Если веса уже лежат на месте — ничего не делаем.
    if target_path.exists():
        print(f"Модель уже есть: {target_path}")
        return

    print(f"Скачивание yolov8n.pt в {target_path}...")

    # YOLO("yolov8n.pt") триггерит скачивание.
    # Ultralytics кладёт веса в текущую рабочую директорию, если их нет.
    YOLO("yolov8n.pt")

    # На случай, если ultralytics положил файл в cwd, а не в target_path —
    # переносим его на нужное место.
    downloaded_path = Path("yolov8n.pt")
    if downloaded_path.exists() and downloaded_path != target_path:
        # shutil.move умеет перемещать и через границы файловых систем.
        shutil.move(str(downloaded_path), str(target_path))

    print(f"Готово: {target_path}")


if __name__ == "__main__":
    download_if_needed()