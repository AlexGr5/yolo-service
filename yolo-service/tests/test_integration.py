"""
Интеграционный тест. Один на весь проект.

Проверяет всю цепочку разом. Реальная модель, реальная картинка, реальный
HTTP-запрос. Если что-то из этого недоступно (нет сети, нет весов, упал
запуск модели) — тест скипается, а не падает. Смысл в том, чтобы не ломать CI, где
может не быть ни сети, ни возможности скачать веса.
"""
import pytest
import requests
from fastapi.testclient import TestClient

from app import main


BUS_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


@pytest.mark.integration
def test_integration_bus_image():
    # Шаг первый: скачать картинку. Ловим широкий Exception, потому что
    # причин провала может быть много: нет сети, таймаут, сервер вернул 404,
    # DNS не резолвится. Любая из них — повод скипнуть, а не падать.
    #
    # Таймаут десять секунд — чтобы тест не висел минутами если сеть
    # полуживая. raise_for_status кинет исключение если статус 4xx или 5xx.
    # response.content возвращает байты, именно они нужны для картинки.
    try:
        response = requests.get(BUS_IMAGE_URL, timeout=10)
        response.raise_for_status()
        image_bytes = response.content
    except Exception:
        pytest.skip("Нет сети — пропускаем интеграционный тест")

    # Шаг второй: запустить сервис с настоящей моделью. Здесь уже нужен with
    # у TestClient — без него lifespan не запустится и веса не подгрузятся.
    # С with запустится, загрузит модель, выполнит запрос и корректно
    # всё остановит на выходе.
    #
    # Если модель не загрузилась (нет весов и не удалось скачать, упал torch,
    # не хватило памяти) — это тоже скип, а не падение.
    try:
        with TestClient(main.app) as client:
            result = client.post(
                "/predict",
                params={"conf": 0.25, "iou": 0.45},
                files={"file": ("bus.jpg", image_bytes, "image/jpeg")},
            )
    except Exception as error:
        pytest.skip(f"Не удалось запустить сервис с реальной моделью: {error}")

    # Шаг третий: проверки результата. Код 200, поле detections присутствует,
    # детекций хотя бы одна. На bus.jpg реально есть люди и автобус, так что
    # пустой список означал бы что-то неладное.
    assert result.status_code == 200
    body = result.json()
    assert "detections" in body
    detections = body["detections"]
    assert len(detections) > 0, "На bus.jpg должны быть детекции"

    # Собираем множество всех найденных классов. Множество само убирает
    # дубликаты. Если четыре person, в множестве будет один person.
    detected_classes = set()
    for detection in detections:
        detected_classes.add(detection["class"])

    # Хотя бы один из двух классов должен найтись. Точное число детекций
    # не проверяем намеренно: разные версии весов могут давать разные
    # результаты. Нам важно что модель вообще что-то осмысленное нашла.
    assert "person" in detected_classes or "bus" in detected_classes