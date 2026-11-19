"""
API-тесты с мок-моделью.

Настоящая YOLO здесь не грузится. Её место занимает заглушка, которая
возвращает заранее заданный тензор. Потому что настоящая модель при старте
читает веса с диска и считает на процессоре — это секунды. И результат
зависит от весов, что делает тест недетерминированным. А нам в этих
тестах важно проверить HTTP-контракт: успешный запрос, обработку ошибок,
валидацию параметров. Качество детекции проверяется в интеграционном
тесте, здесь оно не нужно.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def mocked_app(sample_image_bytes):
    # Готовим подмену настоящей модели на заглушку и отдаём тесту готовый
    # TestClient, который уже смотрит на приложение с подменённой моделью.
    #
    # Параметр sample_image_bytes в сигнатуре фактически не используется.
    # Он нужен только для порядка. pytest сначала создаст картинку, потом мок.
    # Можно было бы убрать, но пусть будет.

    mock_model = MagicMock()
    # У настоящей YOLO есть атрибут names — словарь индекс-имя-класса.
    # Нам в тестах хватит двух классов, все восемьдесят COCO не нужны.
    mock_model.names = {0: "person", 1: "car"}

    # Собираем фейковый выход в том же формате, что у настоящей yolov8n:
    # [1, 84, 8400]. Одна детекция класса ноль с уверенностью 0.9
    # в центре картинки.
    num_classes = 80
    num_anchors = 8400
    fake_output_array = np.zeros(
        (1, 4 + num_classes, num_anchors), dtype=np.float32
    )
    fake_output_array[0, 0, 0] = 320.0
    fake_output_array[0, 1, 0] = 320.0
    fake_output_array[0, 2, 0] = 100.0
    fake_output_array[0, 3, 0] = 100.0
    fake_output_array[0, 4, 0] = 0.9

    fake_output = torch.from_numpy(fake_output_array)

    # В нашем коде inference.run_inference зовёт model.model(input_tensor).
    # Значит атрибут model у мока должен быть вызываемым и возвращать
    # наш фейковый тензор. MagicMock с return_value именно это и делает:
    # при любом вызове с любыми аргументами возвращает указанное значение.
    mock_model.model = MagicMock(return_value=fake_output)

    # Кладём мок в глобальное состояние приложения. Дальше эндпоинты будут
    # брать модель оттуда и попадут на наш мок вместо настоящей.
    main.app_state["model"] = mock_model
    main.app_state["class_names"] = mock_model.names

    # TestClient без with не запускает lifespan. Это здесь критично, потому что
    # если бы запускал, то при старте подгрузил бы настоящую модель
    # и перетёр бы наш мок. А так lifespan пропускается, мок остаётся живым.
    client = TestClient(main.app)
    yield client

    # После теста убираем мок, чтобы следующий тест стартовал с чистого листа.
    main.app_state["model"] = None
    main.app_state["class_names"] = None


def test_health_endpoint():
    # Проверяем что эндпоинт здоровья отвечает и возвращает status ok.
    # Мок-модель тут не нужна: эндпоинт health не зависит от модели.
    # Это специально так — в Docker он должен отвечать даже пока модель
    # грузится, иначе healthcheck убьёт контейнер раньше времени.
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # Проверяем только наличие ключа, а не значение. Значение может быть
    # true или false в зависимости от того, загружена ли модель.
    assert "model_loaded" in body


def test_predict_returns_detections(mocked_app, sample_image_bytes):
    # Успешный запрос на предсказание. Проверяем код 200 и структуру
    # ответа. Значения bbox здесь не проверяем. Они зависят от метаданных
    # letterbox конкретной картинки, а математика пересчёта уже покрыта
    # в тестах постпроцесса. Здесь важно только что HTTP-слой работает.
    response = mocked_app.post(
        "/predict",
        # params уходят в query-строку, то есть после вопросительного знака
        # в URL. Это параметры conf и iou.
        params={"conf": 0.5, "iou": 0.5},
        # files — это multipart/form-data. Ключ file совпадает с именем поля,
        # которое ждёт FastAPI. Значение — кортеж из имени файла, содержимого
        # и MIME-типа. Именно так браузер отправляет файл формы.
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "detections" in body
    assert len(body["detections"]) == 1

    # Проверяем что мок-ответ дошёл до клиента без искажений. Класс
    # превратился в person через словарь names, уверенность проброшена,
    # у bbox четыре координаты.
    detection = body["detections"][0]
    assert detection["class"] == "person"
    assert detection["confidence"] == pytest.approx(0.9, abs=1e-5)
    assert len(detection["bbox"]) == 4


def test_predict_empty_file_returns_400(mocked_app, empty_bytes):
    # Пустой файл. Сервис должен вернуть 400 с понятным сообщением,
    # а не упасть с 500. Проверяем и код, и текст ошибки, чтобы отличить
    # эту ошибку от других 400-х.
    response = mocked_app.post(
        "/predict",
        files={"file": ("empty.png", empty_bytes, "image/png")},
    )
    assert response.status_code == 400
    # lower приводит текст к нижнему регистру, чтобы тест не упал если
    # в сообщении поменяется регистр первой буквы.
    assert "пустой" in response.json()["detail"].lower()


def test_predict_not_an_image_returns_400(mocked_app, not_an_image_bytes):
    # Пришёл файл, который не является картинкой. Pillow при декодировании
    # упадёт, сервис должен превратить это в 400 с сообщением про декодирование.
    response = mocked_app.post(
        "/predict",
        files={"file": ("text.txt", not_an_image_bytes, "text/plain")},
    )
    assert response.status_code == 400
    assert "декодировать" in response.json()["detail"].lower()


def test_predict_huge_file_returns_413(mocked_app, huge_bytes):
    # Файл больше лимита. Проверка размера срабатывает до декодирования,
    # поэтому тип содержимого неважен — шлём просто бинарные данные.
    # Ожидаем код 413 (Payload Too Large) с сообщением про размер.
    response = mocked_app.post(
        "/predict",
        files={"file": ("huge.bin", huge_bytes, "application/octet-stream")},
    )
    assert response.status_code == 413
    assert "большой" in response.json()["detail"].lower()


def test_predict_invalid_conf_returns_422(mocked_app, sample_image_bytes):
    # Передали conf равный 2.0, что выходит за диапазон 0 до 1.
    # FastAPI с валидацией ge и le в описании параметра сам вернёт 422
    # ещё до выполнения нашего кода. Тест фиксирует что эта валидация
    # включена и не забыта.
    #
    # Параметр в URL пишу прямо в строке а не через params — так виднее
    # что именно тестируется.
    response = mocked_app.post(
        "/predict?conf=2.0",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 422