"""
Юнит-тесты на постпроцесс, то есть на пересчёт координат.

Модель здесь не грузится вообще. Мы подаём руками готовый тензор в том же
формате, что возвращает настоящая YOLO, и смотрим что вернёт наша функция
postprocess_detections.
"""
import numpy as np
import pytest
import torch

from app import inference


def make_fake_raw_output(detections, num_classes, num_anchors):
    # Собираем тензор такой же формы, как у настоящей yolov8n: [1, 84, 8400].
    # Только с заранее известным содержимым. Настоящая модель выдаёт такие же
    # числа, но мы не знаем заранее какие. А с фейком знаем точно, поэтому
    # можем предсказать что должно получиться после пересчёта. Это и есть
    # смысл юнит-теста: контролируемый вход, предсказуемый выход.
    total_columns = 4 + num_classes

    # float32 — важно, потому что настоящая модель возвращает именно float32.
    # На float64 могут вылезти мелкие расхождения при сравнении.
    output = np.zeros((1, total_columns, num_anchors), dtype=np.float32)

    for index in range(len(detections)):
        detection = detections[index]

        # Первые четыре канала — координаты бокса: центр по X, центр по Y,
        # ширина и высота. Именно в таком формате их выдаёт YOLO.
        output[0, 0, index] = detection["cx"]
        output[0, 1, index] = detection["cy"]
        output[0, 2, index] = detection["w"]
        output[0, 3, index] = detection["h"]

        # Канал класса считается как четыре плюс номер класса. Класс 0 попадёт
        # в канал 4, класс 1 в канал 5, и так далее до 83 для класса 79.
        # Ставим score только в канал нужного класса, остальные оставляем нулями.
        output[0, 4 + detection["class_id"], index] = detection["score"]

    # torch.from_numpy не копирует данные, а просто оборачивает numpy в torch.
    # Для теста этого достаточно, менять содержимое мы не собираемся.
    return torch.from_numpy(output)


def test_bbox_recalculated_to_original_coordinates():
    # Это главный тест. Если он зелёный, значит bbox в ответе сервиса будут
    # в пикселях исходного изображения, а не в координатах 640 на 640.
    #
    # Сценарий: исходник 1280 на 640. После letterbox до 640 на 640 получаем
    # масштаб 0.5 и вертикальный паддинг 160 (по 160 сверху и снизу, потому что
    # после масштабирования высота стала 320, и до 640 не хватает 320 пополам).
    #
    # Детекция в координатах 640 на 640: центр 320 320, размер 100 на 100.
    # В формате углов это 270 270 370 370. После обратного пересчёта должно
    # получиться 540 220 740 420 в пикселях исходника.
    raw_output = make_fake_raw_output(
        detections=[
            {"cx": 320.0, "cy": 320.0, "w": 100.0, "h": 100.0,
             "class_id": 0, "score": 0.9},
        ],
        num_classes=80,
        num_anchors=10,
    )

    # Метаданные letterbox. Масштаб 0.5 значит картинку уменьшили вдвое.
    # Паддинг по Y 160 значит сверху добавили 160 пикселей серого.
    # По X паддинга нет, картинка после масштаба занимает всю ширину.
    meta = {
        "scale": 0.5,
        "pad_x": 0.0,
        "pad_y": 160.0,
        "original_width": 1280,
        "original_height": 640,
    }

    detections = inference.postprocess_detections(
        raw_output=raw_output,
        meta=meta,
        conf_threshold=0.5,
        iou_threshold=0.5,
        class_names={0: "person"},
    )

    # Ждём ровно одну детекцию класса person с уверенностью 0.9.
    assert len(detections) == 1
    result = detections[0]
    assert result["class"] == "person"
    assert result["confidence"] == pytest.approx(0.9, abs=1e-5)

    # Проверяем все четыре координаты.
    assert result["bbox"][0] == pytest.approx(540.0, abs=0.1)
    assert result["bbox"][1] == pytest.approx(220.0, abs=0.1)
    assert result["bbox"][2] == pytest.approx(740.0, abs=0.1)
    assert result["bbox"][3] == pytest.approx(420.0, abs=0.1)


def test_low_confidence_filtered_out():
    # Детекция с уверенностью 0.1 должна быть отброшена, если порог 0.5.
    # Проверяем что фильтр по conf реально работает, а не просто стоит в коде
    # мёртвым грузом.
    raw_output = make_fake_raw_output(
        detections=[
            {"cx": 320.0, "cy": 320.0, "w": 100.0, "h": 100.0,
             "class_id": 0, "score": 0.1},
        ],
        num_classes=80,
        num_anchors=10,
    )
    meta = {
        "scale": 1.0, "pad_x": 0.0, "pad_y": 0.0,
        "original_width": 640, "original_height": 640,
    }

    detections = inference.postprocess_detections(
        raw_output=raw_output,
        meta=meta,
        conf_threshold=0.5,
        iou_threshold=0.5,
        class_names={0: "person"},
    )

    # Пустой список — фильтр сработал как надо.
    assert len(detections) == 0


def test_bbox_clipped_to_image_borders():
    # Если модель предсказала бокс, вылезающий за левый или верхний край,
    # после пересчёта координаты могут уйти в минус. Это некрасиво и ломает
    # клиентов, которые рисуют bbox. Поэтому координаты нужно обрезать по нулю.
    #
    # Здесь центр 10 10 и размер 200 на 200, значит левый верхний угол
    # в минус девяносто. После клиппинга должен стать нулём.
    raw_output = make_fake_raw_output(
        detections=[
            {"cx": 10.0, "cy": 10.0, "w": 200.0, "h": 200.0,
             "class_id": 0, "score": 0.9},
        ],
        num_classes=80,
        num_anchors=10,
    )
    meta = {
        "scale": 1.0, "pad_x": 0.0, "pad_y": 0.0,
        "original_width": 640, "original_height": 640,
    }

    detections = inference.postprocess_detections(
        raw_output=raw_output,
        meta=meta,
        conf_threshold=0.5,
        iou_threshold=0.5,
        class_names={0: "person"},
    )

    # Левый верхний угол должен упереться в ноль, а не уйти в минус.
    assert len(detections) == 1
    assert detections[0]["bbox"][0] == 0.0
    assert detections[0]["bbox"][1] == 0.0


def test_nms_suppresses_overlapping_boxes():
    # Модель почти всегда предсказывает несколько похожих боксов на один
    # и тот же объект. NMS должен оставить только самый уверенный.
    #
    # Даём два почти одинаковых бокса. Центры различаются на пять пикселей,
    # размер одинаковый сто на сто. IoU между ними около 0.82, что больше
    # порога 0.5. Значит менее уверенный должен быть отброшен.
    raw_output = make_fake_raw_output(
        detections=[
            {"cx": 320.0, "cy": 320.0, "w": 100.0, "h": 100.0,
             "class_id": 0, "score": 0.9},
            {"cx": 325.0, "cy": 325.0, "w": 100.0, "h": 100.0,
             "class_id": 0, "score": 0.5},
        ],
        num_classes=80,
        num_anchors=10,
    )
    meta = {
        "scale": 1.0, "pad_x": 0.0, "pad_y": 0.0,
        "original_width": 640, "original_height": 640,
    }

    detections = inference.postprocess_detections(
        raw_output=raw_output,
        meta=meta,
        conf_threshold=0.3,
        iou_threshold=0.5,
        class_names={0: "person"},
    )

    # Остался один бокс — это правильно. Но важно проверить и уверенность.
    # Если NMS случайно оставил слабый бокс, len тоже будет один. Проверка
    # на 0.9 гарантирует что оставлен именно более уверенный.
    assert len(detections) == 1
    assert detections[0]["confidence"] == pytest.approx(0.9, abs=1e-5)


def test_letterbox_and_reverse_recalculation():
    # Отдельно проверяем саму функцию letterbox_image, без участия постпроцесса.
    # Исходник 1280 на 480. Приводим к 640 на 640. Масштаб должен стать 0.5
    # (по ширине упираемся в 640, по высоте остаётся место), а по вертикали
    # появится паддинг по 200 сверху и снизу.
    #
    # np.full создаёт массив нужной формы со всеми одинаковыми значениями.
    # Здесь белый цвет везде, но для проверки размеров и паддинга конкретный
    # цвет не важен.
    image = np.full((480, 1280, 3), 255, dtype=np.uint8)

    padded, scale, pad_x, pad_y = inference.letterbox_image(
        image, target_size=640, padding_value=114
    )

    # Проверяем и метаданные (масштаб и паддинги), и форму результата.
    # Если кто-то случайно сломает округление и получит картинку 639 на 640,
    # тест поймает это на assert по shape.
    assert scale == pytest.approx(0.5, abs=1e-6)
    assert pad_x == pytest.approx(0.0, abs=1e-6)
    assert pad_y == pytest.approx(200.0, abs=1e-6)
    assert padded.shape[0] == 640
    assert padded.shape[1] == 640