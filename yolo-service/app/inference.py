"""
Модуль inference: загрузка модели, препроцесс, инференс, постпроцесс.

Здесь вся ручная обработка изображения. Можно было бы вызвать model.predict
из ultralytics и получить готовые детекции одной строкой. Но тогда непонятно
что происходит внутри, и нечего тестировать. Поэтому все шаги — letterbox,
NMS, пересчёт координат — реализованы вручную. Каждый шаг можно
отдельно проверить.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

from app import config


def load_model(model_path):
    # Загружает модель YOLOv8. Вызывать нужно ровно один раз при старте
    # сервиса, а не на каждый запрос. Загрузка занимает секунды, и повторять
    # её на каждый запрос — самый быстрый способ убить производительность.
    #
    # model_path — путь к файлу весов. Если файл существует, грузим его.
    # Если нет, просим ultralytics скачать по имени yolov8n.pt.
    #
    # Импорт внутри функции специально. Если импортировать ultralytics наверху
    # файла, то тесты постпроцесса, которым модель вообще не нужна, всё равно
    # потянут за собой всю библиотеку. А она тяжёлая. Пусть грузится только
    # когда действительно нужна.
    from ultralytics import YOLO

    if Path(model_path).exists():
        model = YOLO(model_path)
    else:
        # Ultralytics сам сходит на сервер и скачает веса. Кладёт их либо
        # в кэш, либо в текущую рабочую директорию — зависит от версии.
        model = YOLO("yolov8n.pt")

    # Для обучения оставляют train, для предсказаний нужен eval.
    model.model.eval()

    return model


def letterbox_image(image_array, target_size, padding_value):
    # Приводит изображение к квадрату target_size на target_size с сохранением
    # пропорций. Проще говоря, если исходник не квадратный, добиваем его
    # серыми полосами до квадрата, но не растягиваем сам объект.
    #
    # Зачем не просто cv2.resize до квадрата: на растянутой картинке люди
    # становятся низкими и широкими, машины приплюснутыми. Модель на таком
    # не обучалась и будет ошибаться. Letterbox сохраняет пропорции,
    # а пустоту забивает нейтральным серым.
    #
    # image_array — картинка в формате высота-ширина-каналы, тип uint8,
    # три канала. Цвет RGB, не BGR.
    # target_size — желаемая сторона квадрата, у нас 640.
    # padding_value — цвет заливки полей, 0 до 255.
    #
    # Возвращаем четыре штуки: готовую квадратную картинку и три числа —
    # масштаб и отступы по каждой оси. Эти три числа сохраняются и потом
    # используются в постпроцессе чтобы пересчитать координаты обратно.

    # Размеры исходника.
    original_height = image_array.shape[0]
    original_width = image_array.shape[1]

    # Коэффициент масштабирования по каждой оси считаем отдельно.
    # Берём минимальный, чтобы картинка целиком влезла в квадрат. Если взять
    # максимальный, одна сторона упрётся в 640 а вторая вылезет за границу.
    scale_for_height = target_size / original_height
    scale_for_width = target_size / original_width
    if scale_for_height < scale_for_width:
        scale = scale_for_height
    else:
        scale = scale_for_width

    # Новые размеры после масштабирования. Округляем до целых пикселей.
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))

    # Меняем размер. Билинейная интерполяция — стандартный метод, быстрый
    # и достаточный для уменьшения картинки.
    resized = cv2.resize(
        image_array,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    # Сколько пикселей не хватает до квадрата по каждой оси.
    total_pad_horizontal = target_size - new_width
    total_pad_vertical = target_size - new_height

    # Делим пополам. Симметричный паддинг — это соглашение, которое сильно
    # упрощает обратный пересчёт в постпроцессе. Если бы вся пустота шла
    # снизу, пришлось бы помнить каждую сторону отдельно.
    # Целочисленное деление на два даёт левую или верхнюю половину.
    # Остаток достаётся правой или нижней.
    pad_left = total_pad_horizontal // 2
    pad_right = total_pad_horizontal - pad_left
    pad_top = total_pad_vertical // 2
    pad_bottom = total_pad_vertical - pad_top

    # Добиваем до квадрата. copyMakeBorder создаёт новую картинку нужного
    # размера и заливает поля указанным цветом.
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(padding_value, padding_value, padding_value),
    )

    return padded, scale, float(pad_left), float(pad_top)


def preprocess_image(image_array, target_size):
    # Готовит картинку к подаче в модель. Делает четыре вещи по порядку:
    # первое — letterbox, второе — нормализация пикселей к диапазону 0 до 1,
    # третье — перестановка осей из HWC в CHW, четвёртое — добавление
    # размерности батча.
    #
    # image_array — RGB-картинка, высота-ширина-каналы, uint8, значения 0..255.
    # Именно RGB. Если читал через OpenCV, тот отдаёт BGR, надо конвертировать
    # через cv2.cvtColor(image, cv2.COLOR_BGR2RGB). Модель на BGR-картинках
    # просто портит результат.
    #
    # target_size — входной размер модели, 640 для yolov8n.
    #
    # Возвращаем тензор и словарь с метаданными letterbox. Метаданные нужны
    # в постпроцессе чтобы вернуть координаты в исходную систему.

    # Шаг первый: letterbox до квадрата. Заодно получаем масштаб и отступы.
    padded, scale, pad_x, pad_y = letterbox_image(
        image_array, target_size, config.PADDING_COLOR
    )

    # Шаг второй: нормализация. Делим на 255 чтобы получить диапазон 0..1.
    # astype(np.float32) обязателен: без него деление выполнится в целых
    # числах и получится чёрная картинка из нулей (uint8 // 255 всегда 0).
    normalized = padded.astype(np.float32) / 255.0

    # Шаг третий: порядок осей. numpy и Pillow хранят картинку как
    # высота-ширина-каналы. PyTorch ждёт каналы-высота-ширина. Транспонирование
    # переставляет оси, данные те же, просто по-другому разложены в памяти.
    # Если пропустить, PyTorch либо ругнётся на форму, либо молча обработает
    # мусор.
    channels_first = np.transpose(normalized, (2, 0, 1))

    # Шаг четвёртый: батч. Нейросети всегда работают с батчем картинок.
    # Даже одну картинку надо оформить как батч из одной штуки, добавив
    # единицу спереди.
    batched = np.expand_dims(channels_first, axis=0)

    # Оборачиваем numpy в torch. from_numpy не копирует данные, только
    # оборачивает, поэтому это быстро.
    tensor = torch.from_numpy(batched)

    # Метаданные для обратного пересчёта. Всё что нужно помнить о том как
    # мы исказили картинку чтобы потом суметь вернуть координаты обратно.
    meta = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "original_width": int(image_array.shape[1]),
        "original_height": int(image_array.shape[0]),
    }

    return tensor, meta


def run_inference(model, input_tensor):
    # Прогоняет тензор через модель и возвращает сырой выход — без встроенного
    # постпроцесса ultralytics.
    #
    # Обычно model.predict сам делает и NMS, и пересчёт координат. Нам это
    # не подходит, мы хотим делать постпроцесс сами. Поэтому обращаемся
    # к внутренней нейросети напрямую через model.model, минуя удобную обёртку.
    #
    # model — загруженная YOLO.
    # input_tensor — тензор [1, 3, 640, 640] из препроцесса.
    #
    # Возвращаем сырой тензор формы [1, 4 + число_классов, число_якорей].
    # Для yolov8n с COCO это [1, 84, 8400]: 4 координаты, 80 классов, 8400 якорей.

    # torch.no_grad отключает построение графа вычислений. Нам не нужны
    # градиенты для предсказаний, а без этого флага PyTorch зря тратит память.
    with torch.no_grad():
        raw_output = model.model(input_tensor)

    # В некоторых версиях forward возвращает кортеж или список. Нам нужен
    # первый элемент — это и есть тензор предсказаний.
    if isinstance(raw_output, (list, tuple)):
        raw_output = raw_output[0]

    return raw_output


def compute_iou(box_a, box_b):
    # Считает Intersection over Union двух боксов. Это метрика насколько
    # два прямоугольника перекрываются. Ноль значит не пересекаются вообще,
    # единица значит идентичны. Нужна для NMS: если две рамки сильно
    # перекрываются, менее уверенную удаляем.
    #
    # box_a и box_b — списки из четырёх чисел в формате x1 y1 x2 y2.
    # Возвращаем число от нуля до единицы.

    # Координаты пересечения. Левый верхний угол пересечения — это максимум
    # из левых верхних углов обоих боксов. Правый нижний — минимум из правых
    # нижних.
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    # Ширина и высота пересечения. Если отрицательные — значит пересечения
    # нет, обрезаем до нуля.
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Площади каждого бокса.
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Площадь объединения по формуле включений-исключений. Просто сумма
    # площадей минус площадь пересечения, иначе пересечение посчитается
    # дважды.
    union_area = area_a + area_b - inter_area

    # Защита от деления на ноль. Если оба бокса вырождены в точку, объединение
    # будет нулевым. Тогда считаем что перекрытия нет.
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def apply_nms_per_class(boxes, scores, class_ids, iou_threshold):
    # Non-Maximum Suppression — подавление перекрывающихся боксов. Модель почти
    # всегда предсказывает несколько похожих боксов на один объект. NMS оставляет
    # самый уверенный, остальные дубликаты выкидывает.
    #
    # Делаем NMS отдельно по каждому классу, а не глобально. Потому что, если человек
    # стоит перед машиной, их боксы сильно перекрываются но это разные объекты.
    # Глобальный NMS подавил бы менее уверенного. NMS по классам сравнивает
    # человека только с человеком, машину только с машиной. Оба сохраняются.
    #
    # boxes — список боксов в формате x1 y1 x2 y2. В какой системе координат
    # неважно, главное чтобы все в одной.
    # scores — список уверенностей, по одному числу на бокс.
    # class_ids — список номеров классов, по одному числу на бокс.
    # iou_threshold — порог IoU. Если перекрытие больше порога, менее
    # уверенный бокс выкидывается.
    #
    # Возвращаем список индексов оставшихся боксов.

    # Собираем какие классы вообще встретились. Множество само уберёт дубликаты.
    unique_classes = set()
    for class_id in class_ids:
        unique_classes.add(class_id)

    kept_indices = []

    # Обрабатываем каждый класс независимо.
    for current_class in unique_classes:
        # Индексы всех боксов этого класса.
        indices_of_class = []
        for idx in range(len(class_ids)):
            if class_ids[idx] == current_class:
                indices_of_class.append(idx)

        # Сортируем по убыванию уверенности. Первым идёт самый уверенный,
        # его точно оставляем.
        indices_of_class.sort(key=lambda i: scores[i], reverse=True)

        # Идём по списку и выкидываем тех кто сильно перекрывается
        # с уже оставленными.
        while len(indices_of_class) > 0:
            # Лучший в текущем списке — оставляем.
            best_idx = indices_of_class[0]
            kept_indices.append(best_idx)

            # Остальные сравниваем с лучшим по IoU. Тех у кого перекрытие
            # маленькое — оставляем на следующий круг.
            remaining = []
            # Цикл от единицы потому что нулевой уже обработан.
            for i in range(1, len(indices_of_class)):
                candidate_idx = indices_of_class[i]
                iou = compute_iou(boxes[best_idx], boxes[candidate_idx])
                if iou <= iou_threshold:
                    remaining.append(candidate_idx)
            # Переходим к следующему кругу с оставшимися.
            indices_of_class = remaining

    return kept_indices


def postprocess_detections(raw_output, meta, conf_threshold, iou_threshold, class_names):
    # Превращает сырой выход модели в список нормальных детекций в координатах
    # исходного изображения.
    #
    # Шаги по порядку:
    #   первое — разложить тензор по якорям, чтобы с каждым работать отдельно
    #   второе — для каждого якоря найти класс с максимальной вероятностью
    #   третье — отбросить якоря с уверенностью ниже порога
    #   четвёртое — перевести cx cy w h в x1 y1 x2 y2
    #   пятое — NMS чтобы убрать дубликаты
    #   шестое — пересчитать координаты обратно в исходник
    #   седьмое — обрезать по границам картинки
    #
    # raw_output — сырой тензор от модели, [1, 4 + число_классов, число_якорей].
    # meta — метаданные letterbox из препроцесса: масштаб, отступы, размеры
    # исходника.
    # conf_threshold — порог уверенности от нуля до единицы.
    # iou_threshold — порог IoU для NMS от нуля до единицы.
    # class_names — словарь номер-класса к имени. Обычно model.names.
    #
    # Возвращаем список словарей, по одному на детекцию. У каждого три поля:
    # имя класса, уверенность, bbox из четырёх координат в пикселях исходника.

    # Убираем размерность батча и переставляем оси. Было [1, 84, 8400],
    # после первого шага [84, 8400], после транспонирования [8400, 84].
    # Теперь каждая строка — это один якорь со всеми его числами.
    prediction_tensor = raw_output[0]
    prediction_tensor = prediction_tensor.transpose(0, 1)

    # tolist превращает тензор в обычные списки Python. С ними удобнее
    # и быстрее работать через индексы.
    predictions = prediction_tensor.tolist()

    # Сколько колонок в строке: 4 координаты плюс сколько-то классов.
    number_of_columns = len(predictions[0])
    number_of_classes = number_of_columns - 4

    # Сюда складываем тех кто прошёл фильтр по уверенности.
    candidate_boxes = []
    candidate_scores = []
    candidate_class_ids = []

    # Проходим по всем якорям.
    for candidate_index in range(len(predictions)):
        candidate = predictions[candidate_index]

        # Координаты в формате центра и размеров. Модель именно так их выдаёт.
        center_x = candidate[0]
        center_y = candidate[1]
        box_width = candidate[2]
        box_height = candidate[3]

        # Ищем класс с максимальной вероятностью. Классы начинаются с индекса
        # четыре, потому что первые четыре числа это координаты. Класс ноль
        # лежит в позиции четыре, класс один в позиции пять, и так далее.
        best_class_id = 0
        best_score = candidate[4]
        for class_offset in range(1, number_of_classes):
            score = candidate[4 + class_offset]
            if score > best_score:
                best_score = score
                best_class_id = class_offset

        # Фильтр по уверенности. Из 8400 якорей большинство это фон,
        # модель на них всё равно выдаёт какие-то вероятности но низкие.
        # Без фильтра тащили бы в NMS тысячи мусорных боксов.
        if best_score < conf_threshold:
            continue

        # Перевод из центра и размеров в углы. Центр минус половина стороны
        # даёт левый верхний угол, плюс половина даёт правый нижний.
        # Координаты пока всё ещё в системе 640 на 640.
        x1 = center_x - box_width / 2.0
        y1 = center_y - box_height / 2.0
        x2 = center_x + box_width / 2.0
        y2 = center_y + box_height / 2.0

        candidate_boxes.append([x1, y1, x2, y2])
        candidate_scores.append(best_score)
        candidate_class_ids.append(best_class_id)

    # Если после фильтра ничего не осталось, возвращаем пустой список.
    if len(candidate_boxes) == 0:
        return []

    # NMS по классам, чтобы убрать дубликаты на один объект.
    kept_indices = apply_nms_per_class(
        candidate_boxes,
        candidate_scores,
        candidate_class_ids,
        iou_threshold,
    )

    # Обратный пересчёт координат. Достаём из метаданных всё что нужно.
    scale = meta["scale"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    original_width = meta["original_width"]
    original_height = meta["original_height"]

    detections = []
    for index in kept_indices:
        box = candidate_boxes[index]

        # Убираем паддинг и делим на масштаб. Это обратная операция к тому
        # что делал letterbox. Ссначала вычитание педдинга, потом
        # деление на масштаб. Именно в таком порядке, потому что в препроцессе
        # было наоборот — сначала умножение на масштаб, потом добавление
        # педдинга. Обратные операции идут в обратном порядке.
        #
        # Если забыть этот пересчёт, bbox будут в координатах 640 на 640,
        # а картинка например 1920 на 1080. Все рамки съедутся в левый
        # верхний угол.
        x1 = (box[0] - pad_x) / scale
        y1 = (box[1] - pad_y) / scale
        x2 = (box[2] - pad_x) / scale
        y2 = (box[3] - pad_y) / scale

        # Клиппинг по границам исходника. После вычитания паддинга координаты
        # могли уйти в минус или вылезти за правый нижний край. Бокс
        # с отрицательной координатой бессмысленен, клиент такого не ждёт.
        if x1 < 0.0:
            x1 = 0.0
        if y1 < 0.0:
            y1 = 0.0
        if x2 > float(original_width):
            x2 = float(original_width)
        if y2 > float(original_height):
            y2 = float(original_height)

        # Собираем итоговую детекцию. Имя класса достаём из словаря,
        # уверенность и координаты уже готовы.
        detections.append({
            "class": class_names[candidate_class_ids[index]],
            "confidence": candidate_scores[index],
            "bbox": [x1, y1, x2, y2],
        })

    return detections