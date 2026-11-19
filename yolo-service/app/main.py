"""
FastAPI-приложение. REST API для детекции объектов.
"""
import io
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from PIL import Image

from app import config, inference


# Глобальное состояние приложения. Заполняется при старте сервиса и живёт
# всё время работы. Модель кладём сюда чтобы не грузить её на каждый запрос.
# class_names тоже здесь, чтобы эндпоинты не лазили в модель за словарём
# каждый раз — он не меняется.
app_state = {
    "model": None,
    "class_names": None,
}


@asynccontextmanager
async def lifespan(app):
    # Управление жизненным циклом. Код до yield выполняется при старте,
    # код после yield при остановке. FastAPI сам вызовет эту функцию
    # в нужный момент.
    #
    # Здесь грузим модель ровно один раз при старте.
    print("Загрузка модели...")
    model = inference.load_model(config.MODEL_PATH)
    app_state["model"] = model
    app_state["class_names"] = model.names
    print("Модель загружена.")

    # yield — разделитель. Всё до него это startup, всё после это shutdown.
    # FastAPI держит приложение живым пока мы стоим на этой строчке.
    yield

    # Остановка. Обнуляем ссылки чтобы освободить память.
    print("Остановка сервиса, освобождение модели.")
    app_state["model"] = None
    app_state["class_names"] = None


# Создаём приложение. Параметры title и description попадут в Swagger UI,
# который автоматически открывается на /docs. Удобно для проверки руками.
app = FastAPI(
    title="YOLOv8 Detection Service",
    description="Сервис детекции объектов на изображениях на базе YOLOv8n.",
    version="1.0.0",
    # Передаём функцию lifespan чтобы FastAPI знал что делать при старте
    # и остановке.
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    # Проверка работоспособности. Возвращает статус ok и загружена ли модель.
    #
    # Этот эндпоинт не требует загруженной модели и работает всегда, даже
    # во время старта. Зачем: Docker healthcheck дёргает его каждые тридцать
    # секунд. Если бы он падал пока модель грузится, Docker убил бы контейнер
    # не дождавшись загрузки. Поэтому health отвечает всегда.
    #
    # model_loaded показывает true или false, но на статус ответа это
    # не влияет. Клиент сам смотрит на это поле если ему важно.
    return {
        "status": "ok",
        "model_loaded": app_state["model"] is not None,
    }


@app.post("/predict")
async def predict(
    # Файл с картинкой. File с тремя точками значит поле обязательно,
    # без него FastAPI вернёт 422 автоматически.
    file: UploadFile = File(...),
    # Порог уверенности. Query значит параметр приходит в query-строке.
    # ge и le задают допустимый диапазон. Если пришлют что-то вне 0..1,
    # FastAPI сам вернёт 422, код даже не выполнится.
    conf: float = Query(config.DEFAULT_CONF_THRESHOLD, ge=0.0, le=1.0),
    # Порог IoU для NMS. То же самое, тоже от нуля до единицы.
    iou: float = Query(config.DEFAULT_IOU_THRESHOLD, ge=0.0, le=1.0),
):
    # Принимает одно изображение плюс два параметра, возвращает список
    # детекций в координатах исходника.
    #
    # Ошибки которые может вернуть: 400 на пустой файл или не-картинку,
    # 413 на слишком большой файл, 422 на невалидные conf или iou
    # (это уже автоматически через валидацию FastAPI).

    # Читаем не больше чем лимит плюс один байт. Зачем плюс один: если файл
    # больше лимита, мы про это узнаем по длине. Читать гигабайты целиком
    # в память только чтобы потом сказать что они слишком большие — плохая
    # идея.
    image_bytes = await file.read(config.MAX_FILE_SIZE_BYTES + 1)

    # Пустой файл. Отдельная проверка потому что пустота это частая ошибка
    # клиента, лучше вернуть понятное сообщение а не пытаться декодировать.
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Файл пустой")

    # Файл больше лимита. Мы прочитали лимит плюс один, значит если длина
    # больше лимита — он точно превышен, даже если на самом деле файл ещё
    # длиннее.
    if len(image_bytes) > config.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой (максимум {config.MAX_FILE_SIZE_BYTES} байт)",
        )

    # Пытаемся декодировать картинку. Любая ошибка превращается в 400.
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        # load заставляет Pillow реально прочитать пиксели. Без него open
        # может пройти успешно даже на битом файле, потому что читает только
        # заголовок. А упадёт потом, когда до пикселей дойдёт. load сразу
        # заставляет всё прочитать, тогда ошибка ловится здесь в одном месте.
        pil_image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")

    # Принудительно приводим к RGB. Картинка могла быть RGBA с прозрачностью,
    # чёрно-белая с одним каналом, палитровая. Все приводим к стандартному
    # RGB из трёх каналов, потому что модель ждёт именно его.
    pil_image = pil_image.convert("RGB")

    # PIL Image в numpy массив. Формат высота-ширина-каналы, тип uint8.
    image_array = np.array(pil_image)

    # Препроцесс: letterbox, нормализация, порядок осей, батч.
    input_tensor, meta = inference.preprocess_image(
        image_array, config.MODEL_INPUT_SIZE
    )

    # Прогоняем через модель и получаем сырой выход.
    raw_output = inference.run_inference(app_state["model"], input_tensor)

    # Постпроцесс: фильтр по уверенности, NMS, пересчёт координат в исходник.
    detections = inference.postprocess_detections(
        raw_output=raw_output,
        meta=meta,
        conf_threshold=conf,
        iou_threshold=iou,
        class_names=app_state["class_names"],
    )

    # Возвращаем результат. Размеры исходника тоже полезны клиенту: по ним
    # можно сразу нарисовать bbox без повторного открытия файла.
    return {
        "image_width": meta["original_width"],
        "image_height": meta["original_height"],
        "detections": detections,
    }