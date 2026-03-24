FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# СНАЧАЛА устанавливаем конфликтующий av нужной версии через conda
RUN conda install -y -c conda-forge "av=11.*"

# Копируем список зависимостей
COPY requirements.txt .

# ТЕПЕРЬ запускаем pip (убираем флаг --only-binary, так как av уже установлен)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код проекта
COPY src/main.py .
COPY src/silero_vad.onnx .

# Запускаем сервер
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
