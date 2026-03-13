FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Устанавливаем Python и системные зависимости
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    ffmpeg pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Делаем python3.10 версией по умолчанию
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код проекта
# Копируем код проекта из папки src
COPY src/main.py .
COPY src/silero_vad.onnx .

# Запускаем сервер
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
