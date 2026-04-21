# Базовый образ с Python, PyTorch 2.1.0 и CUDA
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Отключаем интерактивные запросы (tzdata и др.)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

# Системные зависимости (ffmpeg для работы со звуком)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev \
    ffmpeg pkg-config libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости (кешируются при неизменном requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Копируем весь проект
COPY backend/ ./backend/

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001"]
