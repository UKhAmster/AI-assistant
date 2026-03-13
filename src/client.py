import asyncio
import websockets
import wave
import json
import sys
import os

# Принудительно очищаем переменные прокси для этого скрипта
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)

# Настройки
WS_URL = "ws://127.0.0.1:8001/ws"
CHUNK_BYTES = 1024
SAMPLE_RATE = 16000

async def receive_responses(websocket):
    """Фоновая задача для приема ответов от сервера"""
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            entities = data.get("entities", {})
            print("\n[SERVER RESPONSE]:")
            print(json.dumps(entities, ensure_ascii=False, indent=2))
            print("")
    except websockets.exceptions.ConnectionClosed:
        print("\nСоединение закрыто сервером.")

async def stream_audio(file_path: str):
    """Стриминг WAV файла на сервер"""
    try:
        wf = wave.open(file_path, 'rb')
    except Exception as e:
        print(f"Ошибка открытия файла: {e}")
        return

    # Проверка формата
    if wf.getframerate() != SAMPLE_RATE or wf.getsampwidth() != 2 or wf.getnchannels() != 1:
        print("ВНИМАНИЕ: Файл должен быть 16kHz, 16-bit, Mono!")
        print(f"Текущие параметры: {wf.getframerate()}Hz, {wf.getsampwidth()*8}-bit, {wf.getnchannels()} channels")
        wf.close()
        return

    async with websockets.connect(WS_URL) as websocket:
        print(f"Подключено к {WS_URL}. Начинаем стриминг...")
        
        # Запускаем слушателя в фоне
        receiver_task = asyncio.create_task(receive_responses(websocket))

        # Читаем и отправляем файл чанками
        while True:
            data = wf.readframes(CHUNK_BYTES // 2) 
            if not data:
                break
            
            await websocket.send(data)
            await asyncio.sleep(0.032) 

        print("\nСтриминг файла завершен. Досылаем 'тишину' для срабатывания VAD...")
        
        # ЭМУЛЯЦИЯ АТС: Шлем абсолютную тишину (нули) в течение 3 секунд
        # Сервер обработает эти нули, поймет, что человек замолчал,
        # отсчитает 1.2 секунды и принудительно запустит распознавание текста.
        silence_chunk = b'\x00' * CHUNK_BYTES
        for _ in range(100):
            await websocket.send(silence_chunk)
            await asyncio.sleep(0.032)

        print("\nОжидание завершения распознавания последних фраз сервера (до 10 секунд)...")
        # Ожидаем в течение 10 секунд, пока сервер не пришлет все ответы
        for _ in range(100):
            await asyncio.sleep(0.1)

        receiver_task.cancel()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python client.py <путь_к_wav_файлу>")
        sys.exit(1)
        
    wav_file = sys.argv[1]
    asyncio.run(stream_audio(wav_file))