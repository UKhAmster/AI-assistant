import asyncio
import websockets
import wave
import json
import sys

# Настройки (замени IP на адрес твоей виртуалки, если запускаешь не на ней)
# Например: WS_URL = "ws://192.168.1.100:8001/ws"
WS_URL = "ws://127.0.0.1:8001/ws" 
CHUNK_BYTES = 1024
SAMPLE_RATE = 16000

async def receive_responses(websocket):
    """Фоновая задача для приема ответов от сервера"""
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data.get("status") == "success":
                print("\n" + "="*50)
                print(f"🗣️ РАСПОЗНАННЫЙ ТЕКСТ (Whisper):")
                print(f"   {data.get('transcription', '')}\n")
                
                print(f"🧠 ИЗВЛЕЧЕННЫЕ СУЩНОСТИ (Qwen NER):")
                entities = data.get('entities', {})
                print(json.dumps(entities, indent=4, ensure_ascii=False))
                print("="*50 + "\n")
            else:
                print(f"\n[СЕРВЕР]: {data}\n")
                
    except websockets.exceptions.ConnectionClosed:
        print("\nСоединение закрыто сервером.")

async def stream_audio(file_path: str):
    """Стриминг WAV файла на сервер"""
    try:
        wf = wave.open(file_path, 'rb')
    except Exception as e:
        print(f"Ошибка открытия файла: {e}")
        return

    # Строгая проверка формата (16kHz, 16-bit, Mono)
    if wf.getframerate() != SAMPLE_RATE or wf.getsampwidth() != 2 or wf.getnchannels() != 1:
        print("❌ ОШИБКА: Файл должен быть 16kHz, 16-bit, Mono!")
        wf.close()
        return

    async with websockets.connect(WS_URL) as websocket:
        print(f"🔗 Подключено к {WS_URL}. Начинаем стриминг аудио...")
        
        receiver_task = asyncio.create_task(receive_responses(websocket))

        while True:
            data = wf.readframes(CHUNK_BYTES // 2) 
            if not data:
                break
            
            await websocket.send(data)
            await asyncio.sleep(0.032) # Эмуляция реального времени (32мс)

        print("✅ Аудио закончилось. Шлем 'тишину' для срабатывания VAD...")
        
        # Шлем тишину 3 секунды, чтобы сервер понял, что фраза окончена
        silence_chunk = b'\x00' * CHUNK_BYTES
        for _ in range(100):
            await websocket.send(silence_chunk)
            await asyncio.sleep(0.032)

        print("⏳ Ожидаем ответ от LLM (до 15 секунд)...")
        for _ in range(150):
            await asyncio.sleep(0.1)

        receiver_task.cancel()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python client.py <путь_к_wav_файлу>")
        sys.exit(1)
        
    wav_file = sys.argv[1]
    asyncio.run(stream_audio(wav_file))