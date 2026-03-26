import wave, asyncio
async def test():
    from src.main import STTEngine, WhisperModel
    import numpy as np
    model = WhisperModel("small", device="cpu", compute_type="float32")
    stt = STTEngine(model)
    wf = wave.open("test_call.wav", "rb")
    audio_bytes = wf.readframes(wf.getnframes())
    text = await stt.transcribe(audio_bytes)
    print("TRANSCRIPT:", text)
asyncio.run(test())
