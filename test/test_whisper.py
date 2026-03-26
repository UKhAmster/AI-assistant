from faster_whisper import WhisperModel
model = WhisperModel("small", device="cpu", compute_type="float32")
segments, _ = model.transcribe("test_call.wav", language="ru")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
