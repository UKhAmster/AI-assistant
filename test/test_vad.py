import wave
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("src/silero_vad.onnx", providers=["CPUExecutionProvider"])
state = np.zeros((2, 1, 128), dtype=np.float32)

wf = wave.open("perfect_test.wav", "rb")
chunk_bytes = 1024
max_conf = 0
while True:
    data = wf.readframes(512)
    if not data or len(data) < 1024:
        break
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    inputs = {
        'input': np.expand_dims(arr, 0),
        'sr': np.array([16000], dtype=np.int64),
        'state': state
    }
    out, state = session.run(None, inputs)
    if out[0][0] > max_conf:
        max_conf = out[0][0]

print("Max confidence:", max_conf)
