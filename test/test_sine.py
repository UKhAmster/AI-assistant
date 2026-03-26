import numpy as np
import onnxruntime as ort

session = ort.InferenceSession('src/silero_vad.onnx', providers=['CPUExecutionProvider'])
state = np.zeros((2, 1, 128), dtype=np.float32)

t = np.linspace(0, 3, 16000 * 3, endpoint=False)
# 400 Hz loud sine wave
audio = np.sin(2 * np.pi * 400 * t).astype(np.float32)

max_conf = 0
for i in range(0, len(audio), 512):
    chunk = audio[i:i+512]
    if len(chunk) < 512: break
    
    inputs = {
        'input': np.expand_dims(chunk, 0),
        'sr': np.array(16000, dtype=np.int64),
        'state': state
    }
    out, state = session.run(None, inputs)
    if out[0][0] > max_conf:
        max_conf = out[0][0]

print("Max confidence SINE WAVE:", max_conf)
