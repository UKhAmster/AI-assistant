import wave; import numpy as np; import onnxruntime as ort
session = ort.InferenceSession('src/silero_vad.onnx', providers=['CPUExecutionProvider'])

wf = wave.open('perfect_test.wav', 'rb')
data = wf.readframes(wf.getnframes())
arr_all = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

state = np.zeros((2, 1, 128), dtype=np.float32)

max_conf = 0.0
for i in range(0, len(arr_all), 512):
    chunk = arr_all[i:i+512]
    if len(chunk) < 512: break
    max_val = np.max(np.abs(chunk))
    if max_val > 0.005: 
        vad_chunk = chunk / max_val
    else:
        vad_chunk = chunk
    out, state = session.run(None, {'input': np.expand_dims(vad_chunk, 0), 'sr': np.array([16000], dtype=np.int64), 'state': state})
    if out[0][0] > max_conf: max_conf = float(out[0][0])
print(max_conf)
