import wave; import numpy as np; import onnxruntime as ort
session = ort.InferenceSession('src/silero_vad.onnx', providers=['CPUExecutionProvider'])
state = np.zeros((2, 1, 128), dtype=np.float32)
wf = wave.open('test_call.wav', 'rb')
max_conf = 0
for _ in range(500):
    data = wf.readframes(512)
    if len(data) < 1024: break
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    
    # NORMALIZATION AGC
    max_val = np.max(np.abs(arr))
    if max_val > 0.001:
        arr = arr / max_val
        
    out, state = session.run(None, {'input': np.expand_dims(arr, 0), 'sr': np.array(16000, dtype=np.int64), 'state': state})
    if out[0][0] > max_conf: max_conf = out[0][0]

print('Max Conf with AGC:', max_conf)
