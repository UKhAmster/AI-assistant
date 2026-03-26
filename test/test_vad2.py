import wave; import numpy as np; import onnxruntime as ort
session = ort.InferenceSession('src/silero_vad.onnx', providers=['CPUExecutionProvider'])
state = np.zeros((2, 1, 128), dtype=np.float32)
wf = wave.open('test_call.wav', 'rb')
max_conf1 = 0; max_conf2 = 0
for _ in range(500):
    data = wf.readframes(512)
    if len(data) < 1024: break
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    
    # 1D array
    out1, state1 = session.run(None, {'input': np.expand_dims(arr, 0), 'sr': np.array([16000], dtype=np.int64), 'state': state})
    if out1[0][0] > max_conf1: max_conf1 = out1[0][0]
    
    # Scalar
    out2, state2 = session.run(None, {'input': np.expand_dims(arr, 0), 'sr': np.array(16000, dtype=np.int64), 'state': state})
    if out2[0][0] > max_conf2: max_conf2 = out2[0][0]
    
    state = state2 # advance state using scalar output

print('Max 1D:', max_conf1, 'Max Scalar:', max_conf2)
