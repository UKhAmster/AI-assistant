import wave; import numpy as np; import onnxruntime as ort
session = ort.InferenceSession('src/silero_vad.onnx', providers=['CPUExecutionProvider'])

def run_test(filename):
    wf = wave.open(filename, 'rb')
    data = wf.readframes(wf.getnframes())
    arr_all = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    state = np.zeros((2, 1, 128), dtype=np.float32)
    
    is_speaking = False
    silence_count = 0
    speech_detected = 0
    
    threshold = 0.001
    SILENCE_DUR = 1.2
    
    print(f"--- {filename} ---")
    for i in range(0, len(arr_all), 512):
        chunk = arr_all[i:i+512]
        if len(chunk) < 512: break
        
        out, state = session.run(None, {'input': np.expand_dims(chunk, 0), 'sr': np.array([16000], dtype=np.int64), 'state': state})
        conf = float(out[0][0])
        has_speech = conf > threshold
        
        if not is_speaking:
            if has_speech:
                is_speaking = True
                silence_count = 0
                print(f"Start speech at {i/16000:.2f}s, Conf: {conf:.4f}")
        else:
            if has_speech:
                silence_count = 0
            else:
                silence_count += 1
                if silence_count * 512 / 16000 >= SILENCE_DUR:
                    print(f"End speech at {i/16000:.2f}s")
                    is_speaking = False
                    silence_count = 0

run_test('test_call.wav')
run_test('perfect_test.wav')
