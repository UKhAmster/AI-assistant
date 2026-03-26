import torch
import torchaudio
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
wav = read_audio('test_call.wav')
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
print("Speech timestamps:", speech_timestamps)
