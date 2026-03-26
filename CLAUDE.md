# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Russian-language voice assistant for a college admissions office. It handles incoming "calls" over WebSocket, performing real-time speech-to-text, LLM-powered dialogue, and text-to-speech synthesis. The assistant (named "Ксения") can create CRM tickets (Bitrix24 leads) when a caller requests a callback.

**Stack:** FastAPI + WebSocket + Faster-Whisper (STT) + Silero VAD + Silero TTS + vLLM (Qwen 2.5-14B) + Bitrix24

---

## Architecture

The system is a **FastAPI WebSocket server** (`src/main.py`) with a pipeline of four engines initialized at startup:

1. **VAD** — Silero VAD (ONNX, CPU) detects speech/silence boundaries in 512-sample chunks
2. **STT** — Faster-Whisper (CUDA, float16) transcribes detected speech segments to Russian text
3. **LLM** — Qwen2.5-14B-Instruct-AWQ via vLLM, accessed through OpenAI-compatible API (`AsyncOpenAI`). Uses function calling (`create_ticket` tool) to capture caller name/phone/intent
4. **TTS** — Silero TTS v4 (CUDA) synthesizes Russian speech responses at 24kHz

Audio format throughout: **16kHz, 16-bit, mono PCM**.

### Data flow per dialogue turn

```
Caller (phone/browser)
        │  WebSocket — raw PCM bytes (16kHz, 16-bit, mono)
        ▼
┌─────────────────────────────────────────────────────┐
│                  FastAPI (main.py)                   │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌────────────────┐   │
│  │ VADEngine│   │ STTEngine│   │   LLMAgent     │   │
│  │  Silero  │──▶│  Whisper │──▶│  Qwen 2.5-14B  │   │
│  │  (ONNX)  │   │  (CUDA)  │   │  (via vLLM)    │   │
│  └──────────┘   └──────────┘   └───────┬────────┘   │
│                                        │             │
│                               ┌────────▼────────┐   │
│                               │   TTSEngine     │   │
│                               │  Silero v4 (ru) │   │
│                               └────────┬────────┘   │
└────────────────────────────────────────┼─────────────┘
                                         │  raw PCM bytes (24kHz)
                                         ▼
                                   Caller hears response
                                         │
                               (if ticket needed)
                                         ▼
                                  Bitrix24 CRM
                               (create_ticket tool call)
```

Steps for each turn:
1. **VAD** chunks incoming audio; accumulates speech frames, detects end-of-utterance via silence
2. **STT** transcribes the accumulated segment
3. **LLM** receives transcription + full `chat_history`, responds; optionally calls `create_ticket`
4. **TTS** synthesizes the reply text to PCM bytes
5. **WebSocket** streams raw PCM back to the caller; `send_to_bitrix()` fires as a background task

### Key files

| File | Purpose |
|---|---|
| `src/main.py` | All engines (VAD, STT, TTS, LLM) + WebSocket endpoint `/ws` |
| `src/silero_vad.onnx` | Pre-trained Silero VAD weights |
| `src/client.py` | Older test client — streams WAV, prints JSON (expects old JSON response format) |
| `src/client_headless.py` | Updated test client — streams WAV, saves received audio to `answer.wav` |
| `test/` | VAD tuning and diagnostic scripts |

### Key classes in `src/main.py`

| Class | Role |
|---|---|
| `VADEngine` | Wraps Silero ONNX; stateful (carries RNN state between chunks) |
| `STTEngine` | Wraps Faster-Whisper; runs transcription in a thread pool |
| `TTSEngine` | Wraps Silero TTS v4; voice `xenia`, 24kHz, returns raw PCM bytes |
| `LLMAgent` | Manages `chat_history`, system prompt, tool schema; calls vLLM |

### Deployment

Two Docker containers via `docker-compose.yml`:

| Container | Port | Description |
|---|---|---|
| `vllm-server` | 8000 | vLLM serving Qwen2.5-14B-AWQ, 40% GPU VRAM |
| `api-server` | 8001 | This FastAPI app — Whisper + Silero TTS on remaining VRAM |

Both require NVIDIA GPU with CUDA. Base image: `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04`.

---

## Commands

```bash
# Run server locally (without Docker)
uvicorn src.main:app --host 0.0.0.0 --port 8001

# Run via Docker Compose (builds and starts both vLLM + API)
docker compose up --build

# Test with a WAV file (must be 16kHz, 16-bit, mono)
python src/client_headless.py <path_to_wav>   # saves reply to answer.wav
python src/client.py <path_to_wav>            # prints transcription + entities to console

# Convert any audio to the required format
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 test.wav

# Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8000/v1` | vLLM endpoint (set automatically in compose) |
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-14B-Instruct-AWQ` | Model identifier for vLLM requests |

---

## WebSocket Protocol

- **Incoming:** raw PCM bytes (16kHz, 16-bit, mono) in 1024-byte chunks
- **Outgoing:** raw PCM bytes (24kHz, 16-bit, mono) — TTS output
- No JSON framing; purely binary audio in both directions
- A greeting is synthesized and sent to the caller immediately on connect

---

## Known Bugs (fix before production)

- **`chat_history` grows unbounded** — will exceed vLLM's `max_model_len=4096` on long calls and crash. Needs trimming to the last N messages (keep system prompt + recent turns).
- **Concurrent `process_turn` tasks** — rapid speech triggers multiple parallel processing tasks on the same session. Needs an `is_processing` guard flag.
- **`VADEngine.state` is not thread-safe** — the ONNX RNN state is a mutable numpy array mutated in place. Concurrent calls (possible via `create_task`) will corrupt it.

---

## Not Yet Implemented

- **Bitrix24 webhook** — `send_to_bitrix()` is a stub (`asyncio.sleep(1)`). Needs real `httpx.AsyncClient.post()` to the Bitrix24 REST webhook, with error handling and retry logic.
- **Phone number normalization** — LLM may return phone in any format ("восемь девятьсот...", "+7 900 ...", etc.). Needs normalization before sending to CRM.
- **Knowledge base / RAG** — LLM currently answers from general knowledge. College-specific documents (programmes, admission requirements, schedules) need to be indexed and retrieved.
- **Whisper model cache in Docker** — model is re-downloaded on every `docker compose up --build`. Add a volume mount (same pattern as vLLM's `~/.cache/huggingface`).

---

## Future Improvements

- **Barge-in (interruption)** — stop TTS playback when the caller starts speaking mid-response
- **Dialogue logging** — persist transcripts per session for analysis and fine-tuning
- **Latency metrics** — instrument each stage (VAD→STT→LLM→TTS) for monitoring

---

## Notes

- The project language (comments, prompts, logs) is primarily **Russian**
- `remote_ops.py`, `check_docker.py`, `patch_dockerfile.py`, `patch_remote.py` are ad-hoc deployment helper scripts for SSH operations against a remote GPU server — not part of the main application
- Two virtual environments exist: `.venv` (Python 3.10, primary) and `venv` (Python 3.12, secondary/staged)
- `src/client.py` expects the old JSON response format and will not display audio replies correctly — use `src/client_headless.py` for current testing