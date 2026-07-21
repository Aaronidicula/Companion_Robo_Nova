# Companion Robo Nova 🤖

**Nova** is a voice-driven robot companion built for a Raspberry Pi 4 + Arduino, powered by a local LLM running on a laptop. Say "Hey Nova," and it listens, thinks (via a locally-hosted small language model), talks back through offline text-to-speech, and animates its face/gestures on an Arduino-driven display and servos.

Everything runs on the local network — no cloud APIs, no accounts, no data leaving the house. The Pi handles ears/mouth/body; the laptop handles the "brain."

## How it works

The system is split across two machines that find each other automatically on the LAN (via mDNS — no IP addresses to configure):

```
┌─────────────────────────────┐        mDNS/LAN         ┌───────────────────────────────────────┐
│           Laptop            │◄────────────────────────┤              Raspberry Pi 4             │
│                              │                          │                                         │
│  ollama serve                │                          │  robo_orchestrator.py                  │
│   + qwen3:4b-instruct model  │                          │   • openWakeWord ("Hey Nova")          │
│  advertise_ollama.py         │──── broadcasts host ────►│   • SpeechRecognition (Google STT)     │
│   (zeroconf service)         │                          │   • conversation loop + memory (JSON)  │
│                              │                          │   • talks to Ollama over the network   │
└──────────────────────────────┘                          │            │  (MCP, stdio)             │
                                                            │            ▼                           │
                                                            │  mcp_tools_server.py                   │
                                                            │   • owns the Arduino serial connection │
                                                            │   • owns the speaker                   │
                                                            │   • tools: speak / set_emotion /       │
                                                            │     trigger_gesture / end_session /    │
                                                            │     imitate_pose (WIP)                 │
                                                            └───────────────────────────────────────┘
```

1. **Wake word** — `openWakeWord` continuously listens on the Pi's mic for "Hey Nova" using a custom-trained ONNX model (`hey_nova.onnx`).
2. **Listen** — once woken, audio is captured with `SpeechRecognition` and transcribed via Google's speech-to-text, with a "go ahead, I'm listening" audio + visual cue so the child knows exactly when to speak.
3. **Think** — the transcript is sent to a small language model (`qwen3:4b-instruct`) served by `ollama` on the laptop. The model decides what to do entirely through tool calls — there is no hardcoded rule engine.
4. **Act** — the model calls MCP tools exposed by the Pi (`speak`, `set_emotion`, `trigger_gesture`, `end_session`) which the orchestrator forwards to `mcp_tools_server.py`, the only process allowed to touch the Arduino serial port and the speaker.
5. **Speak** — replies are synthesized offline with a resident Piper TTS voice (falling back to the Piper CLI, then to `gTTS` if needed) and played through the robot's speaker while the Arduino animates the mouth/face.
6. **Remember** — conversation history is persisted to a local JSON file so Nova retains context across wake/sleep cycles.

## Repository layout

```
Companion_Robo_Nova/
├── Laptop/
│   └── advertise_ollama.py     # Broadcasts the laptop's Ollama endpoint over mDNS
├── Raspberry_Pi4/
│   ├── robo_orchestrator.py    # Main loop: wake word → STT → LLM chat → tool calls
│   ├── mcp_tools_server.py     # MCP tool server: Arduino serial, TTS, speaker
│   └── hey_nova.onnx           # Trained openWakeWord model for "Hey Nova"
└── train_wakeword.ipynb        # Colab notebook to train a custom wake-word model
```

### `Laptop/advertise_ollama.py`
Runs alongside `ollama serve` on the laptop. Broadcasts the laptop's Ollama endpoint on the LAN via mDNS (`_nova-ollama._tcp.local.`) so the Pi never needs a hardcoded IP — if the laptop's IP changes (new Wi-Fi, DHCP renewal, moving between rooms), it just re-advertises the current one.

### `Raspberry_Pi4/robo_orchestrator.py`
The main process on the Pi. Responsibilities:
- Discovers the laptop's Ollama service via mDNS (or a manually set `NOVA_LAPTOP_HOST`).
- Runs continuous wake-word detection with `openWakeWord`, downsampling the 48 kHz mic stream to the 16 kHz the model expects.
- Captures and transcribes speech with `speech_recognition`, tuned for children's speech cadence (longer pause thresholds so mid-sentence thinking pauses aren't mistaken for "done talking") and for a noisy Pi environment (dynamic energy threshold, RMS diagnostics).
- Drives the conversation loop against the Ollama model, feeding it MCP tool schemas and looping on tool calls until the model speaks or ends the session.
- Persists conversation memory to `nova_memory.json`, trimmed to a bounded number of messages.
- Forces consistent ALSA capture/playback gain levels on every run (USB audio cards reset to power-on defaults on reboot).
- Spawns `mcp_tools_server.py` as a subprocess and talks to it over the MCP stdio protocol.

### `Raspberry_Pi4/mcp_tools_server.py`
The only process allowed to write to the Arduino serial port or play audio. Exposes the robot's physical actions as MCP tools that the LLM calls:

| Tool | Purpose |
|---|---|
| `speak(text)` | Synthesizes and plays speech through the robot's speaker (resident Piper TTS → Piper CLI → gTTS fallback chain) and animates the mouth. This is the only way the robot produces audible speech. |
| `set_emotion(emotion)` | Sets the robot's face to `happy`, `excited`, `calm`, `concerned`, or `neutral` on the OLED display. |
| `trigger_gesture(gesture)` | Triggers a physical gesture (`wave`, `nod`, `shake`, `cheer`, `point`). |
| `end_session(farewell_text)` | Speaks a farewell and powers down the robot. Called only when the child is clearly saying goodbye. |
| `_set_listening_indicator(active)` | Internal (orchestrator-only) tool that toggles the listening-ears face overlay and plays an audible "go ahead, I'm listening" chirp when the mic opens. |
| `imitate_pose(pose_data)` | Placeholder — not implemented yet. Intended to mirror MediaPipe pose landmarks via the arm/head servos for imitation games. |

Also strips emoji from any text before TTS (the voice otherwise tries to phoneticize them into garbled audio) and generates a short two-tone "ready" chirp once at startup for the listening cue.

### `Raspberry_Pi4/hey_nova.onnx`
A custom-trained `openWakeWord` model for the "Hey Nova" wake phrase, produced by `train_wakeword.ipynb`.

### `train_wakeword.ipynb`
A self-contained Google Colab notebook for training your own `openWakeWord` wake-word model, patched against version-drift in openWakeWord's own official notebook (Python 3.12, torchaudio 2.x, newer Colab images, upstream config schema changes). It clones the required repos, downloads Piper TTS voices and background-noise/negative-sample datasets (FMA, ACAV100M), generates and augments training clips, and runs a hand-rolled PyTorch training loop that mirrors openWakeWord's `auto_train` curriculum (staged learning rate, hard-negative mining, checkpoint ensembling) before exporting the final `.onnx` model. Edit `TARGET_PHRASE` and `MODEL_NAME` near the top and run all cells (~75–90 minutes on a Colab GPU runtime).

## Hardware

- Raspberry Pi 4
- Arduino (owns the OLED face, servos/gestures, and the serial protocol that `mcp_tools_server.py` drives)
- USB microphone, expected to enumerate as ALSA card `RoboMic`
- USB speaker, expected to enumerate as ALSA card `RoboSpk`
- Laptop or desktop on the same LAN, capable of running `ollama` (GPU recommended for lower latency)

## Setup

### 1. Laptop — serve the model
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve &
python3 Laptop/advertise_ollama.py     # keep running — advertises this laptop on the LAN
ollama pull qwen3:4b-instruct-2507-q4_K_M
```
Requires: `pip install zeroconf`

### 2. Raspberry Pi — run the robot
```bash
pip install mcp zeroconf ollama SpeechRecognition PyAudio openwakeword numpy pyserial piper-tts gTTS
python3 Raspberry_Pi4/robo_orchestrator.py
```
- `mcp_tools_server.py` is spawned automatically as a subprocess — you don't run it directly (though you can run it standalone for testing tool calls in isolation).
- Set `NOVA_LAPTOP_HOST=http://<laptop-ip>:11434` to skip mDNS discovery and point at a fixed address.
- Set `PIPER_MODEL_PATH` if your Piper voice `.onnx` file lives somewhere other than next to `mcp_tools_server.py`.
- `ffmpeg` and ALSA utilities (`aplay`, `amixer`) must be available on the Pi.
- A Piper voice model (e.g. `en_GB-semaine-medium.onnx` + its `.onnx.json`) should be placed alongside `mcp_tools_server.py` for offline TTS; without it, `speak()` falls back to `gTTS` (requires network).

Once running, the robot greets you on startup and then waits for "Hey Nova" to start each conversation turn.

## Notes

- The system deliberately has **no rule engine or hardcoded responses** — every reply and action comes from the LLM's tool calls, kept short (one sentence, under 12 words) so the robot stays responsive for a child's attention span.
- Source files include detailed inline "field notes" documenting real debugging sessions (STT mis-hears, turn-taking timing, listening cues) — worth reading in `robo_orchestrator.py` and `mcp_tools_server.py` if you're tuning audio behavior on your own hardware.
