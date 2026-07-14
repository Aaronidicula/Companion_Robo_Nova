#!/usr/bin/env python3
"""
robo_orchestrator.py — runs on the Raspberry Pi.

Everything the robot says or does comes from the SLM running on the
laptop. This script does NOT contain a rule engine, math shortcuts, or
quick facts anymore — every transcript goes straight to the model. The
model decides what to do by calling MCP tools (speak, set_emotion,
trigger_gesture, end_session, imitate_pose); this script just runs the
loop that feeds it, and the tools themselves live in
mcp_tools_server.py, spawned as a subprocess over stdio, because that's
the process with the actual serial connection to the Arduino.

Point the Pi at your laptop automatically — no IP to configure:
    On the laptop:  OLLAMA_HOST=0.0.0.0:11434 ollama serve
                    python3 advertise_ollama.py   (keep running)
                    ollama pull qwen3:8b
    On the Pi:      pip install mcp zeroconf
This script finds the laptop via mDNS at startup (see
discover_laptop_ollama below). Set NOVA_LAPTOP_HOST to override.
"""

import os
import re
import json
import time
import socket
import asyncio
import threading
import subprocess
from datetime import datetime

os.environ['AUDIODEV'] = 'hw:3,0'

import ollama
import speech_recognition as sr
import pyaudio
from zeroconf import Zeroconf, ServiceBrowser, ServiceListener

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import numpy as np
from openwakeword.model import Model as OWWModel

import ctypes

# ── SUPPRESS ALSA/JACK C-LIBRARY NOISE ─────────────────
# PyAudio initializes every ALSA PCM device (including unused ones like
# surround51, iec958, modem, phoneline) to enumerate them, and each
# missing one prints a C-level warning that bypasses Python's stderr
# redirection. This installs a no-op error handler so those messages
# never print, without touching real Python exceptions/tracebacks.
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
    ctypes.c_int, ctypes.c_char_p
)

def _alsa_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(_alsa_error_handler)

try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except OSError:
    pass  # libasound not found — harmless, just skip silencing

# ── CONFIG ────────────────────────────────────────────
MODEL_NAME       = "qwen3:4b-instruct-2507-q4_K_M"  # official non-thinking variant — no <think> blocks, no toggle to fight
USE_THINKING     = None    # this model has no thinking mode at all; None means "don't send the think param"
KEEP_ALIVE       = "30m"   # keep the model resident on the laptop between turns instead of risking an unload/reload
MAX_TOOL_ROUNDS  = 4       # guardrail against tool-call loops

OWW_MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hey_nova.onnx")
OWW_THRESHOLD   = 0.5    # confidence score to trigger wake — tune based on real-world testing
OWW_CHUNK_SIZE  = 1280   # 80ms at 16kHz, openWakeWord's expected frame size

# RoboMic's USB audio card only supports 48kHz, not the 16kHz
# openWakeWord needs — same reason get_voice_input() below uses 48000.
# So we capture at 48kHz and downsample 3:1 before every predict() call.
MIC_SAMPLE_RATE     = 48000
OWW_TARGET_RATE     = 16000
DOWNSAMPLE_FACTOR   = MIC_SAMPLE_RATE // OWW_TARGET_RATE   # 3
OWW_READ_CHUNK      = OWW_CHUNK_SIZE * DOWNSAMPLE_FACTOR   # samples to read at 48kHz per cycle

OLLAMA_SERVICE_TYPE = "_nova-ollama._tcp.local."


class _OllamaListener(ServiceListener):
    """Watches for the laptop's advertise_ollama.py broadcast on the LAN."""
    def __init__(self):
        self.address = None
        self._found = threading.Event()

    def add_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        if info and info.addresses:
            ip = socket.inet_ntoa(info.addresses[0])
            self.address = f"http://{ip}:{info.port}"
            self._found.set()

    def update_service(self, zc, type_, name):
        pass

    def remove_service(self, zc, type_, name):
        pass

    def wait(self, timeout):
        return self._found.wait(timeout)


def discover_laptop_ollama(timeout: int = 15) -> str:
    """Finds the laptop running advertise_ollama.py via mDNS. Set
    NOVA_LAPTOP_HOST env var to skip discovery and use a fixed
    address instead (handy for testing, or if mDNS is blocked)."""
    manual = os.environ.get("NOVA_LAPTOP_HOST")
    if manual:
        print(f"Using manually set NOVA_LAPTOP_HOST={manual}")
        return manual

    print("🔍 Searching for the laptop's Ollama service on the LAN...")
    zc = Zeroconf()
    listener = _OllamaListener()
    ServiceBrowser(zc, OLLAMA_SERVICE_TYPE, listener)
    found = listener.wait(timeout)
    zc.close()
    if not found:
        raise RuntimeError(
            "Could not find the laptop's Ollama service after "
            f"{timeout}s. Make sure advertise_ollama.py is running on "
            "the laptop and both devices are on the same network."
        )
    print(f"✅ Found laptop at {listener.address}")
    return listener.address


client = None  # set in main() once the laptop is discovered


def configure_audio_levels():
    """USB audio cards reset to their power-on default gain on every
    reboot — that's why RoboMic's capture level kept creeping back up
    to ~100% and clipping (peaks pinned at 32767 with zero dynamic
    range, confirmed during wake-word debugging).

    Control names/values below come from a known-good manual boot
    script, not guesswork: on this hardware the capture control is
    named "Mic" (used in capture ["cap"] direction), not "Capture" —
    and Auto Gain Control has to be explicitly disabled or the chip
    keeps re-adjusting gain on its own regardless of what's set here.
    RoboSpk's "Mic" control is a monitor/loopback path, not an input —
    muting it avoids feedback/hum on playback.

    This is deliberately a hard override run every session, not a
    one-time fix — alsactl-based restore services often run too early
    in boot for USB cards and silently lose these settings."""
    try:
        subprocess.run(["amixer", "-c", "RoboMic", "sset", "Mic", "10,10", "cap"],
                        check=True, capture_output=True, text=True)
        subprocess.run(["amixer", "-c", "RoboMic", "sset", "Auto Gain Control", "off"],
                        check=True, capture_output=True, text=True)
        print("🎚️  RoboMic capture gain forced to 10/28, AGC off")
    except Exception as e:
        print(f"⚠️  Could not set RoboMic levels: {e}")
        print("    Check control names with: amixer -c RoboMic controls")

    try:
        subprocess.run(["amixer", "-c", "RoboSpk", "sset", "Speaker", "27,27"],
                        check=True, capture_output=True, text=True)
        subprocess.run(["amixer", "-c", "RoboSpk", "sset", "Mic", "0"],
                        check=True, capture_output=True, text=True)
        print("🔊 RoboSpk speaker level forced to 27, monitor muted")
    except Exception as e:
        print(f"⚠️  Could not set RoboSpk levels: {e}")
        print("    Check control names with: amixer -c RoboSpk controls")

# ── MEMORY ────────────────────────────────────────────
# Conversation history persists as JSON between runs, so Nova has
# context from earlier sessions instead of waking up blank every time.
# JSON (not CSV) because messages aren't flat rows — tool calls carry
# nested arguments, and JSON keeps that structure intact for the model
# to read back, where CSV would force it into a lossy flat string.
MEMORY_FILE         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nova_memory.json")
MEMORY_MAX_MESSAGES = 40  # how much history the file keeps on disk, across all past sessions
MEMORY_CONTEXT_MESSAGES = 12  # how much of that history actually gets loaded into a fresh
                               # session's starting context — kept separate from the file cap
                               # above so "does Nova remember past sessions" (yes, up to 40) and
                               # "how much does every single model call have to re-process"
                               # (much less, 12) are two independent knobs, not the same one.

# How long to wait in silence for the child to start talking at all,
# vs. the hard cap on a single utterance once they do start (see
# pause_threshold inside get_voice_input for what actually ends a
# phrase early — 0.8s of silence after speech, not this cap).
SESSION_LISTEN_TIMEOUT     = 30
SESSION_PHRASE_TIME_LIMIT  = 12  # was 8 — a bit more headroom for longer kid sentences


def _json_default(o):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    return str(o)


def load_memory() -> list:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE) as f:
                data = json.load(f)
            print(f"🧠 Loaded {len(data)} messages from previous sessions")
            return data
        except Exception as e:
            print(f"⚠️  Could not load memory file, starting fresh: {e}")
    return []


def save_memory(messages: list):
    try:
        # Never persist the system prompt itself here — it's re-added
        # fresh from SYSTEM_PROMPT on every startup in build_initial_messages,
        # so an old saved copy can't go stale and override a future edit.
        trimmed = [m for m in messages if m.get("role") != "system"][-MEMORY_MAX_MESSAGES:]
        with open(MEMORY_FILE, "w") as f:
            json.dump(trimmed, f, indent=2, default=_json_default)
    except Exception as e:
        print(f"⚠️  Could not save memory file: {e}")


def build_initial_messages() -> list:
    history = [m for m in load_memory() if m.get("role") != "system"]
    trimmed_for_context = history[-MEMORY_CONTEXT_MESSAGES:]
    if len(trimmed_for_context) < len(history):
        print(f"🧠 {len(history)} messages on disk, loading last "
              f"{len(trimmed_for_context)} into live context")
    return [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed_for_context


def get_time_of_day_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning!"
    elif 12 <= hour < 17:
        return "Good afternoon!"
    elif 17 <= hour < 21:
        return "Good evening!"
    else:
        return "Good night! ...well, good to see you up so late!"

SYSTEM_PROMPT = (
    "You are Nova, a friendly robot companion that talks to a child. "
    "You have no voice and no face of your own except through tools — "
    "you MUST call the speak tool to say anything out loud, plain text "
    "replies are never heard. Use set_emotion and trigger_gesture when "
    "they fit what you're saying. Keep spoken sentences short and simple. "
    "Call speak only ONCE per turn — say one thing, then stop and wait "
    "for the child to respond. Do not chain multiple things to say in a "
    "row; if you have more to say, save it for after they reply. "
    "Only call end_session when the child is clearly saying goodbye or "
    "goodnight."
)


# ── VOICE INPUT (unchanged from the Pi-local version) ─
def find_mic_device_index() -> int:
    mic_card_num = None
    try:
        with open("/proc/asound/cards") as f:
            for line in f:
                if "RoboMic" in line:
                    mic_card_num = int(line.strip().split()[0])
                    break
    except Exception as e:
        print(f"⚠️  Could not read /proc/asound/cards: {e}")
    if mic_card_num is None:
        print("⚠️  RoboMic not found — udev rule may not have applied")
        return 0
    p = pyaudio.PyAudio()
    found = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0 and f"hw:{mic_card_num}," in info['name']:
            found = i
    p.terminate()
    return found if found is not None else 0

MIC_DEVICE_INDEX = find_mic_device_index()

def get_voice_input(timeout: int = 8, phrase_time_limit: int = 4) -> str:
    r = sr.Recognizer()
    # Dynamic (not a hardcoded number) — auto-calibrates to whatever the
    # room actually sounds like right now. A fixed energy_threshold=50
    # was tuned back when RoboMic's gain was still near 100%; now that
    # gain is correctly forced down to 10/28, that number no longer
    # matches the real noise floor, which is exactly why pause_threshold
    # was never detecting silence and capture kept hitting the
    # phrase_time_limit ceiling on every single call regardless of how
    # long you actually spoke.
    r.dynamic_energy_threshold = True
    r.pause_threshold = 0.8
    mic = sr.Microphone(device_index=MIC_DEVICE_INDEX, sample_rate=48000)
    t0 = time.monotonic()
    try:
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=0.3)
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except sr.WaitTimeoutError:
        return ""
    capture_elapsed = time.monotonic() - t0
    t1 = time.monotonic()
    try:
        text = r.recognize_google(audio, language="en-IN")
        stt_elapsed = time.monotonic() - t1
        print(f"You said: {text}   [capture={capture_elapsed:.1f}s, stt={stt_elapsed:.1f}s]")
        return text.strip()
    except (sr.UnknownValueError, Exception) as e:
        stt_elapsed = time.monotonic() - t1
        print(f"🎤 Voice error: {e}   [capture={capture_elapsed:.1f}s, stt={stt_elapsed:.1f}s]")
        return ""


def downsample_48k_to_16k(chunk: np.ndarray) -> np.ndarray:
    """Cheap 3:1 downsample via block-averaging, which also acts as a
    simple low-pass filter to reduce aliasing before decimating — good
    enough for wake-word feature extraction without adding a scipy
    dependency for a proper resampler."""
    n = len(chunk) - (len(chunk) % DOWNSAMPLE_FACTOR)
    trimmed = chunk[:n].astype(np.int32)
    return trimmed.reshape(-1, DOWNSAMPLE_FACTOR).mean(axis=1).astype(np.int16)


def listen_for_wake_word(oww_model) -> bool:
    """Streams mic audio at its native 48kHz, downsamples to the 16kHz
    openWakeWord expects, and returns True the moment the wake word's
    confidence score crosses OWW_THRESHOLD. Fully offline, no per-cycle
    Google STT calls. Follow-up conversation within an active session
    still uses Google STT via get_voice_input() as before.

    Doesn't assume the prediction dict's key is literally "hey_nova" —
    that depends on how the model was named/trained and isn't worth
    guessing wrong silently. Instead it takes the highest-scoring key
    each frame, whatever it's called, and logs scores periodically so
    you can see real numbers while testing rather than a silent
    "not responding"."""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=MIC_SAMPLE_RATE,
        input=True,
        input_device_index=MIC_DEVICE_INDEX,
        frames_per_buffer=OWW_READ_CHUNK,
    )
    frame_count = 0
    try:
        while True:
            audio_chunk = stream.read(OWW_READ_CHUNK, exception_on_overflow=False)
            audio_48k = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_16k = downsample_48k_to_16k(audio_48k)
            prediction = oww_model.predict(audio_16k)

            if prediction:
                best_name, best_score = max(prediction.items(), key=lambda kv: kv[1])
            else:
                best_name, best_score = None, 0.0

            frame_count += 1
            if frame_count % 12 == 0:  # roughly once a second — enough to watch live, not spammy
                peak = int(np.abs(audio_48k).max())
                rms = float(np.sqrt(np.mean(audio_48k.astype(np.float64) ** 2)))
                print(f"   [audio] peak={peak:5d} rms={rms:7.1f}   [wake scores] {prediction}")

            if best_score >= OWW_THRESHOLD:
                print(f"👂 Wake word detected: {best_name} (score={best_score:.3f})")
                return True
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def load_oww_model():
    """openwakeword's Model() constructor signature has changed across
    package versions — some expect wakeword_model_paths (and infer the
    inference framework from the file extension), older ones expect
    wakeword_models plus an explicit inference_framework. Try the newer
    signature first, fall back to the older one, so this works
    regardless of which version got installed."""
    try:
        return OWWModel(wakeword_model_paths=[OWW_MODEL_PATH])
    except TypeError:
        return OWWModel(wakeword_models=[OWW_MODEL_PATH], inference_framework="onnx")

# ── TOOL SCHEMA CONVERSION (MCP -> Ollama) ─────────────
def mcp_tools_to_ollama_schema(mcp_tools) -> list:
    # Tools prefixed with "_" are internal/system tools (e.g. the
    # listening indicator) — the orchestrator calls them directly via
    # session.call_tool(), but they're never offered to the model.
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
        if not t.name.startswith("_")
    ]

# ── FALLBACK: catch a tool call the model wrote as text ─
# Some smaller models occasionally dump {"name": ..., "arguments": {...}}
# into message.content instead of using native tool_calls. If that
# happens, we try to salvage it rather than silently ignoring the turn.
def salvage_text_tool_call(content: str) -> dict | None:
    if not content:
        return None
    match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}', content)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

# ── AGENT LOOP ─────────────────────────────────────────
async def run_turn(session: ClientSession, ollama_tools: list, messages: list):
    for round_num in range(MAX_TOOL_ROUNDS):
        t0 = time.monotonic()
        chat_kwargs = dict(
            model=MODEL_NAME,
            messages=messages,
            tools=ollama_tools,
            keep_alive=KEEP_ALIVE,
        )
        if USE_THINKING is not None:
            chat_kwargs["think"] = USE_THINKING
        resp = client.chat(**chat_kwargs)
        elapsed = time.monotonic() - t0
        print(f"⏱️  round {round_num + 1}: model call took {elapsed:.1f}s "
              f"(context = {len(messages)} messages)")
        msg = resp["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            salvaged = salvage_text_tool_call(msg.get("content", ""))
            if salvaged:
                tool_calls = [{"function": salvaged}]
            else:
                # Model produced plain text with no tool call — nudge it,
                # since speech must go through the speak tool.
                if msg.get("content"):
                    messages.append({
                        "role": "user",
                        "content": "Remember: call the speak tool to say that out loud."
                    })
                    continue
                break

        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            print(f"🔧 tool call: {name}({args})")
            try:
                result = await session.call_tool(name, args)
                result_text = "".join(getattr(c, "text", "") for c in result.content)
            except Exception as e:
                result_text = f"error: {e}"
            messages.append({
                "role": "tool",
                "content": result_text,
                "name": name,
            })

        if any(c["function"]["name"] == "end_session" for c in tool_calls):
            return True  # session ended, stop the outer loop too

        if any(c["function"]["name"] == "speak" for c in tool_calls):
            # She's said her one thing for this turn — stop here and hand
            # control back to listening, rather than letting the model
            # chain a second speak() (and a second TTS playback) before
            # the child gets a chance to respond.
            return False

    return False


async def run_active_session(session: ClientSession, ollama_tools: list, messages: list,
                              first_input: str | None = None) -> bool:
    """Runs one active listening session: no wake word needed for
    follow-ups, each listen waits up to 30s, and 30s of silence quietly
    ends the session (back to sleep, no shutdown). Returns True if the
    model called end_session during this session (caller should stop
    the whole program), False otherwise.

    first_input lets a session open with something already said —
    used both for a same-breath wake-word follow-up (if that's ever
    reintroduced) and for the startup greeting opening its own
    listening window without requiring "Hey Nova" first."""
    pending_input = first_input

    while True:
        if pending_input:
            user_input = pending_input
            pending_input = None
        else:
            # Ears on only for the actual capture window — not while
            # thinking or while she's talking, so the face honestly
            # reflects "listening now" rather than "session is active."
            await session.call_tool("_set_listening_indicator", {"active": True})
            user_input = get_voice_input(timeout=SESSION_LISTEN_TIMEOUT,
                                          phrase_time_limit=SESSION_PHRASE_TIME_LIMIT)
            await session.call_tool("_set_listening_indicator", {"active": False})

        if not user_input:
            print("😴 30s of silence — session ending, back to sleep.")
            return False

        messages.append({"role": "user", "content": user_input})
        save_memory(messages)

        ended = await run_turn(session, ollama_tools, messages)
        save_memory(messages)

        if ended:
            return True


async def main():
    global client
    configure_audio_levels()

    laptop_host = discover_laptop_ollama()
    client = ollama.Client(host=laptop_host)

    print("🔊 Loading wake word model...")
    oww_model = load_oww_model()
    print("✅ Wake word model loaded")

    server_params = StdioServerParameters(
        command="python3",
        args=["mcp_tools_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            ollama_tools = mcp_tools_to_ollama_schema(mcp_tools)

            messages = build_initial_messages()
            print("🤖 Nova ready — say 'Hey Nova' to start talking.\n")

            greeting = f"{get_time_of_day_greeting()} I'm Nova! Want to talk or play a game?"
            print(f"👋 Startup greeting: {greeting}")
            await session.call_tool("set_emotion", {"emotion": "happy"})
            await session.call_tool("trigger_gesture", {"gesture": "wave"})
            await session.call_tool("speak", {"text": greeting})

            # The startup greeting opens its own 30s listening session,
            # same as a wake word would — no need to say "Hey Nova"
            # again if the child responds right away.
            print("👂 Listening for a response to the startup greeting...")
            ended = await run_active_session(session, ollama_tools, messages)

            while not ended:
                try:
                    # SLEEPING: continuously streaming audio through
                    # openWakeWord until the wake word fires. Nothing
                    # here reaches the model or Google STT.
                    listen_for_wake_word(oww_model)

                    print("👂 Wake word detected: Hey Nova — session active")
                    ended = await run_active_session(session, ollama_tools, messages)

                except KeyboardInterrupt:
                    print("\n👋 Interrupted — shutting down loop.")
                    break
                except Exception as e:
                    import traceback
                    print(f"Error: {e}")
                    traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
