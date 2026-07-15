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

── STT FIX NOTES (2026-07-14) ──────────────────────────────────────
Field log showed energy_threshold was stable across calls (~4400-5500,
nowhere near the 15000 ceiling) — so mis-hears were NOT caused by a
bad calibration snapshot. The tell was capture duration: failed
("couldn't understand it") turns clustered at long captures (10.9s,
12.8s, close to phrase_time_limit=12s) while successful turns were
short (2-4s). That means listen() often isn't finding 0.8s of
continuous silence to end the phrase — something mid-capture (a noise
burst: fan, servo settle, coil whine) is being read as "still
speech" and dragging extra non-speech audio into what gets sent to
STT, garbling it.

Changes made below, in order of confidence:
  1. dynamic_energy_adjustment_damping raised from SR's default 0.15
     to 0.30 — makes the live threshold react more slowly to a brief
     loud burst, so a one-off spike is less likely to get treated as
     "the speaker got louder, raise the bar" and then fail to drop
     back down before real silence follows.
  2. adjust_for_ambient_noise duration restored to 0.5s (was 1.0s) —
     the field data doesn't show this caused the current problem, but
     there's no upside to holding a longer window right after
     playback, and it reduces the chance the tail of Nova's own
     speech gets folded into the "room noise" estimate.
  3. Post-capture RMS of the actual returned AudioData is now logged
     alongside capture_elapsed. This is DIAGNOSTIC, not a fix — it's
     needed to confirm whether #1 actually shortened the long-capture
     failures, since we don't have visibility into audio during
     listen() itself with SR's blocking API.
  4. pause_threshold nudged to 0.9s (was 0.8s) — small extra headroom
     so a short benign gap doesn't get second-guessed as continued
     speech once combined with change #1.

None of this is guaranteed — it's the best-supported fix from the
data collected so far, not a confirmed root cause. Watch the new RMS
log line on the next real session: if long-capture mis-hears persist
and their RMS is elevated throughout (not just spiking briefly), the
noise-burst theory is wrong and the next thing to check is whether a
trigger_gesture/servo call is overlapping the listening window.
─────────────────────────────────────────────────────────────────────

── "COMFORTABLE TURN-TAKING" FIX NOTES (2026-07-15) ────────────────
Field feedback described two distinct, compounding problems that made
talking to Nova feel stressful — like the child had to watch for a
narrow window to speak in:

  A. "Sometimes it doesn't pick it up, I have to say it again."
     Root cause: there was a silent dead zone between the ears
     visually turning on and the mic actually being ready — the
     POST_SPEECH_SETTLE_S sleep plus AMBIENT_CALIBRATION_S ran before
     listen() even started, with zero signal to the child that this
     gap was happening. If they started talking right as the ears
     icon appeared, those first words were spoken into dead air.

  B. "Ears turn off in the middle of speaking, cuts off mid-sentence."
     Root cause: pause_threshold=0.9s is tuned for adult speech
     cadence. A child thinking mid-sentence routinely pauses longer
     than that, so listen() was reading a normal thinking-pause as
     "they're done" and ending the phrase early.

Fixes:
  1. pause_threshold raised 0.9s -> 1.6s, and non_speaking_duration
     added at 0.6s. Together these give real thinking-pauses room
     without needing the (much longer) phrase_time_limit to be the
     only thing capping a turn.
  2. SESSION_PHRASE_TIME_LIMIT raised 12s -> 18s to match the longer
     pause_threshold — otherwise the outer time limit would just
     become the new premature cutoff.
  3. The pre-listen "settle" sleep is now skipped when nothing was
     just played (e.g. right after a wake-word detection, as opposed
     to right after Nova finished speaking) — see
     settle_before_first_listen below. This removes most of the dead
     zone that caused problem A in the first place.
  4. An audible "go ahead, I'm listening" chirp now plays the instant
     the ears turn on (see mcp_tools_server.py's
     _set_listening_indicator), the same pattern Google/Alexa use, so
     the child gets an unambiguous cue instead of having to watch for
     a visual icon while distracted mid-play.
─────────────────────────────────────────────────────────────────────
"""

import os
import re
import json
import time
import socket
import audioop
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
MEMORY_FILE         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nova_memory.json")
MEMORY_MAX_MESSAGES = 40
MEMORY_CONTEXT_MESSAGES = 12

# Raised 12s -> 18s alongside the longer pause_threshold below — the
# outer time limit needs headroom too, or it just becomes the new
# premature cutoff instead of pause_threshold.
SESSION_LISTEN_TIMEOUT     = 30
SESSION_PHRASE_TIME_LIMIT  = 18
MAX_MISHEARD_RETRIES       = 3  # consecutive "heard something, couldn't understand it" attempts
                                 # before giving up and ending the session — without this cap,
                                 # background noise sitting at the calibrated threshold can trigger
                                 # false speech-detected loops that never end on their own

# dynamic_energy_threshold recalibrates every call based on what it just
# heard. Field logs show this room's real ambient noise floor sits
# around 3500-6000 RMS (Pi fan + USB audio hardware noise) — so a
# calibrated threshold in that range is accurate, not broken. Clamping
# it down to some fixed "quiet room" value would do the opposite of
# what's needed: it would make the mic treat that real ambient noise as
# speech. ENERGY_THRESHOLD_CEILING below is a diagnostic tripwire for a
# truly pathological spike (e.g. something briefly very loud right next
# to the mic), not a normal-operation clamp.
ENERGY_THRESHOLD_CEILING   = 15000

# Gives any acoustic reverb/echo from RoboSpk's just-finished playback a
# moment to fully decay in the room before the mic starts sampling
# ambient noise for calibration — without this, the calibration step
# right after Nova finishes speaking can pick up her own trailing audio
# as "the room's noise floor" and calibrate too high.
#
# NOTE: this settle sleep is now SKIPPED when nothing was just played
# (e.g. right after wake-word detection) — see settle_before_first_listen
# in run_active_session(). It's still applied after the startup greeting
# and after every one of Nova's own spoken turns, where the echo risk is
# real.
POST_SPEECH_SETTLE_S       = 0.5

# How long adjust_for_ambient_noise() samples before every listen().
# Restored to 0.5s (was 1.0s) — field data didn't show this caused the
# mis-hear problem, but a shorter window still reduces the odds of
# folding in a stray noise event, and gets the mic listening sooner.
AMBIENT_CALIBRATION_S      = 0.5


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
    "they fit what you're saying. "
    "Never use emojis in the speak text — the robot's voice reads them "
    "out loud as strange sounds, which confuses the child. Express "
    "excitement or feeling through set_emotion and trigger_gesture "
    "instead, and through the words themselves. "
    "Every spoken reply must be ONE short sentence, under 12 words. No "
    "exceptions — not two sentences, not a sentence plus a question, "
    "just one short sentence. Longer replies take much longer to "
    "actually speak out loud, which the child experiences as Nova "
    "going quiet and unresponsive. Say the single most important thing "
    "and stop. "
    "Call speak only ONCE per turn — say one thing, then stop and wait "
    "for the child to respond. Do not chain multiple things to say in a "
    "row; if you have more to say, save it for after they reply. "
    "Only call end_session when the child is clearly saying goodbye or "
    "goodnight."
)


# ── VOICE INPUT ────────────────────────────────────────
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

# Created once, reused across every call — dynamic_energy_threshold only
# actually stabilizes if it gets to see audio across many calls over time.
_recognizer = sr.Recognizer()
_recognizer.dynamic_energy_threshold = True
# Raised 0.9s -> 1.6s (see "COMFORTABLE TURN-TAKING" notes above). A
# child thinking mid-sentence routinely pauses longer than adult
# cadence; 0.9s was reading that as "done talking" and cutting them off
# mid-thought.
_recognizer.pause_threshold = 1.6
# New: keeps a little trailing silence in the capture itself, softening
# exactly where the cutoff lands rather than only changing when it
# triggers.
_recognizer.non_speaking_duration = 0.6
# Default is 0.15. Raising this makes the live energy threshold react
# MORE SLOWLY to a sudden loud burst mid-capture (fan surge, servo
# settle, coil whine) — the working theory for why some captures were
# running out to phrase_time_limit instead of ending on real silence.
_recognizer.dynamic_energy_adjustment_damping = 0.30


def _rms_of_audiodata(audio: "sr.AudioData") -> float:
    """Best-effort RMS of the captured AudioData, for diagnostic
    logging only — lets us see, after the fact, whether a long capture
    was long because of sustained noise throughout (bad calibration /
    genuinely noisy room) vs. a brief burst (supports the damping fix)
    vs. actual continued speech (kid just talked for a while, not a
    bug at all)."""
    try:
        return audioop.rms(audio.get_raw_data(), audio.sample_width)
    except Exception:
        return -1.0


def get_voice_input(timeout: int = 8, phrase_time_limit: int = 4) -> str | None:
    """Returns the transcribed text, None if nobody spoke at all within
    timeout (true silence), or "" if speech was captured but couldn't
    be transcribed (background noise, mumbling, STT hiccup). That
    distinction matters to the caller: real silence should end a
    session, a failed transcription should just prompt a retry."""
    mic = sr.Microphone(device_index=MIC_DEVICE_INDEX, sample_rate=48000)
    t0 = time.monotonic()
    try:
        with mic as source:
            _recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_CALIBRATION_S)
            if _recognizer.energy_threshold > ENERGY_THRESHOLD_CEILING:
                print(f"⚠️  energy_threshold {_recognizer.energy_threshold:.0f} is unusually "
                      f"high (ceiling {ENERGY_THRESHOLD_CEILING}) — leaving it as calibrated, "
                      f"just flagging in case this session sounds off")
            print(f"   [mic] energy_threshold={_recognizer.energy_threshold:.0f}")
            audio = _recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except sr.WaitTimeoutError:
        return None  # true silence — nobody started speaking at all
    capture_elapsed = time.monotonic() - t0
    capture_rms = _rms_of_audiodata(audio)
    ran_full_length = capture_elapsed >= (phrase_time_limit - 0.5)
    if ran_full_length:
        print(f"⚠️  capture ran ~to phrase_time_limit ({capture_elapsed:.1f}s / "
              f"{phrase_time_limit}s) — either genuinely long speech, or listen() "
              f"never found {_recognizer.pause_threshold}s of silence. "
              f"captured_rms={capture_rms:.0f}")
    t1 = time.monotonic()
    try:
        text = _recognizer.recognize_google(audio, language="en-IN")
        stt_elapsed = time.monotonic() - t1
        print(f"You said: {text}   [capture={capture_elapsed:.1f}s, stt={stt_elapsed:.1f}s, "
              f"rms={capture_rms:.0f}]")
        return text.strip()
    except sr.UnknownValueError:
        stt_elapsed = time.monotonic() - t1
        print(f"🎤 Heard something but couldn't understand it   "
              f"[capture={capture_elapsed:.1f}s, stt={stt_elapsed:.1f}s, rms={capture_rms:.0f}]")
        return ""  # captured audio, just not intelligible — not silence
    except Exception as e:
        stt_elapsed = time.monotonic() - t1
        print(f"🎤 Voice error: {e}   [capture={capture_elapsed:.1f}s, stt={stt_elapsed:.1f}s, "
              f"rms={capture_rms:.0f}]")
        return ""


def downsample_48k_to_16k(chunk: np.ndarray) -> np.ndarray:
    n = len(chunk) - (len(chunk) % DOWNSAMPLE_FACTOR)
    trimmed = chunk[:n].astype(np.int32)
    return trimmed.reshape(-1, DOWNSAMPLE_FACTOR).mean(axis=1).astype(np.int16)


def listen_for_wake_word(oww_model) -> bool:
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
            if frame_count % 12 == 0:
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
    try:
        return OWWModel(wakeword_model_paths=[OWW_MODEL_PATH])
    except TypeError:
        return OWWModel(wakeword_models=[OWW_MODEL_PATH], inference_framework="onnx")

def mcp_tools_to_ollama_schema(mcp_tools) -> list:
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
            return True

        if any(c["function"]["name"] == "speak" for c in tool_calls):
            return False

    return False


async def run_active_session(session: ClientSession, ollama_tools: list, messages: list,
                              first_input: str | None = None,
                              settle_before_first_listen: bool = True) -> bool:
    """
    settle_before_first_listen: whether to apply POST_SPEECH_SETTLE_S
    before the very FIRST listen() call of this session. Pass False
    when nothing was just played through RoboSpk (e.g. right after a
    wake-word detection) — there's no echo to let decay, so that sleep
    was pure dead air the child had to wait through before the mic
    actually opened. Pass True (the default) right after Nova has
    spoken — the startup greeting, or any turn where run_turn() just
    called speak() — where the echo-decay reasoning still applies.
    Every listen() after the first one in a session still gets the
    settle, since by then Nova's tool calls (speak, gestures) may have
    played audio in between.
    """
    pending_input = first_input
    misheard_streak = 0
    is_first_listen = True

    while True:
        if pending_input:
            user_input = pending_input
            pending_input = None
        else:
            # Ears on only for the actual capture window — not while
            # thinking or while she's talking, so the face honestly
            # reflects "listening now" rather than "session is active."
            # The tool call itself also fires the audible "go ahead"
            # chirp (see mcp_tools_server.py) at the exact moment the
            # mic is about to open.
            await session.call_tool("_set_listening_indicator", {"active": True})
            if not (is_first_listen and not settle_before_first_listen):
                time.sleep(POST_SPEECH_SETTLE_S)  # let any speaker echo decay before calibrating
            user_input = get_voice_input(timeout=SESSION_LISTEN_TIMEOUT,
                                          phrase_time_limit=SESSION_PHRASE_TIME_LIMIT)
            await session.call_tool("_set_listening_indicator", {"active": False})
            is_first_listen = False

        if user_input is None:
            print("😴 30s of silence — session ending, back to sleep.")
            # Ears off (above) and face back to a resting expression —
            # otherwise the OLED is left showing whatever emotion the
            # last spoken line set, even though Nova is no longer
            # actively listening or engaged.
            await session.call_tool("set_emotion", {"emotion": "neutral"})
            return False

        if user_input == "":
            misheard_streak += 1
            if misheard_streak >= MAX_MISHEARD_RETRIES:
                print(f"😴 {misheard_streak} misheard attempts in a row — "
                      f"giving up for now, back to sleep.")
                await session.call_tool("set_emotion", {"emotion": "neutral"})
                return False
            print(f"🎤 Didn't catch that — listening again "
                  f"({misheard_streak}/{MAX_MISHEARD_RETRIES}).")
            continue

        misheard_streak = 0
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
        env=os.environ.copy(),
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

            print("👂 Listening for a response to the startup greeting...")
            # Nova just spoke the greeting — echo-decay settle is warranted here.
            ended = await run_active_session(session, ollama_tools, messages)

            while not ended:
                try:
                    listen_for_wake_word(oww_model)

                    print("👂 Wake word detected: Hey Nova — session active")
                    # Nothing was just played through RoboSpk — skip the
                    # settle sleep on the first listen so the mic opens
                    # (and the "go ahead" chirp fires) immediately.
                    ended = await run_active_session(session, ollama_tools, messages,
                                                      settle_before_first_listen=False)

                except KeyboardInterrupt:
                    print("\n👋 Interrupted — shutting down loop.")
                    break
                except Exception as e:
                    import traceback
                    print(f"Error: {e}")
                    traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
