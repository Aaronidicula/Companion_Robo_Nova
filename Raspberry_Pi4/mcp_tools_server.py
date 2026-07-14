#!/usr/bin/env python3
"""
mcp_tools_server.py — runs on the Raspberry Pi.

Exposes the robot's physical actions as MCP tools. This process owns the
Arduino serial connection and the speaker — nothing else in the system is
allowed to write to serial or play audio directly. The laptop-side model
never touches hardware; it only ever asks (via a tool call) for one of
these functions to run, and the *orchestrator* on the Pi is the one that
actually invokes them here.

Run standalone for testing:
    python3 mcp_tools_server.py
Normally it's spawned as a subprocess by robo_orchestrator.py over stdio.
"""

import os
import re
import sys
import time
import wave
import subprocess

os.environ['JACK_NO_AUDIO_RESERVATION'] = '1'
os.environ['JACK_NO_START_SERVER'] = '1'

import serial
from mcp.server.fastmcp import FastMCP

import pishutdown_script  # your existing Pi shutdown helper


def log(msg: str):
    """Use this instead of print() anywhere in this file. Over MCP's
    stdio transport, stdout IS the JSON-RPC protocol channel between
    this process and robo_orchestrator.py — a plain print() writes into
    that same stream, which the parent reads as protocol messages, not
    something that shows up in your terminal. This is exactly why none
    of this file's print() output (Arduino connect status, Piper
    fallback messages, TTS timing) ever appeared in any log — it had
    nowhere visible to go. stderr is untouched by the protocol and is
    inherited straight through to the terminal, so that's where
    diagnostic output belongs."""
    print(msg, file=sys.stderr, flush=True)

# ── CONFIG ────────────────────────────────────────────
SERIAL_PORT      = "/dev/ttyACM0"
SERIAL_BAUD      = 9600
SERIAL_CMD_DELAY = 0.08

ALLOWED_EMOTIONS = {"happy", "excited", "calm", "concerned", "neutral"}
ALLOWED_GESTURES = {"wave", "nod", "shake", "cheer", "point"}

# ── PIPER TTS CONFIG ────────────────────────────────────
# Local, offline TTS — no network dependency, no API key. PIPER_MODEL_PATH
# must point at the .onnx voice file you downloaded (plus its .onnx.json
# sibling, which piper expects to sit next to it). PIPER_BIN is only used
# by the CLI fallback path below, in case the Python API can't load the
# voice for some reason.
PIPER_BIN = os.environ.get("PIPER_BIN", "piper")
PIPER_MODEL_PATH = os.environ.get(
    "PIPER_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "en_GB-semaine-medium.onnx"),
)
PIPER_TIMEOUT_S = 30

# Load the voice ONCE, at process startup, and keep it resident for the
# life of this server — this is the actual fix for the flat ~7-9s
# "synth" time that showed up in the logs regardless of sentence length.
# That flat cost was never per-word synthesis; it was subprocess.run()
# spawning a brand-new `piper` process on every single speak() call,
# which reloads the whole ONNX model from disk from scratch each time.
# PiperVoice.load() below does that load exactly once; every call to
# voice.synthesize_wav() afterward reuses the already-loaded model, the
# same way Ollama's keep_alive keeps a model resident on the laptop
# instead of reloading it per request.
_piper_voice = None
if os.path.exists(PIPER_MODEL_PATH):
    try:
        from piper import PiperVoice
        _t_load = time.monotonic()
        _piper_voice = PiperVoice.load(PIPER_MODEL_PATH)
        log(f"✅ Piper voice loaded and resident ({time.monotonic() - _t_load:.1f}s): "
            f"{PIPER_MODEL_PATH}")
    except Exception as e:
        log(f"⚠️  Could not load Piper voice at startup ({e}) — "
            f"speak() will fall back to the piper CLI / gTTS per call")
else:
    log(f"⚠️  Piper model not found at {PIPER_MODEL_PATH} — "
        f"speak() will fall back to the piper CLI / gTTS per call")

# ── EMOJI STRIPPING ──────────────────────────────────────
# The model keeps decorating replies with emoji ("Let's play! 🎯🚀✨").
# Piper doesn't skip these — it tries to phoneticize or stumbles on them,
# which is exactly the garbled/odd-pause audio being heard. Stripping
# happens here, at the TTS boundary, rather than only relying on a
# system-prompt instruction — a small model will drift back to emoji
# over a long conversation no matter what it's told, so this is the
# layer that actually guarantees clean audio regardless of what text
# comes in.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs (incl. extended-A, supplemental)
    "\U00002600-\U000027BF"  # misc symbols and dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicator symbols (flag emoji)
    "\U00002190-\U000021FF"  # arrows
    "\U00002B00-\U00002BFF"  # misc symbols and arrows
    "\U0001F000-\U0001F02F"  # mahjong/dominoes (rarely used, cheap to cover)
    "\U0000FE0F"             # variation selector-16 (emoji presentation)
    "\U0000200D"             # zero-width joiner (emoji sequences)
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    cleaned = _EMOJI_PATTERN.sub("", text)
    # Emoji removal can leave doubled-up spaces or trailing whitespace
    # ("Let's play!  " after stripping "🎯🚀✨") — collapse those too.
    return re.sub(r"\s+", " ", cleaned).strip()


mcp = FastMCP("robo-actions")

# ── SERIAL ────────────────────────────────────────────
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    log(f"✅ Arduino connected on {SERIAL_PORT}")
except Exception as e:
    log(f"⚠️  Arduino not connected: {e} — tools will no-op on serial")


def serial_send(command: str):
    if not (ser and ser.is_open):
        return
    try:
        ser.write(f"{command}\n".encode())
        time.sleep(SERIAL_CMD_DELAY)
    except Exception as e:
        log(f"Serial write error: {e}")


# ── TOOLS ─────────────────────────────────────────────
@mcp.tool()
def speak(text: str) -> str:
    """Speak a short sentence out loud through the robot's speaker and
    animate the mouth while talking. This is the ONLY way the robot
    produces audible speech — plain text replies are not heard by the
    child, so anything meant to be said out loud must go through this
    tool."""
    try:
        text = strip_emoji(text)
        tts_path = "resident-piper"
        t_synth = time.monotonic()

        if _piper_voice is not None:
            # Primary path: reuse the already-loaded model, no reload.
            try:
                with wave.open("/tmp/response.wav", "wb") as wav_file:
                    _piper_voice.synthesize_wav(text, wav_file)
            except Exception as e:
                log(f"🗣️  Resident Piper synth failed ({e}) — "
                    f"falling back to piper CLI for this line")
                tts_path = None
        else:
            tts_path = None

        if tts_path is None and os.path.exists(PIPER_MODEL_PATH):
            # Secondary path: the old CLI subprocess call — slow (reloads
            # the model from disk every time) but doesn't depend on the
            # Python API having loaded successfully at startup.
            tts_path = "cli-piper"
            try:
                result = subprocess.run(
                    [PIPER_BIN, "--model", PIPER_MODEL_PATH, "--output_file", "/tmp/response.wav"],
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=PIPER_TIMEOUT_S,
                )
                if result.returncode != 0:
                    log(f"🗣️  Piper CLI error (exit {result.returncode}): {result.stderr.strip()}")
                    tts_path = None
            except Exception as e:
                log(f"🗣️  Piper CLI failed ({e}) — falling back to gTTS for this line")
                tts_path = None
        elif tts_path is None:
            log(f"🗣️  Piper model not found at {PIPER_MODEL_PATH} — falling back to gTTS")

        synth_elapsed = time.monotonic() - t_synth

        t_convert = time.monotonic()
        if tts_path in ("resident-piper", "cli-piper"):
            # Normalize Piper's raw output (volume, sample rate, channels) to
            # match what aplay -D plughw:RoboSpk,0 expects downstream.
            os.system(
                "ffmpeg -y -i /tmp/response.wav -af 'volume=3.0' "
                "-ar 48000 -ac 1 /tmp/response_clean.wav 2>/dev/null"
            )
        else:
            # Final fallback: gTTS, only reached if Piper isn't configured
            # or every Piper path above failed — keeps the robot talking
            # no matter what, at the cost of needing network access.
            from gtts import gTTS
            gTTS(text=text, lang='en', slow=False).save("/tmp/response.mp3")
            os.system(
                "ffmpeg -y -i /tmp/response.mp3 -af 'volume=3.0' "
                "-ar 48000 -ac 1 /tmp/response_clean.wav 2>/dev/null"
            )
        convert_elapsed = time.monotonic() - t_convert

        log(f"🗣️  TTS timing: synth={synth_elapsed:.1f}s (path={tts_path or 'gtts'}), "
            f"convert={convert_elapsed:.1f}s — audio starts after this point")

        serial_send("SPEAK_START")
        os.system("aplay -D plughw:RoboSpk,0 /tmp/response_clean.wav 2>/dev/null")
        serial_send("SPEAK_END")
        return "spoken"
    except Exception as e:
        return f"error: {e}"


@mcp.tool()
def set_emotion(emotion: str) -> str:
    """Set the robot's face to one of: happy, excited, calm, concerned,
    neutral. Use this to reflect how the robot should feel about its
    own reply — it changes the face on the OLED display."""
    emotion = emotion.lower().strip()
    if emotion not in ALLOWED_EMOTIONS:
        return f"error: emotion must be one of {sorted(ALLOWED_EMOTIONS)}"
    serial_send(emotion)
    return f"emotion set to {emotion}"


@mcp.tool()
def trigger_gesture(gesture: str) -> str:
    """Trigger a one-off physical gesture on the robot's arms/head. One
    of: wave, nod, shake, cheer, point."""
    gesture = gesture.lower().strip()
    if gesture not in ALLOWED_GESTURES:
        return f"error: gesture must be one of {sorted(ALLOWED_GESTURES)}"
    serial_send(f"GESTURE:{gesture}")
    return f"gesture {gesture} triggered"


@mcp.tool()
def end_session(farewell_text: str) -> str:
    """End the conversation and power down the robot. Call this only
    when the child is clearly saying goodbye / goodnight / done talking.
    farewell_text is what the robot will say before shutting down."""
    speak(farewell_text)
    serial_send("END")
    if ser and ser.is_open:
        ser.close()
    log("👋 Session ended by model — triggering shutdown")
    pishutdown_script.trigger_system_halt()
    return "session ended"


@mcp.tool()
def _set_listening_indicator(active: bool) -> str:
    """Internal system tool — not for model use. Called by the
    orchestrator (not the model) to toggle the listening-ears overlay
    on the face while capturing audio after the wake word."""
    serial_send("LISTEN_ON" if active else "LISTEN_OFF")
    return "ok"


@mcp.tool()
def imitate_pose(pose_data: dict) -> str:
    """PLACEHOLDER — not implemented yet. Will accept MediaPipe pose
    landmarks and mirror the pose using the arm/head servos, for
    imitation games and simple dance moves. Currently a no-op."""
    log(f"[imitate_pose placeholder] received: {pose_data}")
    return "not implemented yet"


if __name__ == "__main__":
    serial_send("LOADING")
    serial_send("READY")  # no local model warm-up in this process anymore to gate this on —
                           # without this, the Arduino stays in STATE_LOADING forever and
                           # redraws the loading screen every 400ms, overwriting any emotion face
    mcp.run(transport="stdio")
