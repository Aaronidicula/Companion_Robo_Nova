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
import time
import subprocess

os.environ['JACK_NO_AUDIO_RESERVATION'] = '1'
os.environ['JACK_NO_START_SERVER'] = '1'

import serial
from gtts import gTTS
from mcp.server.fastmcp import FastMCP

import pishutdown_script  # your existing Pi shutdown helper

# ── CONFIG ────────────────────────────────────────────
SERIAL_PORT      = "/dev/ttyACM0"
SERIAL_BAUD      = 9600
SERIAL_CMD_DELAY = 0.08

ALLOWED_EMOTIONS = {"happy", "excited", "calm", "concerned", "neutral"}
ALLOWED_GESTURES = {"wave", "nod", "shake", "cheer", "point"}

mcp = FastMCP("robo-actions")

# ── SERIAL ────────────────────────────────────────────
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    print(f"✅ Arduino connected on {SERIAL_PORT}")
except Exception as e:
    print(f"⚠️  Arduino not connected: {e} — tools will no-op on serial")


def serial_send(command: str):
    if not (ser and ser.is_open):
        return
    try:
        ser.write(f"{command}\n".encode())
        time.sleep(SERIAL_CMD_DELAY)
    except Exception as e:
        print(f"Serial write error: {e}")


# ── TOOLS ─────────────────────────────────────────────
@mcp.tool()
def speak(text: str) -> str:
    """Speak a short sentence out loud through the robot's speaker and
    animate the mouth while talking. This is the ONLY way the robot
    produces audible speech — plain text replies are not heard by the
    child, so anything meant to be said out loud must go through this
    tool."""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("/tmp/response.mp3")
        os.system(
            "ffmpeg -y -i /tmp/response.mp3 -af 'volume=3.0' "
            "-ar 48000 -ac 1 /tmp/response_clean.wav 2>/dev/null"
        )
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
    print("👋 Session ended by model — triggering shutdown")
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
    print(f"[imitate_pose placeholder] received: {pose_data}")
    return "not implemented yet"


if __name__ == "__main__":
    serial_send("LOADING")
    mcp.run(transport="stdio")
