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
import socket
import asyncio
import threading

os.environ['AUDIODEV'] = 'hw:3,0'

import ollama
import speech_recognition as sr
import pyaudio
from zeroconf import Zeroconf, ServiceBrowser, ServiceListener

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── CONFIG ────────────────────────────────────────────
MODEL_NAME       = "qwen3:8b"   # drop to qwen3:4b if the laptop is CPU-only / low RAM
USE_THINKING     = True         # improves tool selection accuracy for qwen3
MAX_TOOL_ROUNDS  = 4            # guardrail against tool-call loops

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

# ── MEMORY ────────────────────────────────────────────
# Conversation history persists as JSON between runs, so Nova has
# context from earlier sessions instead of waking up blank every time.
# JSON (not CSV) because messages aren't flat rows — tool calls carry
# nested arguments, and JSON keeps that structure intact for the model
# to read back, where CSV would force it into a lossy flat string.
MEMORY_FILE         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nova_memory.json")
MEMORY_MAX_MESSAGES = 40  # trim oldest turns beyond this so context doesn't grow unbounded


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
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history

SYSTEM_PROMPT = (
    "You are Nova, a friendly robot companion that talks to a child. "
    "You have no voice and no face of your own except through tools — "
    "you MUST call the speak tool to say anything out loud, plain text "
    "replies are never heard. Use set_emotion and trigger_gesture when "
    "they fit what you're saying. Keep spoken sentences short and simple. "
    "Only call end_session when the child is clearly saying goodbye or "
    "goodnight."
)

# Wake word — the model never sees anything until one of these is heard.
# STT mishears "Nova" fairly often, so a few common near-misses are
# included; extend this list as you notice new ones in your logs.
WAKE_PATTERNS = ["hey nova", "hi nova", "hey nover", "a nova"]


def check_wake_word(text: str):
    """Returns (woke: bool, remainder: str|None). remainder is whatever
    was said right after the wake phrase in the same breath, e.g.
    'hey nova what's the weather' -> remainder = 'what's the weather'."""
    t = text.lower()
    for phrase in WAKE_PATTERNS:
        idx = t.find(phrase)
        if idx != -1:
            remainder = t[idx + len(phrase):].strip(" ,.!?")
            return True, (remainder or None)
    return False, None


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
    r.dynamic_energy_threshold = False
    r.energy_threshold = 50
    r.pause_threshold = 0.8
    mic = sr.Microphone(device_index=MIC_DEVICE_INDEX, sample_rate=48000)
    try:
        with mic as source:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except sr.WaitTimeoutError:
        return ""
    try:
        text = r.recognize_google(audio, language="en-IN")
        print(f"You said: {text}")
        return text.strip()
    except (sr.UnknownValueError, Exception) as e:
        print(f"🎤 Voice error: {e}")
        return ""

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
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=ollama_tools,
            think=USE_THINKING,
        )
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

    return False


async def main():
    global client
    laptop_host = discover_laptop_ollama()
    client = ollama.Client(host=laptop_host)

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

            while True:
                try:
                    # SLEEPING: short bursts, quietly, until the wake word
                    # shows up. Nothing here reaches the model.
                    heard = get_voice_input(timeout=8, phrase_time_limit=4)
                    if not heard:
                        continue

                    woke, remainder = check_wake_word(heard)
                    if not woke:
                        continue

                    print("👂 Wake word detected: Hey Nova — session active")
                    await session.call_tool("_set_listening_indicator", {"active": True})

                    # ACTIVE SESSION: no wake word needed for follow-ups.
                    # Each listen waits up to 30s for the child to speak
                    # again; if nothing comes, the session quietly ends
                    # and we go back to waiting for "Hey Nova" — no
                    # shutdown, just back to sleep.
                    pending_input = remainder
                    session_ended_by_model = False

                    while True:
                        user_input = pending_input or get_voice_input(timeout=30, phrase_time_limit=8)
                        pending_input = None

                        if not user_input:
                            print("😴 30s of silence — session ending, back to sleep.")
                            break

                        messages.append({"role": "user", "content": user_input})
                        save_memory(messages)

                        ended = await run_turn(session, ollama_tools, messages)
                        save_memory(messages)

                        if ended:
                            session_ended_by_model = True
                            break

                    await session.call_tool("_set_listening_indicator", {"active": False})

                    if session_ended_by_model:
                        break  # end_session tool already triggered Pi shutdown

                except KeyboardInterrupt:
                    print("\n👋 Interrupted — shutting down loop.")
                    break
                except Exception as e:
                    import traceback
                    print(f"Error: {e}")
                    traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
