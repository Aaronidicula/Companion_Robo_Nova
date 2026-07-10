#!/usr/bin/env python3
"""
advertise_ollama.py — runs on the laptop, alongside `ollama serve`.

Broadcasts the laptop's Ollama endpoint on the LAN via mDNS so the Pi
never needs a hardcoded IP address. If the laptop's IP changes (new
wifi network, DHCP renewal, moving between home/office, etc.) this
just re-advertises the current one — nothing to update by hand on
the Pi side.

Usage (run both, laptop side):
    OLLAMA_HOST=0.0.0.0:11434 ollama serve &
    python3 advertise_ollama.py

Requires: pip install zeroconf
"""

import socket
import time

from zeroconf import ServiceInfo, Zeroconf

SERVICE_TYPE = "_nova-ollama._tcp.local."
SERVICE_NAME = f"nova-laptop.{SERVICE_TYPE}"
PORT = 11434


def get_local_ip() -> str:
    """LAN IP the laptop would use to reach the internet — this is the
    IP the Pi needs, not 127.0.0.1 or a Docker-internal address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def main():
    ip = get_local_ip()
    info = ServiceInfo(
        SERVICE_TYPE,
        SERVICE_NAME,
        addresses=[socket.inet_aton(ip)],
        port=PORT,
        properties={},
    )
    zc = Zeroconf()
    zc.register_service(info)
    print(f"📡 Advertising Ollama at {ip}:{PORT} on the LAN as {SERVICE_NAME}")
    print("   Leave this running alongside `ollama serve`. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        zc.unregister_service(info)
        zc.close()


if __name__ == "__main__":
    main()
