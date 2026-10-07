"""Audio device discovery on top of `soundcard`.

"Loopback" devices capture what the computer is playing (Discord, VTT music,
the game) and "mics" capture the local player. On Windows loopback uses WASAPI,
on Linux the PulseAudio/PipeWire monitor sources, and on macOS a virtual driver
such as BlackHole is required.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class DeviceInfo:
    id: str
    name: str
    kind: Literal["loopback", "mic"]
    is_default: bool = False


class AudioUnavailableError(RuntimeError):
    pass


def is_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


@cache
def _soundcard() -> Any:
    try:
        import soundcard
    except Exception as e:  # missing libpulse, no audio server, etc.
        raise AudioUnavailableError(f"Audio backend unavailable: {e}") from e
    return soundcard


def list_devices() -> list[DeviceInfo]:
    sc = _soundcard()
    out: list[DeviceInfo] = []
    try:
        default_speaker = sc.default_speaker()
    except Exception:
        default_speaker = None
    try:
        default_mic = sc.default_microphone()
    except Exception:
        default_mic = None

    for m in sc.all_microphones(include_loopback=True):
        if m.isloopback:
            is_default = bool(
                default_speaker is not None
                and (m.id == default_speaker.id or m.name == default_speaker.name)
            )
            out.append(DeviceInfo(str(m.id), m.name, "loopback", is_default))
        else:
            is_default = bool(default_mic is not None and m.id == default_mic.id)
            out.append(DeviceInfo(str(m.id), m.name, "mic", is_default))
    if not any(d.kind == "loopback" and d.is_default for d in out):
        # Linux monitors are named "Monitor of <speaker>"; match them by name.
        out = [
            DeviceInfo(d.id, d.name, d.kind, True)
            if d.kind == "loopback"
            and default_speaker is not None
            and default_speaker.name in d.name
            else d
            for d in out
        ]
    return out


def open_loopback(device_id: str | None) -> Any:
    sc = _soundcard()
    if device_id:
        return sc.get_microphone(device_id, include_loopback=True)
    speaker = sc.default_speaker()
    return sc.get_microphone(speaker.id, include_loopback=True)


def open_mic(device_id: str | None) -> Any:
    sc = _soundcard()
    if device_id:
        return sc.get_microphone(device_id, include_loopback=False)
    return sc.default_microphone()
