"""Audio device discovery and capture, on top of `soundcard` (PortAudio as a fallback).

"Loopback" devices capture what the computer is playing (Discord, VTT music,
the game) and "mics" capture the local player. On Windows loopback uses WASAPI,
on Linux the PulseAudio/PipeWire monitor sources, and on macOS a virtual driver
such as BlackHole is required.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

log = logging.getLogger(__name__)


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


def _find_loopback(device_id: str | None) -> Any:
    sc = _soundcard()
    if device_id:
        return sc.get_microphone(device_id, include_loopback=True)
    speaker = sc.default_speaker()
    return sc.get_microphone(speaker.id, include_loopback=True)


def _find_mic(device_id: str | None) -> Any:
    sc = _soundcard()
    if device_id:
        return sc.get_microphone(device_id, include_loopback=False)
    return sc.default_microphone()


# --- capture ------------------------------------------------------------------

# Preferred PortAudio host APIs, best first. WASAPI and Core Audio give full
# device names and native rates; MME truncates names to 31 characters.
HOST_API_PREFERENCE = ["Windows WASAPI", "Core Audio", "Windows DirectSound", "MME", "ALSA"]


class Capture(Protocol):
    """An open input stream."""

    backend: str
    samplerate: int

    def read(self, frames: int) -> np.ndarray: ...

    def close(self) -> None: ...


class _SoundcardCapture:
    backend = "soundcard"
    samplerate = 48_000

    def __init__(self, device: Any, blocksize: int):
        self._ctx = device.recorder(samplerate=self.samplerate, blocksize=blocksize)
        self._rec = self._ctx.__enter__()

    def read(self, frames: int) -> np.ndarray:
        return self._rec.record(numframes=frames)

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)


class _PortAudioCapture:
    """Fallback for microphones `soundcard` cannot open.

    soundcard's WASAPI backend only accepts devices whose shared-mode mix format
    is WAVE_FORMAT_EXTENSIBLE float; some USB mics (e.g. Razer Seiren V3 Mini)
    report plain PCM and fail with an AssertionError.
    """

    backend = "portaudio"

    def __init__(self, name: str):
        import sounddevice as sd

        index, info = _portaudio_input(sd, name)
        self.samplerate = int(info["default_samplerate"])
        self._stream = sd.InputStream(
            device=index, channels=1, samplerate=self.samplerate, dtype="float32"
        )
        self._stream.start()

    def read(self, frames: int) -> np.ndarray:
        data, _overflowed = self._stream.read(frames)
        return data

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


def _portaudio_input(sd: Any, name: str) -> tuple[int, dict[str, Any]]:
    apis = [a["name"] for a in sd.query_hostapis()]
    candidates = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] < 1:
            continue
        # MME cuts names at 31 characters, so also accept a prefix match.
        if info["name"] == name or (len(info["name"]) >= 31 and name.startswith(info["name"])):
            api = apis[info["hostapi"]]
            rank = HOST_API_PREFERENCE.index(api) if api in HOST_API_PREFERENCE else 99
            candidates.append((rank, index, info))
    if not candidates:
        raise AudioUnavailableError(f"PortAudio has no input device named '{name}'.")
    _, index, info = min(candidates, key=lambda c: c[0])
    return index, info


def _describe(error: BaseException) -> str:
    return str(error) or type(error).__name__


def open_capture(
    kind: Literal["loopback", "mic"], device_id: str | None, blocksize: int
) -> Capture:
    """Open a device for recording, falling back to PortAudio for awkward mics.

    Must be called on the thread that will read from it (WASAPI uses COM).
    """
    device = _find_loopback(device_id) if kind == "loopback" else _find_mic(device_id)
    try:
        return _SoundcardCapture(device, blocksize)
    except Exception as first:
        if kind == "loopback":
            raise AudioUnavailableError(
                f"Could not open '{device.name}': {_describe(first)}"
            ) from first
        try:
            capture = _PortAudioCapture(device.name)
        except Exception as second:
            raise AudioUnavailableError(
                f"Could not open the microphone '{device.name}' "
                f"(soundcard: {_describe(first)}; portaudio: {_describe(second)})."
            ) from second
        log.info("using PortAudio for %s (soundcard failed: %r)", device.name, first)
        return capture
