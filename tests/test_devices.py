from __future__ import annotations

import pytest

from chronicler.audio import devices


class FakeSoundcardMic:
    name = "Microphone (Razer Seiren V3 Mini)"

    def recorder(self, samplerate, blocksize):
        raise AssertionError  # what soundcard does for non-EXTENSIBLE mix formats


class FakeSd:
    def __init__(self, inputs):
        self._inputs = inputs

    def query_hostapis(self):
        return [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}]

    def query_devices(self):
        return self._inputs


RAZER_INPUTS = [
    {
        "name": "Microphone (Razer Seiren V3 Min",
        "hostapi": 0,
        "max_input_channels": 1,
        "default_samplerate": 44100.0,
    },
    {
        "name": "Microphone (Razer Seiren V3 Mini)",
        "hostapi": 1,
        "max_input_channels": 1,
        "default_samplerate": 44100.0,
    },
    {
        "name": "Microphone (Razer Seiren V3 Mini)",
        "hostapi": 2,
        "max_input_channels": 1,
        "default_samplerate": 48000.0,
    },
    {
        "name": "Speakers (Razer)",
        "hostapi": 2,
        "max_input_channels": 0,
        "default_samplerate": 48000.0,
    },
]


def test_portaudio_prefers_wasapi_and_accepts_truncated_mme_names() -> None:
    index, info = devices._portaudio_input(
        FakeSd(RAZER_INPUTS), "Microphone (Razer Seiren V3 Mini)"
    )
    assert index == 2 and info["default_samplerate"] == 48000.0
    mme_only = FakeSd(RAZER_INPUTS[:1])
    index, _ = devices._portaudio_input(mme_only, "Microphone (Razer Seiren V3 Mini)")
    assert index == 0


def test_portaudio_unknown_device() -> None:
    with pytest.raises(devices.AudioUnavailableError):
        devices._portaudio_input(FakeSd(RAZER_INPUTS), "Some Other Mic")


def test_open_capture_falls_back_to_portaudio_for_mics(monkeypatch) -> None:
    monkeypatch.setattr(devices, "_find_mic", lambda device_id: FakeSoundcardMic())
    opened = []

    class FakePortAudio:
        backend = "portaudio"
        samplerate = 48_000

        def __init__(self, name):
            opened.append(name)

    monkeypatch.setattr(devices, "_PortAudioCapture", FakePortAudio)
    capture = devices.open_capture("mic", None, blocksize=4800)
    assert capture.backend == "portaudio"
    assert opened == ["Microphone (Razer Seiren V3 Mini)"]


def test_open_capture_reports_both_failures(monkeypatch) -> None:
    monkeypatch.setattr(devices, "_find_mic", lambda device_id: FakeSoundcardMic())

    def broken(name):
        raise RuntimeError("no PortAudio")

    monkeypatch.setattr(devices, "_PortAudioCapture", broken)
    with pytest.raises(
        devices.AudioUnavailableError, match="soundcard: AssertionError; portaudio: no PortAudio"
    ):
        devices.open_capture("mic", None, blocksize=4800)


def test_loopback_does_not_fall_back(monkeypatch) -> None:
    monkeypatch.setattr(devices, "_find_loopback", lambda device_id: FakeSoundcardMic())
    with pytest.raises(devices.AudioUnavailableError, match="Could not open"):
        devices.open_capture("loopback", None, blocksize=4800)


class ConstantCapture:
    backend = "fake"
    samplerate = 48_000

    def __init__(self, value: float):
        self.value = value

    def read(self, frames: int):
        import time

        import numpy as np

        time.sleep(frames / self.samplerate)
        return np.full((frames, 2), self.value, dtype=np.float32)

    def close(self) -> None:
        pass


def test_live_mix_has_headroom_and_flags_clipping(monkeypatch) -> None:
    import numpy as np

    from chronicler.audio.source import LiveSource

    values = {"loopback": 0.8, "mic": 1.0}
    monkeypatch.setattr(
        devices, "open_capture", lambda kind, device_id, blocksize: ConstantCapture(values[kind])
    )
    source = LiveSource(None, None, mic_enabled=True)
    source.start()
    blocks = [source.read() for _ in range(15)]
    clipping = source.clipping()
    source.stop()

    mixed = np.concatenate(blocks)
    steady = mixed[mixed.size // 2 :]
    # (0.8 + 1.0) / 2 = 0.9: summed without clipping at 1.0.
    assert np.allclose(np.median(steady), 0.9, atol=0.02)
    assert clipping == {"system": False, "mic": True}
