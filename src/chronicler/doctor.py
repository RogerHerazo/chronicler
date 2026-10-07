"""Health checks: is everything Chronicler needs installed, running and fast enough?

Each check returns ok / warn / fail plus a concrete fix hint. A failing
*blocking* check prevents starting a session.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal

from chronicler.audio import devices
from chronicler.campaign.notes import DEFAULT_TOKEN_BUDGET, load_notes
from chronicler.config import Settings
from chronicler.transcribe import MODEL_SIZES_MB, cuda_device_count, model_is_cached, plan_device

Status = Literal["ok", "warn", "fail", "skip"]

# A 15-minute chunk needs roughly 2.5 GB of free disk per hour of FLAC + WAV.
MIN_FREE_GB = 3.0
REQUIRED_PACKAGES = [
    ("faster_whisper", "faster-whisper"),
    ("ctranslate2", "ctranslate2"),
    ("av", "av"),
    ("soundcard", "soundcard"),
    ("soundfile", "soundfile"),
    ("soxr", "soxr"),
    ("numpy", "numpy"),
    ("anthropic", "anthropic"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("keyring", "keyring"),
]


@dataclass
class CheckResult:
    id: str
    title: str
    status: Status
    detail: str
    fix: str = ""
    blocking: bool = True
    seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class Check:
    id: str
    title: str
    run: Callable[[], tuple[Status, str, str]]  # status, detail, fix
    blocking: bool = True
    slow: bool = False


def _python() -> tuple[Status, str, str]:
    v = sys.version_info
    detail = f"Python {v.major}.{v.minor}.{v.micro} on {platform.system()} {platform.machine()}"
    if v < (3, 11):
        return "fail", detail, "Install Python 3.11 or newer (https://www.python.org/downloads/)."
    return "ok", detail, ""


def _packages() -> tuple[Status, str, str]:
    missing, broken, found = [], [], []
    for module, dist in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            missing.append(dist)
            continue
        except Exception as e:  # installed but its native library is missing
            broken.append(f"{dist} ({type(e).__name__}: {e})")
            continue
        try:
            found.append(f"{dist} {importlib.metadata.version(dist)}")
        except importlib.metadata.PackageNotFoundError:
            found.append(dist)
    if missing:
        return (
            "fail",
            f"Missing: {', '.join(missing)}",
            "Reinstall Chronicler: uv tool install --reinstall chronicler-dnd",
        )
    if broken:
        hint = "Reinstall Chronicler."
        if any(b.startswith("soundcard") for b in broken) and platform.system() == "Linux":
            hint = "Install the PulseAudio client library: sudo apt install libpulse0"
        return "fail", f"Failed to load: {'; '.join(broken)}", hint
    return "ok", ", ".join(found), ""


def _audio(settings: Settings) -> Callable[[], tuple[Status, str, str]]:
    def run() -> tuple[Status, str, str]:
        if devices.is_wsl():
            return (
                "fail",
                "Running inside WSL, which cannot capture Windows audio.",
                "Install and run Chronicler on Windows itself (PowerShell: "
                "uv tool install chronicler-dnd, then chronicler).",
            )
        try:
            found = devices.list_devices()
        except devices.AudioUnavailableError as e:
            return "fail", str(e), "Make sure a sound server is running and audio devices exist."
        loopbacks = [d for d in found if d.kind == "loopback"]
        mics = [d for d in found if d.kind == "mic"]
        if not loopbacks:
            fix = "Chronicler records what your computer plays through a loopback device."
            if platform.system() == "Darwin":
                fix = (
                    "macOS needs a virtual audio driver: install BlackHole "
                    "(https://existential.audio/blackhole/), create a Multi-Output Device "
                    "with your speakers + BlackHole, and pick BlackHole as the system device."
                )
            return "fail", "No loopback (system audio) device found.", fix
        chosen = next((d for d in loopbacks if d.id == settings.loopback_device), None)
        chosen = chosen or next((d for d in loopbacks if d.is_default), loopbacks[0])
        detail = f"System audio: {chosen.name}"
        if settings.mic_enabled:
            if not mics:
                return (
                    "warn",
                    detail + ". No microphone found.",
                    "Connect a microphone, or turn the mic off in Settings if your own voice "
                    "already comes through system audio.",
                )
            mic = next((d for d in mics if d.id == settings.mic_device), None)
            mic = mic or next((d for d in mics if d.is_default), mics[0])
            detail += f". Microphone: {mic.name}"
        else:
            detail += ". Microphone: off"
        return "ok", detail, ""

    return run


def _gpu(settings: Settings) -> Callable[[], tuple[Status, str, str]]:
    def run() -> tuple[Status, str, str]:
        if settings.whisper_device == "cpu":
            return "ok", "GPU disabled in Settings; transcribing on the CPU.", ""
        count = cuda_device_count()
        if count == 0:
            status: Status = "fail" if settings.whisper_device == "cuda" else "warn"
            return (
                status,
                "No CUDA GPU detected; transcription will run on the CPU (slower).",
                "With an NVIDIA GPU, install the latest driver. CPU mode works but uses a "
                "smaller model.",
            )
        try:
            import ctranslate2

            # Loading cuBLAS/cuDNN happens lazily; force it with a tiny allocation.
            ctranslate2.get_supported_compute_types("cuda")
            from faster_whisper import WhisperModel

            if model_is_cached("tiny"):
                WhisperModel("tiny", device="cuda", compute_type="float16")
        except Exception as e:
            return (
                "fail",
                f"CUDA GPU found but it cannot be used: {e}",
                "Install the NVIDIA cuBLAS and cuDNN 9 libraries for CUDA 12 "
                "(pip install nvidia-cublas-cu12 nvidia-cudnn-cu12, or see the faster-whisper "
                "README), or set the transcription device to CPU in Settings.",
            )
        return "ok", f"{count} CUDA device(s) available.", ""

    return run


def _model(settings: Settings) -> Callable[[], tuple[Status, str, str]]:
    def run() -> tuple[Status, str, str]:
        plan = plan_device(settings.whisper_model, settings.whisper_device)
        if model_is_cached(plan.model):
            return "ok", f"Whisper '{plan.model}' is downloaded ({plan.device}).", ""
        size = MODEL_SIZES_MB.get(plan.model)
        size_txt = f" (~{size} MB)" if size else ""
        return (
            "warn",
            f"Whisper '{plan.model}' is not downloaded yet{size_txt}.",
            "It downloads automatically the first time you transcribe. Use 'Download now' to "
            "get it before your session.",
        )

    return run


def _speed(settings: Settings) -> Callable[[], tuple[Status, str, str]]:
    def run() -> tuple[Status, str, str]:
        import numpy as np

        from chronicler.transcribe import Transcriber

        plan = plan_device(settings.whisper_model, settings.whisper_device)
        if not model_is_cached(plan.model):
            return "skip", "Skipped until the Whisper model is downloaded.", ""
        # 30 s of a synthetic voice-band signal keeps the VAD from skipping it.
        sr = 16_000
        t = np.arange(sr * 30) / sr
        audio = (0.1 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(
            np.float32
        )
        rtf = Transcriber(plan, language="en").benchmark(audio)
        budget = settings.chunk_minutes * 60
        minutes = rtf * budget / 60
        detail = (
            f"{plan.model} on {plan.device}: about {minutes:.1f} min to transcribe a "
            f"{settings.chunk_minutes:g}-minute part."
        )
        if rtf > 0.8:
            return (
                "fail",
                detail + " That cannot keep up with live play.",
                "Choose a smaller Whisper model in Settings (e.g. 'base' or 'small').",
            )
        if rtf > 0.4:
            return "warn", detail + " Results will lag behind.", "Consider a smaller model."
        return "ok", detail, ""

    return run


def _llm(settings: Settings) -> Callable[[], tuple[Status, str, str]]:
    def run() -> tuple[Status, str, str]:
        from chronicler.llm import make_provider

        status = make_provider(settings).check()
        return ("ok" if status.ok else "fail"), status.detail, status.fix_hint

    return run


def _storage(settings: Settings) -> Callable[[], tuple[Status, str, str]]:
    def run() -> tuple[Status, str, str]:
        problems = []
        for label, path in (
            ("Data folder", settings.data_dir),
            ("Export folder", settings.resolved_export_dir),
        ):
            try:
                path.mkdir(parents=True, exist_ok=True)
                probe = path / ".chronicler-write-test"
                probe.write_text("ok")
                probe.unlink()
            except OSError as e:
                problems.append(f"{label} {path} is not writable ({e.strerror or e})")
        if problems:
            return "fail", "; ".join(problems), "Pick a different folder in Settings."
        free_gb = shutil.disk_usage(settings.data_dir).free / 1e9
        detail = f"Data in {settings.data_dir}, exports in {settings.resolved_export_dir}. "
        detail += f"{free_gb:.0f} GB free."
        if free_gb < MIN_FREE_GB:
            return "warn", detail, "A 4-hour session needs about 3 GB; free up some space."
        return "ok", detail, ""

    return run


def _notes(notes_dir: str | None) -> Callable[[], tuple[Status, str, str]]:
    def run() -> tuple[Status, str, str]:
        if not notes_dir:
            return (
                "skip",
                "No campaign notes folder set. The built-in tracker will learn as you play.",
                "",
            )
        bundle = load_notes(notes_dir)
        if bundle.error:
            return "fail", bundle.error, "Fix the folder path in Settings."
        if not bundle.files:
            return "warn", f"No .md or .txt files in {notes_dir}.", ""
        detail = f"{len(bundle.files)} file(s), about {bundle.tokens:,} tokens."
        if bundle.truncated:
            return (
                "warn",
                detail + f" {len(bundle.skipped)} file(s) skipped to stay under the "
                f"{DEFAULT_TOKEN_BUDGET:,}-token budget.",
                "Point the folder at your most important notes (characters, places, plot).",
            )
        return "ok", detail, ""

    return run


def build_checks(settings: Settings, notes_dir: str | None = None) -> list[Check]:
    return [
        Check("python", "Python", _python),
        Check("packages", "Libraries", _packages),
        Check("audio", "Audio devices", _audio(settings)),
        Check("gpu", "GPU acceleration", _gpu(settings), blocking=False),
        Check("model", "Whisper model", _model(settings), blocking=False),
        Check("speed", "Transcription speed", _speed(settings), slow=True),
        Check("llm", "Analysis provider", _llm(settings)),
        Check("storage", "Storage", _storage(settings)),
        Check("notes", "Campaign notes", _notes(notes_dir), blocking=False),
    ]


def run_check(check: Check) -> CheckResult:
    start = time.perf_counter()
    try:
        status, detail, fix = check.run()
    except Exception as e:
        status, detail, fix = "fail", f"Check crashed: {type(e).__name__}: {e}", ""
    return CheckResult(
        check.id,
        check.title,
        status,
        detail,
        fix,
        check.blocking,
        round(time.perf_counter() - start, 2),
    )


def blocking_failures(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.status == "fail" and r.blocking]
