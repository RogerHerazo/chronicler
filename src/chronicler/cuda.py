"""Make NVIDIA's CUDA libraries from pip wheels visible to CTranslate2.

faster-whisper (via CTranslate2) needs cuBLAS 12 and cuDNN 9 at runtime. The
`chronicler-dnd[cuda]` extra installs them as pip wheels (`nvidia-cublas-cu12`,
`nvidia-cudnn-cu12`), but those land in site-packages where the dynamic loader
does not look. We preload them (Linux) or register their folders (Windows)
before the first model is created.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import platform
from functools import cache
from pathlib import Path

log = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
# Libraries CTranslate2 opens by name on first GPU use.
REQUIRED = (
    ["cublas64_12.dll", "cudnn_ops64_9.dll"]
    if IS_WINDOWS
    else ["libcublas.so.12", "libcudnn_ops.so.9"]
)


def nvidia_lib_dirs() -> list[Path]:
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return []
    sub = "bin" if IS_WINDOWS else "lib"
    dirs: list[Path] = []
    for root in spec.submodule_search_locations:
        for pkg in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            candidate = Path(root) / pkg / sub
            if candidate.is_dir():
                dirs.append(candidate)
    return dirs


@cache
def preload() -> list[Path]:
    """Idempotently expose pip-installed CUDA libraries. Returns the folders used."""
    dirs = nvidia_lib_dirs()
    if not dirs:
        return []
    if IS_WINDOWS:
        for d in dirs:
            os.add_dll_directory(str(d))  # type: ignore[attr-defined]
        os.environ["PATH"] = os.pathsep.join([*map(str, dirs), os.environ.get("PATH", "")])
        return dirs
    # Linux: load every shared object globally. Some depend on each other, so
    # retry until a pass makes no progress.
    pending = [f for d in dirs for f in sorted(d.glob("lib*.so*"))]
    while pending:
        failed = []
        for lib in pending:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(lib)
        if len(failed) == len(pending):
            log.debug("could not preload: %s", [f.name for f in failed])
            break
        pending = failed
    return dirs


def missing_libraries() -> list[str]:
    preload()
    missing = []
    for name in REQUIRED:
        try:
            ctypes.CDLL(name)
        except OSError:
            missing.append(name)
    return missing


INSTALL_HINT = (
    'Install the CUDA libraries: uv tool install --reinstall "chronicler-dnd[cuda]" '
    '(or pip install "chronicler-dnd[cuda]"). Also make sure your NVIDIA driver is up to date. '
    "Or set the transcription device to CPU in Settings."
)
