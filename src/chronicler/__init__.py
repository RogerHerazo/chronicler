"""Chronicler: a live, local D&D session scribe."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chronicler-dnd")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0"
