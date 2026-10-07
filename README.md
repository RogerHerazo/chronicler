# Chronicler

A live, local scribe for your D&D table.

Chronicler records your session from your own computer, transcribes it locally with Whisper, and every few minutes updates a running recap, the people and places that came up, and the open threads.
You see it in your browser while you play, and you get Markdown notes and a full transcript when the session ends.

- **Live, not after the fact.** Every 15 minutes (configurable) the latest part of the session is transcribed and analyzed, so the recap is never more than one part behind.
- **Local first.** Audio never leaves your computer. Transcription runs on your GPU or CPU.
- **Your choice of brain.** Analysis runs on Claude (best quality, needs an API key) or on Ollama (free and fully offline).
- **Remembers your world.** A built-in campaign tracker learns NPCs, places, items, factions and plot threads across sessions. You can also point it at a folder of existing campaign notes.
- **Plays well with your notes.** Every session gets `live_notes.md`, `transcript.txt` and a final `summary.md`, written wherever you like (an Obsidian vault, for example).
- **Health check built in.** On start, Chronicler checks your audio devices, GPU, Whisper model, transcription speed and AI provider, and tells you exactly how to fix anything that is wrong.

## How it works

```
your speakers + mic ──> Chronicler records ──> every 15 min: Whisper transcribes the part
                                                      └──> Claude / Ollama updates recap, NPCs, threads
                                                              └──> live page in your browser + Markdown files
```

Chronicler captures two sources and mixes them:

- **System audio** (loopback): everything your computer plays, such as Discord voices, your VTT and music.
- **Your microphone**: your own voice, which Discord does not play back to you.

For an in-person table, put a laptop with a decent microphone in the middle and turn system audio down.

## Install

Chronicler needs Python 3.11 or newer.
The easiest way to install it is with [uv](https://docs.astral.sh/uv/getting-started/installation/).

### Windows (recommended)

With an NVIDIA GPU:

```powershell
uv tool install "chronicler-dnd[cuda]"
chronicler
```

Without an NVIDIA GPU:

```powershell
uv tool install chronicler-dnd
chronicler
```

Until the first PyPI release, install straight from GitHub instead:

```powershell
uv tool install "chronicler-dnd[cuda] @ git+https://github.com/RogerHerazo/chronicler"
```

Your browser opens on the health check.
When everything is green, create a campaign and press **Start new session**.

### Linux

```bash
sudo apt install libpulse0   # PulseAudio / PipeWire client library
uv tool install "chronicler-dnd[cuda]"   # or without [cuda]
chronicler
```

System audio is captured from the PulseAudio or PipeWire "monitor" of your speakers.

### macOS

macOS cannot record system audio without a virtual audio driver.

1. Install [BlackHole](https://existential.audio/blackhole/) (2ch is enough).
2. In **Audio MIDI Setup**, create a **Multi-Output Device** with your speakers and BlackHole, and use it as your output.
3. Run `uv tool install chronicler-dnd`, then `chronicler`, and pick BlackHole as the system audio device in Settings.

Transcription runs on the CPU on a Mac, using a smaller Whisper model by default.

### WSL

Chronicler cannot capture Windows audio from inside WSL.
Install and run it on Windows itself.

## Setting up the AI

### Claude (default)

1. Create an API key at [console.anthropic.com](https://console.anthropic.com/settings/keys).
2. Paste it in **Settings → Analysis**. It is stored in your operating system's keychain, never in a file.
   You can also set the `ANTHROPIC_API_KEY` environment variable instead.

Chronicler uses `claude-opus-5-5` by default.
You can switch to a cheaper model in Settings.
The campaign context is prompt-cached, so each 15-minute part costs only a few cents.

### Ollama (free, offline)

1. Install [Ollama](https://ollama.com) and pull a model with at least 14B parameters, for example `ollama pull qwen3:14b`.
2. In **Settings → Analysis**, choose **Ollama** and enter the model name.

Smaller models work, but their notes are noticeably weaker.
If Whisper and the model share one GPU, make sure both fit in its memory.

## Using it

1. **Health**: fix anything red. Yellow items are optional.
2. **Settings**: pick your devices and press **Test for 3 seconds** while something plays and you speak. Set the spoken language if you know it.
3. **Live**: press **Start new session** when the game starts.
   The first recap appears after the first part is analyzed.
4. **Campaign**: confirm, rename, merge or dismiss the names Chronicler suggests.
   Confirmed names are fed back in, so later sessions spell them right.
5. When the game ends, press **Stop recording**, then **Finish & summarize** to write the final summary from the full transcript.

If you stopped by accident, **Continue this session** resumes recording into the same session.
If Chronicler crashed, any part that was recorded but not yet processed is picked up on the next start.

### Where things are saved

| What | Where |
| --- | --- |
| Settings | `chronicler` folder in your user config directory |
| Database, session audio (FLAC) and chunk WAVs | `chronicler` folder in your user data directory (shown on the Health page) |
| `live_notes.md`, `transcript.txt`, `summary.md` | The export folder from Settings, one folder per session |

### Command line

```text
chronicler                    start the app and open the browser
chronicler doctor             run the health check in the terminal (exit code 1 on problems)
chronicler doctor --quick     skip the transcription speed test
chronicler replay FILE        feed an existing recording through the pipeline as if it were live
chronicler replay FILE --headless --speed 0 --summarize
                              process a recording without the UI, as fast as possible
```

`replay` is handy for trying Chronicler on an old recording, for example one made with OBS.

## Privacy

- Audio is recorded and transcribed on your computer and never uploaded.
- With Claude, the transcript text of each part, your campaign notes and the tracker's names are sent to the Anthropic API.
- With Ollama, nothing leaves your computer.
- Everyone at the table should know they are being recorded.

## Troubleshooting

The health check shows a fix for every problem it finds.
The most common ones are below.

- **"CUDA libraries are missing"**: reinstall with the `[cuda]` extra, `uv tool install --reinstall "chronicler-dnd[cuda]"`, and update your NVIDIA driver.
- **"No loopback (system audio) device found"**: on macOS, install BlackHole (see above). On Linux, install `libpulse0`.
- **The system audio meter stays flat**: on Windows, Chronicler records your *default* output device. If Discord plays to a headset, pick that headset in Settings.
- **Transcription is too slow for live play**: choose a smaller Whisper model in Settings, or analyze every 20 minutes.
- **Names are misspelled**: confirm or rename them on the Campaign page, or add a notes folder with the correct spellings.

## Development

```bash
git clone https://github.com/RogerHerazo/chronicler
cd chronicler
uv sync --extra cuda      # or plain `uv sync` without an NVIDIA GPU
uv run pytest
uv run chronicler
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/architecture.md](docs/architecture.md).

## License

[MIT](LICENSE)
