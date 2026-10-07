# Architecture

Chronicler is one Python process that serves a local web UI and runs a recording and analysis pipeline.

```
LiveSource (loopback + mic, 16 kHz mono)          FileReplaySource (for replay and tests)
            └──────────────┬───────────────────────────────┘
                     Recorder thread
          writes recording_NN.flac and chunk_NNN.wav,
          cutting each chunk at the quietest moment near the target length
                           │ on_chunk
                     Pipeline (asyncio)
          one worker, chunks strictly in order:
          transcribing ──> analyzing ──> done | failed
               │                │
         Transcriber       LLMProvider (Claude | Ollama)
        (faster-whisper)         │
                           Store (SQLite): sessions, chunks, entities, threads
                                 │
                 export.py ──> live_notes.md, transcript.txt, summary.md
                 EventBus  ──> Server-Sent Events ──> browser (htmx)
```

## Modules

| Module | Responsibility |
| --- | --- |
| `audio/devices.py` | Lists loopback and microphone devices through `soundcard`. |
| `audio/source.py` | `LiveSource` mixes the devices paced by the wall clock, so a silent device (WASAPI loopback delivers nothing while nothing plays) cannot stall the timeline. `FileReplaySource` streams a file through PyAV. |
| `audio/recorder.py` | Buffers audio, writes the session FLAC and emits chunk WAVs. |
| `transcribe.py` | Loads the Whisper model once and transcribes chunks with absolute session timestamps. |
| `cuda.py` | Makes the pip-installed cuBLAS and cuDNN libraries visible to CTranslate2. |
| `llm/` | Provider-neutral Pydantic output models, shared prompts, and the Claude and Ollama providers. |
| `campaign/store.py` | SQLite schema, migrations and tracker operations (find by alias, merge). |
| `campaign/notes.py` | Loads an optional folder of Markdown notes within a token budget. |
| `pipeline.py` | Orchestrates recording, processing, tracker updates, exports and events. |
| `doctor.py` | Health checks with status, detail and fix hint. |
| `web/` | FastAPI app, Jinja templates, htmx, one stylesheet. |
| `cli.py` | `chronicler`, `chronicler doctor`, `chronicler replay`. |

## Design decisions

- **Chunks are processed strictly in order.** Each analysis receives the previous rolling recap, the known entities and the open threads, and returns an updated recap. A retried older chunk updates the tracker but never rolls the recap back.
- **Re-analysis is idempotent.** Tracker rows produced by a chunk are deleted before the chunk's analysis is applied again.
- **Crash safety.** Chunks are written to disk and recorded in the database before processing. On start, sessions left in `recording` become `stopped`, and unfinished chunks are queued again.
- **Prompt caching.** The system prompt (instructions plus campaign notes) is identical for every call in a session; per-chunk context goes in the user message. With Claude, the system block is cached for an hour.
- **Suggested vs confirmed.** Everything the model proposes enters the tracker as `suggested`. Users confirm, rename, merge or dismiss. Suggested and confirmed names are both fed back as canonical spellings.
