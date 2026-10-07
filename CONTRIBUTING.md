# Contributing to Chronicler

Thanks for helping.
Bug reports, fixes and new features are all welcome.

## Setup

```bash
git clone https://github.com/RogerHerazo/chronicler
cd chronicler
uv sync --extra cuda      # or plain `uv sync` without an NVIDIA GPU
uv run pre-commit install
```

## Before you open a pull request

```bash
uv run ruff format
uv run ruff check
uv run pyright
uv run pytest
```

CI runs the same checks on Windows, macOS and Linux.

To try a change end to end without a sound card or a live game, replay a recording:

```bash
uv run chronicler replay path/to/session.mp3 --speed 20
```

Set `CHRONICLER_CONFIG=/some/scratch/config.toml` to keep test runs away from your real campaigns.
The config file can point `data_dir` and `export_dir` at scratch folders.

## Guidelines

- Keep the app local first: audio must never leave the user's machine.
- Every problem the app can detect should come with a concrete fix hint, in the UI and in the health check.
- The UI has no build step: Jinja templates, htmx and one stylesheet. Check new screens in light and dark mode and at phone width.
- Add tests for behaviour you change. Tests must not need a GPU, a sound card or network access.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:` and so on).
  Releases and the changelog are generated from them, so never edit `CHANGELOG.md` by hand.

## Adding an LLM provider

See [docs/providers.md](docs/providers.md).
