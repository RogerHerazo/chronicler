"""Command line entry point."""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from chronicler import __version__
from chronicler.config import config_path, load_settings
from chronicler.doctor import CheckResult, blocking_failures, build_checks, run_check

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Chronicler: a live, local D&D session scribe.",
)
console = Console()

ICONS = {"ok": "[green]✓[/]", "warn": "[yellow]![/]", "fail": "[red]✕[/]", "skip": "[dim]–[/]"}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "faster_whisper", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _free_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise typer.BadParameter(f"No free port found from {preferred}.")


def _print_results(results: list[CheckResult]) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(width=2)
    table.add_column(style="bold", no_wrap=True)
    table.add_column()
    for r in results:
        detail = r.detail + (f"\n[dim]→ {r.fix}[/]" if r.fix and r.status != "ok" else "")
        table.add_row(ICONS[r.status], r.title, detail)
    console.print(table)


def _run_doctor(include_slow: bool) -> list[CheckResult]:
    from chronicler.campaign.store import Store

    settings = load_settings()
    notes_dir = None
    if settings.active_campaign_id is not None and settings.db_path.exists():
        store = Store(settings.db_path)
        campaign = store.get_campaign(settings.active_campaign_id)
        notes_dir = campaign.notes_dir if campaign else None
        store.close()
    checks = [c for c in build_checks(settings, notes_dir) if include_slow or not c.slow]
    return [run_check(c) for c in checks]


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    port: int = typer.Option(8765, help="Port for the local web UI."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open a browser."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    """Start Chronicler and open it in your browser."""
    if ctx.invoked_subcommand is not None:
        return
    _setup_logging(verbose)
    console.print(f"[bold]Chronicler[/] {__version__}")
    with console.status("Running health checks…"):
        results = _run_doctor(include_slow=False)
    _print_results(results)
    _serve(port, open_browser=not no_browser, initial_results=results)


def _serve(
    port: int,
    *,
    open_browser: bool,
    initial_results: list[CheckResult] | None = None,
    replay: tuple[Path, float] | None = None,
) -> None:
    import uvicorn

    from chronicler.audio.source import FileReplaySource
    from chronicler.web.app import create_app
    from chronicler.web.state import AppState

    state = AppState(load_settings(), config_path())
    for r in initial_results or []:
        state.doctor_results[r.id] = r

    async def on_startup(s: AppState) -> None:
        if replay is None:
            return
        campaign = s.active_campaign() or s.store.create_campaign("Replay")
        s.settings.active_campaign_id = campaign.id
        path, speed = replay
        await s.pipeline.start_recording(
            campaign.id, FileReplaySource(path, speed), s.settings.chunk_minutes * 60
        )

    host = "127.0.0.1"
    port = _free_port(host, port)
    url = f"http://{host}:{port}/{'live' if replay else 'doctor'}"
    console.print(f"\nOpen [bold cyan]{url}[/] · press Ctrl+C to quit\n")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(
        create_app(state, on_startup=on_startup),
        host=host,
        port=port,
        log_level="warning",
    )


@app.command()
def doctor(
    quick: bool = typer.Option(False, "--quick", help="Skip the transcription speed test."),
) -> None:
    """Check that everything Chronicler needs is installed and working."""
    _setup_logging(False)
    with console.status("Running health checks…"):
        results = _run_doctor(include_slow=not quick)
    _print_results(results)
    failures = blocking_failures(results)
    if failures:
        console.print(f"\n[red]{len(failures)} problem(s) must be fixed before recording.[/]")
        raise typer.Exit(1)
    console.print("\n[green]Ready to record.[/]")


@app.command()
def replay(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, help="Recording to replay."),
    speed: float = typer.Option(10.0, help="Playback speed; 0 = as fast as possible."),
    chunk_minutes: float | None = typer.Option(None, help="Override the chunk length."),
    campaign: str | None = typer.Option(None, help="Campaign name (created if missing)."),
    headless: bool = typer.Option(
        False, "--headless", help="No web UI: process the file and exit."
    ),
    summarize: bool = typer.Option(
        False, "--summarize", help="Headless only: write the final summary at the end."
    ),
    port: int = typer.Option(8765),
    no_browser: bool = typer.Option(False, "--no-browser"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Feed an existing recording through the pipeline as if it were live."""
    _setup_logging(verbose)
    settings = load_settings()
    if chunk_minutes:
        settings.chunk_minutes = chunk_minutes
    if campaign:
        from chronicler.campaign.store import Store

        store = Store(settings.db_path)
        match = next((c for c in store.list_campaigns() if c.name == campaign), None)
        settings.active_campaign_id = (match or store.create_campaign(campaign)).id
        store.close()
    if not headless:
        from chronicler.config import save_settings

        save_settings(settings)
        _serve(port, open_browser=not no_browser, replay=(audio, speed))
        return
    asyncio.run(_replay_headless(audio, speed, settings, summarize))


async def _replay_headless(audio: Path, speed: float, settings: object, summarize: bool) -> None:
    from chronicler.audio.source import FileReplaySource
    from chronicler.config import Settings
    from chronicler.export import session_export_dir
    from chronicler.web.state import AppState

    assert isinstance(settings, Settings)
    state = AppState(settings)
    pipeline = state.pipeline
    await pipeline.start()
    campaign = state.active_campaign() or state.store.create_campaign("Replay")
    session = await pipeline.start_recording(
        campaign.id, FileReplaySource(audio, speed), settings.chunk_minutes * 60
    )

    def log_event(event: dict[str, object]) -> None:
        if event["event"] == "chunk":
            console.print(f"  chunk {event['chunk_id']}: {event['status']}", highlight=False)
            if event.get("error"):
                console.print(f"    [red]{event['error']}[/]")

    queue = pipeline.bus.subscribe()

    async def printer() -> None:
        while True:
            log_event(await queue.get())

    printer_task = asyncio.create_task(printer())
    rec = pipeline.recording
    assert rec is not None
    await asyncio.to_thread(rec.recorder.finished.wait)
    await pipeline.wait_idle()
    if summarize:
        console.print("Writing the session summary…")
        await pipeline.finish_session(session.id)
    printer_task.cancel()
    await pipeline.shutdown()
    session_row = state.store.get_session(session.id)
    assert session_row is not None
    out = session_export_dir(settings.resolved_export_dir, session_row)
    console.print(f"\nDone. Exports in [bold]{out}[/]")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)
