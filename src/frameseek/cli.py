"""Command-line entry point: `frameseek <command>`."""

from __future__ import annotations

import math
import shutil
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn
from rich.table import Table

from frameseek.config import load_config
from frameseek.devices import pick_device
from frameseek.ingest import ingest as run_ingest
from frameseek.ingest import require_tools
from frameseek.shots import build_shots, contact_sheet
from frameseek.models.vision import (
    Done,
    OllamaError,
    OllamaVision,
    Status,
    Timing,
    Token,
    build_vision_model,
)

app = typer.Typer(help="Ask questions about anything in a video.", no_args_is_help=True)

# Progress goes to stderr, results go to stdout, so
# `frameseek describe img.jpg > answer.txt` saves only the answer.
log = Console(stderr=True)

DEFAULT_PROMPT = (
    "Describe this image in 2-3 sentences. Mention the main objects, "
    "their colours, any people, and any visible text."
)


@app.command()
def doctor() -> None:
    """Check that config, tools, device and the vision model are ready."""
    cfg = load_config()
    ok = True
    log.print(f"config     : vision={cfg.vision.backend}:{cfg.vision.model}")
    log.print(f"device     : {pick_device(cfg.device)} (requested: {cfg.device})")

    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        log.print(f"{tool:<11}: {found or '[red]MISSING[/red] (brew install ffmpeg)'}")
        ok &= bool(found)

    model = build_vision_model(cfg.vision)
    if isinstance(model, OllamaVision):
        try:
            model.health()
            loaded = "loaded in memory" if model.is_loaded() else "not loaded yet"
            log.print(f"ollama     : ok ({cfg.vision.host}, model {loaded})")
        except OllamaError as exc:
            log.print(f"ollama     : [red]FAIL[/red] - {exc}")
            ok = False

    if not ok:
        raise typer.Exit(code=1)


# ---------- ingest ----------

def _ingest_with_progress(cfg, url: str, force: bool = False):
    """Run ingest with live progress bars. Returns (meta, was_cached, seconds)."""
    try:
        require_tools()
    except RuntimeError as exc:
        log.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)

    start = time.perf_counter()
    progress = Progress(
        TextColumn("{task.description:<14}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=log,
        transient=True,
    )
    tasks: dict[str, int] = {}

    def on_progress(ev: dict) -> None:
        stage = ev["stage"]
        if stage == "download":
            name = f"{ev['stream']} stream"
            if name not in tasks:
                tasks[name] = progress.add_task(name, total=ev["total"] or None)
            progress.update(tasks[name], completed=ev["done"], total=ev["total"] or None)
        else:
            label = {
                "resolve": "Finding best video <= {}p + audio".format(cfg.ingest.max_height),
                "merge": "Merging video + audio into one file",
                "probe": "Reading codec, fps and keyframe spacing (ffprobe)",
            }[stage]
            progress.console.print(f"[dim][{time.perf_counter() - start:5.1f}s] {label}[/dim]")

    try:
        with progress:
            meta, cached = run_ingest(
                url, cfg.data_dir, max_height=cfg.ingest.max_height,
                force=force, on_progress=on_progress,
            )
    except ValueError as exc:
        log.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)
    return meta, cached, time.perf_counter() - start


@app.command()
def ingest(
    url: str = typer.Argument(..., help="YouTube URL (any form) or 11-character video ID"),
    force: bool = typer.Option(False, "--force", help="Download again even if cached"),
) -> None:
    """Download a YouTube video once into data/<video_id>/ and describe it."""
    cfg = load_config()
    meta, cached, elapsed = _ingest_with_progress(cfg, url, force)

    if cached:
        log.print(f"[green]Cached[/green] - already ingested, nothing to do ({elapsed:.1f}s)")
    else:
        log.print(f"[green]Ingested[/green] in {elapsed:.1f}s")

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("id", meta.id)
    table.add_row("title", meta.title)
    table.add_row("channel", meta.channel)
    table.add_row("duration", f"{meta.duration_s / 60:.1f} min")
    table.add_row("video", f"{meta.width}x{meta.height} @ {meta.fps:g} fps, {meta.vcodec}")
    table.add_row("audio", meta.acodec or "none")
    kf = f"every {meta.keyframe_interval_s:g}s" if meta.keyframe_interval_s else "unknown"
    table.add_row("keyframes", kf)
    subs = ", ".join(meta.manual_subtitles) or "none"
    table.add_row("captions", f"manual: {subs}; auto: {'yes' if meta.has_auto_captions else 'no'}")
    table.add_row("saved to", str(cfg.data_dir / meta.id / meta.file))
    log.print(table)


# ---------- shots ----------

@app.command()
def shots(
    url: str = typer.Argument(..., help="YouTube URL (any form) or 11-character video ID"),
    force: bool = typer.Option(False, "--force", help="Re-detect even if cached"),
    sheet: bool = typer.Option(True, "--sheet/--no-sheet", help="Write contact_sheet.jpg"),
) -> None:
    """Split a video into shots and extract keyframes (runs ingest first if needed)."""
    cfg = load_config()
    meta, _, _ = _ingest_with_progress(cfg, url)
    video_dir = cfg.data_dir / meta.id

    start = time.perf_counter()
    progress = Progress(
        TextColumn("{task.description:<20}"),
        BarColumn(),
        TextColumn("{task.fields[detail]}"),
        console=log,
        transient=True,
    )
    tasks: dict[str, int] = {}

    def on_progress(ev: dict) -> None:
        if ev["stage"] == "detect":
            if "detect" not in tasks:
                tasks["detect"] = progress.add_task("Detecting cuts", total=ev["total"], detail="")
            progress.update(tasks["detect"], completed=ev["done"],
                            detail=f"{ev['done']:.0f}/{ev['total']:.0f}s of video")
        elif ev["stage"] == "extract":
            if "extract" not in tasks:
                tasks["extract"] = progress.add_task("Extracting keyframes", total=ev["total"], detail="")
            progress.update(tasks["extract"], completed=ev["done"],
                            detail=f"{ev['done']}/{ev['total']} frames")

    with progress:
        shot_list, cached = build_shots(
            video_dir, meta.file, params=cfg.shots, force=force, on_progress=on_progress
        )
    elapsed = time.perf_counter() - start

    if cached:
        log.print(f"[green]Cached[/green] - shots already detected with these settings ({elapsed:.1f}s)")
    else:
        log.print(f"[green]Detected[/green] in {elapsed:.1f}s")

    lengths = sorted(s.duration_s for s in shot_list)
    n_kf = sum(len(s.keyframes) for s in shot_list)
    longest = max(shot_list, key=lambda s: s.duration_s)
    every_2s = math.ceil(meta.duration_s / 2)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("shots", str(len(shot_list)))
    table.add_row("shot length", f"median {lengths[len(lengths) // 2]:.1f}s, "
                                 f"shortest {lengths[0]:.1f}s, longest {lengths[-1]:.1f}s "
                                 f"(shot {longest.index})")
    table.add_row("keyframes", f"{n_kf} (vs {every_2s} if sampling every 2s: "
                               f"{every_2s / max(n_kf, 1):.1f}x fewer to caption)")
    table.add_row("saved to", str(video_dir / "keyframes"))
    if sheet:
        table.add_row("contact sheet", str(contact_sheet(video_dir, shot_list)))
    log.print(table)


# ---------- describe ----------

def _print_timing(t: Timing) -> None:
    log.print()
    log.print("[bold]Where the time went[/bold]")
    log.print(f"  load model    {t.load_s:6.1f}s")
    log.print(f"  read input    {t.prefill_s:6.1f}s  ({t.prefill_tokens} tokens, prefill)")
    log.print(
        f"  write answer  {t.decode_s:6.1f}s  "
        f"({t.decode_tokens} tokens, {t.decode_tok_per_s:.1f} tok/s, decode)"
    )
    log.print(f"  total         {t.total_s:6.1f}s")


@app.command()
def describe(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to an image"),
    prompt: str = typer.Option(DEFAULT_PROMPT, "--prompt", "-p", help="What to ask about it"),
) -> None:
    """Send one image to the vision model and stream its answer."""
    cfg = load_config()
    model = build_vision_model(cfg.vision)
    start = time.perf_counter()

    def stamp(msg: str) -> None:
        log.print(f"[dim][{time.perf_counter() - start:5.1f}s] {msg}[/dim]")

    spinner = log.status("Starting...")
    spinner.start()
    first_token = True
    try:
        for event in model.stream(image, prompt):
            if isinstance(event, Status):
                stamp(event.message)
                spinner.update(event.message + "...")
            elif isinstance(event, Token):
                if first_token:
                    spinner.stop()
                    stamp("First word arrived, streaming answer:")
                    log.print()
                    first_token = False
                sys.stdout.write(event.text)
                sys.stdout.flush()
            elif isinstance(event, Done):
                sys.stdout.write("\n")
                _print_timing(event.timing)
    except OllamaError as exc:
        spinner.stop()
        log.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)
    finally:
        spinner.stop()


if __name__ == "__main__":
    app()
