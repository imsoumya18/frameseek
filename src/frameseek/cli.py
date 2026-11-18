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
from frameseek.devices import ctranslate2_device, pick_device
from frameseek.ingest import ingest as run_ingest
from frameseek.ingest import require_tools
from frameseek.shots import build_shots, contact_sheet
from frameseek.transcribe import NoCaptionsError, build_transcript
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
    """Check tools, devices and the vision model. Each engine picks its device separately."""
    cfg = load_config()
    ok = True
    log.print(f"config      : vision={cfg.vision.backend}:{cfg.vision.model}, "
              f"whisper={cfg.transcribe.whisper_model}")

    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        log.print(f"{tool:<12}: {found or '[red]MISSING[/red] (brew install ffmpeg)'}")
        ok &= bool(found)

    log.print("[bold]devices[/bold] (requested: {})".format(cfg.device))
    torch_dev = pick_device(cfg.device)
    try:
        import torch  # noqa: F401
        note = ""
    except ImportError:
        note = " [dim](PyTorch not installed yet; arrives with SigLIP2 in Section 6)[/dim]"
    log.print(f"  PyTorch models : {torch_dev}{note}")
    wd = cfg.transcribe.device if cfg.transcribe.device != "auto" else cfg.device
    log.print(f"  Whisper        : {ctranslate2_device(wd)} "
              "[dim](CTranslate2: CUDA or CPU; MPS is experimental, opt in with transcribe.device: mps)[/dim]")

    model = build_vision_model(cfg.vision)
    if isinstance(model, OllamaVision):
        try:
            model.health()
        except OllamaError as exc:
            log.print(f"  Vision (Ollama): [red]FAIL[/red] - {exc}")
            raise typer.Exit(code=1)
        res = model.residency()
        if res is None:
            log.print("  Vision (Ollama): ok, model not loaded right now "
                      "[dim](run `frameseek describe` once, then doctor again to see where it runs)[/dim]")
        else:
            size, vram = res
            share = vram / size if size else 0.0
            where = ("GPU" if share >= 0.99 else
                     "CPU" if share <= 0.01 else f"split: {share:.0%} GPU / {1 - share:.0%} CPU")
            log.print(f"  Vision (Ollama): {where} [dim]({vram / 1e9:.1f} of {size / 1e9:.1f} GB "
                      "in GPU memory; Ollama chooses this itself)[/dim]")

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


# ---------- transcribe ----------

def _fmt_t(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


@app.command()
def transcribe(
    url: str = typer.Argument(..., help="YouTube URL (any form) or 11-character video ID"),
    source: str = typer.Option(None, "--source", help="auto | youtube | whisper (overrides config)"),
    model: str = typer.Option(None, "--model", help="Whisper model, e.g. small, medium, large-v3-turbo"),
    language: str = typer.Option(None, "--language", help="Force a language code, e.g. bn"),
    device: str = typer.Option(None, "--device", help="Whisper device: auto | cpu | cuda | mps"),
    force: bool = typer.Option(False, "--force", help="Re-transcribe even if cached"),
    lines: int = typer.Option(8, "--lines", help="How many transcript lines to preview"),
) -> None:
    """Timestamped speech: YouTube captions if available, otherwise local Whisper."""
    cfg = load_config()
    if source:
        cfg.transcribe.source = source
    if model:
        cfg.transcribe.whisper_model = model
    if language:
        cfg.transcribe.language = language
    if device:
        cfg.transcribe.device = device
    meta, _, _ = _ingest_with_progress(cfg, url)
    video_dir = cfg.data_dir / meta.id

    start = time.perf_counter()
    progress = Progress(
        TextColumn("{task.description:<14}"),
        BarColumn(),
        TextColumn("{task.fields[detail]}"),
        console=log,
        transient=True,
    )
    task: list[int] = []

    def stamp(msg: str) -> None:
        progress.console.print(f"[dim][{time.perf_counter() - start:5.1f}s] {msg}[/dim]")

    def on_progress(ev: dict) -> None:
        st = ev["stage"]
        if st == "captions":
            stamp("Checking YouTube for captions")
        elif st == "load_model":
            stamp(f"Loading Whisper '{ev['model']}' on {ev['device']} (downloads once on first use)")
        elif st == "check_language":
            stamp(f"Checking which language the audio is in (Whisper {cfg.transcribe.language_model})")
        elif st == "language_check":
            if ev["captions"] is None:
                use = "will transcribe in it" if ev["probability"] >= 0.5 else "too unsure, letting Whisper decide"
                stamp(f"Audio is '{ev['heard']}' ({ev['probability']:.0%}); {use}")
            else:
                verdict = "matches" if ev["captions"].split("-")[0] == ev["heard"] else "does NOT match"
                stamp(f"Audio is '{ev['heard']}' ({ev['probability']:.0%}); "
                      f"captions are '{ev['captions']}': {verdict}")
        elif st == "language":
            how = "given to the model" if ev.get("forced") else f"detected, {ev['probability']:.0%} confident"
            stamp(f"Language: {ev['language']} ({how}); "
                  f"speech found in {ev['speech_s']:.0f}s of {ev['total_s']:.0f}s")
        elif st == "transcribe":
            if not task:
                task.append(progress.add_task("Transcribing", total=ev["total"], detail=""))
            progress.update(task[0], completed=ev["done"],
                            detail=f"{_fmt_t(ev['done'])} / {_fmt_t(ev['total'])}")

    try:
        with progress:
            tr, cached = build_transcript(
                video_dir, meta, cfg.transcribe, device_pref=cfg.device,
                force=force, on_progress=on_progress,
            )
    except (NoCaptionsError, ValueError) as exc:
        log.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)
    elapsed = time.perf_counter() - start

    if cached:
        log.print(f"[green]Cached[/green] - transcript already built with these settings ({elapsed:.1f}s)")
    else:
        log.print(f"[green]Transcribed[/green] in {elapsed:.1f}s")

    coverage = tr.speech_seconds() / meta.duration_s if meta.duration_s else 0.0
    has_words = any(s.words for s in tr.segments)
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("source", tr.source)
    if tr.note:
        table.add_row("why", tr.note)
    table.add_row("language", f"{tr.language}: {tr.language_from} "
                              f"[dim](YouTube metadata says {meta.language or 'nothing'})[/dim]")
    table.add_row("segments", f"{len(tr.segments)}, about {tr.word_count()} words")
    table.add_row("word timing", "yes" if has_words else "no (segment-level only)")
    table.add_row("speech", f"{coverage:.0%} of the video")
    table.add_row("saved to", str(video_dir / "transcript.json"))
    log.print(table)

    if tr.segments and lines > 0:
        log.print()
        for s in tr.segments[:lines]:
            log.print(f"[dim]{_fmt_t(s.start):>7}[/dim]  {s.text}")
        if len(tr.segments) > lines:
            log.print(f"[dim]   ... {len(tr.segments) - lines} more[/dim]")


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
