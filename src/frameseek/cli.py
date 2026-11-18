"""Command-line entry point: `frameseek <command>`."""

from __future__ import annotations

import math
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from frameseek.caption import CAPTION_PROMPT, CAPTION_SCHEMA, caption_keyframes
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
        note = " [dim](PyTorch not installed: run uv sync)[/dim]"
    log.print(f"  PyTorch models : {torch_dev}{note} [dim](SigLIP2, e5 embeddings)[/dim]")
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
    table.add_row("title", escape(meta.title))
    table.add_row("channel", escape(meta.channel))
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

def _shots_with_progress(cfg, meta, force: bool = False):
    """Run shot detection with progress bars. Returns (shots, was_cached, seconds)."""
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
    return shot_list, cached, time.perf_counter() - start


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
    shot_list, cached, elapsed = _shots_with_progress(cfg, meta, force)

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
            log.print(f"[dim]{_fmt_t(s.start):>7}[/dim]  {escape(s.text)}")
        if len(tr.segments) > lines:
            log.print(f"[dim]   ... {len(tr.segments) - lines} more[/dim]")


# ---------- caption ----------

@app.command()
def caption(
    url: str = typer.Argument(..., help="YouTube URL (any form) or 11-character video ID"),
    force: bool = typer.Option(False, "--force", help="Re-caption every frame"),
    side: int = typer.Option(None, "--side", help="Image size sent to the model (overrides config)"),
    show: int = typer.Option(6, "--show", help="How many captions to preview"),
) -> None:
    """Describe every keyframe as structured JSON (runs ingest and shots first if needed)."""
    cfg = load_config()
    if side:
        cfg.caption.max_image_side = side
    meta, _, _ = _ingest_with_progress(cfg, url)
    shot_list, _, _ = _shots_with_progress(cfg, meta)
    video_dir = cfg.data_dir / meta.id

    model_name = cfg.caption.model or cfg.vision.model
    model = build_vision_model(replace(cfg.vision, model=model_name))
    if isinstance(model, OllamaVision):
        try:
            model.health()
        except OllamaError as exc:
            log.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(code=1)

    start = time.perf_counter()
    progress = Progress(
        TextColumn("Captioning"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[detail]}"),
        console=log,
        transient=True,
    )
    task: list[int] = []

    def on_progress(ev: dict) -> None:
        if not task:
            task.append(progress.add_task("cap", total=ev["total"], detail=""))
        rec = ev["record"]
        if rec.get("caption"):
            t = rec["timing"]
            detail = f"{(t['prefill_s'] + t['decode_s']):.1f}s  {escape(rec['caption']['summary'][:48])}"
        else:
            detail = "[red]error, will retry next run[/red]"
        progress.update(task[0], completed=ev["done"], detail=detail)

    with progress:
        result = caption_keyframes(video_dir, shot_list, model, model_name, cfg.caption,
                                   force=force, on_progress=on_progress)
    elapsed = time.perf_counter() - start
    if model_name != (cfg.answer.model or cfg.vision.model) and result.newly_done:
        model.unload()   # free the captioning model; the answering model is a different one

    timed = [r["timing"] for r in result.records if "timing" in r]
    per_frame = [t["prefill_s"] + t["decode_s"] for t in timed]
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("frames", f"{len(result.records)} captioned "
                            f"({result.newly_done} new, {result.from_cache} cached, {result.errors} errors)")
    if per_frame:
        table.add_row("per frame", f"{sum(per_frame) / len(per_frame):.1f}s average "
                                   f"({sum(t['prefill_tokens'] for t in timed) / len(timed):.0f} input tokens, "
                                   f"{sum(t['decode_tokens'] for t in timed) / len(timed):.0f} output tokens)")
    table.add_row("model", model_name)
    table.add_row("this run", f"{elapsed:.0f}s")
    table.add_row("saved to", str(video_dir / "captions.jsonl"))
    log.print(table)

    for r in result.records[:show]:
        c = r["caption"]
        text = f"  [dim]text: {escape(', '.join(c['text_on_screen']))}[/dim]" if c["text_on_screen"] else ""
        log.print(f"[dim]{_fmt_t(r['t']):>7}[/dim]  {escape('[' + c['shot'] + ']')} {escape(c['summary'])}{text}")


@app.command(name="bench-vision")
def bench_vision(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, help="A keyframe to test with"),
    sides: str = typer.Option("512,1024", "--sides", help="Comma-separated sizes"),
    model_name: str = typer.Option(None, "--model", help="Vision model to test, e.g. qwen3-vl:4b-instruct"),
) -> None:
    """Caption one image at several sizes: real token counts, timing and output on this machine."""
    import tempfile

    from PIL import Image

    cfg = load_config()
    name = model_name or cfg.caption.model or cfg.vision.model
    model = build_vision_model(replace(cfg.vision, model=name))
    sizes = [int(x) for x in sides.split(",")]

    # Warm up on a *different* image: reusing the test image would let Ollama's
    # prompt cache skip the first row's prefill and report a fake 0.1 s.
    with tempfile.TemporaryDirectory() as tmp, \
            log.status(f"Loading {name} (warm-up on a throwaway image)..."):
        dummy = Path(tmp) / "warmup.jpg"
        Image.effect_noise((640, 360), 64).convert("RGB").save(dummy)
        try:
            model.structured(dummy, CAPTION_PROMPT, CAPTION_SCHEMA, max_side=min(sizes), num_predict=8)
        except OllamaError:
            pass   # 8 tokens can't finish the JSON; we only wanted the model loaded

    table = Table(title=f"{image.name} with {name}")
    for col in ("side", "input tokens", "read (prefill)", "output tokens", "write (decode)", "total"):
        table.add_column(col, justify="right")
    table.add_column("summary / text seen", max_width=50)

    import random

    for side in sizes:
        # Ollama reuses cached work for an input it has seen before, even from an
        # earlier run. Changing one random pixel makes every row a genuine cold read.
        with tempfile.TemporaryDirectory() as tmp, log.status(f"Captioning at {side}px..."):
            fresh = Path(tmp) / f"bench{image.suffix}"
            with Image.open(image) as img:
                img = img.convert("RGB")
                img.putpixel((random.randrange(img.width), random.randrange(img.height)),
                             tuple(random.randrange(256) for _ in range(3)))
                img.save(fresh, quality=95)
            cap, t = model.structured(fresh, CAPTION_PROMPT, CAPTION_SCHEMA, max_side=side)
        seen = escape(cap["summary"]) + (f"  [dim]text: {escape(', '.join(cap['text_on_screen']))}[/dim]"
                                         if cap["text_on_screen"] else "")
        table.add_row(str(side), str(t.prefill_tokens), f"{t.prefill_s:.1f}s", str(t.decode_tokens),
                      f"{t.decode_s:.1f}s", f"{t.prefill_s + t.decode_s:.1f}s", seen)
    log.print(table)
    log.print("[dim]Rows run one after another: on a fanless laptop, later rows can slow down as it heats up.[/dim]")


# ---------- index and search ----------

def _embedders(cfg):
    from frameseek.models.embed import ImageEmbedder, TextEmbedder
    device = pick_device(cfg.device)
    return (TextEmbedder(cfg.embed.text_model, device, cfg.embed.batch_size),
            ImageEmbedder(cfg.embed.image_model, device, cfg.embed.batch_size), device)


def _load_inputs(cfg, url):
    """Captions and transcript must already exist: they are slow, so index never runs them silently."""
    from frameseek.caption import cache_key, load_done
    from frameseek.transcribe import load_transcript

    meta, _, _ = _ingest_with_progress(cfg, url)
    shot_list, _, _ = _shots_with_progress(cfg, meta)
    video_dir = cfg.data_dir / meta.id

    model_name = cfg.caption.model or cfg.vision.model
    done = load_done(video_dir / "captions.jsonl", cache_key(model_name, cfg.caption.max_image_side))
    n_kf = sum(len(s.keyframes) for s in shot_list)
    if not done:
        log.print(f"[red]error:[/red] no captions yet. Run: frameseek caption {meta.id}")
        raise typer.Exit(code=1)
    if len(done) < n_kf:
        log.print(f"[yellow]warning:[/yellow] only {len(done)} of {n_kf} keyframes are captioned; "
                  f"indexing those. Run `frameseek caption {meta.id}` to finish.")

    transcript = load_transcript(video_dir, cfg.transcribe)
    if transcript is None:
        log.print(f"[yellow]warning:[/yellow] no transcript with current settings; indexing frames only. "
                  f"Run: frameseek transcribe {meta.id}")
    return meta, video_dir, list(done.values()), transcript


@app.command()
def index(
    url: str = typer.Argument(..., help="YouTube URL (any form) or 11-character video ID"),
    force: bool = typer.Option(False, "--force", help="Rebuild even if cached"),
) -> None:
    """Embed captions, frames and speech into a searchable vector index."""
    from frameseek.index import build_index

    cfg = load_config()
    meta, video_dir, records, transcript = _load_inputs(cfg, url)
    text_emb, image_emb, device = _embedders(cfg)

    from frameseek.hf import APPROX_SIZE

    start = time.perf_counter()
    progress = Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
        MofNCompleteColumn(), TimeElapsedColumn(),
        console=log, transient=True,
    )
    task: list[int] = []
    loading: list[int] = []   # the current download/load line, removed when it finishes

    def stamp(msg: str) -> None:
        progress.console.print(f"[dim][{time.perf_counter() - start:5.1f}s] {msg}[/dim]")

    def done_loading() -> None:
        while loading:
            progress.remove_task(loading.pop())

    def on_progress(ev: dict) -> None:
        st = ev["stage"]
        if st != "load":
            done_loading()
        if st == "load":
            done_loading()
            name = ev["model"]
            if ev["download"]:
                label = f"Downloading {name} ({APPROX_SIZE.get(name, 'size unknown')}, first run only)"
            else:
                label = f"Loading {name} on {device}"
            stamp(label)
            loading.append(progress.add_task(label, total=None))   # spinner + elapsed time
        elif st == "embed_captions":
            stamp(f"Embedding {ev['count']} captions (e5)")
        elif st == "embed_frames":
            if not task:
                task.append(progress.add_task("Embedding frames", total=ev["total"]))
            progress.update(task[0], completed=ev["done"])
        elif st == "embed_speech":
            stamp(f"Embedding {ev['count']} speech chunks (e5)")
        elif st == "write":
            stamp("Writing LanceDB tables")

    with progress:
        stats, cached = build_index(video_dir, records, transcript, text_emb, image_emb,
                                    cfg.embed, force=force, on_progress=on_progress)
    elapsed = time.perf_counter() - start

    if cached:
        log.print(f"[green]Cached[/green] - index already built from these inputs ({elapsed:.1f}s)")
    else:
        log.print(f"[green]Indexed[/green] in {elapsed:.1f}s")
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("frames", f"{stats.frames} (caption vector + image vector each)")
    table.add_row("speech", f"{stats.speech_chunks} chunks of ~{cfg.embed.speech_chunk_s:g}s")
    table.add_row("models", f"{cfg.embed.text_model}, {cfg.embed.image_model} on {device}")
    table.add_row("saved to", str(video_dir / "index.lance"))
    log.print(table)


@app.command()
def search(
    url: str = typer.Argument(..., help="YouTube URL (any form) or 11-character video ID"),
    question: str = typer.Argument(..., help='e.g. "woman in a red saree" or "লাল শাড়ি"'),
    k: int = typer.Option(5, "--k", help="Results per source"),
) -> None:
    """Raw retrieval from each source (captions, frames, speech), before fusion and answering."""
    from frameseek.index import search as run_search
    from frameseek.ingest import parse_video_id

    cfg = load_config()
    video_dir = cfg.data_dir / parse_video_id(url)
    if not (video_dir / "index.lance").exists():
        log.print(f"[red]error:[/red] no index yet. Run: frameseek index {url}")
        raise typer.Exit(code=1)

    text_emb, image_emb, device = _embedders(cfg)
    for emb in (text_emb, image_emb):
        verb = "Loading" if emb.is_downloaded() else "Downloading (first run only)"
        with log.status(f"{verb} {emb.name} on {device}..."):
            emb.load()
    start = time.perf_counter()
    results = run_search(video_dir, question, text_emb, image_emb, k=k)
    log.print(f"[dim]searched in {(time.perf_counter() - start) * 1000:.0f} ms[/dim]")

    sections = [
        ("captions", "Captions (question text vs caption text, e5)"),
        ("frames", "Frames (question text vs pixels, SigLIP2)"),
        ("speech", "Speech (question text vs what was said, e5)"),
    ]
    for key, title in sections:
        table = Table(title=title, title_justify="left", show_edge=False, pad_edge=False)
        table.add_column("score", justify="right")
        table.add_column("time", justify="right")
        table.add_column("content")
        for r in results[key]:
            when = _fmt_t(r["start"]) if key == "speech" else _fmt_t(r["t"])
            content = escape(r["text"] if key == "speech" else r["summary"])
            if key != "speech":
                content += f" [dim]({r['file']})[/dim]"
            table.add_row(f"{r['score']:.3f}", when, content)
        if not results[key]:
            table.add_row("", "", "[dim](no data)[/dim]")
        log.print(table)
        log.print()
    log.print("[dim]Scores are only comparable within one table: each model has its own scale "
              "(SigLIP2 ~0.1-0.2, e5 ~0.7-0.9). ask combines them as z-scores.[/dim]")


# ---------- ask and ui ----------

@app.command()
def ask(
    url: str = typer.Argument(..., help="YouTube URL (any form) or 11-character video ID"),
    question: str = typer.Argument(..., help='e.g. "what colour is the guitarist\'s shirt?"'),
    mode: str = typer.Option(None, "--mode", help="look (one frame at a time + zoom) or compare (top 3 together); default from config"),
) -> None:
    """Answer a question about a prepared video, with the moment and evidence."""
    from frameseek.answer import describe_parts, fmt_time
    from frameseek.index import build_index
    from frameseek.pipeline import Asker

    cfg = load_config()
    mode = mode or cfg.answer.mode
    meta, video_dir, records, transcript = _load_inputs(cfg, url)
    asker = Asker(cfg)
    for emb in (asker.text, asker.image):
        verb = "Loading" if emb.is_downloaded() else "Downloading (first run only)"
        with log.status(f"{verb} {emb.name}..."):
            emb.load()
    with log.status("Checking the search index..."):
        build_index(video_dir, records, transcript, asker.text, asker.image, cfg.embed)

    if mode == "look":
        log.print(f"[bold]Looking[/bold] at up to {cfg.answer.look_n} moments, best first, with {asker.model_name}")
        status = log.status("Searching...")
        status.start()

        def show(step) -> None:
            mark = "[green]yes[/green]" if step.present else "[red]no[/red]"
            what = f"{step.kind} #{step.rank} {fmt_time(step.moment.t)} {step.moment.file[:-4]}"
            extra = f" [dim]crop {step.crop.name}[/dim]" if step.crop else ""
            secs = f"{step.timing.total_s:.0f}s" if step.timing else ""
            log.print(f"  {what:32} {mark}  {escape(step.seen[:70])}{extra} [dim]{secs}[/dim]")
            status.update(f"{'Zooming' if step.kind == 'look' and step.present else 'Looking'}...")

        try:
            res = asker.ask(video_dir, question, mode, on_step=show)
        finally:
            status.stop()
        log.print()
    else:
        n = cfg.answer.top_n
        with log.status(f"Searching, then {asker.model_name} reads the top {n} moments "
                        f"(~{n * 1240} image tokens)..."):
            res = asker.ask(video_dir, question, mode)
        table = Table(title="Candidate moments shown to the model", title_justify="left",
                      show_edge=False, pad_edge=False)
        for col, just in (("#", "right"), ("time", "right"), ("z-scores", "left"), ("caption", "left")):
            table.add_column(col, justify=just)
        for i, m in enumerate(res.answer.candidates, start=1):
            mark = " [green]<- chosen[/green]" if res.answer.moment is m else ""
            table.add_row(str(i), fmt_time(m.t), describe_parts(m), escape(m.summary) + mark)
        log.print(table)
        log.print()
    a = res.answer

    colour = {"high": "green", "medium": "yellow", "low": "red"}.get(a.confidence, "white")
    log.print(f"[bold]Answer[/bold] [{colour}]({a.confidence} confidence)[/{colour}]")
    print(a.text)
    if a.evidence:
        log.print(f"[dim]Evidence: {escape(a.evidence)}[/dim]")
    if a.moment:
        log.print(f"[dim]At {fmt_time(a.moment.t)}: https://youtu.be/{meta.id}?t={int(a.moment.t)}  "
                  f"(frame {a.moment.file})[/dim]")
    else:
        log.print("[dim]No moment answers this.[/dim]")
    if a.crop:
        log.print(f"[dim]Read from the zoomed crop: {a.crop}[/dim]")
    t = a.timing
    calls = f"{len(a.steps)} model calls, " if a.steps else ""
    log.print(f"[dim]search {res.search_ms:.0f} ms; {calls}read {t.prefill_tokens} tokens in "
              f"{t.prefill_s:.1f}s, wrote {t.decode_tokens} in {t.decode_s:.1f}s; total {res.total_s:.1f}s[/dim]")


@app.command("eval")
def eval_cmd(
    url: str = typer.Argument(..., help="YouTube URL (any form) or 11-character video ID"),
    file: Path = typer.Option(None, "--file", help="Question set; default evals/<video_id>.yaml"),
    answer: bool = typer.Option(False, "--answer", help="Also run the answering model on every question (slow)"),
    mode: str = typer.Option(None, "--mode", help="look or compare; default from config"),
) -> None:
    """Score retrieval (and optionally answers) against a hand-checked question set."""
    from frameseek.answer import describe_parts, fmt_time, fuse
    from frameseek.evaluate import first_hit, is_hit, load_eval, summarise, text_ok
    from frameseek.models.vision import OllamaError
    from frameseek.index import build_index, score_all
    from frameseek.pipeline import Asker

    cfg = load_config()
    meta, video_dir, records, transcript = _load_inputs(cfg, url)
    path = file or Path("evals") / f"{meta.id}.yaml"
    if not path.exists():
        log.print(f"[red]No question set at {path}[/red]")
        raise typer.Exit(1)
    _, items = load_eval(path)
    asker = Asker(cfg)
    for emb in (asker.text, asker.image):
        with log.status(f"Loading {emb.name}..."):
            emb.load()
    with log.status("Checking the search index..."):
        build_index(video_dir, records, transcript, asker.text, asker.image, cfg.embed)

    n = cfg.answer.top_n
    table = Table(title=f"Retrieval on {len(items)} questions (hit = right moment in the top {n})",
                  title_justify="left", show_edge=False, pad_edge=False)
    for col in ("", "rank", "kind", "question", "top moment"):
        table.add_column(col)
    if answer:
        table.add_column("answer")
    mode = mode or cfg.answer.mode
    ranks, chose_right, abstained, seconds, calls, texts = [], 0, 0, [], [], []
    for it in items:
        frames, speech, scores = score_all(video_dir, it.q, asker.text, asker.image)
        moments = fuse(frames, speech, scores)
        r = first_hit(moments, it, speech)
        ranks.append(r)
        top = moments[0] if moments else None
        row = ["[green]✓[/green]" if r and r <= n else "[red]✗[/red]", str(r or "-"), it.kind, escape(it.q),
               f"{fmt_time(top.t)} {top.file[:-4]} ({describe_parts(top)})" if top else ""]
        if answer:
            try:
                with log.status(f"Answering ({mode}): {it.q}"):
                    a = asker.answer(video_dir, it.q, moments, mode)
                seconds.append(a.timing.total_s)
                calls.append(len(a.steps) or 1)
            except OllamaError as e:   # one failed question shouldn't end a 20-minute run
                row.append(f"[red]error[/red]: {escape(str(e)[:70])}")
                table.add_row(*row)
                continue
            if a.moment is None:
                abstained += 1
                verdict = "[yellow]no moment[/yellow]"
            elif is_hit(a.moment, it, speech):
                chose_right += 1
                verdict = "[green]right moment[/green]"
            else:
                verdict = "[red]wrong moment[/red]"
            ok = text_ok(a.text, it)
            if ok is not None:
                texts.append(ok)
                verdict += " [green]text ok[/green]" if ok else f" [red]text should include {escape(', '.join(it.contains))}[/red]"
            row.append(f"{verdict}: {escape(a.text[:70])}" + (f" [dim]({len(a.steps)} calls)[/dim]" if a.steps else ""))
        table.add_row(*row)
    log.print(table)
    s = summarise(items, ranks, n)
    kinds = "  ".join(f"{k} {ok}/{tot}" for k, (ok, tot) in s.by_kind.items())
    log.print(f"\n[bold]hit@1 {s.hit1}/{s.n}   hit@{n} {s.hit_top}/{s.n}   MRR {s.mrr:.2f}[/bold]   [dim]{kinds}[/dim]")
    if answer:
        log.print(f"[bold]{mode}: answering model picked the right moment {chose_right}/{s.n}[/bold], "
                  f"said none answers it {abstained}/{s.n}  [dim](check the answer text by eye)[/dim]")
        if texts:
            log.print(f"[bold]exact answer text right {sum(texts)}/{len(texts)}[/bold] [dim](questions with `contains`)[/dim]")
        if seconds:
            log.print(f"[dim]model time per question: mean {sum(seconds) / len(seconds):.0f}s, "
                      f"max {max(seconds):.0f}s; calls per question: mean {sum(calls) / len(calls):.1f}[/dim]")


@app.command()
def ui(
    port: int = typer.Option(7860, "--port", help="Port for the local web app"),
) -> None:
    """Open the web app: paste a link, prepare it, ask questions."""
    from frameseek.app import launch
    launch(port=port)


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
