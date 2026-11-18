"""Command-line entry point: `frameseek <command>`."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer
from rich.console import Console

from frameseek.config import load_config
from frameseek.devices import pick_device
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

# Progress goes to stderr, the answer goes to stdout, so
# `frameseek describe img.jpg > answer.txt` saves only the answer.
log = Console(stderr=True)

DEFAULT_PROMPT = (
    "Describe this image in 2-3 sentences. Mention the main objects, "
    "their colours, any people, and any visible text."
)


@app.command()
def doctor() -> None:
    """Check that config, device and the vision model are ready."""
    cfg = load_config()
    log.print(f"config     : vision={cfg.vision.backend}:{cfg.vision.model}")
    log.print(f"device     : {pick_device(cfg.device)} (requested: {cfg.device})")

    model = build_vision_model(cfg.vision)
    if isinstance(model, OllamaVision):
        try:
            model.health()
            loaded = "loaded in memory" if model.is_loaded() else "not loaded yet"
            log.print(f"ollama     : ok ({cfg.vision.host}, model {loaded})")
        except OllamaError as exc:
            log.print(f"ollama     : [red]FAIL[/red] - {exc}")
            raise typer.Exit(code=1)


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
