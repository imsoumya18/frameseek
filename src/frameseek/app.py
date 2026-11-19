"""Local web app: paste a YouTube link, prepare it once, then ask questions.

Run with `frameseek ui`. One screen, no scrolling, in two columns:

  left    the link and a checklist of the five preparation steps (done, running
          with progress, waiting), then the question and the answer card, which
          shows each look, zoom and check live while it works
  right   the YouTube player at the answer's moment, the crop the answer was
          read from, and the candidate moments (click one to jump there)

Plain system fonts, no theme fonts. Everything is also logged to the terminal
the app was started from.
"""

from __future__ import annotations

import html
import queue
import threading
import time

from rich.console import Console
from rich.markup import escape

from frameseek.answer import fmt_time
from frameseek.config import load_config
from frameseek.ingest import load_meta, parse_video_id
from frameseek.pipeline import Asker, prepare
from frameseek.progress import STEPS, Checklist, from_disk, mmss

_cfg = load_config()
_asker: Asker | None = None    # created at startup; keeps the models loaded between questions
_log = Console(stderr=True)
_quiet = False

SYSTEM_FONT = ["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", "Arial",
               "sans-serif"]


def _say(msg: str) -> None:
    """Terminal log line with a time stamp (silenced by `frameseek ui --quiet`)."""
    if not _quiet:
        _log.print(f"[dim]{time.strftime('%H:%M:%S')}[/dim] {msg}")


CSS = """
.gradio-container { max-width: 1680px !important; width: 100% !important; margin: 0 auto !important;
                    padding: 0 24px !important; }
footer { display: none !important; }
.fs-top { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap;
          padding: 0 2px; margin: -8px 0 -12px 0; font-size: 1.05rem; }
.fs-top b { font-size: 1.15rem; }
.fs-top .by { font-size: .85rem; color: var(--body-text-color-subdued); display: flex; gap: 14px;
              align-items: center; }
.fs-top a { color: var(--body-text-color-subdued) !important; text-decoration: none !important;
            display: inline-flex; align-items: center; gap: 5px; }
.fs-top a:hover { color: var(--body-text-color) !important; text-decoration: underline !important; }
.fs-top a b { font-size: inherit; font-weight: 600; color: var(--body-text-color); }
.fs-top svg { width: 15px; height: 15px; fill: currentColor; }
.fs-box { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 12px;
          background: var(--block-background-fill); font-size: .9rem; }
.fs-head { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
.fs-head .t { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fs-head .fs-muted { white-space: nowrap; }
.fs-muted { color: var(--body-text-color-subdued); }
.fs-bar { height: 6px; border-radius: 3px; background: var(--border-color-primary); overflow: hidden; }
.fs-bar > div { height: 100%; background: var(--color-accent); }
.fs-step { display: grid; grid-template-columns: 22px auto 1fr; column-gap: 6px; padding: 2px 0; align-items: baseline; }
.fs-step .d { font-size: .8rem; color: var(--body-text-color-subdued); text-align: right;
              overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fs-step .fs-bar { grid-column: 2 / 4; margin: 2px 0 3px 0; }
.fs-step.waiting .n { color: var(--body-text-color-subdued); }
.fs-answer { font-size: 1.05rem; line-height: 1.45; margin: 0 0 8px 0; font-weight: 600; }
.fs-chips { display: flex; flex-wrap: wrap; gap: 6px; font-size: .8rem; margin-bottom: 6px; }
.fs-chip { padding: 1px 8px; border-radius: 999px; border: 1px solid var(--border-color-primary); }
.fs-high { background: rgba(34,197,94,.15); border-color: rgba(34,197,94,.5); }
.fs-medium { background: rgba(234,179,8,.15); border-color: rgba(234,179,8,.5); }
.fs-low { background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.45); }
/* the feed takes whatever height is left on the screen, and scrolls inside */
.fs-feed { max-height: max(80px, calc(100vh - 612px)); overflow-y: auto; font-size: .82rem; line-height: 1.6;
           border-top: 1px solid var(--border-color-primary); margin-top: 6px; padding-top: 6px; }
.fs-feed .ok { color: rgb(22,163,74); } .fs-feed .no { color: rgb(220,38,38); } .fs-feed .un { color: rgb(202,138,4); }
/* 16:9 and as wide as the column, but never taller than the screen leaves room for (no page
   scrolling); when the height is the limit, it narrows and stays centred */
.fs-player { position: relative; width: min(100%, calc((100vh - 382px) * 16 / 9)); aspect-ratio: 16 / 9;
             margin: 0 auto; border-radius: 8px; overflow: hidden; background: #111; }
.fs-player iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; }
.fs-player .ph { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
                 color: #aaa; font-size: .95rem; padding: 20px; text-align: center; }
.fs-label p { margin: 0; font-size: .82rem; color: var(--body-text-color-subdued); }
#moments, #moments .gallery-container, #moments .grid-wrap { height: auto !important; min-height: 0 !important;
  max-height: none !important; }
#moments .grid-container { gap: 6px !important; padding: 0 !important; }
#moments .grid-wrap { padding: 6px !important; }
#moments .thumbnail-item { aspect-ratio: 16 / 9 !important; height: auto !important; background: #000; }
#moments .thumbnail-item img { object-fit: contain !important; }
#moments .caption-label { font-size: .75rem; padding: 0 4px; }
"""


# ---------- HTML pieces ----------

def _bar(fraction: float) -> str:
    return f'<div class="fs-bar"><div style="width:{max(0.0, min(fraction, 1.0)) * 100:.0f}%"></div></div>'


def _checklist_html(c: Checklist, title: str = "", note: str = "") -> str:
    icon = {"done": "✅", "running": "⏳", "failed": "❌", "waiting": "⬜"}
    rows = []
    for key, label in STEPS:
        s = c.steps[key]
        bar = _bar(s.fraction) if s.state == "running" and s.fraction is not None else ""
        partial = _bar(s.fraction) if s.state == "waiting" and s.fraction else ""
        detail = f'<span class="d">{html.escape(s.detail)}</span>' if s.detail else ""
        rows.append(f'<div class="fs-step {s.state}"><span>{icon[s.state]}</span>'
                    f'<span class="n">{label}</span>{detail}{bar or partial}</div>')
    head = (f'<div class="fs-head"><span class="t">{html.escape(title) or "Preparation"}</span>'
            f'<span class="fs-muted">{c.done} of {len(STEPS)} · {c.overall * 100:.0f}%</span></div>'
            f'{_bar(c.overall)}<div style="height:6px"></div>')
    foot = f'<div class="fs-muted" style="margin-top:6px">{note}</div>' if note else ""
    return f'<div class="fs-box">{head}{"".join(rows)}{foot}</div>'


def _player(video_id: str = "", start: int = 0, autoplay: bool = True) -> str:
    """The YouTube embed. A freshly pasted link loads paused at 0:00; an answer plays from its moment."""
    if not video_id:
        return ('<div class="fs-player"><div class="ph">Paste a YouTube link: the video appears here, and '
                'each answer plays from its moment.</div></div>')
    src = (f"https://www.youtube.com/embed/{html.escape(video_id)}?start={start}"
           f"&autoplay={int(autoplay)}&rel=0")
    return (f'<div class="fs-player"><iframe src="{src}" '
            'allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen></iframe></div>')


def _card(text: str, chips: list[tuple[str, str]] = (), feed: list[str] = (), footer: str = "",
          muted: bool = False) -> str:
    chip_html = "".join(f'<span class="fs-chip {cls}">{html.escape(label)}</span>' for label, cls in chips)
    style = ' style="font-weight:400" class="fs-answer fs-muted"' if muted else ' class="fs-answer"'
    feed_html = (f'<div class="fs-feed">{"<br>".join(feed)}'
                 f'{"<br><br>" + footer if footer else ""}</div>') if feed or footer else ""
    return (f'<div class="fs-box"><p{style}>{html.escape(text)}</p>'
            f'{"<div class=fs-chips>" + chip_html + "</div>" if chip_html else ""}{feed_html}</div>')


REPO = "https://github.com/imsoumya18/frameseek"
AUTHOR = ("Soumya", "https://github.com/imsoumya18")
_GITHUB_MARK = ('<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 '
                '5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94'
                '-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87'
                '.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 '
                '2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 '
                '2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 '
                '2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>')


def _header() -> str:
    """App name on the left; who made it and the source code on the right."""
    return (f'<div class="fs-top"><span><b>Frameseek</b> · ask anything said or shown in a video</span>'
            f'<span class="by"><a href="{AUTHOR[1]}" target="_blank" rel="noopener">Developed by '
            f'<b>{AUTHOR[0]}</b></a><a href="{REPO}" target="_blank" rel="noopener">{_GITHUB_MARK}'
            f'Source on GitHub</a></span></div>')


IDLE_CARD = _card("Prepare a video, then ask anything that's said or shown in it.", muted=True)


def _step_line(s) -> str:
    """One line of the live feed, e.g. ✓ zoom 1:01 · a man wearing a pink cap · 26 s"""
    if s.kind == "check" and s.check:
        mark, cls = {"verified": ("✓", "ok"), "unverified": ("~", "un")}.get(s.check.status, ("✗", "no"))
        what = f"check ({s.source})"
        detail = s.check.reason or s.seen
        if s.check.quote_score is not None:
            detail = f"quote “{s.quote[:50]}” {s.check.quote_score:.2f} · {detail}"
    else:
        mark, cls = ("✓", "ok") if s.present else ("✗", "no")
        what, detail = s.kind, s.seen
    secs = f" · {s.timing.total_s:.0f} s" if s.timing else ""
    return (f'<span class="{cls}">{mark}</span> {what} {fmt_time(s.moment.t)} '
            f'<span class="fs-muted">· {html.escape(detail[:140])}{secs}</span>')


def _skip():
    import gradio as gr
    return gr.skip()


# ---------- the checklist ----------

def _title(video_dir) -> str:
    meta = load_meta(video_dir)
    return f"{meta.title} · {mmss(meta.duration_s)}" if meta else ""


def checklist_for(url: str) -> str:
    """Pasting a link fills the checklist from what's already on disk."""
    if not url.strip():
        return _checklist_html(Checklist(), note="Paste a YouTube link above.")
    try:
        video_id = parse_video_id(url)
    except ValueError as exc:
        return _checklist_html(Checklist(), note=html.escape(str(exc)))
    video_dir = _cfg.data_dir / video_id
    c = from_disk(video_dir)
    if c.done == len(STEPS):
        note = "Ready. Ask a question below."
    elif c.done:
        note = "Press <b>Prepare</b> to finish the remaining steps."
    else:
        note = "Not prepared yet. Press <b>Prepare</b>: the first time takes a while (mostly describing keyframes)."
    return _checklist_html(c, _title(video_dir) or video_id, note)


def on_link(url: str, current: str):
    """A new link: redraw the checklist, load that video (paused), and clear the old answer.

    Fires on every keystroke, so the player and answer only change when the video
    id itself changes (pasting the same link again doesn't reload anything).
    """
    try:
        video_id = parse_video_id(url) if url.strip() else ""
    except ValueError:
        video_id = ""
    checklist = checklist_for(url)
    if video_id == current:
        return checklist, _skip(), _skip(), _skip(), _skip(), _skip(), _skip()
    return checklist, _player(video_id, 0, autoplay=False), IDLE_CARD, None, [], {}, video_id


def prepare_video(url: str):
    """Generator: the checklist, redrawn on every progress event while prepare() runs in a thread."""
    if not url.strip():
        yield _checklist_html(Checklist(), note="Paste a YouTube link first.")
        return
    try:
        video_id = parse_video_id(url)
    except ValueError as exc:
        yield _checklist_html(Checklist(), note=html.escape(str(exc)))
        return

    video_dir = _cfg.data_dir / video_id
    c = from_disk(video_dir)
    title = _title(video_dir) or video_id
    events: queue.Queue = queue.Queue()
    outcome: dict = {}
    _say(f"[bold]Preparing[/bold] {video_id}")

    def work() -> None:
        try:
            outcome["result"] = prepare(url, _cfg, on_event=lambda *e: events.put(e))
        except Exception as exc:   # show any failure in the UI instead of a silent thread
            outcome["error"] = exc
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()
    yield _checklist_html(c, title, "Working…")
    last = 0.0
    while (ev := events.get()) is not None:
        key, state, fraction, detail = ev
        if state == "done" and c.steps[key].state != "done":
            _say(f"  ✓ {dict(STEPS)[key]} [dim]{escape(detail)}[/dim]")
        c.set(key, state, fraction, detail)
        if key == "download" and state == "done":
            title = _title(video_dir) or title
        now = time.perf_counter()
        if state == "done" or now - last > 0.25:   # don't redraw more than ~4 times a second
            last = now
            yield _checklist_html(c, title, "Working…")

    if "error" in outcome:
        running = next((k for k, s in c.steps.items() if s.state == "running"), None)
        if running:
            c.set(running, "failed", None, str(outcome["error"])[:200])
        _say(f"[red]Preparing failed:[/red] {escape(str(outcome['error']))}")
        yield _checklist_html(c, title, "Something went wrong; the step marked ❌ says what.")
        return
    _say(f"[green]Ready[/green] {escape(title)}")
    yield _checklist_html(c, title, "Ready. Ask a question below.")


# ---------- asking ----------

def ask_question(url: str, question: str):
    """Generator: (card, player, evidence, moments, state), updated as each step finishes."""
    nothing = (_skip(), _skip(), _skip(), _skip())
    if not question.strip():
        yield (_card("Type a question first.", muted=True), *nothing)
        return
    try:
        video_id = parse_video_id(url)
    except ValueError as exc:
        yield (_card(str(exc), muted=True), *nothing)
        return
    video_dir = _cfg.data_dir / video_id
    if not (video_dir / "index.lance").exists():
        yield (_card("Prepare this video first.", muted=True), *nothing)
        return

    asker = _get_asker()
    updates: queue.Queue = queue.Queue()
    outcome: dict = {}
    _say(f"[bold]Question[/bold] ({video_id}): {escape(question)}")

    def work() -> None:
        try:
            outcome["result"] = asker.ask(video_dir, question, on_step=updates.put)
        except Exception as exc:
            outcome["error"] = exc
        finally:
            updates.put(None)

    threading.Thread(target=work, daemon=True).start()
    feed: list[str] = []
    working = [(f"checking up to {_cfg.answer.look_n} moments, best first", "fs-chip")]
    yield (_card("Searching…", working), *nothing)
    while (step := updates.get()) is not None:
        feed.append(_step_line(step))
        _say(f"  {step.kind} {fmt_time(step.moment.t)} {'yes' if step.present else 'no'} "
             f"[dim]{escape(step.seen[:80])}[/dim]")
        verb = {"look": "Looking at", "zoom": "Zooming into", "check": "Checking"}[step.kind]
        yield (_card(f"{verb} {fmt_time(step.moment.t)}…", working, feed), *nothing)

    if "error" in outcome:
        _say(f"[red]Asking failed:[/red] {escape(str(outcome['error']))}")
        yield (_card(f"Something went wrong: {outcome['error']}", [("error", "fs-low")], feed), *nothing)
        return

    res = outcome["result"]
    a = res.answer
    t = a.timing
    calls = len([s for s in a.steps if s.timing]) or 1
    footer = (f'<span class="fs-muted">search {res.search_ms:.0f} ms · {calls} model calls · '
              f"read {t.prefill_tokens} tokens in {t.prefill_s:.0f} s · wrote {t.decode_tokens} in "
              f"{t.decode_s:.0f} s · total {res.total_s:.0f} s</span>")
    _say(f"[green]Answer[/green] ({a.confidence}, {res.total_s:.0f}s): {escape(a.text)}")

    chips: list[tuple[str, str]] = []
    if a.moment:
        chips.append((f"▶ {fmt_time(a.moment.t)}", "fs-chip"))
        chips.append((f"{a.confidence} confidence", f"fs-{a.confidence}"))
        checks = [s.check.status for s in a.steps if s.kind == "check" and s.check and s.present]
        if checks:
            verified = checks[-1] == "verified"
            chips.append(("evidence checked" if verified else "not independently checked",
                          "fs-high" if verified else "fs-medium"))
    else:
        chips.append(("no moment answers this", "fs-low"))

    start = int(a.moment.t) if a.moment else int(a.candidates[0].t) if a.candidates else 0
    evidence = str(a.crop) if a.crop else str(video_dir / "keyframes" / a.moment.file) if a.moment else None
    moments = [(str(video_dir / "keyframes" / m.file), f"{fmt_time(m.t)}{' ✓' if m is a.moment else ''}")
               for m in a.candidates]
    state = {"video_id": video_id, "times": [int(m.t) for m in a.candidates]}
    yield (_card(a.text, chips, feed, footer), _player(video_id, start), evidence, moments, state)


def seek(state: dict, index: int | None):
    """Clicking a moment restarts the player there."""
    times = (state or {}).get("times", [])
    if index is None or not 0 <= index < len(times):
        return _skip()
    return _player(state["video_id"], times[index])


def _get_asker() -> Asker:
    global _asker
    if _asker is None:
        _asker = Asker(_cfg)
    return _asker


def warm_up() -> None:
    """Load the embedders before the first question, and say whether Ollama is reachable."""
    asker = _get_asker()
    for emb in (asker.text, asker.image):
        start = time.perf_counter()
        _say(f"Loading {emb.name}…")
        emb.load()
        _say(f"  loaded in {time.perf_counter() - start:.1f}s")
    try:
        ok = asker.vision.health() if hasattr(asker.vision, "health") else True
    except Exception:
        ok = False
    _say(f"Answering model {asker.model_name}: "
         + ("[green]Ollama reachable[/green]" if ok else "[red]Ollama not reachable: run `ollama serve`[/red]"))


# ---------- layout ----------

def build():
    import gradio as gr

    with gr.Blocks(title="Frameseek", fill_width=True) as demo:
        gr.HTML(_header(), padding=False)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=380):
                url = gr.Textbox(show_label=False, placeholder="Paste a YouTube link", submit_btn="Prepare",
                                 autofocus=True)
                checklist = gr.HTML(checklist_for(""))
                q = gr.Textbox(show_label=False, submit_btn="Ask",
                               placeholder="Ask anything said or shown: English, Bengali, Hindi…")
                card = gr.HTML(IDLE_CARD)
            with gr.Column(scale=7, min_width=480):
                player = gr.HTML(_player())
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=160):
                        gr.Markdown("Read from", elem_classes="fs-label")
                        evidence = gr.Image(show_label=False, type="filepath", height=170, buttons=[],
                                            interactive=False)
                    with gr.Column(scale=5, min_width=280):
                        gr.Markdown("Moments, best first · click one to jump there", elem_classes="fs-label")
                        moments = gr.Gallery(show_label=False, columns=3, height="auto", object_fit="contain",
                                             allow_preview=False, buttons=[], elem_id="moments")
        state = gr.State({})          # the answer's moments, for click-to-jump
        shown = gr.State("")          # the video id the player is showing

        def on_select(st: dict, evt: gr.SelectData):
            return seek(st, evt.index)
        # This module uses `from __future__ import annotations`, so the hint above is only a
        # string; Gradio needs the real class to know it should pass the click event in.
        on_select.__annotations__["evt"] = gr.SelectData

        # The checklist and the answer card show their own progress, so Gradio's spinners stay hidden.
        preparing = url.submit(prepare_video, inputs=url, outputs=checklist, show_progress="hidden")
        # A new link stops a preparation that's still drawing the old one's checklist.
        url.change(on_link, inputs=[url, shown], outputs=[checklist, player, card, evidence, moments, state, shown],
                   show_progress="hidden", cancels=[preparing])
        q.submit(ask_question, inputs=[url, q], outputs=[card, player, evidence, moments, state],
                 show_progress="hidden")
        moments.select(on_select, inputs=state, outputs=player)
    return demo


def theme():
    import gradio as gr
    return gr.themes.Base(primary_hue="blue", neutral_hue="gray", font=SYSTEM_FONT,
                          font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
                          radius_size="sm", spacing_size="sm", text_size="md")


def launch(port: int = 7860, quiet: bool = False) -> None:
    global _quiet
    _quiet = quiet
    warm_up()
    _say(f"Opening http://127.0.0.1:{port}")
    build().launch(server_port=port, inbrowser=True, quiet=True, footer_links=[],
                   allowed_paths=[str(_cfg.data_dir.resolve())],   # keyframes and crops live here
                   theme=theme(), css=CSS)
