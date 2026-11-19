"""Local web app: paste a YouTube link, prepare it once, then ask questions.

Run with `frameseek ui`. Preparing streams each pipeline step; asking shows the
answer, a YouTube player that starts at the chosen moment, the evidence frame
and the other candidate moments the model was shown.
"""

from __future__ import annotations

import html
import queue
import threading

from frameseek.answer import describe_parts, fmt_time
from frameseek.config import load_config
from frameseek.ingest import parse_video_id
from frameseek.pipeline import Asker, prepare

_cfg = load_config()
_asker: Asker | None = None   # created on first question; keeps models loaded


def _player(video_id: str, start: int) -> str:
    src = f"https://www.youtube.com/embed/{html.escape(video_id)}?start={start}&autoplay=1&rel=0"
    return (f'<iframe width="100%" height="320" src="{src}" frameborder="0" '
            'allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen></iframe>')


def prepare_video(url: str):
    """Generator: yields a status line per step while prepare() runs in a thread."""
    if not url.strip():
        yield "Paste a YouTube link first."
        return
    try:
        parse_video_id(url)
    except ValueError as exc:
        yield f"**Error:** {exc}"
        return

    updates: queue.Queue = queue.Queue()
    outcome: dict = {}

    def work() -> None:
        try:
            outcome["result"] = prepare(url, _cfg, on_status=updates.put)
        except Exception as exc:   # surface any failure in the UI instead of a silent thread
            outcome["error"] = exc
        finally:
            updates.put(None)

    threading.Thread(target=work, daemon=True).start()
    log: list[str] = []
    while (msg := updates.get()) is not None:
        if log and msg.split(":")[0] == log[-1].split(":")[0]:
            log[-1] = msg            # replace "Captioning keyframes: 3/88" instead of appending
        else:
            log.append(msg)
        yield "\n".join(f"- {line}" for line in log)

    if "error" in outcome:
        yield "\n".join(f"- {line}" for line in log) + f"\n\n**Error:** {outcome['error']}"
    else:
        p = outcome["result"]
        yield "\n".join(f"- {line}" for line in log) + f"\n\n**Ready:** {p.title}"


def ask_question(url: str, question: str):
    global _asker
    if not question.strip():
        return "Type a question first.", "", None, []
    video_id = parse_video_id(url)
    video_dir = _cfg.data_dir / video_id
    if not (video_dir / "index.lance").exists():
        return "Prepare this video first.", "", None, []

    if _asker is None:
        _asker = Asker(_cfg)
    res = _asker.ask(video_dir, question)
    a = res.answer

    t = a.timing
    looked = "".join(f"  \n{s.kind} {fmt_time(s.moment.t)}: "
                     + (s.check.status if s.check else 'yes' if s.present else 'no') + f" · {s.seen}"
                     for s in a.steps)
    md = (f"### {a.text}\n\n**Confidence:** {a.confidence}"
          + (f"  \n**Evidence:** {a.evidence}" if a.evidence else "")
          + (f"  \n**Moment:** {fmt_time(a.moment.t)}" if a.moment else "  \n*No moment answers this.*")
          + (f"\n\n**Looked at:**{looked}" if looked else "")
          + f"\n\n<sub>search {res.search_ms:.0f} ms · {len([st for st in a.steps if st.timing]) or 1} model calls · read {t.prefill_tokens} tokens in "
            f"{t.prefill_s:.1f}s · wrote {t.decode_tokens} in {t.decode_s:.1f}s · total {res.total_s:.1f}s</sub>")
    start = int(a.moment.t) if a.moment else int(a.candidates[0].t) if a.candidates else 0
    # The zoomed crop when there is one: that's what the answer was read from.
    evidence = str(a.crop) if a.crop else str(video_dir / "keyframes" / a.moment.file) if a.moment else None
    gallery = [(str(video_dir / "keyframes" / m.file),
                f"{i}. {fmt_time(m.t)} · {describe_parts(m)}{' · chosen' if m is a.moment else ''}")
               for i, m in enumerate(a.candidates, start=1)]
    return md, _player(video_id, start), evidence, gallery


def build():
    import gradio as gr

    with gr.Blocks(title="frameseek") as demo:
        gr.Markdown("# frameseek\nAsk anything about a video, said or shown. Runs locally.")
        with gr.Row():
            url = gr.Textbox(label="YouTube link", placeholder="https://youtu.be/...", scale=4)
            prep = gr.Button("Prepare video", scale=1)
        status = gr.Markdown()
        with gr.Row():
            q = gr.Textbox(label="Question (English or Bengali)",
                           placeholder="What colour is the guitarist's shirt?", scale=4)
            go = gr.Button("Ask", variant="primary", scale=1)
        with gr.Row():
            with gr.Column(scale=3):
                answer = gr.Markdown()
                player = gr.HTML()
            with gr.Column(scale=2):
                evidence = gr.Image(label="Evidence (zoomed crop when there is one)", type="filepath")
        candidates = gr.Gallery(label="Candidate moments, best first", columns=3, height=220)

        prep.click(prepare_video, inputs=url, outputs=status)
        go.click(ask_question, inputs=[url, q], outputs=[answer, player, evidence, candidates])
        q.submit(ask_question, inputs=[url, q], outputs=[answer, player, evidence, candidates])
    return demo


def launch(port: int = 7860) -> None:
    build().launch(server_port=port, inbrowser=True)
