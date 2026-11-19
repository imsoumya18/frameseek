"""Check an answer's evidence before trusting it.

Two different errors need two different checks:

  made-up evidence   the quoted words aren't in the source at all
                     ("the text on screen says 'আমি তো প্রেমিক'" about a title
                     card that says অনেকটা ভালোবাসা / সৃজিত মুখার্জি)
                     -> check 1, in code: is the quote in the source?

  unsupported claim  the quote is real but doesn't answer the question
                     ('ধরিগান খালিমা' is in the transcript, but it isn't about the dhak)
                     -> check 2, a text-only model call: does it answer the question?

A visual answer quotes nothing, so check 1 has nothing to test. Check 2 still
runs on the model's own description of the frame: that can't prove the
description is true (the zoom is the second look for that), but it catches an
answer that doesn't address the question at all ("a man and a woman dancing"
for a question that quotes a lyric).

Check 1 is fuzzy because Whisper's Bengali is badly misspelled: a line really
sung as "এবার পুজোতে তোমার সাথে" is transcribed "এবার পুজো তে তুমার শাতে".
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from frameseek.answer import fmt_time
from frameseek.models.vision import Timing, VisionModel


def normalise(text: str) -> str:
    """Compose characters, drop punctuation, symbols, spaces and control characters, casefold.

    Characters are dropped by Unicode category, not with re's \\w: Python does
    not count Bengali vowel signs (ো, ু, category Mc/Mn) as word characters, so
    re.sub(r"[^\\w\\s]", "", "তোমাকে") gives "তমক" and every comparison would
    silently ignore vowels.
    """
    text = unicodedata.normalize("NFC", text)
    return "".join(c for c in text if unicodedata.category(c)[0] not in "PSZC").casefold()


def quote_score(quote: str, source: str) -> float:
    """Best similarity (0-1) between the quote and any same-length stretch of the source.

    Sliding the quote along the source means a short quote from a long
    transcript chunk can still score 1.0.
    """
    q, s = normalise(quote), normalise(source)
    if not q or not s:
        return 0.0
    if len(s) <= len(q):
        return SequenceMatcher(None, q, s).ratio()
    return max(SequenceMatcher(None, q, s[i:i + len(q)]).ratio() for i in range(len(s) - len(q) + 1))


SUPPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "supports": {"type": "boolean"},
    },
    "required": ["reason", "supports"],
}


def support_prompt(question: str, quote: str, where: str, t: float) -> str:
    return "\n".join([
        "You are checking whether a piece of evidence from a video answers a question.",
        "",
        f"Question: {question}",
        f"Evidence ({where} at {fmt_time(t)}): {quote}",
        "",
        "The evidence may come from an automatic transcript with spelling mistakes; judge what",
        "it says, not its spelling.",
        "- reason: one short sentence on what the evidence says and whether that answers the question.",
        "- supports: true only if the evidence itself answers the question. If it is about",
        "  something else, or only answers it with outside knowledge, false.",
    ])


@dataclass
class Check:
    """The result of verifying one moment's evidence."""

    status: str              # "verified", "failed" or "unverified" (no independent source to check against)
    quote_score: float | None = None
    supports: bool | None = None
    reason: str = ""
    timing: Timing | None = None

    @property
    def passed(self) -> bool:
        return self.status != "failed"


def check_relevance(model: VisionModel, question: str, description: str, t: float,
                    num_predict: int = 200) -> Check:
    """Check 2 alone, for a visual answer: does the model's description answer the question?"""
    out, timing = model.structured(None, support_prompt(question, description, "what the frame shows", t),
                                   SUPPORT_SCHEMA, num_predict=num_predict)
    ok = bool(out.get("supports"))
    return Check("verified" if ok else "failed", None, ok, out.get("reason", "").strip(), timing)


def check_evidence(
    model: VisionModel, question: str, quote: str, source: str, where: str, t: float,
    threshold: float, num_predict: int = 200,
) -> Check:
    """Check 1 (quote in source) then check 2 (quote answers the question).

    No quote means a purely visual claim: nothing to check here (the zoom was its
    second look). No source text means the quote can't be checked independently:
    it's kept, but marked unverified.
    """
    quote = quote.strip()
    if not quote:
        return Check("unverified", reason="no quote: a visual claim")
    if not source.strip():
        return Check("unverified", reason=f"nothing to check the quote against in the {where}")
    score = quote_score(quote, source)
    if score < threshold:
        return Check("failed", score, None, f"quote not found in the {where} (similarity {score:.2f})")
    out, timing = model.structured(None, support_prompt(question, quote, where, t), SUPPORT_SCHEMA,
                                   num_predict=num_predict)
    ok = bool(out.get("supports"))
    return Check("verified" if ok else "failed", score, ok, out.get("reason", "").strip(), timing)
