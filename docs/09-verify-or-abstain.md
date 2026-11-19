# 9. Verify or abstain

Code: [`verify.py`](../src/frameseek/verify.py), the checks inside [`look.py`](../src/frameseek/look.py)

A wrong answer costs more than "I can't tell": the user acts on it. This chapter
makes Frameseek check the evidence behind an answer and, when the check fails,
keep looking or say it can't tell.

## Two ways an answer can be ungrounded

Re-reading the transcript for the failures of Chapters 7 and 8 shows two
different errors:

| Case | The quote | In the source? | The error |
|---|---|---|---|
| "The text on screen says 'আমি তো প্রেমিক'" (title card) | আমি তো প্রেমিক | **No**: the card says অনেকটা ভালোবাসা / সৃজিত মুখার্জি | **Made-up evidence** |
| "They sing about the dhak in 'ধরিগান খালিমা'" | ধরিগান খালিমা | **Yes**: Whisper's line at 2:35-2:55 | **Unsupported claim**: the line isn't about the dhak |
| "The speech mentions 'Durga Maje'" | Durga Maje | Close: the transcript has দুগা মাজে / দুকা মাজে | Probably a fair reading of a garbled line |

A string comparison can catch the first and never the second, because the
string matches. So there are two checks, in this order:

1. **Is the quote in the source?** Code, no model. Fast, and impossible to talk
   round.
2. **Does the quote answer the question?** A text-only model call. Only runs if
   check 1 passed.

Purely visual claims ("a man in a pink cap") quote nothing, so check 1 has
nothing to test. The zoom from Chapter 8 is their second look, and check 2 still
runs on the model's own description of the frame. That can't prove the
description true, but it catches an answer that doesn't address the question at
all (see the first full run below).

## Making the model say what it quotes

Free text can't be checked reliably ("the speech says 'X', which means..."). So
the look and zoom schemas have a `quote` field: the exact words that give the
answer, copied in their own script, or `""` for a visual answer. The code checks
that field, not the prose.

The source depends on where the model says the answer is:

| `source` | The quote is checked against |
|---|---|
| `speech` | The transcript chunk playing at that moment |
| `image` (text on screen) | The on-screen text the 4B captioner read at indexing time (`screen_text`, a new index column) |

The captioner is an independent reader: a different model, run before the
question existed. If it read no text on that frame, the quote can't be checked
and the answer is kept but marked **unverified**. Checking the 8B's quote against
its own reading would prove nothing.

## Check 1: fuzzy, because Whisper misspells

Whisper writes the line "এবার পুজোতে তোমার সাথে" as "এবার পুজো তে তুমার শাতে". An
exact match would reject a lyric that really is sung. `quote_score` slides the
quote along the source and keeps the best similarity (difflib's ratio, 0 to 1).

Measured on this song's transcript:

| Line | Score |
|---|---|
| Real lines, written correctly (7) | **0.82 to 1.0** |
| Lines that aren't in the song (6, made up for the test) | **0.43 to 0.54** |
| দুর্গা মায়ে (what "দুগা মাজে" might be) | 0.64 |
| The title-card claim, against the card's on-screen text | 0.33 |

The threshold, `answer.quote_threshold: 0.7`, sits in the gap. Raising it to
0.95 wouldn't stop any more made-up lines (they already fail), but it would
reject four of the seven real ones, including "এবার পুজোতে তোমার সাথে", which
every eval run answered correctly. Stricter is only safer while it still blocks
wrong answers; past that it only throws away right ones. This threshold comes
from one song; Chapter 10 checks it on more.

### A Bengali pitfall

The usual way to strip punctuation, `re.sub(r"[^\w\s]", "", text)`, **deletes
Bengali vowel signs**: Python does not count ো or ু (Unicode categories Mc and
Mn, combining marks) as word characters. "তোমাকে" becomes "তমক", and every
comparison silently ignores vowels. The first scores for this chapter were
inflated by exactly this. `normalise` now drops characters by Unicode category
(punctuation, symbols, separators, control) instead.

## Check 2: does it answer the question?

A text-only call to the same 8B:

> Question: when do they sing about the dhak?
> Evidence (speech at 2:43): ধরিগান খালিমা ...
> reason, supports: true only if the evidence itself answers the question

No image means no ~1,240 image tokens: about 200 tokens, a few seconds. And it
judges one narrow thing, which is easier than answering the question was.

## Abstaining

A failed check counts as "not here", and the loop moves to the next moment.
After `look_n` moments without a passing answer:

- **"Can't tell"** if something was found but failed its checks
- **"Not found"** if nothing was found at all

Either way the answer lists what was seen, so the user can judge.

Scoring now has three outcomes: right, wrong, abstained. **Risk** is how often
an answer is wrong; **coverage** is how often there is an answer at all.
Stricter checks lower risk and coverage together, and the useful checks are the
ones that remove wrong answers without removing right ones. `frameseek eval
--answer` reports answered / right / wrong / abstained and how many candidate
answers the checks rejected; `answer.verify: false` turns the checks off for
comparison.

## First full run

`frameseek eval --answer --mode look`, 21 questions, Qwen3-VL 8B, checks on.

| | Look, no checks (Ch. 8) | Look + checks |
|---|---|---|
| Right moment | 19 | 17 |
| Wrong moment | 2 | 1 |
| Abstained ("can't tell") | 0 | 2 |
| Errors | 0 | 1 |
| Model time per question | mean 50 s | mean 80 s, max 239 s |

What changed, question by question:

- **"when do they sing about the dhak?"**: wrong → **can't tell**. Two moments
  quoted real lines (check 1: 1.00) that aren't about the dhak; check 2 rejected
  both.
- **"end credits"**: wrong (title card) → **can't tell**.
- **"আমি তো প্রেমিক"**: right; the lyric's quote was verified.
- **"এবার পুজোতে তোমার সাথে"**: right → **wrong**. The answer was "a man and a
  woman are dancing in a temple": a visual claim with no quote, so no check ran.
  This is why check 2 now also runs on visual answers.
- **"দুর্গা প্রতিমা কোথায় দেখা যায়?"**: right → **error**. The new `quote` field made
  the Bengali output longer than `num_predict: 300` and the JSON was cut off.
  Now 500, and a failed call counts as "not here" for that moment instead of
  ending the question.

Abstaining is the expensive path: checking all six moments plus failed checks
took up to 239 s, against ~25 s for an answer found and verified on the first
look.

### Second run, after those fixes

| | Look, no checks (Ch. 8) | Checks, run 1 | **Checks, run 2** |
|---|---|---|---|
| Right moment | 19 | 17 | **18** |
| Wrong moment | 2 | 1 | **1** |
| Can't tell | 0 | 2 | **2** |
| Errors | 0 | 1 | **0** |
| Accuracy when it answers | 19/21 = 90% | 17/18 = 94% | **18/19 = 95%** |
| Model time per question | 50 s | 80 s | 88 s (3.7 calls) |

- The Bengali Durga question is right again (no cut-off at 500 tokens).
- All 13 visual questions stayed right: check 2 on descriptions rejected none
  of them.
- **"এবার পুজোতে তোমার সাথে" is still wrong**, and this time check 2 passed it.
  The question is a bare lyric line, not a question, and the judge accepted "a
  couple dancing at a puja" as answering "this puja, with you". A relevance
  check can't help when the question itself is ambiguous. Knowing that a bare
  Bengali line is probably a lyric to search for in the speech is query
  understanding: Chapters 10 and 11.

The trade: the two answers that were wrong without checks became "can't tell",
one new wrong answer appeared and stayed, and each question costs about 38 s
more. When Frameseek answers, it is right 95% of the time instead of 90%.

One wording problem the checks can't see: "the phrase 'ঢাক পেটাতে দুখান কাঠি' is
spoken by a group of people on a porch". The moment is right, but in a music
video the voice is the singer's on the soundtrack, not necessarily the people in
the frame. The answer claims more than the evidence shows.

## What this can't catch

- **A wrong visual claim that the zoom also believes.** Check 2 sees the
  model's own description, so it judges relevance, not truth.
- **A quote from a moment the captioner read no text on.** Kept as unverified.
- **A misreading close to the truth.** The title answer সুজিত মুখার্জি passes
  both checks (it matches the captioner's own misreading, and it does answer
  the question) but the screen says সৃজিত. That's reading, not grounding:
  Chapter 11.

## Check your understanding

1. For "when do they sing about the dhak?", the quote scores 1.0 against the
   transcript. Which check catches the error, and why can't the other?
2. Raise `quote_threshold` from 0.7 to 0.95. What happens to "এবার পুজোতে তোমার
   সাথে", and what does that do to risk and coverage?
