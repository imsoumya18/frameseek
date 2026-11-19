# 10. Benchmark: three videos, three languages

Code: [`evaluate.py`](../src/frameseek/evaluate.py), `frameseek eval`;
question sets in [`evals/`](../evals), per-question results in [`evals/results/`](../evals/results)

Chapters 7-9 tuned frameseek on one 4.8-minute Bengali music video and 21
questions. This chapter checks whether the results hold on different videos.

## The videos

| Video | Language | Length | Keyframes | Speech source | What it stresses |
|---|---|---|---|---|---|
| [Infra, *Khoshmejaje*](https://youtu.be/nRWZBq2hcls) | Bengali | 4:46 | 88 | Whisper (lyrics, badly garbled) | Fast cuts, small details, sung speech |
| [Obama's back-to-school speech, with stock footage](https://youtu.be/PGUdWfB8nLg) | English | 11:09 | 145 | YouTube auto captions | Burned-in subtitles, stock footage |
| [Naman Sharma on *Belonging*](https://youtu.be/5qbVFbjAj1c) | Hindi | 15:32 | 184 | Whisper (YouTube's captions were in the wrong language) | Devanagari graphics, numbers, talking head |

Indexing cost on a 16 GB MacBook Air: 43 minutes for the Obama video and 83 for
the Hindi one, almost all of it captioning (17-20 s per keyframe) and, for
Hindi, Whisper (19 minutes for 15.5 minutes of audio). The Hindi video's YouTube
captions were labelled Bengali; the language check from Chapter 4 caught it.

## Writing the questions

Each question set was written by looking at every keyframe and the transcript,
and each answer is checked by eye:

- **frames**: the keyframes that show the answer; **spans**: when it is said
- **contains**: exact strings the answer must include (numbers, names)
- **absent**: something that is *not* in the video. The right answer is "not
  found"; answering anyway counts as wrong. Without these, a system that always
  answers would look perfect.

Some questions are built as traps:

- **One name, three spellings** (Obama video): the Harry Potter cover says
  J.K. ROWLING, the burned-in subtitle "J K Rawlings", YouTube's captions
  "Rollins". The answer counts only if it says Rowling.
- **Screen vs speech** (Hindi video): the graphics say **55** errors and
  **1921**; Whisper heard "75" and "1931". The expected answers follow the
  screen.

## Retrieval

| Video | Questions | hit@1 | hit@3 | MRR |
|---|---|---|---|---|
| Bengali music video | 21 | 16 | 18 | 0.83 |
| Obama (English) | 16 | 13 | **16** | 0.89 |
| Hindi commentary | 18 | 15 | **18** | 0.91 |

The two new videos look better than the first, and part of that is the test,
not the system:

- Both have their speech **burned into the picture as subtitles**, and the
  captioner copies on-screen text. Even "speech" questions get matched through
  the captions.
- The questions were written after looking at the same frames and captions the
  system uses, which tends to produce questions it can find.

The music video, with no subtitles and garbled lyrics, is the more honest test.

## Answers (look mode with checks)

| Video | In the video | Right | Wrong | Can't tell | Not in the video: said none | Exact text | Time per question |
|---|---|---|---|---|---|---|---|
| Bengali music video | 21 | 18 | 1 | 2 | (none asked) | 2/3 | 88 s |
| Obama (English) | 16 | 10 | **0** | **6** | 2/2 | 5/5 | 161 s |
| Hindi commentary | 18 | 15 | **0** | 3 | 2/2 | 6/7 | 127 s |
| **All** | **55** | **43** | **1** | **11** | **4/4** | **13/15** | |

- **When frameseek answers, it is right 43 times out of 44 (98%).** It answers
  44 of the 55 answerable questions (80% coverage).
- **All four things that aren't in the videos were declined** ("a cat on a
  sofa", "a dog", "a cricket match", "someone on a bicycle"), even though search
  always returns a best match (for the cat: Barack and Michelle Obama on a sofa).
- **The traps:** "who wrote Harry Potter?" → "J.K. Rowling", read from the cover.
  The English questions about the Hindi graphics got 1921, 1925, 2020, 353 and
  55, all from the screen.

### Where it fell short

**Too cautious on the Obama video: 6 of 16 "can't tell", all visual and easy**
(graduates throwing caps, the American flag, the chalkboard, the old basketball
photo, the photo of a young boy, students on the grass). Retrieval had the right
frame first for five of them. The checks rejected 13 candidate answers on this
video, against 5 on the Hindi one. The suspected cause, still to confirm: nearly
every frame has a burned-in subtitle, the look step quotes it, check 1 confirms
the subtitle is on screen, and check 2 then asks whether *the subtitle* answers
"graduates throwing caps", which it doesn't. A visual answer is being judged
by an unrelated quote. Per-question results now record each step's quote and
check reason, so the next run will show it.

**The Hindi-language question trusted the transcript over the screen.** "किताब में
आरएसएस की स्थापना का साल क्या बताया गया?" got **1931**, Whisper's mishearing, while the
same question in English got 1921 from the graphic. A Hindi question pulls the
model toward the Hindi transcript. Both are "evidence"; nothing yet decides which
source to trust for a number.

**Abstaining is expensive.** A "can't tell" means looking at all six moments plus
failed checks: up to 347 s. With six of them, the Obama video averaged 161 s per
question.

## What this benchmark doesn't cover

- **Other answer modes** on the new videos (look without checks, compare): the
  first video's comparison is in Chapter 9; the full grid is an open issue.
- **The Tamil video** ([eLS-9_FBORk](https://youtu.be/eLS-9_FBORk), 34:35):
  a third script and a long video, about an hour of captioning.
- **Questions written without looking at the captions**, and videos without
  burned-in subtitles.
- **Repeat runs.** Each number is one run at temperature 0. A prompt change
  moves individual answers (Chapter 7), so differences of one or two questions
  are within noise.

## Check your understanding

1. Retrieval found the right moment in the top 3 for every question on the
   Obama video, yet only 10 of 16 were answered. Where in the pipeline did the
   other six go, and why might burned-in subtitles be the cause?
2. Why do the "not in the video" questions matter more than they look, given
   that there are only four of them?
