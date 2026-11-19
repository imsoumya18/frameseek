# 6. Embeddings and vector search

Code: [`src/frameseek/index.py`](../src/frameseek/index.py),
[`src/frameseek/models/embed.py`](../src/frameseek/models/embed.py)

## Embeddings

An embedding turns a piece of text or an image into a list of numbers (here 384
or 768 of them) so that **things with similar meaning point in similar
directions**. "red saree", "লাল শাড়ি" and a photo of a woman in a red saree
should all land in the same neighbourhood; "man playing guitar" lands elsewhere.
Search then becomes geometry: embed the question, find the stored vectors
pointing the most similar way.

## Cosine similarity

Vectors are compared by the angle between them:

| Cosine | Meaning |
|---|---|
| 1.0 | Same direction: same meaning |
| 0 | At right angles: unrelated |
| Negative | Opposite directions |

A vector's length often reflects things like text length rather than meaning,
so every vector is normalised to length 1. Cosine similarity is then just a dot
product.

## Contrastive learning: CLIP and SigLIP

SigLIP2 has an image encoder and a text encoder trained together on billions of
image and alt-text pairs, so both produce vectors in the same space. Training
**pulls together** each image and its own caption and **pushes apart** the image
and every other caption in the batch.

| | CLIP | SigLIP / SigLIP2 |
|---|---|---|
| Training signal | Pick the right caption out of the whole batch (softmax) | For each pair independently: match or not (sigmoid) |
| Consequence | Needs very large batches | Works with smaller batches, scales better |
| Text | Mostly English | SigLIP2 adds multilingual training |

## Three sources, because each one misses things

| Table / vector | Model | Finds | Misses |
|---|---|---|---|
| `frames.txt_vec`: caption text | multilingual-e5 | Facts the captioner wrote: colours, counts, on-screen text | Anything the captioner left out |
| `frames.img_vec`: frame pixels | SigLIP2 | Visual content no caption mentions | Small text, counting, fine detail |
| `speech.txt_vec`: transcript chunks | multilingual-e5 | What was said or sung | Anything only shown |

Measured with the real models on three test frames and two speech lines:

| Question | Captions | Frames | Speech |
|---|---|---|---|
| "something red" | red frame | red frame | (nothing relevant) |
| "লাল রঙ" (Bengali: red colour) | red frame | red frame | |
| "the sun in the sky" | wrong (no caption mentions the sun) | **sun frame** | |
| "dhak drum" | | | **lyric line** |
| "ট্রেন স্টেশন" (Bengali: train station) | | | **English line** |

Each source rescued at least one question the others could not answer.

Note what else this shows: **every table always returns something.** For
"something red", the speech table still returned a dhak lyric as its "best"
match. Nearest is not the same as relevant; Chapter 7 deals with this.

### On a real video

Top result per source on the 4.8-minute Bengali music video (88 keyframes, 8 speech chunks):

| Question | Captions | Frames | Speech |
|---|---|---|---|
| "man playing guitar" | guitar shots | same guitar shots | (nothing relevant) |
| "guy in a pink cap" | miss: no caption mentions the cap | **the right frame, #1** | |
| "Japanese text on a t-shirt" | **the right frame, #1** (text was captioned) | miss: cannot read text | |
| "দুর্গা প্রতিমা" (Bengali: Durga idol) | "Durga idols" caption, #1 | **top 4 are all idol frames** | |
| "ঢাক পেটাতে" (Bengali lyric) | noise | dance scenes | **the exact lyric line, #1** |

The pink cap and the Japanese text are on **the same frame**, and each was found
by a different source. Two further patterns:

- A title card with Bengali on-screen text ranked high in caption search for every
  Bengali question: the script-matching bias described below.
- When a source has nothing relevant, its scores bunch together (0.785, 0.784,
  0.784, 0.784, 0.783); a real match stands out from the rest. The **gap** between
  the top result and the others signals whether a source found anything.

### Asymmetric retrieval: the e5 prefixes

multilingual-e5 was trained with `query: ` on questions and `passage: ` on stored
text, because a short question and a longer passage are different kinds of text.
On five test questions, the prefixes gave 4 correct top matches; without them, 3.

### Scores are not comparable across models

SigLIP2's cosine scores for good matches are around 0.1 to 0.2; e5's sit between
0.7 and 0.9 even for poor ones. A raw 0.13 from SigLIP2 can be a strong match
while 0.80 from e5 can be a weak one. Ranking within each source is meaningful;
comparing raw scores across sources is not. Chapter 7 combines them by rank.

### Empty text matches everything ("hubness")

The last keyframe of the test video is pure black, and the vision model rightly
left every caption field empty. Flattened, that became `". Shot: wide."`, and it
ranked 4th for "man playing guitar" with a score of 0.82. Text embedders place
near-empty text in a generic region that is moderately close to every query; a
vector that turns up near everything is called a **hub**.

Frames whose caption has no content (no summary, objects, text or people) are now
flagged (`has_text = false`) and filtered out of caption search *before* the top k
are taken (`prefilter`). They stay in the table, so pixel search can still find them.

### Known weakness: cross-lingual text search

"লাল শাড়ি পরা মেয়ে" (girl in a red saree) matched a **Bengali lyric line**
(0.81) over the English caption about a woman in red (0.77). The model partly
matches on script: Bengali text looks like Bengali text. `multilingual-e5-base`
made the same mistake at twice the size, so this is not a model-size problem.

Mitigations in place:

- Captions and speech are separate tables, so lyrics never compete directly with
  captions.
- SigLIP2 handles Bengali visual questions reasonably ("লাল রঙ" found the red frame).

Chapter 11 tackles it directly by translating the question with the local LLM
before searching.

## Exact search vs approximate nearest neighbour (ANN)

**Exact search** compares the question with every stored vector. For one video
(~140 vectors x 768 numbers) that is about 100,000 multiply-adds: microseconds,
and always correct.

**ANN indexes** avoid checking everything:

| Index | Idea |
|---|---|
| HNSW | A graph linking each vector to its near neighbours, plus "highway" links between distant regions. Search hops greedily toward the question. |
| IVF-PQ | Cluster the vectors, search only the nearest clusters, store compressed vectors. |

They trade a little **recall** (occasionally missing the true nearest; a greedy
walk can stop at a local best), extra memory, build time and update cost for
large speed-ups. They matter from roughly hundreds of thousands of vectors, i.e.
thousands of videos.

Frameseek uses exact search. LanceDB can add an ANN index later without changing
the search code.

## Storage

`data/<id>/index.lance/` (LanceDB, files on disk, no server):

```
frames   file, t, shot, summary, text, img_vec[768], txt_vec[384]
speech   start, end, text, txt_vec[384]
```

- Transcript lines are merged into chunks of about 15 seconds (split at silences
  longer than 3 s, `[Music]`-style markers dropped): a single line like "চলো" is
  too short to carry meaning.
- LanceDB only treats a column as searchable vectors if it is a **fixed-size list
  of float32**; plain Python lists of floats become variable-length float64 and
  are rejected. The vectors are written through an explicit Arrow schema.
- `index_meta.json` stores a hash of all inputs (models, captions, transcript,
  chunk size). If it matches, the index is reused.

## Check your understanding

1. Give one question caption search would answer but frame-vector search would
   probably miss, and one the other way round.
2. With about 140 vectors, would you build an HNSW index? When would you, and
   what do you give up?
3. SigLIP2 gives a match 0.13 and e5 gives another 0.80. Which is the better
   match, and why can't you tell from those numbers?
