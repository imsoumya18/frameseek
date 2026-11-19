# frameseek learning notes

These notes explain the concepts behind each part of frameseek, in the order the
project was built. Each chapter covers the ideas first, then points to the code
that implements them, so you can read along with the source.

| # | Chapter | Code |
|---|---|---|
| 1 | [Setup and architecture](01-setup-and-architecture.md) | `config.py`, `devices.py`, `models/vision.py`, `cli.py` |
| 2 | [Video fundamentals and ingestion](02-video-fundamentals.md) | `ingest.py` |
| 3 | [Shot detection](03-shot-detection.md) | `shots.py` |
| 4 | [Speech to text](04-speech-to-text.md) | `transcribe.py` |
| 5 | [Vision-language models](05-vision-language-models.md) | `caption.py`, `models/vision.py` |
| 6 | [Embeddings and vector search](06-embeddings-and-search.md) | `index.py`, `models/embed.py` |
| 7 | [Retrieval, answering and the UI](07-retrieval-and-answering.md) | `answer.py`, `pipeline.py`, `app.py` |
| 8 | [Looking again: grounding, crop and zoom](08-look-again.md) | `look.py`, `zoom.py` |
| 9 | [Verify or abstain](09-verify-or-abstain.md) | `verify.py`, `look.py` |
| 10 | Benchmark and evaluation | |
| 11 | Indian scripts and languages | |
| 12 | Results and lessons | |

Each chapter ends with a few questions to check your understanding.
