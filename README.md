# frameseek

Ask questions about anything in a video, whether it was **said** or only **shown**,
and get the answer with the moment and the frame as evidence.

> "What colour was the car?" · "When does the idol first appear?" · "gaadi ka rang kya tha?"

frameseek runs entirely on your own machine with open-source models, and is being
built with Indian-language videos (Hindi, Bengali, Telugu and code-mixed speech) in mind.

**Status:** work in progress. Download, shot detection and speech-to-text work today;
frame captioning, search and answering are next. See the [roadmap](#roadmap).

---

## Why

Most "chat with a video" tools search the **transcript**, so they can only find
what someone said. Questions about what is *on screen* (a colour, a sign, an
object, who is in the frame) need the frames themselves. Doing that well and
cheaply on a laptop, and for Indian-language content, is the goal of this project.

## How it works

frameseek splits the work into two phases:

```mermaid
flowchart LR
    subgraph Indexing["Indexing: once per video"]
        A[YouTube link] --> B[Download] --> C[Shots + keyframes] --> D[Captions + speech] --> E[(Index)]
    end
    subgraph Asking["Asking: every question"]
        Q[Question] --> S[Search] --> L[Look at frames] --> R[Answer + timestamp + frame]
    end
    E -.-> S
```

Expensive work happens once at indexing time, so answering stays fast.

| Step | What it does | Status |
|---|---|---|
| Download | Fetches the video once (≤720p) into a per-video cache, keyed by video ID | Done |
| Shots + keyframes | Splits the video at cuts and keeps one frame per shot, plus extra frames in long shots | Done |
| Speech | Word-timestamped transcript: YouTube captions if their language checks out, else local Whisper | Done |
| Captions | A local vision-language model describes each keyframe | Next |
| Index | Frame, caption and transcript embeddings in a local vector database | Planned |
| Search + answer | Finds candidate moments, looks again at those frames, answers with evidence | Planned |

## Quickstart

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/),
and [Ollama](https://ollama.com) for the vision model.

```bash
git clone https://github.com/imsoumya18/frameseek.git
cd frameseek
uv sync
ollama pull qwen3-vl:8b-instruct

uv run frameseek doctor          # checks ffmpeg, device and the vision model
```

### Commands available today

```bash
# Download a video once (any YouTube URL form works; re-runs are instant)
uv run frameseek ingest "https://youtu.be/<video_id>"

# Split it into shots and extract keyframes (runs ingest first if needed)
uv run frameseek shots "https://youtu.be/<video_id>"

# Timestamped speech: YouTube captions if available, otherwise local Whisper
uv run frameseek transcribe "https://youtu.be/<video_id>"

# Ask the vision model about any single image
uv run frameseek describe path/to/image.jpg -p "What colour is the car?"
```

Everything for a video is cached in `data/<video_id>/`:

```
data/<video_id>/
├── video.mp4            downloaded video
├── meta.json            duration, fps, codecs, keyframe spacing, available captions
├── shots.json           shot boundaries and chosen keyframes (with the settings used)
├── keyframes/           one JPEG per keyframe
├── transcript.json      speech with word timestamps, its source and language
└── contact_sheet.jpg    every keyframe in a grid, for checking results by eye
```

Each step skips itself when its output already exists, and re-runs automatically
when the settings that produced it change.

## Measured so far

On a MacBook Air (16 GB), with a 4.8-minute music video:

| | Result |
|---|---|
| Shot detection | 84 shots found in 8.3 s (about 34x faster than real time) |
| Keyframes | 88, versus 143 if sampling a frame every 2 s |
| Vision model, one frame | ~20 s warm (14.5 s reading the image, 6.1 s writing the answer) |
| Speech, Bengali song | YouTube captions came out in Hindi (wrong uploader tag). Whisper `small` detects Bengali, `large-v3-turbo` transcribes it in Bengali script (~144 s) |

The last row is why the next steps index frames at low resolution with short
captions, and spend high-resolution compute only on the frames a question needs.
The reasoning is in [docs/01](docs/01-setup-and-architecture.md#where-the-time-goes-load-prefill-decode).

## Configuration

All models and settings live in [`config.yaml`](config.yaml). Swapping a model or
tuning shot detection is a one-line change, no code edits needed:

```yaml
vision:
  model: qwen3-vl:8b-instruct
shots:
  min_shot_s: 0.6     # ignore "cuts" closer together than this (camera flashes)
  every_s: 10.0       # extra keyframes inside long shots
```

Each engine picks its own device: PyTorch models use CUDA, Apple Silicon (MPS) or CPU;
Whisper uses CUDA or CPU; the vision model runs wherever Ollama puts it (Metal on a Mac).
`frameseek doctor` shows all three.

## Documentation

The [`docs/`](docs/README.md) folder explains the concepts behind each part, in
the order they were built, with diagrams and self-check questions:

- [Setup and architecture](docs/01-setup-and-architecture.md): two-phase design, model servers vs in-process models, prefill vs decode
- [Video fundamentals and ingestion](docs/02-video-fundamentals.md): containers and codecs, I/P/B frames, seeking, atomic caching
- [Shot detection](docs/03-shot-detection.md): change scores, adaptive thresholds, precision vs recall
- [Speech to text](docs/04-speech-to-text.md): spectrograms, encoder-decoder, word timestamps, hallucinations

See the [docs index](docs/README.md) for all chapters.

## Roadmap

- [x] 1. Setup and architecture
- [x] 2. Video download and caching
- [x] 3. Shot detection and keyframes
- [x] 4. Speech to text
- [ ] 5. Frame captioning with a vision-language model
- [ ] 6. Embeddings and vector search
- [ ] 7. Search, answering and a web UI (first end-to-end demo)
- [ ] 8. Looking again: zooming into frames and locating objects
- [ ] 9. Verifying answers, or saying "can't tell"
- [ ] 10. Benchmark against transcript-only search and hosted models
- [ ] 11. Indian scripts and languages
- [ ] 12. Results write-up

## Development

```bash
uv run pytest        # runs without network or a real model; uses fakes and a generated test video
```

```
src/frameseek/
├── cli.py           commands
├── config.py        typed config loaded from config.yaml
├── devices.py       CUDA / MPS / CPU selection
├── ingest.py        download, probe, cache
├── shots.py         shot detection, keyframes, contact sheet
├── transcribe.py    YouTube captions or Whisper, word timestamps
└── models/vision.py vision model interface + Ollama backend
```

## License

[MIT](LICENSE). Models are downloaded separately and keep their own licences
(Qwen3-VL is Apache-2.0).
