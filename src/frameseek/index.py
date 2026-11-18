"""Build and search the per-video vector index.

data/<video_id>/index.lance/ holds two LanceDB tables:

    frames   one row per keyframe: SigLIP2 image vector (img_vec) and an e5
             vector of the caption text (txt_vec), plus time, shot and caption
    speech   one row per ~15 s chunk of transcript: e5 vector, start, end, text

Captions and speech live in separate tables so that, for example, Bengali
lyrics never compete directly with English captions for a Bengali question.

index_meta.json stores a hash of every input (models, captions, transcript,
chunking); when it matches, the index is reused.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from frameseek.config import EmbedConfig
from frameseek.ingest import write_json_atomic
from frameseek.transcribe import Transcript, is_non_speech

ProgressFn = Callable[[dict[str, Any]], None]


# ---------- turning captions and speech into searchable text ----------

def caption_text(c: dict[str, Any]) -> str:
    """Flatten a structured caption into one sentence-like passage for the text embedder."""
    parts = [c.get("summary", "").rstrip(".") + "."]
    if c.get("shot"):
        parts.append(f"Shot: {c['shot']}.")
    people = c.get("people") or {}
    if people.get("count"):
        parts.append(f"People: {people['count']}, {people.get('description', '')}".rstrip(", ") + ".")
    objs = [f"{o.get('colour', '')} {o.get('name', '')}".strip() for o in c.get("objects", [])]
    if objs:
        parts.append("Objects: " + ", ".join(objs) + ".")
    if c.get("text_on_screen"):
        parts.append("Text on screen: " + ", ".join(c["text_on_screen"]) + ".")
    for key in ("setting", "action"):
        if c.get(key):
            parts.append(f"{key.capitalize()}: {c[key]}.")
    return " ".join(parts)


def has_content(c: dict[str, Any]) -> bool:
    """False for captions the model left empty (black frames, pure blur).

    Their flattened text is near-empty, and text embedders place near-empty text
    close to everything ("hubness"): a black frame scored 0.82 for "man playing
    guitar". Such frames stay in the table for pixel search but are excluded
    from caption search.
    """
    return bool(c.get("summary", "").strip() or c.get("objects") or c.get("text_on_screen")
                or (c.get("people") or {}).get("count"))


def chunk_speech(segments, target_s: float = 15.0, max_gap_s: float = 3.0) -> list[dict[str, Any]]:
    """Merge consecutive transcript segments into ~target_s chunks.

    Single lines ("চলো") are too short to carry meaning on their own. A chunk
    ends when it reaches target_s or when there is a silence longer than
    max_gap_s. Sound markers like "[Music]" are dropped.
    """
    chunks: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for s in segments:
        if is_non_speech(s.text) or not s.text.strip():
            continue
        if cur and (s.start - cur["end"] > max_gap_s or cur["end"] - cur["start"] >= target_s):
            chunks.append(cur)
            cur = None
        if cur is None:
            cur = {"start": s.start, "end": s.end, "text": s.text.strip()}
        else:
            cur["end"] = max(cur["end"], s.end)
            cur["text"] += " " + s.text.strip()
    if cur:
        chunks.append(cur)
    return chunks


def inputs_key(cfg: EmbedConfig, records: list[dict], transcript: Transcript | None) -> str:
    blob = json.dumps({
        "text_model": cfg.text_model, "image_model": cfg.image_model,
        "chunk_s": cfg.speech_chunk_s,
        "frames": [(r["file"], r["caption"]) for r in records],
        "speech": [(s.start, s.end, s.text) for s in transcript.segments] if transcript else None,
        "version": 2,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------- LanceDB tables ----------

def _vectors(arr: np.ndarray):
    """LanceDB only searches fixed-size float32 lists, not plain Python lists of floats."""
    import pyarrow as pa
    dim = arr.shape[1] if arr.ndim == 2 and arr.shape[0] else 0
    flat = pa.array(arr.astype("float32").ravel(), pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def _write_tables(db_path: Path, frames: dict, speech: dict) -> None:
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(db_path))
    db.create_table("frames", data=pa.table({
        "file": pa.array(frames["file"], pa.string()),
        "t": pa.array(frames["t"], pa.float64()),
        "shot": pa.array(frames["shot"], pa.int32()),
        "summary": pa.array(frames["summary"], pa.string()),
        "text": pa.array(frames["text"], pa.string()),
        "has_text": pa.array(frames["has_text"], pa.bool_()),
        "img_vec": _vectors(frames["img_vec"]),
        "txt_vec": _vectors(frames["txt_vec"]),
    }), mode="overwrite")
    if speech["text"]:
        db.create_table("speech", data=pa.table({
            "start": pa.array(speech["start"], pa.float64()),
            "end": pa.array(speech["end"], pa.float64()),
            "text": pa.array(speech["text"], pa.string()),
            "txt_vec": _vectors(speech["txt_vec"]),
        }), mode="overwrite")
    elif "speech" in _table_names(db):
        db.drop_table("speech")


def _table_names(db) -> list[str]:
    """list_tables() returns a listing object in newer LanceDB, a plain list in older ones."""
    if hasattr(db, "list_tables"):
        listing = db.list_tables()
        return list(getattr(listing, "tables", listing))
    return list(db.table_names())


# ---------- build ----------

@dataclass
class IndexStats:
    frames: int
    speech_chunks: int
    seconds: float


def build_index(
    video_dir: Path,
    records: list[dict[str, Any]],
    transcript: Transcript | None,
    text_emb,
    image_emb,
    cfg: EmbedConfig,
    force: bool = False,
    on_progress: ProgressFn = lambda _: None,
) -> tuple[IndexStats, bool]:
    """Embed captions, frames and speech into data/<id>/index.lance. Returns (stats, was_cached)."""
    meta_path = video_dir / "index_meta.json"
    key = inputs_key(cfg, records, transcript)
    if not force and meta_path.exists() and (video_dir / "index.lance").exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("key") == key:
            return IndexStats(meta["frames"], meta["speech_chunks"], 0.0), True

    start = time.perf_counter()
    records = sorted(records, key=lambda r: r["t"])

    for emb, name in ((text_emb, cfg.text_model), (image_emb, cfg.image_model)):
        cached = getattr(emb, "is_downloaded", lambda: True)()
        on_progress({"stage": "load", "model": name, "download": not cached})
        emb.load()
    on_progress({"stage": "loaded"})

    on_progress({"stage": "embed_captions", "count": len(records)})
    txt_vec = text_emb.passages([caption_text(r["caption"]) for r in records])
    img_vec = image_emb.images([video_dir / "keyframes" / r["file"] for r in records], on_progress)

    chunks = chunk_speech(transcript.segments, cfg.speech_chunk_s) if transcript else []
    on_progress({"stage": "embed_speech", "count": len(chunks)})
    sp_vec = text_emb.passages([c["text"] for c in chunks])

    on_progress({"stage": "write"})
    _write_tables(
        video_dir / "index.lance",
        frames={"file": [r["file"] for r in records], "t": [r["t"] for r in records],
                "shot": [r["shot"] for r in records],
                "summary": [r["caption"]["summary"] for r in records],
                "text": [caption_text(r["caption"]) for r in records],
                "has_text": [has_content(r["caption"]) for r in records],
                "img_vec": img_vec, "txt_vec": txt_vec},
        speech={"start": [c["start"] for c in chunks], "end": [c["end"] for c in chunks],
                "text": [c["text"] for c in chunks], "txt_vec": sp_vec},
    )
    stats = IndexStats(len(records), len(chunks), time.perf_counter() - start)
    write_json_atomic(meta_path, {"key": key, "frames": stats.frames, "speech_chunks": stats.speech_chunks,
                                  "text_model": cfg.text_model, "image_model": cfg.image_model})
    return stats, False


# ---------- reading rows back ----------

def load_rows(video_dir: Path) -> tuple[list[dict], list[dict]]:
    """All frames and speech chunks without their vectors, sorted by time."""
    import lancedb

    db = lancedb.connect(str(video_dir / "index.lance"))
    names = _table_names(db)
    ft = db.open_table("frames").to_arrow()
    cols = [c for c in ["file", "t", "shot", "summary", "text", "has_text"] if c in ft.schema.names]
    frames = ft.select(cols).to_pylist()
    speech = (db.open_table("speech").to_arrow().select(["start", "end", "text"]).to_pylist()
              if "speech" in names else [])
    return sorted(frames, key=lambda r: r["t"]), sorted(speech, key=lambda r: r["start"])


def score_all(video_dir: Path, question: str, text_emb, image_emb) -> tuple[list[dict], list[dict], dict]:
    """Exact cosine similarity of the question with every frame, caption and speech chunk.

    Returns (frames, speech, scores) with frames and speech sorted by time and
    scores aligned to them. Fusion needs the whole distribution, not a top-k:
    "how far above the rest" is what makes scores from different models
    comparable. Exact search over a few thousand rows is a millisecond of numpy.
    """
    import lancedb

    db = lancedb.connect(str(video_dir / "index.lance"))
    names = _table_names(db)
    ft = db.open_table("frames").to_arrow().to_pylist()
    ft.sort(key=lambda r: r["t"])
    st = sorted(db.open_table("speech").to_arrow().to_pylist(), key=lambda r: r["start"]) if "speech" in names else []

    def unit(rows: list[dict], col: str) -> np.ndarray:
        m = np.asarray([r[col] for r in rows], dtype=np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    def unit_vec(v) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    q_txt = unit_vec(text_emb.queries([question])[0])
    q_img = unit_vec(image_emb.texts([question])[0])
    scores: dict[str, np.ndarray] = {}
    if ft:
        cap = unit(ft, "txt_vec") @ q_txt
        if "has_text" in ft[0]:
            cap = np.where([r["has_text"] for r in ft], cap, np.nan)   # empty captions never match
        scores["captions"] = cap
        scores["frames"] = unit(ft, "img_vec") @ q_img
    if st:
        scores["speech"] = unit(st, "txt_vec") @ q_txt

    keep = ["file", "t", "shot", "summary", "text", "has_text"]
    frames = [{k: r[k] for k in keep if k in r} for r in ft]
    speech = [{k: r[k] for k in ("start", "end", "text")} for r in st]
    return frames, speech, scores


# ---------- search ----------

def search(video_dir: Path, question: str, text_emb, image_emb, k: int = 5) -> dict[str, list[dict]]:
    """Raw retrieval from each source separately. Section 7 fuses these into one ranking.

    Returns {"captions": [...], "frames": [...], "speech": [...]}, each sorted by
    cosine similarity (score = 1 - cosine distance).
    """
    import lancedb

    db = lancedb.connect(str(video_dir / "index.lance"))
    names = _table_names(db)
    q_txt = text_emb.queries([question])[0]
    q_img = image_emb.texts([question])[0]

    def run(table: str, column: str, vec: np.ndarray, where: str | None = None) -> list[dict]:
        if table not in names:
            return []
        tbl = db.open_table(table)
        q = tbl.search(vec, vector_column_name=column).distance_type("cosine")
        if where and where in tbl.schema.names:   # older indexes lack the column: search unfiltered
            q = q.where(where, prefilter=True)    # filter first, then take the top k
        rows = q.limit(k).to_list()
        for r in rows:
            r["score"] = 1.0 - r.pop("_distance")
            r.pop("img_vec", None)
            r.pop("txt_vec", None)
        return rows

    return {
        "captions": run("frames", "txt_vec", q_txt, where="has_text"),  # question vs caption text (e5)
        "frames": run("frames", "img_vec", q_img),     # question text vs frame pixels (SigLIP2)
        "speech": run("speech", "txt_vec", q_txt),     # question text vs what was said (e5)
    }
