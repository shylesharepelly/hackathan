Cell 1 — Install Dependencies
!pip install pymupdf sentence-transformers msgspec scipy tiktoken openai gradio torch numpy scikit-learn
Cell 2 — Imports + Setup
from __future__ import annotations

import re
import json
import time
import hashlib
import logging
import asyncio
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Optional
from enum import Enum

import fitz
import msgspec
import numpy as np
import tiktoken
import torch
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import AsyncOpenAI
import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

Path("output").mkdir(exist_ok=True)

# ── device ────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Device: %s", DEVICE)

# ── thread/process pools ──────────────────────────────────────────
IO_POOL  = concurrent.futures.ThreadPoolExecutor(max_workers=2)
CPU_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=2)

# ── tokenizer ─────────────────────────────────────────────────────
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ── embedding cache (in-memory, keyed by pdf sha256) ─────────────
_EMBED_CACHE: dict[str, np.ndarray] = {}

# ── regex patterns pre-compiled ──────────────────────────────────
RE_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[.\s]")
RE_SOFT_BREAK       = re.compile(r"(?<!\n)\n(?!\n)")
RE_MULTI_SPACE      = re.compile(r"[ \t]{2,}")

def _enc_hook(obj):
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"msgspec: unsupported type {type(obj)}")

def _struct_to_dict(obj):
    if isinstance(obj, msgspec.Struct):
        return {f: _struct_to_dict(getattr(obj, f)) for f in obj.__struct_fields__}
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, list):
        return [_struct_to_dict(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: _struct_to_dict(v) for k, v in obj.items()}
    return obj

print("✅ Imports done")
print(f"   device  : {DEVICE}")
print(f"   torch   : {torch.__version__}")
print(f"   msgspec : {msgspec.__version__}")
Cell 3 — All msgspec Structs
from __future__ import annotations


class ChangeType(str, Enum):
    UNCHANGED = "UNCHANGED"
    MODIFIED  = "MODIFIED"
    ADDED     = "ADDED"
    DELETED   = "DELETED"


class Severity(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class RecommendedAction(str, Enum):
    IMMEDIATE_REVIEW  = "IMMEDIATE_REVIEW"
    UPDATE_TRAINING   = "UPDATE_TRAINING"
    UPDATE_PROCEDURES = "UPDATE_PROCEDURES"
    MONITOR           = "MONITOR"
    NO_ACTION         = "NO_ACTION"


# ── Stage 1 ───────────────────────────────────────────────────────

class Segment(msgspec.Struct):
    segment_id:  str
    doc_id:      str
    heading:     str
    depth:       int
    page_no:     int
    text:        str
    token_count: int


class SegmentList(msgspec.Struct):
    doc_id:   str
    source:   str
    segments: list[Segment]


# ── Stage 3 ───────────────────────────────────────────────────────

class AlignedPair(msgspec.Struct):
    change_type:           ChangeType
    similarity_score:      float
    heading:               str
    legacy_segment_id:     Optional[str] = None
    modernized_segment_id: Optional[str] = None
    legacy_text:           Optional[str] = None
    modernized_text:       Optional[str] = None
    page_legacy:           Optional[int] = None
    page_modernized:       Optional[int] = None
    context_before:        Optional[str] = None
    context_after:         Optional[str] = None


class AlignmentResult(msgspec.Struct):
    pairs: list[AlignedPair]


# ── Stage 5 ───────────────────────────────────────────────────────

class LLMAnalysis(msgspec.Struct):
    heading:                str
    what_changed:           str
    compliance_implication: str
    severity:               Severity
    recommended_action:     RecommendedAction
    confidence:             float
    legacy_segment_id:      Optional[str] = None
    modernized_segment_id:  Optional[str] = None
    page_legacy:            Optional[int] = None
    page_modernized:        Optional[int] = None


class LLMResult(msgspec.Struct):
    analyses: list[LLMAnalysis]


# ── Stage 6 ───────────────────────────────────────────────────────

class ChangeSummary(msgspec.Struct):
    unchanged: int
    modified:  int
    added:     int
    deleted:   int


class Report(msgspec.Struct):
    generated_at:   str
    legacy_doc:     str
    modernized_doc: str
    summary:        ChangeSummary
    pairs:          list[AlignedPair]
    analyses:       list[LLMAnalysis]


print("✅ All structs defined")
Cell 4 — fitz Extractor + Segmenter
from __future__ import annotations
from statistics import mode as stat_mode


# ── heading detection ─────────────────────────────────────────────

def _is_bold(flags: int) -> bool:
    return bool(flags & 16)


def _infer_depth(text: str) -> int:
    m = RE_NUMBERED_HEADING.match(text.strip())
    if m:
        return len(m.group(1).split("."))
    return 1


def _detect_heading(
    text: str,
    font_size: float,
    flags: int,
    body_font_size: float,
) -> bool:
    text = text.strip()
    if not text or len(text) > 200:
        return False
    # rule 1 — numbered pattern
    if RE_NUMBERED_HEADING.match(text):
        return True
    # rule 2 — significantly larger font
    if font_size > body_font_size * 1.15:
        return True
    # rule 3 — bold + short line
    if _is_bold(flags) and len(text) < 80:
        return True
    # rule 4 — all caps short line
    if text.isupper() and len(text) < 60:
        return True
    return False


def _body_font_size(page: fitz.Page) -> float:
    sizes = []
    for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                sizes.append(round(span["size"], 1))
    if not sizes:
        return 10.0
    try:
        return stat_mode(sizes)
    except Exception:
        return float(np.median(sizes))


def _normalize(text: str) -> str:
    text = RE_SOFT_BREAK.sub(" ", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


# ── page-level extractor ──────────────────────────────────────────

def _extract_page(page: fitz.Page, page_no: int) -> list[dict]:
    """
    Returns list of dicts:
      {text, font_size, flags, is_heading, depth, page_no, bbox_y}
    """
    body_size = _body_font_size(page)
    page_h    = page.rect.height
    margin    = page_h * 0.07   # skip top/bottom 7% — headers/footers

    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    items  = []

    for block in blocks:
        if block["type"] != 0:
            continue

        # skip header/footer region
        bbox_y = block["bbox"][1]
        if bbox_y < margin or bbox_y > (page_h - margin):
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                text = _normalize(span["text"])
                if not text:
                    continue
                size  = span["size"]
                flags = span["flags"]
                items.append({
                    "text":       text,
                    "font_size":  size,
                    "flags":      flags,
                    "is_heading": _detect_heading(text, size, flags, body_size),
                    "depth":      _infer_depth(text),
                    "page_no":    page_no,
                    "bbox_y":     bbox_y,
                })

    return items


# ── document extractor + segmenter ───────────────────────────────

def extract_and_segment(pdf_path: str, doc_id: str) -> SegmentList:
    """
    Opens PDF with fitz, extracts text blocks page by page,
    detects headings, groups paragraphs under headings,
    returns SegmentList.
    """
    t0  = time.time()
    doc = fitz.open(pdf_path)

    all_items: list[dict] = []
    for page_no, page in enumerate(doc, start=1):
        all_items.extend(_extract_page(page, page_no))
    doc.close()

    segments: list[Segment] = []
    seg_counter = 0

    current_heading = "Preamble"
    current_depth   = 1
    current_page    = 1
    buffer: list[str] = []

    def _flush(heading, depth, page_no):
        nonlocal seg_counter
        text = " ".join(buffer).strip()
        if not text:
            return
        token_count = len(TOKENIZER.encode(text))
        # split on 512 token ceiling, paragraph boundary
        if token_count <= 512:
            segments.append(Segment(
                segment_id  = f"{doc_id}::p{page_no}::s{seg_counter}",
                doc_id      = doc_id,
                heading     = heading,
                depth       = depth,
                page_no     = page_no,
                text        = text,
                token_count = token_count,
            ))
            seg_counter += 1
        else:
            # split into sub-segments at sentence boundary
            sentences = re.split(r"(?<=[.!?])\s+", text)
            chunk, chunk_tokens = [], 0
            for sent in sentences:
                st = len(TOKENIZER.encode(sent))
                if chunk_tokens + st > 512 and chunk:
                    seg_text = " ".join(chunk)
                    segments.append(Segment(
                        segment_id  = f"{doc_id}::p{page_no}::s{seg_counter}",
                        doc_id      = doc_id,
                        heading     = heading,
                        depth       = depth,
                        page_no     = page_no,
                        text        = seg_text,
                        token_count = chunk_tokens,
                    ))
                    seg_counter += 1
                    chunk, chunk_tokens = [sent], st
                else:
                    chunk.append(sent)
                    chunk_tokens += st
            if chunk:
                seg_text = " ".join(chunk)
                segments.append(Segment(
                    segment_id  = f"{doc_id}::p{page_no}::s{seg_counter}",
                    doc_id      = doc_id,
                    heading     = heading,
                    depth       = depth,
                    page_no     = page_no,
                    text        = seg_text,
                    token_count = chunk_tokens,
                ))
                seg_counter += 1
        buffer.clear()

    for item in all_items:
        if item["is_heading"]:
            _flush(current_heading, current_depth, current_page)
            current_heading = item["text"]
            current_depth   = item["depth"]
            current_page    = item["page_no"]
        else:
            buffer.append(item["text"])

    _flush(current_heading, current_depth, current_page)

    elapsed = time.time() - t0
    logger.info(
        "Extracted %s → %d segments in %.2fs",
        Path(pdf_path).name, len(segments), elapsed
    )
    return SegmentList(
        doc_id   = doc_id,
        source   = Path(pdf_path).name,
        segments = segments,
    )


print("✅ Extractor + Segmenter defined")
Cell 5 — Parallel PDF Processing
# ── point these at your actual PDFs ──────────────────────────────
LEGACY_PDF     = "data/legacy_policy.pdf"
MODERNIZED_PDF = "data/modernized_policy.pdf"
# ─────────────────────────────────────────────────────────────────

def _run_extraction(args):
    path, doc_id = args
    return extract_and_segment(path, doc_id)

t0 = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = {
        pool.submit(_run_extraction, (LEGACY_PDF,     "legacy")):     "legacy",
        pool.submit(_run_extraction, (MODERNIZED_PDF, "modernized")): "modernized",
    }
    results = {}
    for future in concurrent.futures.as_completed(futures):
        label = futures[future]
        results[label] = future.result()

legacy_segments     = results["legacy"]
modernized_segments = results["modernized"]

logger.info(
    "Both PDFs parsed in %.2fs — legacy: %d segs, modernized: %d segs",
    time.time() - t0,
    len(legacy_segments.segments),
    len(modernized_segments.segments),
)

print(f"✅ Legacy     : {len(legacy_segments.segments)} segments")
print(f"✅ Modernized : {len(modernized_segments.segments)} segments")
Cell 6 — Batch Embedder
MODEL_NAME = "all-MiniLM-L6-v2"

def _doc_hash(seg_list: SegmentList) -> str:
    combined = "".join(s.text for s in seg_list.segments)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def embed_segments(
    seg_list: SegmentList,
    model: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Embeds all segments. Returns float32 matrix (n, 384).
    Result cached in _EMBED_CACHE by doc hash.
    """
    key = f"{seg_list.doc_id}::{_doc_hash(seg_list)}"
    if key in _EMBED_CACHE:
        logger.info("Embedding cache hit: %s", key)
        return _EMBED_CACHE[key]

    texts  = [s.text for s in seg_list.segments]
    t0     = time.time()
    embs   = model.encode(
        texts,
        batch_size          = batch_size,
        normalize_embeddings= True,
        show_progress_bar   = False,
        device              = DEVICE,
        convert_to_numpy    = True,
    )
    embs = embs.astype(np.float32)
    _EMBED_CACHE[key] = embs

    logger.info(
        "Embedded %d segments for %s in %.2fs shape=%s",
        len(texts), seg_list.doc_id, time.time() - t0, embs.shape
    )
    return embs


print("✅ Embedder defined")
Cell 7 — Parallel Embedding
t0    = time.time()
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
logger.info("Model loaded: %s", MODEL_NAME)

# embed both in parallel using threads
# (model.encode releases GIL during torch ops — threads are fine here)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    fut_legacy = pool.submit(embed_segments, legacy_segments,     model)
    fut_modern = pool.submit(embed_segments, modernized_segments, model)
    legacy_embs  = fut_legacy.result()
    modern_embs  = fut_modern.result()

logger.info("Both embedded in %.2fs", time.time() - t0)

print(f"✅ Legacy embeddings     : {legacy_embs.shape}")
print(f"✅ Modernized embeddings : {modern_embs.shape}")
Cell 8 — Similarity Matrix
def _cosine_matrix_gpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity matrix via GPU matmul if available.
    Vectors must be pre-normalized (they are — normalize_embeddings=True).
    Returns float32 numpy matrix (n_A, n_B).
    """
    if DEVICE == "cuda":
        tA  = torch.from_numpy(A).to("cuda")
        tB  = torch.from_numpy(B).to("cuda")
        sim = torch.mm(tA, tB.T).cpu().numpy()
    else:
        sim = A @ B.T
    return sim.astype(np.float32)


def _tfidf_boost(
    legacy_segs:     SegmentList,
    modernized_segs: SegmentList,
    sim_matrix:      np.ndarray,
    ambiguous_lo:    float = 0.70,
    ambiguous_hi:    float = 0.85,
    weight:          float = 0.15,
) -> np.ndarray:
    """
    For pairs in the ambiguous similarity band, blend in TF-IDF
    lexical similarity to resolve semantic ties.
    Only computes TF-IDF for ambiguous pairs — not the full matrix.
    """
    ambiguous_mask = (sim_matrix >= ambiguous_lo) & (sim_matrix <= ambiguous_hi)
    if not ambiguous_mask.any():
        return sim_matrix

    l_texts = [s.text for s in legacy_segs.segments]
    m_texts = [s.text for s in modernized_segs.segments]
    all_texts = l_texts + m_texts

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8192, sublinear_tf=True)
    tfidf = vec.fit_transform(all_texts)

    l_tfidf = tfidf[:len(l_texts)]
    m_tfidf = tfidf[len(l_texts):]

    rows, cols = np.where(ambiguous_mask)
    boosted = sim_matrix.copy()
    for r, c in zip(rows, cols):
        lex = (l_tfidf[r] @ m_tfidf[c].T).toarray()[0, 0]
        boosted[r, c] = (1 - weight) * sim_matrix[r, c] + weight * lex

    return boosted


t0  = time.time()
sim = _cosine_matrix_gpu(legacy_embs, modern_embs)
sim = _tfidf_boost(legacy_segments, modernized_segments, sim)
logger.info("Similarity matrix %s built in %.2fs", sim.shape, time.time() - t0)

print(f"✅ Similarity matrix : {sim.shape}")
print(f"   max  : {sim.max():.4f}")
print(f"   mean : {sim.mean():.4f}")
print(f"   min  : {sim.min():.4f}")
Cell 9 — Hungarian Alignment + Change Classifier
# ── configurable thresholds ───────────────────────────────────────
THRESHOLD_UNCHANGED = 0.92
THRESHOLD_MODIFIED  = 0.75   # below this → MODIFIED substantive


def _get_context(segments: list[Segment], idx: int) -> tuple[Optional[str], Optional[str]]:
    before = segments[idx - 1].text if idx > 0 else None
    after  = segments[idx + 1].text if idx < len(segments) - 1 else None
    return before, after


def align_and_classify(
    legacy_segs:     SegmentList,
    modernized_segs: SegmentList,
    sim_matrix:      np.ndarray,
    threshold_unchanged: float = THRESHOLD_UNCHANGED,
    threshold_modified:  float = THRESHOLD_MODIFIED,
) -> AlignmentResult:

    t0   = time.time()
    n, m = sim_matrix.shape

    # cost matrix for Hungarian — minimise cost = maximise similarity
    cost   = (1.0 - sim_matrix).astype(np.float64)
    size   = max(n, m)
    padded = np.ones((size, size), dtype=np.float64)
    padded[:n, :m] = cost

    row_ind, col_ind = linear_sum_assignment(padded)

    pairs: list[AlignedPair] = []
    matched_leg = set()
    matched_mod = set()

    l_segs = legacy_segs.segments
    m_segs = modernized_segs.segments

    for r, c in zip(row_ind, col_ind):
        if r >= n or c >= m:
            continue
        score = float(sim_matrix[r, c])
        if score < threshold_modified:
            continue   # too low — will be caught as unmatched

        l_seg = l_segs[r]
        m_seg = m_segs[c]

        change_type = (
            ChangeType.UNCHANGED if score >= threshold_unchanged
            else ChangeType.MODIFIED
        )

        ctx_before, ctx_after = _get_context(l_segs, r)

        pairs.append(AlignedPair(
            change_type           = change_type,
            similarity_score      = round(score, 4),
            heading               = l_seg.heading,
            legacy_segment_id     = l_seg.segment_id,
            modernized_segment_id = m_seg.segment_id,
            legacy_text           = l_seg.text,
            modernized_text       = m_seg.text,
            page_legacy           = l_seg.page_no,
            page_modernized       = m_seg.page_no,
            context_before        = ctx_before,
            context_after         = ctx_after,
        ))
        matched_leg.add(r)
        matched_mod.add(c)

    # unmatched legacy → DELETED
    for r, l_seg in enumerate(l_segs):
        if r not in matched_leg:
            pairs.append(AlignedPair(
                change_type       = ChangeType.DELETED,
                similarity_score  = 0.0,
                heading           = l_seg.heading,
                legacy_segment_id = l_seg.segment_id,
                legacy_text       = l_seg.text,
                page_legacy       = l_seg.page_no,
            ))

    # unmatched modernized → ADDED
    for c, m_seg in enumerate(m_segs):
        if c not in matched_mod:
            pairs.append(AlignedPair(
                change_type           = ChangeType.ADDED,
                similarity_score      = 0.0,
                heading               = m_seg.heading,
                modernized_segment_id = m_seg.segment_id,
                modernized_text       = m_seg.text,
                page_modernized       = m_seg.page_no,
            ))

    logger.info(
        "Alignment done in %.2fs — unchanged:%d modified:%d added:%d deleted:%d",
        time.time() - t0,
        sum(1 for p in pairs if p.change_type == ChangeType.UNCHANGED),
        sum(1 for p in pairs if p.change_type == ChangeType.MODIFIED),
        sum(1 for p in pairs if p.change_type == ChangeType.ADDED),
        sum(1 for p in pairs if p.change_type == ChangeType.DELETED),
    )
    return AlignmentResult(pairs=pairs)


alignment_result = align_and_classify(legacy_segments, modernized_segments, sim)

counts = {ct.value: 0 for ct in ChangeType}
for p in alignment_result.pairs:
    counts[p.change_type.value] += 1

print(f"✅ UNCHANGED : {counts['UNCHANGED']}")
print(f"✅ MODIFIED  : {counts['MODIFIED']}")
print(f"✅ ADDED     : {counts['ADDED']}")
print(f"✅ DELETED   : {counts['DELETED']}")
Cell 10 — Async LLM Reasoner
# vLLM must be running in terminal before this cell:
# vllm serve ./models/Qwen2.5-7B-Instruct --port 8000 --dtype auto --max-model-len 8192

SYSTEM_PROMPT = """You are a compliance analyst reviewing changes between two versions of an internal corporate policy document.
Analyse the provided text pair and respond ONLY with a valid JSON object. No markdown, no explanation outside JSON.

Required schema:
{
  "what_changed": "<one sentence: specific change>",
  "compliance_implication": "<one sentence: stricter / more lenient / neutral and why>",
  "severity": "<HIGH | MEDIUM | LOW>",
  "recommended_action": "<IMMEDIATE_REVIEW | UPDATE_TRAINING | UPDATE_PROCEDURES | MONITOR | NO_ACTION>",
  "confidence": <float 0.0-1.0>
}

Severity rules:
  HIGH   = obligation weakened, control removed, accountability gap introduced
  MEDIUM = obligation tightened, new obligation added, process changed
  LOW    = neutral reword, clarification, no operational impact"""


def _build_prompt(pair: AlignedPair) -> str:
    parts = [f"Section: {pair.heading}"]
    parts.append(f"Change type: {pair.change_type.value}")
    parts.append(f"Similarity score: {pair.similarity_score}")

    if pair.context_before:
        parts.append(f"\n[Context before]\n{pair.context_before}")

    if pair.legacy_text:
        parts.append(f"\n[Legacy text — page {pair.page_legacy}]\n{pair.legacy_text}")
    else:
        parts.append("\n[Legacy text]\n— not present —")

    if pair.modernized_text:
        parts.append(f"\n[Modernized text — page {pair.page_modernized}]\n{pair.modernized_text}")
    else:
        parts.append("\n[Modernized text]\n— removed —")

    if pair.context_after:
        parts.append(f"\n[Context after]\n{pair.context_after}")

    parts.append("\nOutput only the JSON object.")
    return "\n".join(parts)


def _parse_llm_response(raw: str, pair: AlignedPair) -> Optional[LLMAnalysis]:
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data  = json.loads(clean)
        return LLMAnalysis(
            heading               = pair.heading,
            what_changed          = data["what_changed"],
            compliance_implication= data["compliance_implication"],
            severity              = Severity(data["severity"]),
            recommended_action    = RecommendedAction(data["recommended_action"]),
            confidence            = float(data["confidence"]),
            legacy_segment_id     = pair.legacy_segment_id,
            modernized_segment_id = pair.modernized_segment_id,
            page_legacy           = pair.page_legacy,
            page_modernized       = pair.page_modernized,
        )
    except Exception as e:
        logger.warning("LLM parse error: %s | raw: %s", e, raw[:200])
        return None


async def _call_llm(
    client:  AsyncOpenAI,
    pair:    AlignedPair,
    model:   str,
    sem:     asyncio.Semaphore,
    retries: int = 2,
) -> Optional[LLMAnalysis]:
    async with sem:
        for attempt in range(retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model       = model,
                    messages    = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": _build_prompt(pair)},
                    ],
                    max_tokens  = 512,
                    temperature = 0.1,
                )
                raw = resp.choices[0].message.content.strip()
                return _parse_llm_response(raw, pair)
            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.warning("LLM call failed: %s", e)
                    return None


async def run_llm_async(
    alignment_result: AlignmentResult,
    base_url: str = "http://localhost:8000/v1",
    model:    str = "./models/Qwen2.5-7B-Instruct",
    max_concurrent: int = 8,
) -> LLMResult:

    # only MODIFIED, ADDED, DELETED go to LLM
    flagged = [
        p for p in alignment_result.pairs
        if p.change_type != ChangeType.UNCHANGED
    ]
    logger.info("Sending %d pairs to LLM", len(flagged))

    client  = AsyncOpenAI(base_url=base_url, api_key="token-ignored")
    sem     = asyncio.Semaphore(max_concurrent)

    tasks   = [_call_llm(client, pair, model, sem) for pair in flagged]
    results = await asyncio.gather(*tasks)

    analyses = [r for r in results if r is not None]
    logger.info("LLM complete — %d / %d successful", len(analyses), len(flagged))
    return LLMResult(analyses=analyses)


# run
llm_result = asyncio.run(run_llm_async(
    alignment_result,
    base_url = "http://localhost:8000/v1",
    model    = "./models/Qwen2.5-7B-Instruct",
))

sev = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
for a in llm_result.analyses:
    sev[a.severity.value] += 1

print(f"✅ HIGH   : {sev['HIGH']}")
print(f"✅ MEDIUM : {sev['MEDIUM']}")
print(f"✅ LOW    : {sev['LOW']}")
Cell 11 — JSON Export
import difflib


def _word_diff_html(legacy: str, modernized: str) -> tuple[str, str]:
    lw = legacy.split()
    mw = modernized.split()
    matcher = difflib.SequenceMatcher(None, lw, mw, autojunk=False)
    l_html, m_html = [], []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            chunk = " ".join(lw[i1:i2])
            l_html.append(chunk)
            m_html.append(chunk)
        elif op == "replace":
            l_html.append(f'<del>{" ".join(lw[i1:i2])}</del>')
            m_html.append(f'<ins>{" ".join(mw[j1:j2])}</ins>')
        elif op == "delete":
            l_html.append(f'<del>{" ".join(lw[i1:i2])}</del>')
        elif op == "insert":
            m_html.append(f'<ins>{" ".join(mw[j1:j2])}</ins>')
    return " ".join(l_html), " ".join(m_html)


def build_report(
    alignment_result: AlignmentResult,
    llm_result:       LLMResult,
    legacy_doc:       str = LEGACY_PDF,
    modernized_doc:   str = MODERNIZED_PDF,
) -> Report:

    counts = {ct.value: 0 for ct in ChangeType}
    for p in alignment_result.pairs:
        counts[p.change_type.value] += 1

    return Report(
        generated_at   = datetime.now().isoformat(),
        legacy_doc     = Path(legacy_doc).name,
        modernized_doc = Path(modernized_doc).name,
        summary        = ChangeSummary(
            unchanged = counts["UNCHANGED"],
            modified  = counts["MODIFIED"],
            added     = counts["ADDED"],
            deleted   = counts["DELETED"],
        ),
        pairs     = alignment_result.pairs,
        analyses  = llm_result.analyses,
    )


def save_report_json(report: Report, out: str = "output/report.json"):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(
        msgspec.json.encode(_struct_to_dict(report))
    )
    logger.info("Saved → %s (%d bytes)", out, out.stat().st_size)


report = build_report(alignment_result, llm_result)
save_report_json(report)

print(f"✅ output/report.json saved")
print(f"   unchanged : {report.summary.unchanged}")
print(f"   modified  : {report.summary.modified}")
print(f"   added     : {report.summary.added}")
print(f"   deleted   : {report.summary.deleted}")
print(f"   analyses  : {len(report.analyses)}")
Cell 12 — Gradio UI
import gradio as gr
import difflib


# ── helpers ───────────────────────────────────────────────────────

def _severity_color(s: str) -> str:
    return {"HIGH": "#f04060", "MEDIUM": "#f0a030", "LOW": "#00c896"}.get(s, "#888")


def _change_color(c: str) -> str:
    return {
        "MODIFIED":  "#f0a030",
        "ADDED":     "#2196f3",
        "DELETED":   "#f04060",
        "UNCHANGED": "#00c896",
    }.get(c, "#888")


def _html_badge(text: str, color: str) -> str:
    return (
        f'<span style="background:rgba(0,0,0,.3);border:1px solid {color};'
        f'color:{color};padding:2px 8px;border-radius:3px;'
        f'font-size:11px;font-family:monospace">{text}</span>'
    )


def _diff_panes(legacy: Optional[str], modernized: Optional[str]) -> str:
    if not legacy and not modernized:
        return ""

    if not legacy:
        l_html = '<span style="color:#888;font-style:italic">— not present —</span>'
        m_html = f'<span style="background:rgba(33,150,243,.15);color:#5fc8ff;padding:2px 4px">{modernized}</span>'
    elif not modernized:
        l_html = f'<span style="background:rgba(240,64,96,.15);color:#ff8090;text-decoration:line-through;padding:2px 4px">{legacy}</span>'
        m_html = '<span style="color:#888;font-style:italic">— removed —</span>'
    else:
        lw = legacy.split()
        mw = modernized.split()
        matcher = difflib.SequenceMatcher(None, lw, mw, autojunk=False)
        lh, mh = [], []
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                chunk = " ".join(lw[i1:i2])
                lh.append(chunk); mh.append(chunk)
            elif op == "replace":
                lh.append(f'<span style="background:rgba(240,64,96,.2);color:#ff8090;text-decoration:line-through;border-radius:3px;padding:1px 3px">{" ".join(lw[i1:i2])}</span>')
                mh.append(f'<span style="background:rgba(0,200,150,.2);color:#5fffcc;border-radius:3px;padding:1px 3px">{" ".join(mw[j1:j2])}</span>')
            elif op == "delete":
                lh.append(f'<span style="background:rgba(240,64,96,.2);color:#ff8090;text-decoration:line-through;border-radius:3px;padding:1px 3px">{" ".join(lw[i1:i2])}</span>')
            elif op == "insert":
                mh.append(f'<span style="background:rgba(0,200,150,.2);color:#5fffcc;border-radius:3px;padding:1px 3px">{" ".join(mw[j1:j2])}</span>')
        l_html = " ".join(lh)
        m_html = " ".join(mh)

    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0;border:1px solid #1e2535;border-radius:6px;overflow:hidden;margin:8px 0">
      <div style="padding:12px 14px;background:#0f1520;border-right:1px solid #1e2535">
        <div style="font-size:10px;color:#6b7694;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">Legacy</div>
        <div style="font-family:monospace;font-size:12px;line-height:1.7;white-space:pre-wrap;word-break:break-word">{l_html}</div>
      </div>
      <div style="padding:12px 14px;background:#0f1520">
        <div style="font-size:10px;color:#6b7694;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">Modernized</div>
        <div style="font-family:monospace;font-size:12px;line-height:1.7;white-space:pre-wrap;word-break:break-word">{m_html}</div>
      </div>
    </div>"""


def _build_summary_html(report: Report) -> str:
    s = report.summary
    sev = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for a in report.analyses:
        sev[a.severity.value] += 1

    def stat(n, label, color):
        return f"""<div style="background:#111620;border:1px solid #1e2535;border-radius:6px;padding:14px 18px;min-width:100px">
          <div style="font-size:26px;font-weight:700;color:{color};font-family:monospace">{n}</div>
          <div style="font-size:10px;color:#6b7694;margin-top:4px;text-transform:uppercase;letter-spacing:.08em">{label}</div>
        </div>"""

    cards = "".join([
        stat(s.unchanged, "Unchanged",  "#00c896"),
        stat(s.modified,  "Modified",   "#f0a030"),
        stat(s.added,     "Added",      "#2196f3"),
        stat(s.deleted,   "Deleted",    "#f04060"),
        stat(sev["HIGH"],   "High",     "#f04060"),
        stat(sev["MEDIUM"], "Medium",   "#f0a030"),
        stat(sev["LOW"],    "Low",      "#00c896"),
    ])
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap;padding:8px 0">{cards}</div>'


def _build_diff_html(report: Report) -> str:
    # index LLM analyses by segment id for quick lookup
    analysis_map: dict[str, LLMAnalysis] = {}
    for a in report.analyses:
        if a.legacy_segment_id:
            analysis_map[a.legacy_segment_id] = a
        if a.modernized_segment_id:
            analysis_map[a.modernized_segment_id] = a

    html_parts = []
    for pair in report.pairs:
        if pair.change_type == ChangeType.UNCHANGED:
            continue

        ct_color  = _change_color(pair.change_type.value)
        ct_badge  = _html_badge(pair.change_type.value, ct_color)
        score_str = f"cos {pair.similarity_score:.3f}" if pair.similarity_score else ""

        analysis_html = ""
        a = analysis_map.get(pair.legacy_segment_id or "") or \
            analysis_map.get(pair.modernized_segment_id or "")
        if a:
            sev_color = _severity_color(a.severity.value)
            sev_badge = _html_badge(a.severity.value, sev_color)
            act_badge = _html_badge(a.recommended_action.value, "#8a93b0")
            analysis_html = f"""
            <div style="margin-top:10px;padding:10px 12px;background:rgba(255,255,255,.03);
                        border-left:3px solid {sev_color};border-radius:0 4px 4px 0">
              <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px">
                {sev_badge} {act_badge}
                <span style="font-size:10px;color:#6b7694">conf {a.confidence:.0%}</span>
              </div>
              <div style="font-size:12px;margin-bottom:4px"><b>What changed:</b> {a.what_changed}</div>
              <div style="font-size:12px;color:#a0a8c0">{a.compliance_implication}</div>
            </div>"""

        pages = []
        if pair.page_legacy:    pages.append(f"legacy p.{pair.page_legacy}")
        if pair.page_modernized: pages.append(f"modern p.{pair.page_modernized}")
        page_str = " · ".join(pages)

        html_parts.append(f"""
        <div style="background:#111620;border:1px solid #1e2535;border-radius:8px;
                    margin-bottom:14px;overflow:hidden">
          <div style="padding:10px 14px;border-bottom:1px solid #1e2535;
                      background:#161c2a;display:flex;align-items:center;gap:10px">
            {ct_badge}
            <span style="font-size:13px;font-weight:600;flex:1">{pair.heading}</span>
            <span style="font-size:10px;color:#6b7694;font-family:monospace">{score_str}</span>
            <span style="font-size:10px;color:#6b7694">{page_str}</span>
          </div>
          <div style="padding:12px 14px">
            {_diff_panes(pair.legacy_text, pair.modernized_text)}
            {analysis_html}
          </div>
        </div>""")

    return "\n".join(html_parts) if html_parts else "<p style='color:#6b7694'>No changes found.</p>"


# ── full pipeline runner for Gradio ──────────────────────────────

def run_pipeline(legacy_file, modernized_file, progress=gr.Progress()):
    try:
        legacy_path     = legacy_file.name
        modernized_path = modernized_file.name

        progress(0.1, desc="Extracting and segmenting PDFs...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fl = pool.submit(extract_and_segment, legacy_path,     "legacy")
            fm = pool.submit(extract_and_segment, modernized_path, "modernized")
            l_segs = fl.result()
            m_segs = fm.result()

        progress(0.3, desc="Generating embeddings...")
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fl = pool.submit(embed_segments, l_segs, _model)
            fm = pool.submit(embed_segments, m_segs, _model)
            l_embs = fl.result()
            m_embs = fm.result()

        progress(0.5, desc="Computing similarity matrix...")
        _sim = _cosine_matrix_gpu(l_embs, m_embs)
        _sim = _tfidf_boost(l_segs, m_segs, _sim)

        progress(0.65, desc="Aligning sections and classifying changes...")
        _alignment = align_and_classify(l_segs, m_segs, _sim)

        progress(0.75, desc="Running LLM analysis...")
        _llm = asyncio.run(run_llm_async(
            _alignment,
            base_url = "http://localhost:8000/v1",
            model    = "./models/Qwen2.5-7B-Instruct",
        ))

        progress(0.90, desc="Building report...")
        _report = build_report(_alignment, _llm, legacy_path, modernized_path)
        save_report_json(_report, "output/report.json")

        progress(1.0, desc="Done")

        summary_html = _build_summary_html(_report)
        diff_html    = _build_diff_html(_report)

        return summary_html, diff_html, "output/report.json"

    except Exception as e:
        logger.exception("Pipeline error: %s", e)
        return f"<p style='color:red'>Error: {e}</p>", "", None


# ── Gradio layout ─────────────────────────────────────────────────

CSS = """
body, .gradio-container { background:#0a0d12 !important; color:#d8dde8 !important; }
.gr-button-primary { background:#2196f3 !important; border:none !important; }
"""

with gr.Blocks(css=CSS, title="Policy Diff") as demo:
    gr.Markdown("## Policy Document Comparison Tool")
    gr.Markdown("Upload legacy and modernized policy PDFs to generate a compliance diff report.")

    with gr.Row():
        legacy_input     = gr.File(label="Legacy PDF",     file_types=[".pdf"])
        modernized_input = gr.File(label="Modernized PDF", file_types=[".pdf"])

    run_btn = gr.Button("Run Comparison", variant="primary")

    summary_out = gr.HTML(label="Summary")
    diff_out    = gr.HTML(label="Clause Diff View")
    json_out    = gr.File(label="Download report.json")

    run_btn.click(
        fn      = run_pipeline,
        inputs  = [legacy_input, modernized_input],
        outputs = [summary_out, diff_out, json_out],
    )

demo.launch(share=False, server_port=7860)
Cell 13 — Manifest Check
outputs = [
    "output/report.json",
]

print("── Output Manifest ──────────────────────────────────────────")
all_ok = True
for path in outputs:
    p = Path(path)
    if p.exists():
        print(f"  ✅  {path:<40} {p.stat().st_size:>10,} bytes")
    else:
        print(f"  ❌  {path:<40} MISSING")
        all_ok = False

if all_ok:
    print("\n🟢 Pipeline complete")
    print("   Gradio UI : http://localhost:7860")
    print("   Report    : output/report.json")
else:
    print("\n🔴 Some outputs missing — check cells above")