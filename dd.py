Cell 1 — Install Dependencies
!pip install docling pydantic orjson rapidfuzz scikit-learn scipy \
             tiktoken sentence-transformers faiss-cpu openai jinja2 torch
Cell 2 — Imports
from __future__ import annotations

import re
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from enum import Enum

import orjson
import faiss
import torch
import tiktoken
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from jinja2 import Template

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DocItemLabel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

Path("data").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

print("✅ Imports done")
print(f"   torch  : {torch.__version__}")
print(f"   device : {'cuda (ROCm)' if torch.cuda.is_available() else 'cpu'}")
Cell 3 — All Pydantic Models
# ── Stage 1: Ingestion ────────────────────────────────────────────

class SectionNode(BaseModel):
    section_id: str
    heading_text: str
    depth: int
    parent_id: Optional[str] = None
    raw_text: str = ""
    children: list[str] = Field(default_factory=list)


class DocumentTree(BaseModel):
    doc_id: str
    source_path: str
    sections: dict[str, SectionNode] = Field(default_factory=dict)
    root_ids: list[str] = Field(default_factory=list)


# ── Stage 2: Tree Index ───────────────────────────────────────────

class CanonicalNode(BaseModel):
    canonical_key: str
    section_id: str
    heading_text: str
    depth: int
    parent_key: Optional[str] = None
    children_keys: list[str] = Field(default_factory=list)
    raw_text: str = ""
    char_count: int = 0


class TreeIndex(BaseModel):
    doc_id: str
    nodes: dict[str, CanonicalNode] = Field(default_factory=dict)
    sid_to_key: dict[str, str] = Field(default_factory=dict)
    root_keys: list[str] = Field(default_factory=list)


# ── Stage 3: Alignment ────────────────────────────────────────────

class SectionState(str, Enum):
    MATCHED = "MATCHED"
    ADDED   = "ADDED"
    DELETED = "DELETED"


class AlignedPair(BaseModel):
    state: SectionState
    legacy_key: Optional[str] = None
    modernized_key: Optional[str] = None
    legacy_heading: Optional[str] = None
    modernized_heading: Optional[str] = None
    match_score: float = 0.0


class AlignmentMap(BaseModel):
    pairs: list[AlignedPair] = Field(default_factory=list)

    @property
    def matched(self) -> list[AlignedPair]:
        return [p for p in self.pairs if p.state == SectionState.MATCHED]

    @property
    def added(self) -> list[AlignedPair]:
        return [p for p in self.pairs if p.state == SectionState.ADDED]

    @property
    def deleted(self) -> list[AlignedPair]:
        return [p for p in self.pairs if p.state == SectionState.DELETED]


# ── Stage 4: Chunking ─────────────────────────────────────────────

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    canonical_key: str
    heading_text: str
    chunk_index: int
    text: str
    token_count: int


class SectionChunks(BaseModel):
    canonical_key: str
    doc_id: str
    chunks: list[Chunk] = Field(default_factory=list)


class ChunkStore(BaseModel):
    doc_id: str
    sections: dict[str, SectionChunks] = Field(default_factory=dict)


# ── Stage 4: Triage ───────────────────────────────────────────────

class TriageLabel(str, Enum):
    NO_CHANGE   = "NO_CHANGE"
    PARAPHRASE  = "PARAPHRASE"
    SUBSTANTIVE = "SUBSTANTIVE"
    MISSING     = "MISSING"


class ChunkPair(BaseModel):
    legacy_chunk_id: str
    modernized_chunk_id: Optional[str] = None
    legacy_text: str
    modernized_text: Optional[str] = None
    canonical_key: str
    cosine_score: Optional[float] = None
    triage: TriageLabel
    heading_text: str


class TriageResult(BaseModel):
    pairs: list[ChunkPair] = Field(default_factory=list)

    @property
    def for_llm(self) -> list[ChunkPair]:
        return [p for p in self.pairs
                if p.triage in (TriageLabel.PARAPHRASE, TriageLabel.SUBSTANTIVE)]

    @property
    def direct_to_report(self) -> list[ChunkPair]:
        return [p for p in self.pairs
                if p.triage in (TriageLabel.NO_CHANGE, TriageLabel.MISSING)]


# ── Stage 5: LLM ─────────────────────────────────────────────────

class Severity(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class ChangeType(str, Enum):
    OBLIGATION_ADDED   = "OBLIGATION_ADDED"
    OBLIGATION_REMOVED = "OBLIGATION_REMOVED"
    THRESHOLD_CHANGED  = "THRESHOLD_CHANGED"
    SCOPE_BROADENED    = "SCOPE_BROADENED"
    SCOPE_NARROWED     = "SCOPE_NARROWED"
    NEUTRAL_REWORD     = "NEUTRAL_REWORD"
    PROCESS_CHANGED    = "PROCESS_CHANGED"
    OTHER              = "OTHER"


class LLMAnalysis(BaseModel):
    legacy_chunk_id: str
    modernized_chunk_id: Optional[str] = None
    canonical_key: str
    heading_text: str
    triage: str
    cosine_score: Optional[float] = None
    what_changed: str
    compliance_implication: str
    change_type: ChangeType
    severity: Severity
    confidence: float


class LLMResult(BaseModel):
    analyses: list[LLMAnalysis] = Field(default_factory=list)

    @property
    def high(self) -> list[LLMAnalysis]:
        return [a for a in self.analyses if a.severity == Severity.HIGH]

    @property
    def medium(self) -> list[LLMAnalysis]:
        return [a for a in self.analyses if a.severity == Severity.MEDIUM]

    @property
    def low(self) -> list[LLMAnalysis]:
        return [a for a in self.analyses if a.severity == Severity.LOW]


print("✅ All models defined")
Cell 4 — Docling Ingestor (Text Only)
_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)[\.\s]")


def _infer_depth(text: str) -> int:
    m = _HEADING_RE.match(text.strip())
    return len(m.group(1).split(".")) if m else 1


def _make_sid(n: int) -> str:
    return f"s_{n:04d}"


class DoclingIngestor:
    """
    Parses a PDF using Docling.
    Extracts headings and paragraph text only.
    Tables and images are skipped for now.
    """

    def __init__(self, ocr: bool = False):
        opts = PdfPipelineOptions()
        opts.do_ocr             = ocr
        opts.do_table_structure = False   # off — text only
        self._converter = DocumentConverter()

    def ingest(self, path: str | Path, doc_id: str) -> DocumentTree:
        path   = Path(path)
        result = self._converter.convert(str(path))
        tree   = DocumentTree(doc_id=doc_id, source_path=str(path))
        self._build(result.document, tree)
        logger.info("Ingested %s → %d sections", path.name, len(tree.sections))
        return tree

    def save(self, tree: DocumentTree, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(orjson.dumps(tree.model_dump(), option=orjson.OPT_INDENT_2))
        logger.info("Saved → %s", out)

    def _build(self, doc, tree: DocumentTree):
        counter = 0
        stack: list[tuple[str, int]] = []
        current: Optional[str] = None

        HEADINGS = {
            DocItemLabel.SECTION_HEADER,
            DocItemLabel.TITLE,
            DocItemLabel.SUBTITLE,
        }

        for item, _ in doc.iterate_items():
            label = item.label

            if label in HEADINGS:
                text  = item.text.strip()
                if not text:
                    continue
                depth = _infer_depth(text)

                while stack and stack[-1][1] >= depth:
                    stack.pop()

                parent_id = stack[-1][0] if stack else None
                sid       = _make_sid(counter)
                counter  += 1

                node = SectionNode(
                    section_id=sid,
                    heading_text=text,
                    depth=depth,
                    parent_id=parent_id,
                )
                tree.sections[sid] = node

                if parent_id is None:
                    tree.root_ids.append(sid)
                else:
                    tree.sections[parent_id].children.append(sid)

                stack.append((sid, depth))
                current = sid

            elif label == DocItemLabel.TEXT:
                if not current:
                    continue
                t = item.text.strip()
                if not t:
                    continue
                node = tree.sections[current]
                node.raw_text = (node.raw_text + " " + t).strip()

            # Tables, images, lists — skipped for now


print("✅ DoclingIngestor defined")
Cell 5 — Run Ingestion
# ── Point these at your actual PDFs ──────────────────────────────
LEGACY_PDF_PATH     = "data/legacy_policy.pdf"
MODERNIZED_PDF_PATH = "data/modernized_policy.pdf"
# ─────────────────────────────────────────────────────────────────

ingestor = DoclingIngestor(ocr=False)

legacy_tree     = ingestor.ingest(LEGACY_PDF_PATH,     doc_id="legacy")
modernized_tree = ingestor.ingest(MODERNIZED_PDF_PATH, doc_id="modernized")

ingestor.save(legacy_tree,     "output/legacy_tree.json")
ingestor.save(modernized_tree, "output/modernized_tree.json")

print(f"✅ Legacy     : {len(legacy_tree.sections)} sections")
print(f"✅ Modernized : {len(modernized_tree.sections)} sections")
Cell 6 — Tree Index Builder
class TreeIndexBuilder:

    def build(self, tree: DocumentTree) -> TreeIndex:
        index    = TreeIndex(doc_id=tree.doc_id)
        counters: dict[str, int] = {}

        def _walk(sid: str, parent_key: Optional[str]) -> str:
            node      = tree.sections[sid]
            bucket    = parent_key or "__root__"
            counters[bucket] = counters.get(bucket, 0) + 1
            n         = counters[bucket]
            canon_key = f"S{n}" if parent_key is None else f"{parent_key}.{n}"

            cnode = CanonicalNode(
                canonical_key=canon_key,
                section_id=sid,
                heading_text=node.heading_text,
                depth=node.depth,
                parent_key=parent_key,
                raw_text=node.raw_text,
                char_count=len(node.raw_text),
            )
            index.nodes[canon_key] = cnode
            index.sid_to_key[sid]  = canon_key

            if parent_key is None:
                index.root_keys.append(canon_key)

            for child_sid in node.children:
                child_key = _walk(child_sid, canon_key)
                cnode.children_keys.append(child_key)

            return canon_key

        for rid in tree.root_ids:
            _walk(rid, None)

        return index

    def save(self, index: TreeIndex, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(orjson.dumps(index.model_dump(), option=orjson.OPT_INDENT_2))
        logger.info("Saved → %s", out)


builder          = TreeIndexBuilder()
legacy_index     = builder.build(legacy_tree)
modernized_index = builder.build(modernized_tree)

builder.save(legacy_index,     "output/legacy_index.json")
builder.save(modernized_index, "output/modernized_index.json")

print(f"✅ Legacy index     : {len(legacy_index.nodes)} nodes")
print(f"✅ Modernized index : {len(modernized_index.nodes)} nodes")
Cell 7 — Section Aligner
class SectionAligner:

    def __init__(self, match_threshold: float = 0.60):
        self.threshold = match_threshold

    def align(self, legacy: TreeIndex, modernized: TreeIndex) -> AlignmentMap:
        amap     = AlignmentMap()
        leg_keys = list(legacy.nodes.keys())
        mod_keys = list(modernized.nodes.keys())
        n, m     = len(leg_keys), len(mod_keys)

        cost = np.ones((n, m), dtype=np.float32)
        for i, lk in enumerate(leg_keys):
            lh = legacy.nodes[lk].heading_text
            for j, mk in enumerate(mod_keys):
                mh  = modernized.nodes[mk].heading_text
                sim = (0.6 * fuzz.token_sort_ratio(lh, mh) / 100.0 +
                       0.4 * fuzz.ratio(lh, mh) / 100.0)
                cost[i, j] = 1.0 - sim

        size   = max(n, m)
        padded = np.ones((size, size), dtype=np.float32)
        padded[:n, :m] = cost

        row_ind, col_ind = linear_sum_assignment(padded)

        matched_leg: set[str] = set()
        matched_mod: set[str] = set()

        for r, c in zip(row_ind, col_ind):
            if r >= n or c >= m:
                continue
            score = 1.0 - padded[r, c]
            if score >= self.threshold:
                lk = leg_keys[r]
                mk = mod_keys[c]
                amap.pairs.append(AlignedPair(
                    state=SectionState.MATCHED,
                    legacy_key=lk,
                    modernized_key=mk,
                    legacy_heading=legacy.nodes[lk].heading_text,
                    modernized_heading=modernized.nodes[mk].heading_text,
                    match_score=round(score, 4),
                ))
                matched_leg.add(lk)
                matched_mod.add(mk)

        for lk in leg_keys:
            if lk not in matched_leg:
                amap.pairs.append(AlignedPair(
                    state=SectionState.DELETED,
                    legacy_key=lk,
                    legacy_heading=legacy.nodes[lk].heading_text,
                ))

        for mk in mod_keys:
            if mk not in matched_mod:
                amap.pairs.append(AlignedPair(
                    state=SectionState.ADDED,
                    modernized_key=mk,
                    modernized_heading=modernized.nodes[mk].heading_text,
                ))

        return amap

    def save(self, amap: AlignmentMap, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(orjson.dumps(amap.model_dump(), option=orjson.OPT_INDENT_2))
        logger.info("Saved → %s", out)


aligner       = SectionAligner(match_threshold=0.60)
alignment_map = aligner.align(legacy_index, modernized_index)
aligner.save(alignment_map, "output/alignment_map.json")

print(f"✅ MATCHED : {len(alignment_map.matched)}")
print(f"✅ ADDED   : {len(alignment_map.added)}")
print(f"✅ DELETED : {len(alignment_map.deleted)}")
Cell 8 — Paragraph Chunker
class ParagraphChunker:

    def __init__(self, min_tokens: int = 150, max_tokens: int = 300):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")

    def build_store(self, index: TreeIndex, tree: DocumentTree) -> ChunkStore:
        store = ChunkStore(doc_id=index.doc_id)
        for key, cnode in index.nodes.items():
            raw = tree.sections[cnode.section_id].raw_text.strip()
            if not raw:
                continue
            chunks = self._split(raw)
            sc = SectionChunks(canonical_key=key, doc_id=index.doc_id)
            for i, text in enumerate(chunks):
                sc.chunks.append(Chunk(
                    chunk_id=f"{index.doc_id}::{key}::{i}",
                    doc_id=index.doc_id,
                    canonical_key=key,
                    heading_text=cnode.heading_text,
                    chunk_index=i,
                    text=text,
                    token_count=len(self._enc.encode(text)),
                ))
            store.sections[key] = sc
        return store

    def save(self, store: ChunkStore, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(orjson.dumps(store.model_dump(), option=orjson.OPT_INDENT_2))
        logger.info("Saved → %s", out)

    def _split(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if not paragraphs:
            return [text]

        chunks, buffer, buf_tokens = [], [], 0

        for para in paragraphs:
            pt = len(self._enc.encode(para))
            if pt > self.max_tokens:
                if buffer:
                    chunks.append(" ".join(buffer))
                    buffer, buf_tokens = [], 0
                chunks.extend(self._sentence_split(para))
                continue
            if buf_tokens + pt > self.max_tokens and buf_tokens >= self.min_tokens:
                chunks.append(" ".join(buffer))
                buffer, buf_tokens = [para], pt
            else:
                buffer.append(para)
                buf_tokens += pt

        if buffer:
            chunks.append(" ".join(buffer))

        return [c for c in chunks if c.strip()]

    def _sentence_split(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, buffer, buf_tokens = [], [], 0
        for sent in sentences:
            st = len(self._enc.encode(sent))
            if buf_tokens + st > self.max_tokens and buffer:
                chunks.append(" ".join(buffer))
                buffer, buf_tokens = [sent], st
            else:
                buffer.append(sent)
                buf_tokens += st
        if buffer:
            chunks.append(" ".join(buffer))
        return chunks


chunker = ParagraphChunker(min_tokens=150, max_tokens=300)

legacy_chunks     = chunker.build_store(legacy_index,     legacy_tree)
modernized_chunks = chunker.build_store(modernized_index, modernized_tree)

chunker.save(legacy_chunks,     "output/legacy_chunks.json")
chunker.save(modernized_chunks, "output/modernized_chunks.json")

total_l = sum(len(sc.chunks) for sc in legacy_chunks.sections.values())
total_m = sum(len(sc.chunks) for sc in modernized_chunks.sections.values())
print(f"✅ Legacy chunks     : {total_l}")
print(f"✅ Modernized chunks : {total_m}")
Cell 9 — Embedding + FAISS Triage
def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingTriager:

    THRESHOLD_NO_CHANGE  = 0.92
    THRESHOLD_PARAPHRASE = 0.75

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        device = _get_device()
        logger.info("Loading embedding model on %s", device)
        self._model = SentenceTransformer(model_name, device=device)

    def triage(
        self,
        alignment_map: AlignmentMap,
        legacy_chunks: ChunkStore,
        modernized_chunks: ChunkStore,
    ) -> TriageResult:

        result = TriageResult()

        # matched pairs → embed and score
        for pair in alignment_map.matched:
            lk = pair.legacy_key
            mk = pair.modernized_key

            l_sec = legacy_chunks.sections.get(lk)
            m_sec = modernized_chunks.sections.get(mk)

            if not l_sec or not m_sec:
                continue
            if not l_sec.chunks or not m_sec.chunks:
                continue

            l_embs = self._embed([c.text for c in l_sec.chunks])
            m_embs = self._embed([c.text for c in m_sec.chunks])

            index = faiss.IndexFlatIP(m_embs.shape[1])
            index.add(m_embs)
            scores, indices = index.search(l_embs, k=1)

            for i, l_chunk in enumerate(l_sec.chunks):
                nn_idx  = int(indices[i, 0])
                cosine  = float(scores[i, 0])
                m_chunk = m_sec.chunks[nn_idx]

                result.pairs.append(ChunkPair(
                    legacy_chunk_id=l_chunk.chunk_id,
                    modernized_chunk_id=m_chunk.chunk_id,
                    legacy_text=l_chunk.text,
                    modernized_text=m_chunk.text,
                    canonical_key=lk,
                    cosine_score=round(cosine, 4),
                    triage=self._label(cosine),
                    heading_text=pair.legacy_heading or "",
                ))

        # added sections → all chunks MISSING
        for pair in alignment_map.added:
            m_sec = modernized_chunks.sections.get(pair.modernized_key)
            if not m_sec:
                continue
            for m_chunk in m_sec.chunks:
                result.pairs.append(ChunkPair(
                    legacy_chunk_id="",
                    modernized_chunk_id=m_chunk.chunk_id,
                    legacy_text="",
                    modernized_text=m_chunk.text,
                    canonical_key=pair.modernized_key,
                    triage=TriageLabel.MISSING,
                    heading_text=pair.modernized_heading or "",
                ))

        # deleted sections → all chunks MISSING
        for pair in alignment_map.deleted:
            l_sec = legacy_chunks.sections.get(pair.legacy_key)
            if not l_sec:
                continue
            for l_chunk in l_sec.chunks:
                result.pairs.append(ChunkPair(
                    legacy_chunk_id=l_chunk.chunk_id,
                    modernized_chunk_id="",
                    legacy_text=l_chunk.text,
                    modernized_text="",
                    canonical_key=pair.legacy_key,
                    triage=TriageLabel.MISSING,
                    heading_text=pair.legacy_heading or "",
                ))

        return result

    def save(self, result: TriageResult, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(orjson.dumps(result.model_dump(), option=orjson.OPT_INDENT_2))
        logger.info("Saved → %s", out)

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        ).astype(np.float32)

    def _label(self, score: float) -> TriageLabel:
        if score >= self.THRESHOLD_NO_CHANGE:
            return TriageLabel.NO_CHANGE
        if score >= self.THRESHOLD_PARAPHRASE:
            return TriageLabel.PARAPHRASE
        return TriageLabel.SUBSTANTIVE


triager      = EmbeddingTriager(model_name="BAAI/bge-large-en-v1.5")
triage_result = triager.triage(alignment_map, legacy_chunks, modernized_chunks)
triager.save(triage_result, "output/triage_result.json")

counts = {t.value: 0 for t in TriageLabel}
for p in triage_result.pairs:
    counts[p.triage.value] += 1

print(f"✅ NO_CHANGE   : {counts['NO_CHANGE']}")
print(f"✅ PARAPHRASE  : {counts['PARAPHRASE']}")
print(f"✅ SUBSTANTIVE : {counts['SUBSTANTIVE']}")
print(f"✅ MISSING     : {counts['MISSING']}")
print(f"→  To LLM     : {len(triage_result.for_llm)}")
Cell 10 — LLM Reasoner
# vLLM must be running before this cell:
#   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype auto

SYSTEM_PROMPT = """You are a compliance analyst comparing legacy and modernized policy documents.
You will receive two text chunks from the same policy section.
Respond ONLY with a valid JSON object. No markdown, no explanation outside the JSON.

Required schema:
{
  "what_changed": "<one sentence describing the specific change>",
  "compliance_implication": "<one sentence: stricter, more lenient, or neutral>",
  "change_type": "<OBLIGATION_ADDED | OBLIGATION_REMOVED | THRESHOLD_CHANGED | SCOPE_BROADENED | SCOPE_NARROWED | NEUTRAL_REWORD | PROCESS_CHANGED | OTHER>",
  "severity": "<HIGH | MEDIUM | LOW>",
  "confidence": <float 0.0-1.0>
}"""


def _user_prompt(pair: ChunkPair) -> str:
    return f"""Section: {pair.heading_text}
Triage: {pair.triage.value}  |  Cosine similarity: {pair.cosine_score}

LEGACY TEXT:
{pair.legacy_text}

MODERNIZED TEXT:
{pair.modernized_text}

Output only the JSON object."""


class LLMReasoner:

    def __init__(
        self,
        base_url: str  = "http://localhost:8000/v1",
        model: str     = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.1,
        batch_size: int = 8,
        retry_limit: int = 2,
    ):
        self._client      = OpenAI(base_url=base_url, api_key="token-ignored")
        self._model       = model
        self._max_tokens  = max_tokens
        self._temperature = temperature
        self._batch_size  = batch_size
        self._retry_limit = retry_limit

    def reason(self, triage_result: TriageResult) -> LLMResult:
        flagged = triage_result.for_llm
        logger.info("Sending %d chunk pairs to LLM", len(flagged))
        result = LLMResult()

        for i in range(0, len(flagged), self._batch_size):
            batch = flagged[i : i + self._batch_size]
            for pair in batch:
                analysis = self._call(pair)
                if analysis:
                    result.analyses.append(analysis)
            logger.info("Processed %d / %d", min(i + self._batch_size, len(flagged)), len(flagged))

        return result

    def save(self, result: LLMResult, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(orjson.dumps(result.model_dump(), option=orjson.OPT_INDENT_2))
        logger.info("Saved → %s", out)

    def _call(self, pair: ChunkPair) -> Optional[LLMAnalysis]:
        for attempt in range(self._retry_limit + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": _user_prompt(pair)},
                    ],
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                raw = resp.choices[0].message.content.strip()
                return self._parse(raw, pair)
            except Exception as e:
                if attempt < self._retry_limit:
                    time.sleep(2 ** attempt)
                else:
                    logger.warning("LLM call failed for %s: %s", pair.legacy_chunk_id, e)
                    return None

    def _parse(self, raw: str, pair: ChunkPair) -> Optional[LLMAnalysis]:
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data  = json.loads(clean)
            return LLMAnalysis(
                legacy_chunk_id=pair.legacy_chunk_id,
                modernized_chunk_id=pair.modernized_chunk_id,
                canonical_key=pair.canonical_key,
                heading_text=pair.heading_text,
                triage=pair.triage.value,
                cosine_score=pair.cosine_score,
                what_changed=data["what_changed"],
                compliance_implication=data["compliance_implication"],
                change_type=ChangeType(data["change_type"]),
                severity=Severity(data["severity"]),
                confidence=float(data["confidence"]),
            )
        except Exception as e:
            logger.warning("Parse error for %s: %s | raw: %s", pair.legacy_chunk_id, e, raw[:200])
            return None


reasoner   = LLMReasoner(base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-7B-Instruct")
llm_result = reasoner.reason(triage_result)
reasoner.save(llm_result, "output/llm_result.json")

print(f"✅ HIGH   : {len(llm_result.high)}")
print(f"✅ MEDIUM : {len(llm_result.medium)}")
print(f"✅ LOW    : {len(llm_result.low)}")
Cell 11 — Report Generator
REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Policy Diff Report</title>
<style>
  :root {
    --bg:#0a0d12; --surface:#111620; --border:#1e2535;
    --text:#d8dde8; --muted:#6b7694;
    --high:#f04060; --med:#f0a030; --low:#00c896;
    --added:#2196f3; --deleted:#f04060;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:Inter,sans-serif; padding:40px 32px; }
  h1   { font-size:22px; font-weight:700; margin-bottom:6px; }
  h2   { font-size:13px; font-weight:600; margin:32px 0 12px;
         letter-spacing:.06em; text-transform:uppercase; color:#fff; }
  p.sub { font-size:12px; color:var(--muted); margin-bottom:24px; }

  .summary { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:32px; }
  .stat { background:var(--surface); border:1px solid var(--border);
          border-radius:6px; padding:14px 20px; min-width:110px; }
  .stat .n { font-size:26px; font-weight:700; line-height:1; }
  .stat .l { font-size:10px; color:var(--muted); margin-top:4px;
             text-transform:uppercase; letter-spacing:.08em; }
  .stat.high   .n { color:var(--high); }
  .stat.med    .n { color:var(--med);  }
  .stat.low_s  .n { color:var(--low);  }
  .stat.added  .n { color:var(--added);}
  .stat.del    .n { color:var(--high); }

  table { width:100%; border-collapse:collapse; font-size:11.5px; margin-bottom:28px; }
  th { background:var(--surface); padding:8px 12px; text-align:left;
       font-size:10px; text-transform:uppercase; letter-spacing:.08em;
       color:var(--muted); border-bottom:1px solid var(--border); }
  td { padding:9px 12px; border-bottom:1px solid var(--border); vertical-align:top; line-height:1.6; }
  tr:last-child td { border-bottom:none; }

  .badge { display:inline-block; padding:2px 8px; border-radius:3px;
           font-size:10px; font-weight:600; letter-spacing:.05em; }
  .HIGH     { background:rgba(240,64,96,.15);  color:var(--high); }
  .MEDIUM   { background:rgba(240,160,48,.15); color:var(--med);  }
  .LOW      { background:rgba(0,200,150,.15);  color:var(--low);  }
  .MATCHED  { background:rgba(107,118,148,.12);color:var(--muted);}
  .ADDED    { background:rgba(33,150,243,.15); color:var(--added);}
  .DELETED  { background:rgba(240,64,96,.15);  color:var(--high); }

  .chunk { background:var(--surface); border:1px solid var(--border); border-radius:4px;
           padding:8px 10px; font-size:11px; font-family:monospace;
           white-space:pre-wrap; word-break:break-word;
           max-height:110px; overflow:auto; line-height:1.55; }
  .score { font-family:monospace; font-size:11px; color:var(--muted); }
  footer { margin-top:48px; font-size:10px; color:var(--muted); text-align:center; }
</style>
</head>
<body>

<h1>Policy Document Comparison Report</h1>
<p class="sub">Generated {{ generated_at }} &nbsp;·&nbsp; Legacy: {{ legacy_doc }} &nbsp;·&nbsp; Modernized: {{ modern_doc }}</p>

<h2>Executive Summary</h2>
<div class="summary">
  <div class="stat high"> <div class="n">{{ n_high }}</div>    <div class="l">High severity</div></div>
  <div class="stat med">  <div class="n">{{ n_med }}</div>     <div class="l">Medium severity</div></div>
  <div class="stat low_s"><div class="n">{{ n_low }}</div>     <div class="l">Low severity</div></div>
  <div class="stat added"><div class="n">{{ n_added }}</div>   <div class="l">Sections added</div></div>
  <div class="stat del">  <div class="n">{{ n_deleted }}</div> <div class="l">Sections deleted</div></div>
  <div class="stat">      <div class="n">{{ n_matched }}</div> <div class="l">Sections matched</div></div>
</div>

<h2>Section Alignment</h2>
<table>
  <tr><th>State</th><th>Legacy Key</th><th>Legacy Heading</th>
      <th>Modernized Key</th><th>Modernized Heading</th><th>Score</th></tr>
  {% for p in alignment_pairs %}
  <tr>
    <td><span class="badge {{ p.state }}">{{ p.state }}</span></td>
    <td>{{ p.legacy_key or '—' }}</td>
    <td>{{ p.legacy_heading or '—' }}</td>
    <td>{{ p.modernized_key or '—' }}</td>
    <td>{{ p.modernized_heading or '—' }}</td>
    <td class="score">{{ '%.2f'|format(p.match_score) if p.match_score else '—' }}</td>
  </tr>
  {% endfor %}
</table>

<h2>LLM Analysis — All Flagged Changes</h2>
<table>
  <tr><th>Severity</th><th>Section</th><th>Change Type</th>
      <th>What Changed</th><th>Compliance Impact</th><th>Conf.</th></tr>
  {% for a in analyses_sorted %}
  <tr>
    <td><span class="badge {{ a.severity }}">{{ a.severity }}</span></td>
    <td><b>{{ a.canonical_key }}</b><br>
        <span style="color:var(--muted);font-size:10px">{{ a.heading_text }}</span></td>
    <td style="font-size:10.5px">{{ a.change_type }}</td>
    <td>{{ a.what_changed }}</td>
    <td>{{ a.compliance_implication }}</td>
    <td class="score">{{ '%.0f'|format(a.confidence*100) }}%</td>
  </tr>
  {% endfor %}
</table>

<h2>Raw Clause Pairs — High Severity</h2>
<table>
  <tr><th>Section</th><th>Cosine</th><th>Legacy Text</th><th>Modernized Text</th></tr>
  {% for p in high_pairs %}
  <tr>
    <td><b>{{ p.canonical_key }}</b><br>
        <span style="color:var(--muted);font-size:10px">{{ p.heading_text }}</span></td>
    <td class="score">{{ p.cosine_score }}</td>
    <td><div class="chunk">{{ p.legacy_text }}</div></td>
    <td><div class="chunk">{{ p.modernized_text or '— deleted —' }}</div></td>
  </tr>
  {% endfor %}
</table>

<footer>Policy Diff Pipeline &nbsp;·&nbsp; ROCm + vLLM &nbsp;·&nbsp; Python 3.11</footer>
</body>
</html>"""


class ReportGenerator:

    def generate(
        self,
        alignment_map: AlignmentMap,
        triage_result: TriageResult,
        llm_result: LLMResult,
        legacy_doc: str = "legacy_policy.pdf",
        modern_doc: str = "modernized_policy.pdf",
    ) -> str:

        sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        analyses_sorted = sorted(
            llm_result.analyses,
            key=lambda a: (sev_order.get(a.severity.value, 9), -a.confidence)
        )

        high_ids   = {a.legacy_chunk_id for a in llm_result.high}
        high_pairs = [p for p in triage_result.pairs if p.legacy_chunk_id in high_ids]

        html = Template(REPORT_TEMPLATE).render(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            legacy_doc=legacy_doc,
            modern_doc=modern_doc,
            n_high=len(llm_result.high),
            n_med=len(llm_result.medium),
            n_low=len(llm_result.low),
            n_added=len(alignment_map.added),
            n_deleted=len(alignment_map.deleted),
            n_matched=len(alignment_map.matched),
            alignment_pairs=sorted(
                alignment_map.pairs,
                key=lambda p: (p.state.value, p.legacy_key or p.modernized_key or "")
            ),
            analyses_sorted=analyses_sorted,
            high_pairs=high_pairs,
        )
        return html

    def save(self, html: str, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        logger.info("Saved → %s", out)

    def save_json(self, llm_result: LLMResult, triage_result: TriageResult, out: str | Path):
        out = Path(out)
        counts = {t.value: 0 for t in TriageLabel}
        for p in triage_result.pairs:
            counts[p.triage.value] += 1
        artifact = {
            "generated_at": datetime.now().isoformat(),
            "triage_summary": counts,
            "analyses": [a.model_dump() for a in llm_result.analyses],
            "missing_chunks": [
                p.model_dump() for p in triage_result.pairs
                if p.triage == TriageLabel.MISSING
            ],
        }
        out.write_bytes(orjson.dumps(artifact, option=orjson.OPT_INDENT_2))
        logger.info("Saved → %s", out)


reporter = ReportGenerator()
html     = reporter.generate(alignment_map, triage_result, llm_result)

reporter.save(html, "output/report.html")
reporter.save_json(llm_result, triage_result, "output/diff_artifact.json")

print("✅ output/report.html")
print("✅ output/diff_artifact.json")
Cell 12 — Pipeline Manifest Check
outputs = [
    "output/legacy_tree.json",
    "output/modernized_tree.json",
    "output/legacy_index.json",
    "output/modernized_index.json",
    "output/alignment_map.json",
    "output/legacy_chunks.json",
    "output/modernized_chunks.json",
    "output/triage_result.json",
    "output/llm_result.json",
    "output/report.html",
    "output/diff_artifact.json",
]

print("── Output Manifest ──────────────────────────────────────────")
all_ok = True
for path in outputs:
    p = Path(path)
    if p.exists():
        print(f"  ✅  {path:<45} {p.stat().st_size:>9,} bytes")
    else:
        print(f"  ❌  {path:<45} MISSING")
        all_ok = False

if all_ok:
    print("\n🟢 Pipeline complete — open output/report.html")
else:
    print("\n🔴 Some outputs missing — check cells above")