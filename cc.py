Cell 1 — Install Dependencies
# Cell 1: Install all dependencies
!pip install docling pydantic orjson reportlab sentence-transformers faiss-cpu rapidfuzz scikit-learn scipy tiktoken
Cell 2 — models.py
# Cell 2: Data models — run this first, everything imports from here
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class TableCell(BaseModel):
    row: int
    col: int
    text: str


class Table(BaseModel):
    caption: Optional[str] = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    raw_cells: list[TableCell] = Field(default_factory=list)


class ListBlock(BaseModel):
    ordered: bool = False
    items: list[str] = Field(default_factory=list)


class SectionNode(BaseModel):
    section_id: str
    heading_text: str
    depth: int
    parent_id: Optional[str] = None
    raw_text: str = ""
    tables: list[Table] = Field(default_factory=list)
    lists: list[ListBlock] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)


class DocumentTree(BaseModel):
    doc_id: str
    source_path: str
    sections: dict[str, SectionNode] = Field(default_factory=dict)
    root_ids: list[str] = Field(default_factory=list)


print("✅ Models loaded")
Cell 3 — Synthetic Fixture Generator
# Cell 3: Generate synthetic legacy + modernized policy PDFs
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

Path("data").mkdir(exist_ok=True)

def _styles():
    base = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=base["Heading1"], fontSize=14, spaceAfter=6)
    h2 = ParagraphStyle("H2", parent=base["Heading2"], fontSize=12, spaceAfter=4)
    body = ParagraphStyle("Body", parent=base["Normal"], fontSize=10, spaceAfter=6, leading=14)
    return h1, h2, body


LEGACY_SECTIONS = [
    ("1. Purpose and Scope", 1),
    ("1.1 Purpose", 2),
    ("This policy establishes the framework for information security management within the organisation. "
     "All employees, contractors, and third-party users must comply with the requirements set out herein.", 0),
    ("1.2 Scope", 2),
    ("This policy applies to all information assets owned, leased, or operated by the organisation, "
     "including hardware, software, data, and communication systems.", 0),

    ("2. Access Control", 1),
    ("2.1 User Access Management", 2),
    ("Access to information systems shall be granted on a need-to-know basis. "
     "All access requests must be approved by the relevant line manager and the IT Security team. "
     "Privileged access must be reviewed quarterly.", 0),
    ("2.2 Password Policy", 2),
    ("Passwords must be a minimum of 8 characters and changed every 90 days. "
     "Passwords must not be shared under any circumstances. "
     "Multi-factor authentication is required for remote access.", 0),
    ("2.3 Third-Party Access", 2),
    ("Third-party access shall be granted for a maximum period of 6 months. "
     "All third-party connections must be logged and monitored.", 0),

    ("3. Incident Response", 1),
    ("3.1 Reporting", 2),
    ("Security incidents must be reported to the IT helpdesk within 24 hours of discovery. "
     "The incident response team shall assess all reported incidents within 48 hours.", 0),
    ("3.2 Escalation Thresholds", 2),
    ("Incidents affecting more than 50 users shall be escalated to senior management. "
     "Data breaches must be reported to the regulatory authority within 72 hours.", 0),

    ("4. Data Classification", 1),
    ("4.1 Classification Levels", 2),
    ("Data shall be classified into four levels: Public, Internal, Confidential, and Restricted. "
     "All data must be labelled at the point of creation.", 0),
    ("4.2 Handling Requirements", 2),
    ("Restricted data must be encrypted at rest and in transit using AES-128 encryption. "
     "Confidential data must not be stored on personal devices.", 0),

    # Section exists in legacy only (deleted in modernized)
    ("5. Legacy System Controls", 1),
    ("5.1 Risk Assessment", 2),
    ("Legacy systems operating beyond their vendor support lifecycle must undergo annual risk assessment. "
     "Compensating controls must be documented and approved by the CISO.", 0),
]

# Table data for section 3.2
LEGACY_THRESHOLD_TABLE = [
    ["Incident Type",        "Affected Users", "Escalation Target",  "SLA"],
    ["Low severity",         "< 10",           "IT Manager",         "5 days"],
    ["Medium severity",      "10–50",          "IT Security Lead",   "48 hours"],
    ["High severity",        "> 50",           "Senior Management",  "24 hours"],
    ["Regulatory data breach","Any",           "Regulatory Authority","72 hours"],
]

MODERNIZED_SECTIONS = [
    ("1. Purpose and Scope", 1),
    ("1.1 Purpose", 2),
    ("This policy establishes the framework for information security governance and risk management "
     "across the organisation. Compliance is mandatory for all personnel and associated entities.", 0),
    ("1.2 Scope", 2),
    ("This policy applies to all information assets owned, leased, processed, or transmitted by the "
     "organisation, including cloud-hosted infrastructure, SaaS platforms, and supply chain integrations.", 0),

    ("2. Access Control", 1),
    ("2.1 User Access Management", 2),
    ("Access to information systems shall be granted on a least-privilege basis, reviewed monthly. "
     "All access requests require approval from the line manager, IT Security, and the Data Owner. "
     "Privileged access must be reviewed monthly and auto-revoked after 30 days of inactivity.", 0),
    ("2.2 Password and Authentication Policy", 2),   # heading changed
    ("Passwords must be a minimum of 14 characters. Mandatory rotation is removed in favour of "
     "breach-detection monitoring. Passwords must not be shared under any circumstances. "
     "Multi-factor authentication is required for all system access, not only remote sessions.", 0),
    ("2.3 Third-Party and Supply Chain Access", 2),  # heading changed
    ("Third-party access shall be granted for a maximum period of 3 months with mandatory re-approval. "
     "All third-party connections must be logged, monitored, and subject to annual security assessment. "
     "Supply chain partners must provide SOC 2 Type II evidence annually.", 0),

    ("3. Incident Response", 1),
    ("3.1 Reporting", 2),
    ("Security incidents must be reported via the security portal within 4 hours of discovery. "
     "The incident response team shall triage all reported incidents within 2 hours.", 0),
    ("3.2 Escalation Thresholds", 2),
    ("Incidents affecting more than 10 users shall be escalated to senior management. "
     "Data breaches must be reported to the regulatory authority within 48 hours.", 0),

    ("4. Data Classification", 1),
    ("4.1 Classification Levels", 2),
    ("Data shall be classified into five levels: Public, Internal, Confidential, Restricted, and Top Secret. "
     "All data must be labelled at the point of creation using automated tooling.", 0),
    ("4.2 Handling Requirements", 2),
    ("Restricted data must be encrypted at rest and in transit using AES-256 encryption. "
     "Confidential data must not be stored on personal devices or unmanaged cloud storage.", 0),

    # NEW section — not in legacy
    ("5. Cloud Security Controls", 1),
    ("5.1 Cloud Configuration Management", 2),
    ("All cloud-hosted workloads must adhere to the organisation's secure baseline configuration. "
     "Infrastructure-as-code templates must pass automated security scanning prior to deployment. "
     "Cloud security posture must be reviewed quarterly by the Cloud Security team.", 0),
]

MODERNIZED_THRESHOLD_TABLE = [
    ["Incident Type",        "Affected Users", "Escalation Target",  "SLA"],
    ["Low severity",         "< 5",            "IT Manager",         "3 days"],
    ["Medium severity",      "5–10",           "IT Security Lead",   "24 hours"],
    ["High severity",        "> 10",           "Senior Management",  "4 hours"],
    ["Regulatory data breach","Any",           "Regulatory Authority","48 hours"],
]


def _build_pdf(path: str, sections: list, threshold_table: list, title: str):
    h1, h2, body_style = _styles()
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    story = []

    from reportlab.lib.styles import getSampleStyleSheet
    title_style = ParagraphStyle("Title", parent=getSampleStyleSheet()["Title"],
                                 fontSize=16, spaceAfter=16)
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.3*cm))

    insert_table_after = "3.2 Escalation Thresholds"

    for text, level in sections:
        if level == 1:
            story.append(Paragraph(text, h1))
        elif level == 2:
            story.append(Paragraph(text, h2))
        else:
            story.append(Paragraph(text, body_style))
            # inject threshold table after escalation section body
            if insert_table_after in [s for s, l in sections if l == 2]:
                # check previous heading
                pass

        if level == 2 and text == insert_table_after:
            # will be added after next body paragraph — flag
            pass

        # inject table right after the body paragraph following section 3.2
        if level == 0 and "shall be escalated" in text:
            story.append(Spacer(1, 0.2*cm))
            tbl = Table(threshold_table, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#1e2535")),
                ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
                ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
                ("FONTSIZE",     (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor("#f5f5f5"), colors.white]),
                ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
                ("LEFTPADDING",  (0,0), (-1,-1), 6),
                ("RIGHTPADDING", (0,0), (-1,-1), 6),
                ("TOPPADDING",   (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",(0,0), (-1,-1), 4),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.3*cm))

    doc.build(story)


_build_pdf("data/legacy_policy.pdf",    LEGACY_SECTIONS,     LEGACY_THRESHOLD_TABLE,     "Information Security Policy v2.1 (Legacy)")
_build_pdf("data/modernized_policy.pdf", MODERNIZED_SECTIONS, MODERNIZED_THRESHOLD_TABLE, "Information Security Policy v3.0 (Modernized)")

print("✅ PDFs generated:")
print("   data/legacy_policy.pdf")
print("   data/modernized_policy.pdf")

Cell 4 — Ingestor
# Cell 4: Docling ingestor — parses PDF into DocumentTree
import re
import logging
from pathlib import Path
from typing import Optional
import orjson
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DocItemLabel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)[\.\s]")

def _infer_depth(text: str) -> int:
    m = _HEADING_RE.match(text.strip())
    if m:
        return len(m.group(1).split("."))
    return 1

def _make_id(n: int) -> str:
    return f"s_{n:04d}"

def _extract_table(item) -> Table:
    tbl = Table()
    try:
        df = item.export_to_dataframe()
        if df is not None and not df.empty:
            tbl.headers = [str(c) for c in df.columns.tolist()]
            tbl.rows = [[str(v) for v in row] for row in df.values.tolist()]
            return tbl
    except Exception:
        pass
    if hasattr(item, "data") and item.data:
        for cell_row in item.data.grid:
            for c in cell_row:
                tbl.raw_cells.append(TableCell(
                    row=c.start_row_offset_idx,
                    col=c.start_col_offset_idx,
                    text=c.text.strip()
                ))
    return tbl


class DoclingIngestor:

    def __init__(self, ocr: bool = False):
        opts = PdfPipelineOptions()
        opts.do_ocr = ocr
        opts.do_table_structure = True
        self._converter = DocumentConverter()

    def ingest(self, path: str | Path, doc_id: str) -> DocumentTree:
        path = Path(path)
        logger.info("Ingesting %s", path.name)
        result = self._converter.convert(str(path))
        doc = result.document
        tree = DocumentTree(doc_id=doc_id, source_path=str(path))
        self._build(doc, tree)
        logger.info("  → %d sections parsed", len(tree.sections))
        return tree

    def save(self, tree: DocumentTree, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(orjson.dumps(tree.model_dump(), option=orjson.OPT_INDENT_2))
        logger.info("Saved → %s", out)

    def _build(self, doc, tree: DocumentTree):
        counter = 0
        stack: list[tuple[str, int]] = []   # (section_id, depth)
        current: Optional[str] = None

        HEADING_LABELS = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE, DocItemLabel.SUBTITLE}

        for item, _ in doc.iterate_items():
            label = item.label

            if label in HEADING_LABELS:
                text = item.text.strip()
                depth = _infer_depth(text)

                while stack and stack[-1][1] >= depth:
                    stack.pop()

                parent_id = stack[-1][0] if stack else None
                sid = _make_id(counter); counter += 1

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

            elif label == DocItemLabel.TEXT and current:
                node = tree.sections[current]
                t = item.text.strip()
                node.raw_text = (node.raw_text + " " + t).strip()

            elif label == DocItemLabel.TABLE and current:
                tree.sections[current].tables.append(_extract_table(item))

            elif label == DocItemLabel.LIST_ITEM and current:
                node = tree.sections[current]
                if node.lists and not node.lists[-1].ordered:
                    node.lists[-1].items.append(item.text.strip())
                else:
                    node.lists.append(ListBlock(ordered=False, items=[item.text.strip()]))

print("✅ DoclingIngestor defined")
Cell 5 — Run Ingestion
# Cell 5: Run ingestion on both PDFs
Path("output").mkdir(exist_ok=True)

ingestor = DoclingIngestor(ocr=False)

legacy_tree   = ingestor.ingest("data/legacy_policy.pdf",    doc_id="legacy")
modernized_tree = ingestor.ingest("data/modernized_policy.pdf", doc_id="modernized")

ingestor.save(legacy_tree,    "output/legacy_tree.json")
ingestor.save(modernized_tree, "output/modernized_tree.json")

print(f"\n📄 Legacy sections     : {len(legacy_tree.sections)}")
print(f"📄 Modernized sections : {len(modernized_tree.sections)}")
Cell 6 — Inspection Helpers
# Cell 6: Inspect tree — use these to debug ingestion quality

def print_tree(tree: DocumentTree, show_text: bool = False, show_tables: bool = True):
    """Pretty-print section hierarchy."""
    indent = "  "

    def _walk(sid: str):
        node = tree.sections[sid]
        pad = indent * (node.depth - 1)
        text_preview = (node.raw_text[:80] + "…") if node.raw_text else "—"
        print(f"{pad}[{node.section_id}] {'#'*node.depth} {node.heading_text}")
        if show_text:
            print(f"{pad}    ↳ {text_preview}")
        if show_tables and node.tables:
            for i, tbl in enumerate(node.tables):
                print(f"{pad}    📊 Table {i+1}: {len(tbl.rows)} rows × {len(tbl.headers)} cols  headers={tbl.headers}")
        if node.lists:
            for lb in node.lists:
                print(f"{pad}    • List: {len(lb.items)} items")
        for child_id in node.children:
            _walk(child_id)

    print(f"\n{'='*60}")
    print(f"  DocumentTree: {tree.doc_id}  ({len(tree.sections)} sections)")
    print(f"{'='*60}")
    for rid in tree.root_ids:
        _walk(rid)
    print()


def section_stats(tree: DocumentTree):
    """Summary stats for a parsed tree."""
    total_text  = sum(len(n.raw_text) for n in tree.sections.values())
    total_tables = sum(len(n.tables)  for n in tree.sections.values())
    total_lists  = sum(len(n.lists)   for n in tree.sections.values())
    depths = [n.depth for n in tree.sections.values()]
    print(f"doc_id        : {tree.doc_id}")
    print(f"sections      : {len(tree.sections)}")
    print(f"max depth     : {max(depths)}")
    print(f"total chars   : {total_text}")
    print(f"tables found  : {total_tables}")
    print(f"list blocks   : {total_lists}")
    print()


# ── run ───────────────────────────────────────────────────────────
print("── LEGACY ──────────────────────────────────────────────────")
section_stats(legacy_tree)
print_tree(legacy_tree, show_text=True, show_tables=True)

print("── MODERNIZED ──────────────────────────────────────────────")
section_stats(modernized_tree)
print_tree(modernized_tree, show_text=True, show_tables=True)
Cell 7 — Sanity Assertions
# Cell 7: Assertions — catches ingestion regressions immediately

def assert_tree_quality(tree: DocumentTree, expected_min_sections: int = 5):
    errors = []

    if len(tree.sections) < expected_min_sections:
        errors.append(f"Too few sections: {len(tree.sections)} < {expected_min_sections}")

    for sid, node in tree.sections.items():
        # parent consistency
        if node.parent_id and node.parent_id not in tree.sections:
            errors.append(f"[{sid}] parent_id={node.parent_id!r} not found")
        # children consistency
        for cid in node.children:
            if cid not in tree.sections:
                errors.append(f"[{sid}] child {cid!r} not found")
            elif tree.sections[cid].parent_id != sid:
                errors.append(f"[{sid}] child {cid!r} has wrong parent_id")
        # depth > 0
        if node.depth < 1:
            errors.append(f"[{sid}] depth={node.depth} invalid")
        # tables have headers OR raw_cells
        for i, tbl in enumerate(node.tables):
            if not tbl.headers and not tbl.raw_cells:
                errors.append(f"[{sid}] table[{i}] has no headers or raw_cells")

    # root_ids exist
    for rid in tree.root_ids:
        if rid not in tree.sections:
            errors.append(f"root_id {rid!r} not in sections")

    if errors:
        print(f"❌ {tree.doc_id}: {len(errors)} issue(s):")
        for e in errors: print(f"   • {e}")
    else:
        print(f"✅ {tree.doc_id}: all assertions passed ({len(tree.sections)} sections)")

    return len(errors) == 0


ok_legacy      = assert_tree_quality(legacy_tree,      expected_min_sections=8)
ok_modernized  = assert_tree_quality(modernized_tree,  expected_min_sections=8)

if ok_legacy and ok_modernized:
    print("\n🟢 Stage 1 complete — both trees ready for Stage 2 (Tree Index)")




Cell 8 — Tree Index Models
# Cell 8: Tree index models — canonical key scheme on top of DocumentTree
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class CanonicalNode(BaseModel):
    canonical_key: str          # e.g. "S1.2.3"
    section_id: str             # original s_NNNN from DocumentTree
    heading_text: str
    depth: int
    parent_key: Optional[str] = None
    children_keys: list[str] = Field(default_factory=list)
    raw_text: str = ""
    table_count: int = 0
    list_count: int = 0
    char_count: int = 0


class TreeIndex(BaseModel):
    doc_id: str
    nodes: dict[str, CanonicalNode] = Field(default_factory=dict)  # key → node
    sid_to_key: dict[str, str] = Field(default_factory=dict)       # s_NNNN → S1.2.3
    root_keys: list[str] = Field(default_factory=list)


print("✅ TreeIndex models loaded")
Cell 9 — Tree Index Builder
# Cell 9: Builds canonical S1.2.3 key scheme from a DocumentTree

import orjson
from pathlib import Path


class TreeIndexBuilder:
    """
    Walks a DocumentTree and assigns every section a canonical key
    of the form S{d1}.{d2}.{d3}… based on its position in the hierarchy.

    S1          → top-level section 1
    S1.2        → 2nd child of S1
    S1.2.3      → 3rd child of S1.2

    The counter resets per parent, so the key encodes structural
    position, not document-order index.
    """

    def build(self, tree: DocumentTree) -> TreeIndex:
        index = TreeIndex(doc_id=tree.doc_id)
        # sibling counters: parent_key → next child number
        counters: dict[str, int] = {}

        def _walk(sid: str, parent_key: Optional[str]):
            node = tree.sections[sid]

            # assign sibling counter
            bucket = parent_key or "__root__"
            counters[bucket] = counters.get(bucket, 0) + 1
            n = counters[bucket]

            if parent_key is None:
                canon_key = f"S{n}"
            else:
                canon_key = f"{parent_key}.{n}"

            cnode = CanonicalNode(
                canonical_key = canon_key,
                section_id    = sid,
                heading_text  = node.heading_text,
                depth         = node.depth,
                parent_key    = parent_key,
                raw_text      = node.raw_text,
                table_count   = len(node.tables),
                list_count    = len(node.lists),
                char_count    = len(node.raw_text),
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
        print(f"💾 Saved → {out}")


print("✅ TreeIndexBuilder defined")
Cell 10 — Build Both Indexes
# Cell 10: Build canonical indexes for legacy and modernized trees

builder = TreeIndexBuilder()

legacy_index     = builder.build(legacy_tree)
modernized_index = builder.build(modernized_tree)

builder.save(legacy_index,     "output/legacy_index.json")
builder.save(modernized_index, "output/modernized_index.json")

print(f"\n📐 Legacy index     : {len(legacy_index.nodes)} canonical nodes")
print(f"📐 Modernized index : {len(modernized_index.nodes)} canonical nodes")
Cell 11 — Inspection Helpers
# Cell 11: Inspect both indexes — compare structure side by side

def print_index(index: TreeIndex, show_text: bool = False):
    indent = "  "

    def _walk(key: str):
        node = index.nodes[key]
        pad  = indent * (node.depth - 1)
        meta = []
        if node.table_count: meta.append(f"📊{node.table_count}")
        if node.list_count:  meta.append(f"•{node.list_count}")
        if node.char_count:  meta.append(f"{node.char_count}ch")
        meta_str = "  " + " ".join(meta) if meta else ""
        print(f"{pad}{node.canonical_key:<12} {node.heading_text}{meta_str}")
        if show_text and node.raw_text:
            preview = node.raw_text[:90].replace("\n", " ") + "…"
            print(f"{pad}             ↳ {preview}")
        for ck in node.children_keys:
            _walk(ck)

    print(f"\n{'='*62}")
    print(f"  Index: {index.doc_id}  ({len(index.nodes)} nodes)")
    print(f"{'='*62}")
    for rk in index.root_keys:
        _walk(rk)
    print()


def compare_index_structure(idx_a: TreeIndex, idx_b: TreeIndex):
    """
    Side-by-side key comparison.
    Highlights keys present in one index but not the other.
    """
    keys_a = set(idx_a.nodes.keys())
    keys_b = set(idx_b.nodes.keys())
    shared = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    print(f"\n{'='*62}")
    print(f"  Structure Comparison: {idx_a.doc_id} vs {idx_b.doc_id}")
    print(f"{'='*62}")
    print(f"  Shared keys   : {len(shared)}")
    print(f"  Only in {idx_a.doc_id:<12}: {sorted(only_a) or '—'}")
    print(f"  Only in {idx_b.doc_id:<12}: {sorted(only_b) or '—'}")

    print(f"\n  {'Key':<12} {'Legacy heading':<35} {'Modernized heading':<35}")
    print(f"  {'-'*12} {'-'*35} {'-'*35}")
    all_keys = sorted(keys_a | keys_b)
    for k in all_keys:
        a_txt = idx_a.nodes[k].heading_text[:33] if k in idx_a.nodes else "— MISSING —"
        b_txt = idx_b.nodes[k].heading_text[:33] if k in idx_b.nodes else "— MISSING —"
        marker = "  " if k in shared else "⚠ "
        print(f"{marker}  {k:<12} {a_txt:<35} {b_txt:<35}")
    print()


# ── run ──────────────────────────────────────────────────────────
print_index(legacy_index,     show_text=True)
print_index(modernized_index, show_text=True)
compare_index_structure(legacy_index, modernized_index)
Cell 12 — Sanity Assertions
# Cell 12: Assertions for tree index correctness

def assert_index_quality(index: TreeIndex, tree: DocumentTree):
    errors = []

    # every section_id in tree maps to a canonical key
    for sid in tree.sections:
        if sid not in index.sid_to_key:
            errors.append(f"section {sid!r} missing from sid_to_key")

    # every node's parent_key exists
    for key, node in index.nodes.items():
        if node.parent_key and node.parent_key not in index.nodes:
            errors.append(f"[{key}] parent_key={node.parent_key!r} not in index")

    # every children_key exists and points back correctly
    for key, node in index.nodes.items():
        for ck in node.children_keys:
            if ck not in index.nodes:
                errors.append(f"[{key}] child {ck!r} not in index")
            elif index.nodes[ck].parent_key != key:
                errors.append(f"[{key}] child {ck!r} has wrong parent_key={index.nodes[ck].parent_key!r}")

    # root_keys have no parent
    for rk in index.root_keys:
        if index.nodes[rk].parent_key is not None:
            errors.append(f"root key {rk!r} has parent_key set")

    # canonical key format matches depth
    import re
    for key, node in index.nodes.items():
        parts = key.lstrip("S").split(".")
        if len(parts) != node.depth:
            errors.append(f"[{key}] key depth={len(parts)} != node.depth={node.depth}")

    # no duplicate section_ids
    seen_sids = {}
    for key, node in index.nodes.items():
        if node.section_id in seen_sids:
            errors.append(f"Duplicate section_id {node.section_id!r} in {key} and {seen_sids[node.section_id]}")
        seen_sids[node.section_id] = key

    if errors:
        print(f"❌ {index.doc_id}: {len(errors)} issue(s):")
        for e in errors: print(f"   • {e}")
    else:
        print(f"✅ {index.doc_id}: all assertions passed ({len(index.nodes)} nodes)")

    return len(errors) == 0


ok_l = assert_index_quality(legacy_index,     legacy_tree)
ok_m = assert_index_quality(modernized_index, modernized_tree)

if ok_l and ok_m:
    print("\n🟢 Stage 2 complete — indexes ready for Stage 3 (Section Alignment)")



------


Cell 8 — Tree Index Models
# Cell 8: Tree Index Models
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class CanonicalNode(BaseModel):
    canonical_key: str
    section_id: str
    heading_text: str
    depth: int
    parent_key: Optional[str] = None
    children_keys: list[str] = Field(default_factory=list)
    raw_text: str = ""
    table_count: int = 0
    list_count: int = 0
    char_count: int = 0


class TreeIndex(BaseModel):
    doc_id: str
    nodes: dict[str, CanonicalNode] = Field(default_factory=dict)
    sid_to_key: dict[str, str] = Field(default_factory=dict)
    root_keys: list[str] = Field(default_factory=list)


print("✅ TreeIndex models loaded")
Cell 9 — Tree Index Builder
# Cell 9: Tree Index Builder
import orjson
from pathlib import Path


class TreeIndexBuilder:

    def build(self, tree: DocumentTree) -> TreeIndex:
        index = TreeIndex(doc_id=tree.doc_id)
        counters: dict[str, int] = {}

        def _walk(sid: str, parent_key: Optional[str]):
            node = tree.sections[sid]
            bucket = parent_key or "__root__"
            counters[bucket] = counters.get(bucket, 0) + 1
            n = counters[bucket]
            canon_key = f"S{n}" if parent_key is None else f"{parent_key}.{n}"

            cnode = CanonicalNode(
                canonical_key=canon_key,
                section_id=sid,
                heading_text=node.heading_text,
                depth=node.depth,
                parent_key=parent_key,
                raw_text=node.raw_text,
                table_count=len(node.tables),
                list_count=len(node.lists),
                char_count=len(node.raw_text),
            )
            index.nodes[canon_key] = cnode
            index.sid_to_key[sid] = canon_key

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


builder = TreeIndexBuilder()
legacy_index = builder.build(legacy_tree)
modernized_index = builder.build(modernized_tree)
builder.save(legacy_index, "output/legacy_index.json")
builder.save(modernized_index, "output/modernized_index.json")

print(f"✅ Legacy index: {len(legacy_index.nodes)} nodes")
print(f"✅ Modernized index: {len(modernized_index.nodes)} nodes")
Cell 10 — Section Alignment Models
# Cell 10: Section Alignment Models
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


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
    match_score: float = 0.0          # 0–1 fuzzy score


class AlignmentMap(BaseModel):
    pairs: list[AlignedPair] = Field(default_factory=list)

    @property
    def matched(self)  -> list[AlignedPair]:
        return [p for p in self.pairs if p.state == SectionState.MATCHED]

    @property
    def added(self)    -> list[AlignedPair]:
        return [p for p in self.pairs if p.state == SectionState.ADDED]

    @property
    def deleted(self)  -> list[AlignedPair]:
        return [p for p in self.pairs if p.state == SectionState.DELETED]


print("✅ Alignment models loaded")
Cell 11 — Section Aligner
# Cell 11: Section Aligner — bipartite fuzzy heading match
import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment


class SectionAligner:
    """
    Aligns sections from two TreeIndexes using fuzzy heading similarity.
    Uses scipy linear_sum_assignment (Hungarian algorithm) for optimal
    bipartite matching. Falls back to embedding similarity for low-score
    pairs (score < threshold).

    Only leaf + mid-level nodes at matching depths are aligned.
    """

    def __init__(self, match_threshold: float = 0.60):
        self.threshold = match_threshold

    def align(self, legacy: TreeIndex, modernized: TreeIndex) -> AlignmentMap:
        amap = AlignmentMap()

        leg_keys  = list(legacy.nodes.keys())
        mod_keys  = list(modernized.nodes.keys())

        # build cost matrix  (1 - similarity) so minimising cost = maximising similarity
        n, m = len(leg_keys), len(mod_keys)
        cost = np.ones((n, m), dtype=np.float32)

        for i, lk in enumerate(leg_keys):
            lh = legacy.nodes[lk].heading_text
            for j, mk in enumerate(mod_keys):
                mh = modernized.nodes[mk].heading_text
                score = self._similarity(lh, mh)
                cost[i, j] = 1.0 - score

        # pad to square for Hungarian algorithm
        size = max(n, m)
        padded = np.ones((size, size), dtype=np.float32)
        padded[:n, :m] = cost

        row_ind, col_ind = linear_sum_assignment(padded)

        matched_mod = set()
        matched_leg = set()

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

        # unmatched legacy → DELETED
        for lk in leg_keys:
            if lk not in matched_leg:
                amap.pairs.append(AlignedPair(
                    state=SectionState.DELETED,
                    legacy_key=lk,
                    legacy_heading=legacy.nodes[lk].heading_text,
                ))

        # unmatched modernized → ADDED
        for mk in mod_keys:
            if mk not in matched_mod:
                amap.pairs.append(AlignedPair(
                    state=SectionState.ADDED,
                    modernized_key=mk,
                    modernized_heading=modernized.nodes[mk].heading_text,
                ))

        return amap

    def _similarity(self, a: str, b: str) -> float:
        # weighted blend: token_sort handles word reordering, ratio for exact proximity
        return (
            0.6 * fuzz.token_sort_ratio(a, b) / 100.0 +
            0.4 * fuzz.ratio(a, b) / 100.0
        )

    def save(self, amap: AlignmentMap, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(orjson.dumps(amap.model_dump(), option=orjson.OPT_INDENT_2))


aligner = SectionAligner(match_threshold=0.60)
alignment_map = aligner.align(legacy_index, modernized_index)
aligner.save(alignment_map, "output/alignment_map.json")

print(f"✅ Matched : {len(alignment_map.matched)}")
print(f"✅ Added   : {len(alignment_map.added)}")
print(f"✅ Deleted : {len(alignment_map.deleted)}")
Cell 12 — Chunker Models
# Cell 12: Chunker models
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str                   # e.g. "legacy::S2.1::0"
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
    sections: dict[str, SectionChunks] = Field(default_factory=dict)  # key → SectionChunks


print("✅ Chunker models loaded")
Cell 13 — Paragraph Chunker
# Cell 13: Paragraph-boundary chunker, 150–300 tokens, no mid-sentence splits
import re
import tiktoken


class ParagraphChunker:
    """
    Splits a section's raw_text into paragraph-boundary chunks
    targeting 150–300 tokens. Never splits mid-sentence.
    """

    def __init__(self, min_tokens: int = 150, max_tokens: int = 300):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")

    def chunk_store(self, index: TreeIndex, tree: DocumentTree) -> ChunkStore:
        store = ChunkStore(doc_id=index.doc_id)
        for key, cnode in index.nodes.items():
            sid = cnode.section_id
            raw = tree.sections[sid].raw_text.strip()
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

    # ── internal ─────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if not paragraphs:
            return [text]

        chunks, buffer, buf_tokens = [], [], 0

        for para in paragraphs:
            para_tokens = len(self._enc.encode(para))

            # paragraph itself too large → sentence split
            if para_tokens > self.max_tokens:
                if buffer:
                    chunks.append(" ".join(buffer))
                    buffer, buf_tokens = [], 0
                chunks.extend(self._sentence_split(para))
                continue

            if buf_tokens + para_tokens > self.max_tokens and buf_tokens >= self.min_tokens:
                chunks.append(" ".join(buffer))
                buffer, buf_tokens = [para], para_tokens
            else:
                buffer.append(para)
                buf_tokens += para_tokens

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

legacy_chunks     = chunker.chunk_store(legacy_index,     legacy_tree)
modernized_chunks = chunker.chunk_store(modernized_index, modernized_tree)

chunker.save(legacy_chunks,     "output/legacy_chunks.json")
chunker.save(modernized_chunks, "output/modernized_chunks.json")

total_l = sum(len(sc.chunks) for sc in legacy_chunks.sections.values())
total_m = sum(len(sc.chunks) for sc in modernized_chunks.sections.values())

print(f"✅ Legacy chunks     : {total_l}")
print(f"✅ Modernized chunks : {total_m}")


Cell 14 — Embedding + FAISS Triage Models
# Cell 14: Embedding triage models
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TriageLabel(str, Enum):
    NO_CHANGE   = "NO_CHANGE"     # cosine > 0.92
    PARAPHRASE  = "PARAPHRASE"    # 0.75 – 0.92
    SUBSTANTIVE = "SUBSTANTIVE"   # < 0.75
    MISSING     = "MISSING"       # no FAISS match found


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


print("✅ Triage models loaded")
Cell 15 — Embedder + FAISS Triage
# Cell 15: Sentence Transformer embeddings + FAISS cosine triage
# ROCm-backed via torch — sentence-transformers uses torch device automatically
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer


# ── device selection: ROCm → CUDA fallback → CPU ─────────────────
def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"   # ROCm exposes via CUDA API
    return "cpu"


class EmbeddingTriager:
    """
    For each MATCHED section pair from the AlignmentMap:
      1. Embed all legacy chunks and modernized chunks for that section
      2. Build a FAISS IndexFlatIP index over modernized embeddings
      3. For each legacy chunk, find nearest modernized chunk
      4. Assign triage label based on cosine score
    """

    THRESHOLD_NO_CHANGE   = 0.92
    THRESHOLD_PARAPHRASE  = 0.75

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        device = _get_device()
        print(f"  Loading embedding model on {device}…")
        self._model = SentenceTransformer(model_name, device=device)

    def triage(
        self,
        alignment_map: AlignmentMap,
        legacy_chunks: ChunkStore,
        modernized_chunks: ChunkStore,
    ) -> TriageResult:

        result = TriageResult()

        for pair in alignment_map.matched:
            lk = pair.legacy_key
            mk = pair.modernized_key

            l_section = legacy_chunks.sections.get(lk)
            m_section = modernized_chunks.sections.get(mk)

            # section exists in alignment but has no text chunks — skip
            if not l_section or not m_section:
                continue
            if not l_section.chunks or not m_section.chunks:
                continue

            l_texts = [c.text for c in l_section.chunks]
            m_texts = [c.text for c in m_section.chunks]

            l_embs = self._embed(l_texts)   # (n_l, dim)
            m_embs = self._embed(m_texts)   # (n_m, dim)

            # build FAISS index over modernized embeddings
            dim   = m_embs.shape[1]
            index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalised vecs
            index.add(m_embs)

            scores, indices = index.search(l_embs, k=1)  # nearest neighbour per legacy chunk

            for i, l_chunk in enumerate(l_section.chunks):
                nn_idx   = int(indices[i, 0])
                cosine   = float(scores[i, 0])
                m_chunk  = m_section.chunks[nn_idx]

                triage = self._label(cosine)

                result.pairs.append(ChunkPair(
                    legacy_chunk_id=l_chunk.chunk_id,
                    modernized_chunk_id=m_chunk.chunk_id,
                    legacy_text=l_chunk.text,
                    modernized_text=m_chunk.text,
                    canonical_key=lk,
                    cosine_score=round(cosine, 4),
                    triage=triage,
                    heading_text=pair.legacy_heading or "",
                ))

        # ADDED sections → all chunks are MISSING (no legacy counterpart)
        for pair in alignment_map.added:
            mk = pair.modernized_key
            m_section = modernized_chunks.sections.get(mk)
            if not m_section:
                continue
            for m_chunk in m_section.chunks:
                result.pairs.append(ChunkPair(
                    legacy_chunk_id="",
                    modernized_chunk_id=m_chunk.chunk_id,
                    legacy_text="",
                    modernized_text=m_chunk.text,
                    canonical_key=mk,
                    triage=TriageLabel.MISSING,
                    heading_text=pair.modernized_heading or "",
                ))

        # DELETED sections → all chunks are MISSING (no modernized counterpart)
        for pair in alignment_map.deleted:
            lk = pair.legacy_key
            l_section = legacy_chunks.sections.get(lk)
            if not l_section:
                continue
            for l_chunk in l_section.chunks:
                result.pairs.append(ChunkPair(
                    legacy_chunk_id=l_chunk.chunk_id,
                    modernized_chunk_id="",
                    legacy_text=l_chunk.text,
                    modernized_text="",
                    canonical_key=lk,
                    triage=TriageLabel.MISSING,
                    heading_text=pair.legacy_heading or "",
                ))

        return result

    def save(self, result: TriageResult, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(orjson.dumps(result.model_dump(), option=orjson.OPT_INDENT_2))

    def _embed(self, texts: list[str]) -> np.ndarray:
        embs = self._model.encode(
            texts,
            normalize_embeddings=True,   # required for cosine via dot product
            show_progress_bar=False,
            batch_size=32,
        )
        return embs.astype(np.float32)

    def _label(self, score: float) -> TriageLabel:
        if score >= self.THRESHOLD_NO_CHANGE:
            return TriageLabel.NO_CHANGE
        if score >= self.THRESHOLD_PARAPHRASE:
            return TriageLabel.PARAPHRASE
        return TriageLabel.SUBSTANTIVE


triager = EmbeddingTriager(model_name="BAAI/bge-large-en-v1.5")
triage_result = triager.triage(alignment_map, legacy_chunks, modernized_chunks)
triager.save(triage_result, "output/triage_result.json")

counts = {t.value: 0 for t in TriageLabel}
for p in triage_result.pairs:
    counts[p.triage.value] += 1

print(f"✅ NO_CHANGE   : {counts['NO_CHANGE']}")
print(f"✅ PARAPHRASE  : {counts['PARAPHRASE']}")
print(f"✅ SUBSTANTIVE : {counts['SUBSTANTIVE']}")
print(f"✅ MISSING     : {counts['MISSING']}")
print(f"→  Chunks sent to LLM : {len(triage_result.for_llm)}")
Cell 16 — LLM Reasoning Models
# Cell 16: LLM output models
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Severity(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class ChangeType(str, Enum):
    OBLIGATION_ADDED    = "OBLIGATION_ADDED"
    OBLIGATION_REMOVED  = "OBLIGATION_REMOVED"
    THRESHOLD_CHANGED   = "THRESHOLD_CHANGED"
    SCOPE_BROADENED     = "SCOPE_BROADENED"
    SCOPE_NARROWED      = "SCOPE_NARROWED"
    NEUTRAL_REWORD      = "NEUTRAL_REWORD"
    PROCESS_CHANGED     = "PROCESS_CHANGED"
    OTHER               = "OTHER"


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
    confidence: float   # 0–1


class LLMResult(BaseModel):
    analyses: list[LLMAnalysis] = Field(default_factory=list)

    @property
    def high(self)   -> list[LLMAnalysis]: return [a for a in self.analyses if a.severity == Severity.HIGH]
    @property
    def medium(self) -> list[LLMAnalysis]: return [a for a in self.analyses if a.severity == Severity.MEDIUM]
    @property
    def low(self)    -> list[LLMAnalysis]: return [a for a in self.analyses if a.severity == Severity.LOW]


print("✅ LLM models loaded")
Cell 17 — LLM Reasoner (vLLM / Ollama compatible)
# Cell 17: LLM Reasoner — calls vLLM OpenAI-compat endpoint
# vLLM launch command (run in terminal before this cell):
#   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype auto
#
# For Ollama swap base_url to http://localhost:11434/v1

import json
import time
from openai import OpenAI


SYSTEM_PROMPT = """You are a compliance analyst comparing legacy and modernized policy documents.
You will receive two text chunks from the same policy section.
Respond ONLY with a valid JSON object — no markdown, no explanation outside the JSON.

Required JSON schema:
{
  "what_changed": "<one sentence describing the specific change>",
  "compliance_implication": "<one sentence on whether this is stricter, more lenient, or neutral>",
  "change_type": "<one of: OBLIGATION_ADDED | OBLIGATION_REMOVED | THRESHOLD_CHANGED | SCOPE_BROADENED | SCOPE_NARROWED | NEUTRAL_REWORD | PROCESS_CHANGED | OTHER>",
  "severity": "<HIGH | MEDIUM | LOW>",
  "confidence": <float 0.0–1.0>
}"""


def _user_prompt(pair: ChunkPair) -> str:
    return f"""Section: {pair.heading_text}
Triage: {pair.triage.value}  Cosine similarity: {pair.cosine_score}

LEGACY TEXT:
{pair.legacy_text}

MODERNIZED TEXT:
{pair.modernized_text}

Analyse what changed and output only the JSON object."""


class LLMReasoner:

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
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
        print(f"  Sending {len(flagged)} chunk pairs to LLM in batches of {self._batch_size}…")

        result = LLMResult()
        for i in range(0, len(flagged), self._batch_size):
            batch = flagged[i : i + self._batch_size]
            for pair in batch:
                analysis = self._call(pair)
                if analysis:
                    result.analyses.append(analysis)
            print(f"  Processed {min(i + self._batch_size, len(flagged))} / {len(flagged)}")

        return result

    def save(self, result: LLMResult, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(orjson.dumps(result.model_dump(), option=orjson.OPT_INDENT_2))

    # ── internal ──────────────────────────────────────────────────

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
                    print(f"  ⚠ LLM call failed for {pair.legacy_chunk_id}: {e}")
                    return None

    def _parse(self, raw: str, pair: ChunkPair) -> Optional[LLMAnalysis]:
        try:
            # strip accidental markdown fences
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
            print(f"  ⚠ Parse error for {pair.legacy_chunk_id}: {e}\n  Raw: {raw[:200]}")
            return None


reasoner   = LLMReasoner(base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-7B-Instruct")
llm_result = reasoner.reason(triage_result)
reasoner.save(llm_result, "output/llm_result.json")

print(f"\n✅ LLM analyses complete")
print(f"   HIGH   : {len(llm_result.high)}")
print(f"   MEDIUM : {len(llm_result.medium)}")
print(f"   LOW    : {len(llm_result.low)}")
Cell 18 — Report Generator
# Cell 18: Report Generator — HTML + JSON diff artifact
from datetime import datetime
from jinja2 import Template


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
    --added:#2196f3; --deleted:#f04060; --neutral:#6b7694;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Inter',sans-serif; padding:40px 32px; }
  h1   { font-size:22px; font-weight:700; margin-bottom:6px; }
  h2   { font-size:14px; font-weight:600; margin:32px 0 12px; letter-spacing:.04em; color:#fff; }
  p    { font-size:12px; color:var(--muted); margin-bottom:24px; }

  /* summary band */
  .summary { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:32px; }
  .stat { background:var(--surface); border:1px solid var(--border); border-radius:6px;
          padding:14px 20px; min-width:120px; }
  .stat .n { font-size:28px; font-weight:700; line-height:1; }
  .stat .l { font-size:10px; color:var(--muted); margin-top:4px; text-transform:uppercase; letter-spacing:.08em; }
  .stat.high   .n { color:var(--high); }
  .stat.med    .n { color:var(--med);  }
  .stat.low    .n { color:var(--low);  }
  .stat.added  .n { color:var(--added);}
  .stat.del    .n { color:var(--deleted); }

  /* tables */
  table { width:100%; border-collapse:collapse; font-size:11.5px; margin-bottom:24px; }
  th { background:var(--surface); padding:8px 12px; text-align:left;
       font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
       border-bottom:1px solid var(--border); }
  td { padding:9px 12px; border-bottom:1px solid var(--border); vertical-align:top; line-height:1.55; }
  tr:last-child td { border-bottom:none; }

  .badge { display:inline-block; padding:2px 8px; border-radius:3px;
           font-size:10px; font-weight:600; letter-spacing:.06em; }
  .HIGH       { background:rgba(240,64,96,.15);  color:var(--high); }
  .MEDIUM     { background:rgba(240,160,48,.15); color:var(--med);  }
  .LOW        { background:rgba(0,200,150,.15);  color:var(--low);  }
  .ADDED      { background:rgba(33,150,243,.15); color:var(--added);}
  .DELETED    { background:rgba(240,64,96,.15);  color:var(--high); }
  .MATCHED    { background:rgba(107,118,148,.15);color:var(--muted);}

  .chunk-box { background:var(--surface); border:1px solid var(--border);
               border-radius:5px; padding:8px 10px; font-size:11px;
               font-family:'JetBrains Mono',monospace; line-height:1.6;
               white-space:pre-wrap; word-break:break-word; max-height:120px; overflow:auto; }
  .score { font-family:monospace; font-size:11px; color:var(--muted); }
  footer { margin-top:40px; font-size:10px; color:var(--muted); text-align:center; }
</style>
</head>
<body>

<h1>Policy Document Comparison Report</h1>
<p>Generated {{ generated_at }}  ·  Legacy: {{ legacy_doc }}  ·  Modernized: {{ modern_doc }}</p>

<!-- ── Executive Summary ─────────────────────────────── -->
<h2>Executive Summary</h2>
<div class="summary">
  <div class="stat high">  <div class="n">{{ n_high }}</div>   <div class="l">High severity</div></div>
  <div class="stat med">   <div class="n">{{ n_med }}</div>    <div class="l">Medium severity</div></div>
  <div class="stat low">   <div class="n">{{ n_low }}</div>    <div class="l">Low severity</div></div>
  <div class="stat added"> <div class="n">{{ n_added }}</div>  <div class="l">Sections added</div></div>
  <div class="stat del">   <div class="n">{{ n_deleted }}</div><div class="l">Sections deleted</div></div>
  <div class="stat">       <div class="n">{{ n_matched }}</div><div class="l">Sections matched</div></div>
</div>

<!-- ── Section Alignment Overview ───────────────────── -->
<h2>Section Alignment</h2>
<table>
  <tr><th>State</th><th>Legacy Key</th><th>Legacy Heading</th><th>Modernized Key</th><th>Modernized Heading</th><th>Score</th></tr>
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

<!-- ── Flagged Items (LLM analysed) ──────────────────── -->
<h2>Flagged Changes — LLM Analysis</h2>
<table>
  <tr><th>Severity</th><th>Section</th><th>Change Type</th><th>What Changed</th><th>Compliance Impact</th><th>Conf.</th></tr>
  {% for a in analyses_sorted %}
  <tr>
    <td><span class="badge {{ a.severity }}">{{ a.severity }}</span></td>
    <td><b>{{ a.canonical_key }}</b><br><span style="color:var(--muted);font-size:10px">{{ a.heading_text }}</span></td>
    <td style="font-size:10.5px">{{ a.change_type }}</td>
    <td>{{ a.what_changed }}</td>
    <td>{{ a.compliance_implication }}</td>
    <td class="score">{{ '%.0f'|format(a.confidence * 100) }}%</td>
  </tr>
  {% endfor %}
</table>

<!-- ── Raw Chunk Pairs (high severity only) ──────────── -->
<h2>Raw Clause Pairs — High Severity</h2>
<table>
  <tr><th>Section</th><th>Cosine</th><th>Legacy Text</th><th>Modernized Text</th></tr>
  {% for p in high_pairs %}
  <tr>
    <td><b>{{ p.canonical_key }}</b><br><span style="color:var(--muted);font-size:10px">{{ p.heading_text }}</span></td>
    <td class="score">{{ p.cosine_score }}</td>
    <td><div class="chunk-box">{{ p.legacy_text }}</div></td>
    <td><div class="chunk-box">{{ p.modernized_text or '— deleted —' }}</div></td>
  </tr>
  {% endfor %}
</table>

<footer>Policy Diff Pipeline · ROCm + vLLM · Python 3.11</footer>
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
    ) -> dict:

        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        analyses_sorted = sorted(
            llm_result.analyses,
            key=lambda a: (severity_order.get(a.severity.value, 9), -( a.confidence))
        )

        # high-severity chunk pairs for raw view
        high_ids = {a.legacy_chunk_id for a in llm_result.high}
        high_pairs = [p for p in triage_result.pairs if p.legacy_chunk_id in high_ids]

        ctx = dict(
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

        html = Template(REPORT_TEMPLATE).render(**ctx)
        return {"html": html, "context": ctx}

    def save_html(self, html: str, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")

    def save_json_artifact(self, llm_result: LLMResult, triage_result: TriageResult, out: str | Path):
        out = Path(out)
        artifact = {
            "generated_at": datetime.now().isoformat(),
            "triage_summary": {t.value: 0 for t in TriageLabel},
            "analyses": [a.model_dump() for a in llm_result.analyses],
            "missing_chunks": [
                p.model_dump() for p in triage_result.pairs
                if p.triage == TriageLabel.MISSING
            ],
        }
        for p in triage_result.pairs:
            artifact["triage_summary"][p.triage.value] += 1
        out.write_bytes(orjson.dumps(artifact, option=orjson.OPT_INDENT_2))


reporter = ReportGenerator()
report   = reporter.generate(alignment_map, triage_result, llm_result)

reporter.save_html(report["html"], "output/report.html")
reporter.save_json_artifact(llm_result, triage_result, "output/diff_artifact.json")

print("✅ Report saved → output/report.html")
print("✅ Diff artifact → output/diff_artifact.json")
Cell 19 — End-to-End Run Summary
# Cell 19: Pipeline summary — confirm all outputs exist

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

print("── Pipeline Output Manifest ─────────────────────────────────")
all_ok = True
for path in outputs:
    p = Path(path)
    if p.exists():
        size = p.stat().st_size
        print(f"  ✅  {path:<45} {size:>8,} bytes")
    else:
        print(f"  ❌  {path:<45} MISSING")
        all_ok = False

if all_ok:
    print("\n🟢 Full pipeline complete — open output/report.html to review")
else:
    print("\n🔴 Some outputs missingcheck cells above for errors")


The only external dependency that needs to be running before Cell 17 is vLLM — start it with vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype auto in a terminal.
