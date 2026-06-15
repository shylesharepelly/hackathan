import difflib
from jinja2 import Template
from datetime import datetime
from pathlib import Path

# ── word-level diff ───────────────────────────────────────────────

def _word_diff_html(legacy_text: str, modernized_text: str) -> tuple[str, str]:
    """
    Returns (legacy_html, modernized_html) with word-level diff markup.
    Deleted words highlighted red in legacy.
    Added words highlighted green in modernized.
    Unchanged words unstyled.
    """
    legacy_words    = legacy_text.split()
    modernized_words = modernized_text.split()

    matcher = difflib.SequenceMatcher(
        None, legacy_words, modernized_words, autojunk=False
    )

    legacy_html    = []
    modernized_html = []

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            chunk = " ".join(legacy_words[i1:i2])
            legacy_html.append(chunk)
            modernized_html.append(chunk)

        elif opcode == "replace":
            legacy_html.append(
                f'<span class="diff-del">{" ".join(legacy_words[i1:i2])}</span>'
            )
            modernized_html.append(
                f'<span class="diff-add">{" ".join(modernized_words[j1:j2])}</span>'
            )

        elif opcode == "delete":
            legacy_html.append(
                f'<span class="diff-del">{" ".join(legacy_words[i1:i2])}</span>'
            )

        elif opcode == "insert":
            modernized_html.append(
                f'<span class="diff-add">{" ".join(modernized_words[j1:j2])}</span>'
            )

    return " ".join(legacy_html), " ".join(modernized_html)


def _missing_html(text: str, kind: str) -> str:
    """
    kind: 'added' → full green block
          'deleted' → full red block
    """
    css = "diff-add-block" if kind == "added" else "diff-del-block"
    return f'<span class="{css}">{text}</span>'


# ── report template ───────────────────────────────────────────────

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Policy Diff Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

  :root {
    --bg:      #0a0d12;
    --surface: #111620;
    --surface2:#161c2a;
    --border:  #1e2535;
    --border2: #2a3450;
    --text:    #d8dde8;
    --muted:   #6b7694;
    --high:    #f04060;
    --med:     #f0a030;
    --low:     #00c896;
    --added:   #2196f3;
    --deleted: #f04060;
    --mono:    'JetBrains Mono', monospace;
    --sans:    'Inter', sans-serif;
  }

  * { box-sizing:border-box; margin:0; padding:0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    padding: 40px 32px 80px;
    line-height: 1.6;
  }

  /* ── typography ── */
  h1 { font-size:22px; font-weight:700; margin-bottom:6px; }
  h2 {
    font-size:11px; font-weight:600;
    letter-spacing:.1em; text-transform:uppercase;
    color: var(--muted); margin: 40px 0 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  p.sub { font-size:12px; color:var(--muted); margin-bottom:28px; }

  /* ── summary cards ── */
  .summary { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:8px; }
  .stat {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 20px;
    min-width: 110px;
  }
  .stat .n { font-size:26px; font-weight:700; line-height:1; font-family:var(--mono); }
  .stat .l { font-size:10px; color:var(--muted); margin-top:5px;
             text-transform:uppercase; letter-spacing:.08em; }
  .stat.s-high  .n { color:var(--high);   }
  .stat.s-med   .n { color:var(--med);    }
  .stat.s-low   .n { color:var(--low);    }
  .stat.s-added .n { color:var(--added);  }
  .stat.s-del   .n { color:var(--deleted);}

  /* ── tables ── */
  table {
    width:100%; border-collapse:collapse;
    font-size:11.5px; margin-bottom:28px;
  }
  th {
    background: var(--surface);
    padding: 8px 12px; text-align:left;
    font-size:10px; text-transform:uppercase;
    letter-spacing:.08em; color:var(--muted);
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
    line-height: 1.6;
  }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background: rgba(255,255,255,.02); }

  /* ── badges ── */
  .badge {
    display:inline-block; padding:2px 8px; border-radius:3px;
    font-size:10px; font-weight:600; letter-spacing:.05em;
    font-family: var(--mono);
  }
  .HIGH       { background:rgba(240,64,96,.15);   color:var(--high);   }
  .MEDIUM     { background:rgba(240,160,48,.15);  color:var(--med);    }
  .LOW        { background:rgba(0,200,150,.15);   color:var(--low);    }
  .MATCHED    { background:rgba(107,118,148,.12); color:var(--muted);  }
  .ADDED      { background:rgba(33,150,243,.15);  color:var(--added);  }
  .DELETED    { background:rgba(240,64,96,.15);   color:var(--high);   }
  .PARAPHRASE { background:rgba(240,160,48,.12);  color:var(--med);    }
  .SUBSTANTIVE{ background:rgba(240,64,96,.12);   color:var(--high);   }
  .MISSING    { background:rgba(107,118,148,.12); color:var(--muted);  }
  .NO_CHANGE  { background:rgba(0,200,150,.08);   color:var(--low);    }

  /* ── diff view ── */
  .diff-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 20px;
    overflow: hidden;
  }

  .diff-section-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--surface2);
  }

  .diff-section-key {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--text);
  }

  .diff-section-heading {
    font-size: 12px;
    color: var(--muted);
    flex: 1;
  }

  .diff-score {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
  }

  .diff-panes {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
  }

  .diff-pane {
    padding: 14px 16px;
    font-size: 12px;
    font-family: var(--mono);
    line-height: 1.8;
    white-space: pre-wrap;
    word-break: break-word;
    border-right: 1px solid var(--border);
  }

  .diff-pane:last-child { border-right: none; }

  .diff-pane-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--muted);
    margin-bottom: 10px;
    font-family: var(--sans);
    font-weight: 600;
  }

  /* word-level diff highlights */
  .diff-del {
    background: rgba(240,64,96,.22);
    color: #ff8090;
    border-radius: 3px;
    padding: 1px 3px;
    text-decoration: line-through;
    text-decoration-color: rgba(240,64,96,.6);
  }

  .diff-add {
    background: rgba(0,200,150,.18);
    color: #5fffcc;
    border-radius: 3px;
    padding: 1px 3px;
  }

  /* full block for MISSING sections */
  .diff-del-block {
    background: rgba(240,64,96,.12);
    color: #ff8090;
    border-radius: 3px;
    padding: 2px 4px;
    display: block;
    border-left: 3px solid var(--high);
    padding-left: 10px;
    margin: 2px 0;
  }

  .diff-add-block {
    background: rgba(0,200,150,.10);
    color: #5fffcc;
    border-radius: 3px;
    display: block;
    border-left: 3px solid var(--low);
    padding-left: 10px;
    margin: 2px 0;
  }

  .diff-empty {
    color: var(--muted);
    font-style: italic;
    font-size: 11px;
  }

  /* context clause */
  .diff-context {
    font-size: 11px;
    color: var(--muted);
    border-top: 1px dashed var(--border);
    margin-top: 10px;
    padding-top: 8px;
    font-style: italic;
  }

  /* ── misc ── */
  .score { font-family:var(--mono); font-size:11px; color:var(--muted); }

  footer {
    margin-top: 60px;
    font-size: 10px;
    color: var(--muted);
    text-align: center;
    font-family: var(--mono);
    letter-spacing: .04em;
  }

  @media (max-width: 800px) {
    .diff-panes { grid-template-columns: 1fr; }
    .diff-pane { border-right: none; border-bottom: 1px solid var(--border); }
  }
</style>
</head>
<body>

<h1>Policy Document Comparison Report</h1>
<p class="sub">
  Generated {{ generated_at }} &nbsp;·&nbsp;
  Legacy: {{ legacy_doc }} &nbsp;·&nbsp;
  Modernized: {{ modern_doc }}
</p>

<!-- ── Tier 1: Executive Summary ─────────────────────────── -->
<h2>Executive Summary</h2>
<div class="summary">
  <div class="stat s-high"> <div class="n">{{ n_high }}</div>   <div class="l">High severity</div></div>
  <div class="stat s-med">  <div class="n">{{ n_med }}</div>    <div class="l">Medium severity</div></div>
  <div class="stat s-low">  <div class="n">{{ n_low }}</div>    <div class="l">Low severity</div></div>
  <div class="stat s-added"><div class="n">{{ n_added }}</div>  <div class="l">Sections added</div></div>
  <div class="stat s-del">  <div class="n">{{ n_deleted }}</div><div class="l">Sections deleted</div></div>
  <div class="stat">        <div class="n">{{ n_matched }}</div><div class="l">Sections matched</div></div>
  <div class="stat">        <div class="n">{{ n_no_change }}</div><div class="l">Unchanged</div></div>
</div>

<!-- ── Tier 2: Section Alignment ─────────────────────────── -->
<h2>Section Alignment</h2>
<table>
  <tr>
    <th>State</th><th>Legacy ID</th><th>Legacy Heading</th>
    <th>Modernized ID</th><th>Modernized Heading</th><th>Score</th>
  </tr>
  {% for p in alignment_pairs %}
  <tr>
    <td><span class="badge {{ p.state }}">{{ p.state }}</span></td>
    <td class="score">{{ p.legacy_id or '—' }}</td>
    <td>{{ p.legacy_heading or '—' }}</td>
    <td class="score">{{ p.modernized_id or '—' }}</td>
    <td>{{ p.modernized_heading or '—' }}</td>
    <td class="score">{{ '%.3f'|format(p.similarity_score) if p.similarity_score else '—' }}</td>
  </tr>
  {% endfor %}
</table>

<!-- ── Tier 3: LLM Analysis ──────────────────────────────── -->
<h2>Compliance Risk Analysis</h2>
<table>
  <tr>
    <th>Severity</th><th>Section</th><th>Change Type</th>
    <th>What Changed</th><th>Compliance Impact</th>
    <th>Action</th><th>Conf.</th>
  </tr>
  {% for a in analyses_sorted %}
  <tr>
    <td><span class="badge {{ a.severity }}">{{ a.severity }}</span></td>
    <td>
      <b class="score">{{ a.section_id }}</b><br>
      <span style="color:var(--muted);font-size:10.5px">{{ a.heading }}</span>
    </td>
    <td style="font-size:10.5px; font-family:var(--mono)">{{ a.change_type }}</td>
    <td>{{ a.what_changed }}</td>
    <td>{{ a.compliance_implication }}</td>
    <td>
      <span class="badge {{ a.recommended_action }}"
            style="background:rgba(255,255,255,.05);color:var(--text)">
        {{ a.recommended_action }}
      </span>
    </td>
    <td class="score">{{ '%.0f'|format(a.confidence*100) }}%</td>
  </tr>
  {% endfor %}
</table>

<!-- ── Tier 4: Inline Clause Diff View ───────────────────── -->
<h2>Clause-Level Diff — PARAPHRASE &amp; SUBSTANTIVE</h2>

{% for item in diff_items %}
<div class="diff-section">

  <div class="diff-section-header">
    <span class="diff-section-key">{{ item.section_id }}</span>
    <span class="diff-section-heading">{{ item.heading }}</span>
    <span class="badge {{ item.triage }}">{{ item.triage }}</span>
    <span class="diff-score">cos {{ '%.3f'|format(item.cosine_score) if item.cosine_score else '—' }}</span>
  </div>

  <div class="diff-panes">

    <!-- legacy pane -->
    <div class="diff-pane">
      <div class="diff-pane-label">Legacy</div>
      {% if item.legacy_html %}
        {{ item.legacy_html }}
        {% if item.context_before %}
          <div class="diff-context">↑ context: {{ item.context_before[:120] }}…</div>
        {% endif %}
        {% if item.context_after %}
          <div class="diff-context">↓ context: {{ item.context_after[:120] }}…</div>
        {% endif %}
      {% else %}
        <span class="diff-empty">— not present in legacy —</span>
      {% endif %}
    </div>

    <!-- modernized pane -->
    <div class="diff-pane">
      <div class="diff-pane-label">Modernized</div>
      {% if item.modernized_html %}
        {{ item.modernized_html }}
      {% else %}
        <span class="diff-empty">— removed in modernized —</span>
      {% endif %}
    </div>

  </div>
</div>
{% endfor %}

<footer>
  Policy Diff Pipeline &nbsp;·&nbsp; ROCm + vLLM &nbsp;·&nbsp; Python 3.11
</footer>
</body>
</html>"""


# ── report generator class ────────────────────────────────────────

class ReportGenerator:

    def generate(
        self,
        alignment_map: AlignmentMap,
        triage_result: TriageResult,
        llm_result: LLMResult,
        legacy_doc: str = "legacy_policy.pdf",
        modern_doc: str = "modernized_policy.pdf",
    ) -> str:

        # ── summary counts ────────────────────────────────────────
        sev_order  = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        analyses_sorted = sorted(
            llm_result.analyses,
            key=lambda a: (sev_order.get(a.severity.value, 9), -a.confidence)
        )

        counts = {t.value: 0 for t in TriageLabel}
        for p in triage_result.pairs:
            counts[p.triage.value] += 1

        # ── build diff items ──────────────────────────────────────
        diff_items = []

        for pair in triage_result.pairs:

            if pair.triage not in (TriageLabel.PARAPHRASE, TriageLabel.SUBSTANTIVE, TriageLabel.MISSING):
                continue

            legacy_html    = ""
            modernized_html = ""

            if pair.triage == TriageLabel.MISSING:
                # full section added or deleted
                if pair.legacy_text and not pair.modernized_text:
                    legacy_html     = _missing_html(pair.legacy_text, "deleted")
                    modernized_html = ""
                elif pair.modernized_text and not pair.legacy_text:
                    legacy_html     = ""
                    modernized_html = _missing_html(pair.modernized_text, "added")

            else:
                # word-level diff for PARAPHRASE and SUBSTANTIVE
                if pair.legacy_text and pair.modernized_text:
                    legacy_html, modernized_html = _word_diff_html(
                        pair.legacy_text,
                        pair.modernized_text,
                    )

            diff_items.append({
                "section_id":      pair.section_id,
                "heading":         pair.heading,
                "triage":          pair.triage.value,
                "cosine_score":    pair.cosine_score,
                "legacy_html":     legacy_html,
                "modernized_html": modernized_html,
                "context_before":  pair.context_before,
                "context_after":   pair.context_after,
            })

        # sort diff items: SUBSTANTIVE first, then PARAPHRASE, then MISSING
        triage_order = {"SUBSTANTIVE": 0, "PARAPHRASE": 1, "MISSING": 2}
        diff_items.sort(key=lambda x: triage_order.get(x["triage"], 9))

        # ── render ────────────────────────────────────────────────
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
            n_no_change=counts["NO_CHANGE"],
            alignment_pairs=sorted(
                alignment_map.pairs,
                key=lambda p: (p.state.value, p.legacy_id or p.modernized_id or "")
            ),
            analyses_sorted=analyses_sorted,
            diff_items=diff_items,
        )
        return html

    def save(self, html: str, out: str | Path):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        logger.info("Saved → %s", out)

    def save_json(
        self,
        llm_result: LLMResult,
        triage_result: TriageResult,
        alignment_map: AlignmentMap,
        out: str | Path,
    ):
        out = Path(out)
        counts = {t.value: 0 for t in TriageLabel}
        for p in triage_result.pairs:
            counts[p.triage.value] += 1

        artifact = DiffArtifact(
            generated_at=datetime.now().isoformat(),
            legacy_doc="legacy_policy.pdf",
            modernized_doc="modernized_policy.pdf",
            triage_summary=TriageSummary(
                no_change=counts["NO_CHANGE"],
                paraphrase=counts["PARAPHRASE"],
                substantive=counts["SUBSTANTIVE"],
                missing=counts["MISSING"],
            ),
            analyses=llm_result.analyses,
            missing_chunks=[
                p for p in triage_result.pairs
                if p.triage == TriageLabel.MISSING
            ],
        )

        enc = msgspec.json.Encoder()
        out.write_bytes(enc.encode(artifact))
        logger.info("Saved → %s", out)


# ── run ───────────────────────────────────────────────────────────

reporter = ReportGenerator()
html     = reporter.generate(alignment_map, triage_result, llm_result)

reporter.save(html, "output/report.html")
reporter.save_json(llm_result, triage_result, alignment_map, "output/diff_artifact.json")

print("✅ output/report.html")
print("✅ output/diff_artifact.json")