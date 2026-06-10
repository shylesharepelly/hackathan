Cell 1  : Install Packages
Cell 2  : Imports
Cell 3  : Parse PDF using Docling
Cell 4  : Build Section Index
Cell 5  : Align Sections
Cell 6  : Detect Changes
Cell 7  : Retrieve Changed Content
Cell 8  : Qwen Semantic Diff
Cell 9  : Regulatory Impact Analysis
Cell 10 : Dependency Graph
Cell 11 : Final Report



!pip install docling
!pip install pandas
!pip install networkx
!pip install matplotlib
!pip install rapidfuzz
!pip install requests

ollama pull qwen3
ollama serve

import re
import json
import requests
import pandas as pd

from rapidfuzz import fuzz
from docling.document_converter import DocumentConverter

import networkx as nx
import matplotlib.pyplot as plt


def parse_pdf(pdf_path):

    converter = DocumentConverter()

    result = converter.convert(pdf_path)

    markdown = result.document.export_to_markdown()

    return markdown


legacy_text = parse_pdf("legacy.pdf")
modern_text = parse_pdf("modern.pdf")

def build_section_index(text):

    sections = {}

    current_title = "INTRO"

    buffer = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):

            if buffer:
                sections[current_title] = "\n".join(buffer)

            current_title = line

            buffer = []

        else:
            buffer.append(line)

    sections[current_title] = "\n".join(buffer)

    return sections


legacy_index = build_section_index(
    legacy_text
)

modern_index = build_section_index(
    modern_text
)


def align_sections(
        legacy_index,
        modern_index):

    mappings = []

    modern_titles = list(
        modern_index.keys()
    )

    for old_title in legacy_index:

        best_match = None

        best_score = 0

        for new_title in modern_titles:

            score = fuzz.ratio(
                old_title,
                new_title
            )

            if score > best_score:

                best_score = score

                best_match = new_title

        mappings.append({
            "legacy": old_title,
            "modern": best_match,
            "score": best_score
        })

    return pd.DataFrame(mappings)


aligned = align_sections(
    legacy_index,
    modern_index
)

aligned.head()

def detect_changes(
        aligned_df,
        threshold=80):

    changes = []

    for _, row in aligned_df.iterrows():

        if row["score"] < threshold:

            changes.append(row)

    return pd.DataFrame(changes)


changes_df = detect_changes(
    aligned
)

changes_df


def retrieve_changed_sections(
        changes_df,
        legacy_index,
        modern_index):

    payload = []

    for _, row in changes_df.iterrows():

        payload.append({

            "legacy_title":
                row["legacy"],

            "modern_title":
                row["modern"],

            "legacy_text":
                legacy_index[
                    row["legacy"]
                ],

            "modern_text":
                modern_index[
                    row["modern"]
                ]
        })

    return payload

changed_sections = retrieve_changed_sections(
    changes_df,
    legacy_index,
    modern_index
)

OLLAMA_URL = \
"http://localhost:11434/api/generate"


def qwen_compare(
        old_text,
        new_text):

    prompt = f"""
Compare these policy sections.

OLD:
{old_text}

NEW:
{new_text}

Return:

1. Added requirements
2. Removed requirements
3. Modified requirements
4. Summary
"""

    response = requests.post(

        OLLAMA_URL,

        json={
            "model":"qwen3",
            "prompt":prompt,
            "stream":False
        }
    )

    return response.json()["response"]

example

result = qwen_compare(
    changed_sections[0]["legacy_text"],
    changed_sections[0]["modern_text"]
)

print(result)

def regulatory_impact(
        diff_result):

    prompt = f"""
You are a compliance expert.

Analyze impact:

{diff_result}

Return:

- Risk Level
- Compliance Impact
- Recommended Actions
"""

    response = requests.post(

        OLLAMA_URL,

        json={
            "model":"qwen3",
            "prompt":prompt,
            "stream":False
        }
    )

    return response.json()["response"]


G = nx.DiGraph()

G.add_edge(
    "Customer Data",
    "Retention Policy"
)

G.add_edge(
    "Customer Data",
    "Encryption Policy"
)

G.add_edge(
    "Customer Data",
    "Access Control"
)


report = []

for item in changed_sections:

    diff = qwen_compare(
        item["legacy_text"],
        item["modern_text"]
    )

    impact = regulatory_impact(
        diff
    )

    report.append({

        "section":
            item["legacy_title"],

        "diff":
            diff,

        "impact":
            impact
    })


    