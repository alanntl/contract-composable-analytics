"""
Coleridge Initiative: Show Us the Data - SLEGO Services
========================================================
Competition: https://www.kaggle.com/competitions/coleridgeinitiative-show-us-the-data
Problem Type: NLP Text Extraction (dataset mention identification)
Target: PredictionString (pipe-delimited dataset labels found in paper)
ID Column: Id

Task: Identify mentions of datasets within scientific publications.
Predictions should be cleaned dataset label strings found in the paper text.
Metric: Jaccard similarity between predicted and actual dataset labels.

Key Insights from top solution notebooks:
- Solution 1 (srcecde): MinHash LSH for fuzzy string matching of labels in text
- Solution 2 (jamesmcguigan): HuggingFace QA model for extracting dataset references
- Solution 3 (mohitduklan): EDA showing 45 unique dataset_titles, heavy label reuse

The best practical approach: build a string-matching pipeline that:
1. Extracts full text from each test publication JSON
2. Builds a dictionary of known dataset labels from training data
3. Searches for exact and fuzzy matches of labels in paper text
4. Outputs pipe-delimited matched labels per paper

Services follow G1-G6 design principles (reusable, parameterized).
"""

import os
import sys
import json
import re
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# SERVICE 1: EXTRACT PAPER TEXTS FROM JSON FILES
# =============================================================================

@contract(
    inputs={
        "paper_dir": {"format": "directory", "required": True},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Read JSON publication files and extract full text per paper",
    tags=["io", "text", "nlp", "coleridge", "generic"],
    version="1.0.0",
)
def extract_paper_texts(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
    text_column: str = "full_text",
    include_section_titles: bool = True,
) -> str:
    """
    Read JSON publication files from a directory.
    Each JSON file is a list of dicts with 'section_title' and 'text' keys.
    Concatenates all sections into a single full_text per paper.

    Parameters:
        id_column: Name for the paper ID column (derived from filename)
        text_column: Name for the concatenated text column
        include_section_titles: Prepend section titles to each section's text
    """
    paper_dir = inputs["paper_dir"]
    rows = []

    for fname in sorted(os.listdir(paper_dir)):
        if not fname.endswith(".json"):
            continue

        paper_id = fname.replace(".json", "")
        fpath = os.path.join(paper_dir, fname)

        with open(fpath, "r") as f:
            sections = json.load(f)

        texts = []
        for section in sections:
            sec_title = section.get("section_title", "")
            sec_text = section.get("text", "")
            if include_section_titles and sec_title:
                texts.append(f"{sec_title} {sec_text}")
            else:
                texts.append(sec_text)

        full_text = " ".join(texts)
        rows.append({id_column: paper_id, text_column: full_text})

    df = pd.DataFrame(rows)
    _save_data(df, outputs["data"])

    return f"extract_paper_texts: extracted {len(df)} papers from {paper_dir}"


# =============================================================================
# SERVICE 2: BUILD DATASET LABEL INDEX
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "label_index": {"format": "pickle", "schema": {"type": "artifact"}},
    },
    description="Build a lookup index of known dataset labels from training data",
    tags=["preprocessing", "nlp", "coleridge", "generic"],
    version="1.0.0",
)
def build_label_index(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "cleaned_label",
    dataset_title_column: str = "dataset_title",
    dataset_label_column: str = "dataset_label",
) -> str:
    """
    Build a lookup index of known dataset labels from the training CSV.
    Creates a set of all known cleaned labels plus variations.

    Parameters:
        label_column: Column with cleaned dataset label text
        dataset_title_column: Column with dataset title
        dataset_label_column: Column with raw dataset label
    """
    df = _load_data(inputs["data"])

    # Build comprehensive label set
    labels = set()
    clean_to_original = {}

    for _, row in df.iterrows():
        cleaned = str(row.get(label_column, "")).strip()
        if cleaned and cleaned != "nan":
            labels.add(cleaned)

        # Also store original forms for matching
        title = str(row.get(dataset_title_column, "")).strip()
        if title and title != "nan":
            clean_title = _clean_text(title)
            labels.add(clean_title)
            clean_to_original[clean_title] = cleaned

        raw_label = str(row.get(dataset_label_column, "")).strip()
        if raw_label and raw_label != "nan":
            clean_raw = _clean_text(raw_label)
            labels.add(clean_raw)
            clean_to_original[clean_raw] = cleaned

    # Sort by length descending (prefer longer matches first)
    sorted_labels = sorted(labels, key=len, reverse=True)

    artifact = {
        "labels": sorted_labels,
        "clean_to_original": clean_to_original,
        "n_unique_labels": len(labels),
    }

    os.makedirs(os.path.dirname(outputs["label_index"]) or ".", exist_ok=True)
    with open(outputs["label_index"], "wb") as f:
        pickle.dump(artifact, f)

    return f"build_label_index: indexed {len(labels)} unique dataset labels"


# =============================================================================
# SERVICE 3: SEARCH DATASET MENTIONS IN PAPER TEXT
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "label_index": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Search for dataset label mentions in paper text using string matching",
    tags=["nlp", "text-matching", "coleridge", "generic"],
    version="1.0.0",
)
def search_dataset_mentions(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
    text_column: str = "full_text",
    prediction_column: str = "PredictionString",
    min_label_length: int = 5,
    case_sensitive: bool = False,
) -> str:
    """
    Search each paper's text for mentions of known dataset labels.
    Uses exact string matching on cleaned text.

    Parameters:
        id_column: Paper ID column
        text_column: Column with full paper text
        prediction_column: Output column for pipe-delimited predictions
        min_label_length: Minimum characters for a label to be considered
        case_sensitive: Whether matching is case-sensitive
    """
    df = _load_data(inputs["data"])

    with open(inputs["label_index"], "rb") as f:
        label_artifact = pickle.load(f)

    labels = label_artifact["labels"]
    clean_to_original = label_artifact.get("clean_to_original", {})

    results = []
    total_matches = 0

    for _, row in df.iterrows():
        paper_id = row[id_column]
        text = str(row.get(text_column, ""))
        clean_text_content = _clean_text(text)

        found_labels = set()
        for label in labels:
            if len(label) < min_label_length:
                continue

            # Check if label appears in cleaned text
            search_label = label if case_sensitive else label.lower()
            search_text = clean_text_content if case_sensitive else clean_text_content.lower()

            if search_label in search_text:
                # Map back to canonical cleaned_label if available
                canonical = clean_to_original.get(label, label)
                found_labels.add(canonical)

        prediction_str = "|".join(sorted(found_labels)) if found_labels else ""
        results.append({id_column: paper_id, prediction_column: prediction_str})
        total_matches += len(found_labels)

    result_df = pd.DataFrame(results)
    _save_data(result_df, outputs["submission"])

    # Save metrics
    n_with_matches = sum(1 for r in results if r[prediction_column])
    metrics = {
        "method": "string_matching",
        "n_papers": len(results),
        "n_with_matches": n_with_matches,
        "total_matches": total_matches,
        "avg_matches_per_paper": total_matches / max(len(results), 1),
        "match_rate": n_with_matches / max(len(results), 1),
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"search_dataset_mentions: found {total_matches} matches in {len(results)} papers ({n_with_matches} have matches)"


# =============================================================================
# HELPER: CLEAN TEXT (matches competition evaluation function)
# =============================================================================

def _clean_text(txt: str) -> str:
    """
    Clean text using the competition's official cleaning function.
    Strips non-alphanumeric characters, lowercases, and trims whitespace.
    """
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "extract_paper_texts": extract_paper_texts,
    "build_label_index": build_label_index,
    "search_dataset_mentions": search_dataset_mentions,
}
