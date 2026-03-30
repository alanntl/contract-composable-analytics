"""
BM25 Baseline Evaluation
=========================
Uses BM25 (Okapi BM25) to rank pipelines by text relevance.
Same evaluation CSV and Hit@K metrics as the FAISS experiment.
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from rank_bm25 import BM25Okapi
import re


def tokenize(text: str) -> list:
    """Simple whitespace + lowercase tokenizer with basic cleaning."""
    text = text.lower()
    # Remove punctuation except hyphens
    text = re.sub(r'[^\w\s-]', ' ', text)
    tokens = text.split()
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


def load_pipelines(db_path: str):
    """Load pipeline names, descriptions, and metadata from SQLite."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT name, problem_type, task_goal, description, domain
        FROM pipelines
        ORDER BY name
    """).fetchall()
    conn.close()

    pipelines = []
    for name, pt, tg, desc, domain in rows:
        parts = []
        if tg:
            parts.append(tg)
        if desc:
            parts.append(desc)
        if pt:
            parts.append(pt.replace("-", " "))
        if domain:
            parts.append(domain)
        text = " ".join(parts) if parts else name
        pipelines.append({"name": name, "text": text})

    return pipelines


def build_query_text(row):
    """Extract query text from guided_query JSON or fallback to competition name."""
    guided_query = row.get("guided_query", "")
    if guided_query and isinstance(guided_query, str) and guided_query.strip():
        try:
            q = json.loads(guided_query)
            parts = []
            for field in ["task_goal", "data_context", "problem_type", "domain_keywords"]:
                val = q.get(field, "")
                if val:
                    parts.append(str(val))
            if parts:
                return " ".join(parts)
        except (json.JSONDecodeError, TypeError):
            return guided_query
    return row.get("competition_task", "")


def evaluate_bm25(pipelines, df, k_values=[1, 3, 5, 10, 15]):
    """
    Build BM25 index from pipeline corpus, then for each query compute
    BM25 scores and rank pipelines. Compute Hit@K.
    """
    total = len(df)
    pipeline_names = [p["name"] for p in pipelines]
    pipeline_tokens = [tokenize(p["text"]) for p in pipelines]

    # Build BM25 index
    bm25 = BM25Okapi(pipeline_tokens)

    hit_counts = {k: 0 for k in k_values}
    results = []

    for i, (_, row) in enumerate(df.iterrows()):
        target = row["target_pipeline"]
        query_text = build_query_text(row)
        query_tokens = tokenize(query_text)

        # Get BM25 scores
        scores = bm25.get_scores(query_tokens)

        # Rank by descending score
        ranked_indices = np.argsort(scores)[::-1]
        ranked_names = [pipeline_names[idx] for idx in ranked_indices]

        # Compute hits
        for k in k_values:
            if target in ranked_names[:k]:
                hit_counts[k] += 1

        # Store top-15 for output
        result_row = {
            "competition_task": row["competition_task"],
            "target_pipeline": target,
            "query_text": query_text[:200],
        }
        for j in range(min(15, len(ranked_names))):
            result_row[f"bm25_rank{j+1}"] = ranked_names[j]

        # Find rank of target
        try:
            rank = ranked_names.index(target) + 1
            result_row["target_rank"] = rank
        except ValueError:
            result_row["target_rank"] = ""

        results.append(result_row)

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{total}")

    return hit_counts, total, results


def generate_report(total, hit_counts, output_dir):
    """Generate statistics report."""
    stats_path = os.path.join(output_dir, "bm25_baseline_statistics.txt")
    k_values = sorted(hit_counts.keys())

    with open(stats_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BM25 (OKAPI) BASELINE EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total competitions evaluated: {total}\n")
        f.write(f"Algorithm: BM25Okapi (rank_bm25 library, default k1=1.5, b=0.75)\n")
        f.write(f"Tokenizer: Lowercase, punctuation removal, min length 2\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("RETRIEVAL PERFORMANCE\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"{'Metric':<12} | {'Hits':<15} | {'Rate':<10}\n")
        f.write("-" * 45 + "\n")

        for k in k_values:
            count = hit_counts[k]
            pct = 100 * count / total if total > 0 else 0
            f.write(f"Hit@{k:<7} | {count}/{total:<10} | {pct:.1f}%\n")

        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("NOTES\n")
        f.write("-" * 80 + "\n")
        f.write("- Query text: task_goal + data_context + problem_type + domain_keywords from guided_query.\n")
        f.write("- Corpus text: task_goal + description + problem_type + domain from pipelines table.\n")
        f.write("- BM25 is a probabilistic bag-of-words retrieval model (no semantic understanding).\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Statistics report written to: {stats_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    csv_path = os.path.join(parent_dir, "faiss recommendation", "faiss_evaluation.csv")
    db_path = os.path.abspath(os.path.join(parent_dir, "..", "slego_kb.sqlite"))

    print(f"Database: {db_path}")
    print(f"CSV file: {csv_path}")

    # Load data
    pipelines = load_pipelines(db_path)
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(pipelines)} pipelines from DB")
    print(f"Loaded {len(df)} evaluation queries from CSV")

    # Run evaluation
    hit_counts, total, results = evaluate_bm25(pipelines, df)

    # Print results
    print("\n" + "=" * 80)
    print("BM25 (OKAPI) RESULTS")
    print("=" * 80)
    print(f"Total competitions: {total}\n")

    for k in sorted(hit_counts.keys()):
        pct = 100 * hit_counts[k] / total
        print(f"Hit@{k:<2}: {hit_counts[k]}/{total} ({pct:.1f}%)")

    # Save results CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(script_dir, "bm25_evaluation.csv"), index=False)
    print(f"\nResults written to: {os.path.join(script_dir, 'bm25_evaluation.csv')}")

    # Generate report
    generate_report(total, hit_counts, script_dir)


if __name__ == "__main__":
    main()
