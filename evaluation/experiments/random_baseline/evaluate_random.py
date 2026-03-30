"""
Random Baseline Evaluation
===========================
Two baselines:
  1. Uniform Random: randomly sample pipelines from the full corpus.
  2. Problem-type Random: randomly sample from pipelines sharing the same problem type.

Uses the same evaluation CSV and Hit@K metrics as the FAISS experiment.
"""

import os
import json
import random
import sqlite3
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Seed for reproducibility
RANDOM_SEED = 42
NUM_TRIALS = 1000  # Monte Carlo trials for stable estimates


def load_pipeline_corpus(db_path: str):
    """Load all pipeline names and their problem types from SQLite."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT name, problem_type FROM pipelines").fetchall()
    conn.close()

    all_names = [r[0] for r in rows]
    # Build problem_type -> [pipeline_names] mapping
    pt_map = defaultdict(list)
    for name, pt in rows:
        if pt:
            # Normalise to lower for matching
            pt_map[pt.lower().strip()].append(name)
    return all_names, pt_map


def normalise_problem_type(query_pt: str) -> str:
    """Map CSV problem types (e.g., 'Tabular Regression') to DB types (e.g., 'tabular-regression')."""
    if not query_pt:
        return ""
    return query_pt.lower().strip().replace(" ", "-")


def evaluate_random(all_names, pt_map, df, k_values=[1, 3, 5, 10, 15], num_trials=NUM_TRIALS):
    """
    Monte Carlo evaluation for both random baselines.
    Returns average hit counts over num_trials trials.
    """
    total = len(df)
    rng = random.Random(RANDOM_SEED)

    # Accumulators
    uniform_hits = {k: 0.0 for k in k_values}
    pt_hits = {k: 0.0 for k in k_values}

    for trial in range(num_trials):
        trial_uniform = {k: 0 for k in k_values}
        trial_pt = {k: 0 for k in k_values}

        for _, row in df.iterrows():
            target = row["target_pipeline"]
            query_pt = normalise_problem_type(row.get("problem_type", ""))

            # 1. Uniform Random: sample 15 from all pipelines
            max_k = max(k_values)
            sample_uniform = rng.sample(all_names, min(max_k, len(all_names)))
            for k in k_values:
                if target in sample_uniform[:k]:
                    trial_uniform[k] += 1

            # 2. Problem-type Random: sample from same problem type
            pool = pt_map.get(query_pt, all_names)  # fall back to all if no match
            sample_pt = rng.sample(pool, min(max_k, len(pool)))
            for k in k_values:
                if target in sample_pt[:k]:
                    trial_pt[k] += 1

        for k in k_values:
            uniform_hits[k] += trial_uniform[k]
            pt_hits[k] += trial_pt[k]

    # Average over trials
    for k in k_values:
        uniform_hits[k] = uniform_hits[k] / num_trials
        pt_hits[k] = pt_hits[k] / num_trials

    return uniform_hits, pt_hits, total


def generate_report(total, uniform_hits, pt_hits, output_dir):
    """Generate statistics report."""
    k_values = sorted(uniform_hits.keys())

    stats_path = os.path.join(output_dir, "random_baseline_statistics.txt")
    with open(stats_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("RANDOM BASELINE EVALUATION STATISTICS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total competitions evaluated: {total}\n")
        f.write(f"Monte Carlo trials: {NUM_TRIALS}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("COMPARISON: Uniform Random vs Problem-type Random\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"{'Metric':<12} | {'Uniform Random':<25} | {'Problem-type Random':<25}\n")
        f.write("-" * 70 + "\n")

        for k in k_values:
            u_avg = uniform_hits[k]
            u_pct = 100 * u_avg / total if total > 0 else 0
            p_avg = pt_hits[k]
            p_pct = 100 * p_avg / total if total > 0 else 0

            f.write(f"Hit@{k:<7} | {u_avg:.1f}/{total} ({u_pct:.1f}%)      | {p_avg:.1f}/{total} ({p_pct:.1f}%)\n")

        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("NOTES\n")
        f.write("-" * 80 + "\n")
        f.write("- Uniform Random: samples k pipelines uniformly from all 107 pipelines.\n")
        f.write("- Problem-type Random: samples k pipelines from the same problem type.\n")
        f.write(f"- Results averaged over {NUM_TRIALS} Monte Carlo trials for stability.\n")
        f.write("- These baselines establish the lower bound for retrieval performance.\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Statistics report written to: {stats_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    csv_path = os.path.join(parent_dir, "faiss recommendation", "faiss_evaluation.csv")
    db_path = os.path.join(parent_dir, "..", "slego_kb.sqlite")

    # Resolve to absolute
    db_path = os.path.abspath(db_path)

    print(f"Database: {db_path}")
    print(f"CSV file: {csv_path}")

    # Load data
    all_names, pt_map = load_pipeline_corpus(db_path)
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(all_names)} pipelines from DB")
    print(f"Loaded {len(df)} evaluation queries from CSV")
    print(f"Problem types in DB: {list(pt_map.keys())}")
    print(f"\nRunning {NUM_TRIALS} Monte Carlo trials...")

    uniform_hits, pt_hits, total = evaluate_random(all_names, pt_map, df)

    # Print results
    print("\n" + "=" * 80)
    print("RANDOM BASELINE RESULTS")
    print("=" * 80)
    for k in sorted(uniform_hits.keys()):
        u_pct = 100 * uniform_hits[k] / total
        p_pct = 100 * pt_hits[k] / total
        print(f"Hit@{k:<2}:  Uniform={u_pct:.1f}%  |  Problem-type={p_pct:.1f}%")

    generate_report(total, uniform_hits, pt_hits, script_dir)


if __name__ == "__main__":
    main()
