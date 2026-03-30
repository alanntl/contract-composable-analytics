"""
FAISS Evaluation for faiss_evaluation.csv
==========================================
Adapted from run_evaluation_faiss_only.py
Uses guided queries from llm_recommendation_evaluation.csv
"""

import os
import csv
import json
import sqlite3
import numpy as np

# Load .env file if exists
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(env_path)
except ImportError:
    pass

try:
    import faiss
except ImportError:
    print("Install faiss: pip install faiss-cpu")
    exit(1)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
except ImportError:
    genai = None


class FAISSOnlyRecommender:
    def __init__(self, db_path: str, provider: str, api_key: str):
        self.db_path = db_path
        self.provider = provider.lower()
        self.pipelines = []

        if self.provider == "openai":
            self.embed_model = "text-embedding-3-small"
            self.embed_dim = 1536
            self.table_name = "pipeline_embeddings_openai"
            self.client = OpenAI(api_key=api_key)
        else:
            self.embed_model = "models/text-embedding-004"
            self.embed_dim = 768
            self.table_name = "pipeline_embeddings_gemini"
            self.client = genai.Client(api_key=api_key)

        self._load_data()

    def _load_data(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"""
            SELECT p.id, p.name, p.problem_type, p.task_goal, p.description,
                   pe.description_embedding, pe.task_goal_embedding
            FROM pipelines p
            JOIN {self.table_name} pe ON p.id = pe.pipeline_id
            WHERE pe.description_embedding IS NOT NULL
        """).fetchall()
        conn.close()

        # Collect unique problem types to embed
        unique_problem_types = set()
        for row in rows:
            if row["problem_type"]:
                unique_problem_types.add(row["problem_type"])

        # Embed all unique problem types
        print(f"[{self.provider.upper()}] Embedding {len(unique_problem_types)} unique problem types...")
        problem_type_embeddings = {}
        for pt in unique_problem_types:
            problem_type_embeddings[pt] = self._embed(pt)

        for row in rows:
            desc_vec = np.frombuffer(row["description_embedding"], dtype=np.float32)
            norm = np.linalg.norm(desc_vec)
            if norm > 0:
                desc_vec = desc_vec / norm

            task_vec = np.frombuffer(row["task_goal_embedding"], dtype=np.float32)
            norm = np.linalg.norm(task_vec)
            if norm > 0:
                task_vec = task_vec / norm

            # Get pre-computed problem_type embedding
            pt_vec = problem_type_embeddings.get(row["problem_type"])
            if pt_vec is None:
                pt_vec = np.zeros(self.embed_dim, dtype=np.float32)

            self.pipelines.append({
                "name": row["name"],
                "problem_type": row["problem_type"],
                "task_goal": row["task_goal"],
                "description": row["description"],
                "desc_embedding": desc_vec,
                "task_embedding": task_vec,
                "problem_type_embedding": pt_vec
            })

        print(f"[{self.provider.upper()}] Loaded {len(self.pipelines)} pipelines")

    def _embed(self, text: str) -> np.ndarray:
        if self.provider == "openai":
            result = self.client.embeddings.create(input=text, model=self.embed_model)
            vec = np.array(result.data[0].embedding, dtype=np.float32)
        else:
            from google.genai import types
            result = self.client.models.embed_content(
                model=self.embed_model,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            vec = np.array(result.embeddings[0].values, dtype=np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def recommend(self, query: str, problem_type: str = None, k: int = 15,
                  problem_type_weight: float = 0.3, description_weight: float = 0.7):
        """
        Multi-index FAISS:
        Combines scores from problem_type and description indices with configurable weights.
        Final score = (problem_type_weight * pt_score) + (description_weight * desc_score)
        """
        n = len(self.pipelines)

        # Build description index
        desc_vectors = np.stack([p["desc_embedding"] for p in self.pipelines])
        desc_index = faiss.IndexFlatIP(desc_vectors.shape[1])
        desc_index.add(desc_vectors)

        # Search description index
        query_vec = self._embed(query).reshape(1, -1)
        desc_scores, desc_indices = desc_index.search(query_vec, k=n)

        # Create score dictionary for description (indexed by pipeline position)
        desc_score_map = {int(idx): float(score) for idx, score in zip(desc_indices[0], desc_scores[0])}

        # If problem_type provided, also search problem_type index
        if problem_type:
            pt_vectors = np.stack([p["problem_type_embedding"] for p in self.pipelines])
            pt_index = faiss.IndexFlatIP(pt_vectors.shape[1])
            pt_index.add(pt_vectors)

            query_pt_vec = self._embed(problem_type).reshape(1, -1)
            pt_scores, pt_indices = pt_index.search(query_pt_vec, k=n)

            # Create score dictionary for problem_type
            pt_score_map = {int(idx): float(score) for idx, score in zip(pt_indices[0], pt_scores[0])}

            # Combine scores with weights
            combined_scores = []
            for i in range(n):
                pt_score = pt_score_map.get(i, 0.0)
                d_score = desc_score_map.get(i, 0.0)
                combined = (problem_type_weight * pt_score) + (description_weight * d_score)
                combined_scores.append((i, combined))
        else:
            # No problem_type, use description scores only
            combined_scores = [(i, desc_score_map.get(i, 0.0)) for i in range(n)]

        # Sort by combined score descending
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top K pipeline names
        return [self.pipelines[idx]["name"] for idx, _ in combined_scores[:k]]


def run_evaluation(rec, rows, provider: str):
    """
    Evaluate FAISS recommendations using guided queries and return updated rows.
    """
    # Column name prefix based on provider
    prefix = f"{provider}_faiss"
    raw_col = f"{provider}_raw_faiss"

    # Metrics
    hit_counts = {k: 0 for k in [1, 3, 5, 10, 15]}
    total = len(rows)

    for i, row in enumerate(rows):
        # Support both column naming conventions
        competition = row.get("competition_task") or row.get("Competition", "")
        target = row.get("target_pipeline") or row.get("Traget pipeline", "")
        guided_query = row.get("guided_query", "")
        problem_type = row.get("problem_type") or row.get("Task Classification", "")

        # Parse guided query if available
        # Note: Excludes data_context and input_data_path to focus on task semantics
        if guided_query and isinstance(guided_query, str) and guided_query.strip():
            try:
                query_data = json.loads(guided_query)
                query_text = f"{query_data.get('task_goal', '')} {query_data.get('domain_keywords', '')}"
                problem_type = query_data.get("problem_type", "") or problem_type
            except (json.JSONDecodeError, TypeError):
                query_text = guided_query
        else:
            # Fallback to competition name
            query_text = competition

        # Get top 15 recommendations
        top15 = rec.recommend(query_text, problem_type=problem_type, k=15)

        # Find rank of target in results (1-indexed, or empty if not found)
        try:
            rank = top15.index(target) + 1
            row[raw_col] = str(rank)
        except ValueError:
            row[raw_col] = ""

        # Fill rank columns
        for j, rec_name in enumerate(top15):
            row[f"{prefix} Rank{j+1}"] = rec_name

        # Fill remaining columns if less than 15 results
        for j in range(len(top15), 15):
            row[f"{prefix} Rank{j+1}"] = ""

        # Calculate hits
        for k in hit_counts.keys():
            if target in top15[:k]:
                hit_counts[k] += 1

        if (i + 1) % 10 == 0:
            print(f"[{provider.upper()}] Processed {i + 1}/{total}")

    # Print metrics
    print("\n" + "=" * 80)
    print(f"FAISS Evaluation Results ({provider.upper()}) - Using Guided Queries")
    print("=" * 80)
    print(f"Total competitions: {total}")
    print()

    for k in [1, 3, 5, 10, 15]:
        pct = 100 * hit_counts[k] / total if total > 0 else 0
        print(f"Hit@{k:<2}: {hit_counts[k]}/{total} ({pct:.1f}%)")

    return rows, hit_counts


def generate_statistics_report(total: int, gemini_hits: dict, openai_hits: dict, output_path: str):
    """
    Generate a statistics text file with evaluation results table.
    """
    from datetime import datetime

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FAISS EVALUATION STATISTICS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total competitions evaluated: {total}\n")
        f.write("\n")

        # Comparison Table
        f.write("-" * 80 + "\n")
        f.write("COMPARISON TABLE: Hit Rate by Provider\n")
        f.write("-" * 80 + "\n")
        f.write("\n")

        # Table header
        f.write(f"{'Metric':<12} | {'Gemini':<20} | {'OpenAI':<20} | {'Winner':<10}\n")
        f.write("-" * 70 + "\n")

        k_values = [1, 3, 5, 10, 15]
        for k in k_values:
            gemini_count = gemini_hits.get(k, 0) if gemini_hits else 0
            openai_count = openai_hits.get(k, 0) if openai_hits else 0

            gemini_pct = 100 * gemini_count / total if total > 0 else 0
            openai_pct = 100 * openai_count / total if total > 0 else 0

            gemini_str = f"{gemini_count}/{total} ({gemini_pct:.1f}%)" if gemini_hits else "N/A"
            openai_str = f"{openai_count}/{total} ({openai_pct:.1f}%)" if openai_hits else "N/A"

            if gemini_hits and openai_hits:
                if gemini_count > openai_count:
                    winner = "Gemini"
                elif openai_count > gemini_count:
                    winner = "OpenAI"
                else:
                    winner = "Tie"
            else:
                winner = "-"

            f.write(f"Hit@{k:<7} | {gemini_str:<20} | {openai_str:<20} | {winner:<10}\n")

        f.write("\n")

        # Summary Statistics
        f.write("-" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write("\n")

        if gemini_hits:
            gemini_avg = sum(gemini_hits.values()) / len(gemini_hits) / total * 100 if total > 0 else 0
            f.write(f"Gemini Average Hit Rate: {gemini_avg:.1f}%\n")
            f.write(f"  - Best performance: Hit@15 with {gemini_hits.get(15, 0)}/{total} ({100*gemini_hits.get(15,0)/total:.1f}%)\n")

        if openai_hits:
            openai_avg = sum(openai_hits.values()) / len(openai_hits) / total * 100 if total > 0 else 0
            f.write(f"OpenAI Average Hit Rate: {openai_avg:.1f}%\n")
            f.write(f"  - Best performance: Hit@15 with {openai_hits.get(15, 0)}/{total} ({100*openai_hits.get(15,0)/total:.1f}%)\n")

        f.write("\n")

        # Detailed breakdown by K
        f.write("-" * 80 + "\n")
        f.write("DETAILED BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        f.write("\n")

        for k in k_values:
            f.write(f"Hit@{k}:\n")
            if gemini_hits:
                gemini_count = gemini_hits.get(k, 0)
                gemini_miss = total - gemini_count
                f.write(f"  Gemini: {gemini_count} hits, {gemini_miss} misses\n")
            if openai_hits:
                openai_count = openai_hits.get(k, 0)
                openai_miss = total - openai_count
                f.write(f"  OpenAI: {openai_count} hits, {openai_miss} misses\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Statistics report written to: {output_path}")


def main():
    import pandas as pd

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "faiss_evaluation.csv")
    stats_path = os.path.join(script_dir, "faiss_evaluation_statistics.txt")

    # Database path
    db_path = "/Users/an/Documents/kaggle_pipeline/app/slego_kb.sqlite"

    print(f"Using database: {db_path}")
    print(f"CSV file: {csv_path}")

    # Read faiss_evaluation CSV
    df = pd.read_csv(csv_path)
    rows = df.to_dict('records')
    total = len(rows)

    print(f"Loaded {total} competitions")

    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    gemini_hits = None
    openai_hits = None

    # Run Gemini evaluation
    if gemini_key:
        print("\n[GEMINI] Initializing...")
        try:
            rec_gemini = FAISSOnlyRecommender(db_path, "gemini", gemini_key)
            rows, gemini_hits = run_evaluation(rec_gemini, rows, "gemini")
        except Exception as e:
            print(f"[GEMINI] Error: {e}")
            import traceback
            traceback.print_exc()

    # Run OpenAI evaluation
    if openai_key:
        print("\n[OPENAI] Initializing...")
        try:
            rec_openai = FAISSOnlyRecommender(db_path, "openai", openai_key)
            rows, openai_hits = run_evaluation(rec_openai, rows, "openai")
        except Exception as e:
            print(f"[OPENAI] Error: {e}")
            import traceback
            traceback.print_exc()

    if not openai_key and not gemini_key:
        print("Set OPENAI_API_KEY or GOOGLE_API_KEY environment variable")
        return

    # Write results back to CSV
    result_df = pd.DataFrame(rows)
    result_df.to_csv(csv_path, index=False)
    print(f"\nResults written to: {csv_path}")

    # Generate statistics report
    generate_statistics_report(total, gemini_hits, openai_hits, stats_path)


if __name__ == "__main__":
    main()
