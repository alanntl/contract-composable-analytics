"""
LLM Reranking Evaluation
========================
Takes top-K FAISS results and uses Gemini 3 Flash to rerank to top 3.
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List

# Load .env file if exists
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(env_path)
except ImportError:
    pass

# Google GenAI imports
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Install google-genai: pip install google-genai")
    exit(1)


class LLMReranker:
    """Gemini 3 Flash based reranker to select top 3 from candidates."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3-flash-preview"

    def get_pipeline_details(self, pipeline_names: List[str]) -> List[dict]:
        """Fetch pipeline details from database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        details = []
        for name in pipeline_names:
            row = conn.execute("""
                SELECT name, description, problem_type, domain, specification, sample_input_schema
                FROM pipelines
                WHERE name = ?
            """, (name,)).fetchone()

            if row:
                details.append({
                    "name": row["name"],
                    "description": row["description"] or "",
                    "problem_type": row["problem_type"] or "",
                    "domain": row["domain"] or "",
                    "specification": row["specification"] or "",
                    "sample_input_schema": row["sample_input_schema"] or ""
                })
            else:
                details.append({"name": name, "description": "Not found", "problem_type": "", "domain": "", "specification": "", "sample_input_schema": ""})

        conn.close()
        return details

    def rerank(self, query: dict, candidate_names: List[str]) -> List[str]:
        """Rerank candidates and return top 3."""
        candidates = self.get_pipeline_details(candidate_names)

        # Build prompt - NO pipeline names shown to LLM
        system_prompt = """You are an expert ML pipeline recommender. Given a user's query and a list of candidate pipelines, select the 3 BEST matching pipelines.

IMPORTANT: Return your response in this EXACT JSON format:
{
    "top3": [1, 2, 3],
    "reasoning": "Brief explanation of why these 3 were selected"
}

Return the pipeline NUMBERS (1-15) of your top 3 choices.
Match pipelines based on how well their description aligns with the user's requirements."""

        # Format candidate info - NO NAMES, only numbers
        candidate_text = ""
        for i, c in enumerate(candidates, 1):
            candidate_text += f"""
Pipeline {i}:
- Problem Type: {c.get('problem_type', 'N/A')}
- Domain: {c.get('domain', 'N/A')}
- Description: {c.get('description', 'N/A')}
- Specification: {c.get('specification', 'N/A')}
- Data Input Schema: {c.get('sample_input_schema', 'N/A')}
"""

        user_prompt = f"""{system_prompt}

USER QUERY:
{json.dumps(query, indent=2)}

CANDIDATE PIPELINES (select best 3 by number):
{candidate_text}

Return JSON with top3 pipeline numbers and reasoning."""

        # Call Gemini 3 Flash
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_prompt)],
                ),
            ]

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
            )

            content = response.text

            # Parse JSON response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            result = json.loads(json_str)
            top3_numbers = result.get("top3", [])[:3]

            # Convert pipeline numbers to names
            valid_top3 = []
            for num in top3_numbers:
                try:
                    idx = int(num) - 1  # Convert 1-based to 0-based
                    if 0 <= idx < len(candidates):
                        valid_top3.append(candidates[idx]["name"])
                except (ValueError, TypeError):
                    pass

            # If some numbers invalid, fill with first candidates
            while len(valid_top3) < 3 and len(candidates) > len(valid_top3):
                for c in candidates:
                    if c["name"] not in valid_top3:
                        valid_top3.append(c["name"])
                        break

            return valid_top3[:3]

        except Exception as e:
            print(f"LLM error: {e}")
            # Fallback: return first 3 candidates
            return [c["name"] for c in candidates[:3]]


def generate_statistics_report(total: int, faiss_hits: dict, rerank_hits: dict, output_path: str):
    """Generate statistics comparing FAISS vs LLM rerank."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM RERANKING EVALUATION STATISTICS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: gemini-3-flash-preview\n")
        f.write(f"Total competitions evaluated: {total}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("COMPARISON: FAISS Top-3 vs LLM Reranked Top-3\n")
        f.write("-" * 80 + "\n")
        f.write("\n")

        f.write(f"{'Metric':<12} | {'FAISS Top-3':<20} | {'LLM Rerank':<20} | {'Improvement':<12}\n")
        f.write("-" * 70 + "\n")

        for k in [1, 3]:
            faiss_count = faiss_hits.get(k, 0)
            rerank_count = rerank_hits.get(k, 0)

            faiss_pct = 100 * faiss_count / total if total > 0 else 0
            rerank_pct = 100 * rerank_count / total if total > 0 else 0
            improvement = rerank_pct - faiss_pct

            faiss_str = f"{faiss_count}/{total} ({faiss_pct:.1f}%)"
            rerank_str = f"{rerank_count}/{total} ({rerank_pct:.1f}%)"
            imp_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"

            f.write(f"Hit@{k:<7} | {faiss_str:<20} | {rerank_str:<20} | {imp_str:<12}\n")

        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("DETAILED ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write("\n")

        f.write(f"FAISS Hit@3:     {faiss_hits.get(3, 0)} hits, {total - faiss_hits.get(3, 0)} misses\n")
        f.write(f"LLM Rerank Hit@3: {rerank_hits.get(3, 0)} hits, {total - rerank_hits.get(3, 0)} misses\n")
        f.write("\n")

        f.write("Note: LLM reranker uses FAISS top-15 as input candidates.\n")
        f.write("Maximum possible Hit@3 is bounded by FAISS Hit@15.\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Statistics report written to: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Input: FAISS evaluation results
    faiss_csv_path = os.path.join(parent_dir, "faiss recommendation", "faiss_evaluation.csv")

    # Output files
    output_csv_path = os.path.join(script_dir, "llm_rerank_evaluation.csv")
    stats_path = os.path.join(script_dir, "llm_rerank_statistics.txt")

    # Database path
    db_path = "/Users/an/Documents/kaggle_pipeline/app/slego_kb.sqlite"

    print(f"Input FAISS results: {faiss_csv_path}")
    print(f"Database: {db_path}")

    # Read FAISS results
    df = pd.read_csv(faiss_csv_path)
    print(f"Loaded {len(df)} competitions")

    # Check for API key
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not gemini_key:
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        return

    print("Using LLM: gemini-3-flash-preview")

    # Initialize reranker
    reranker = LLMReranker(db_path)

    # Prepare output dataframe
    output_rows = []

    # Metrics
    faiss_hits = {1: 0, 3: 0}
    rerank_hits = {1: 0, 3: 0}
    total = len(df)

    for i, row in df.iterrows():
        competition = row.get("competition_task", "")
        target = row.get("target_pipeline", "")
        guided_query_str = row.get("guided_query", "")
        problem_type = row.get("problem_type", "")

        # Parse guided query
        try:
            query = json.loads(guided_query_str) if guided_query_str else {}
        except:
            query = {"task_goal": competition, "problem_type": problem_type}

        # Get FAISS top 15 candidates (using Gemini FAISS results)
        faiss_candidates = []
        for j in range(1, 16):
            col = f"gemini_faiss Rank{j}"
            if col in row and pd.notna(row[col]):
                faiss_candidates.append(row[col])

        # Get FAISS top 3 for comparison
        faiss_top3 = faiss_candidates[:3]

        # Calculate FAISS hits
        if target in faiss_top3[:1]:
            faiss_hits[1] += 1
        if target in faiss_top3:
            faiss_hits[3] += 1

        # Run LLM reranking
        if faiss_candidates:
            try:
                rerank_top3 = reranker.rerank(query, faiss_candidates[:15])
            except Exception as e:
                print(f"Rerank error for {competition}: {e}")
                rerank_top3 = faiss_top3
        else:
            rerank_top3 = []

        # Pad to 3 if needed
        while len(rerank_top3) < 3:
            rerank_top3.append("")

        # Calculate rerank hits
        if target in rerank_top3[:1]:
            rerank_hits[1] += 1
        if target in rerank_top3:
            rerank_hits[3] += 1

        # Find raw rank in reranked list
        try:
            rerank_raw = rerank_top3.index(target) + 1 if target in rerank_top3 else ""
        except:
            rerank_raw = ""

        # Build output row
        output_row = {
            "competition_task": competition,
            "problem_type": problem_type,
            "guided_query": guided_query_str,
            "target_pipeline": target,
            "gemini_faiss_rank1": faiss_top3[0] if len(faiss_top3) > 0 else "",
            "gemini_faiss_rank2": faiss_top3[1] if len(faiss_top3) > 1 else "",
            "gemini_faiss_rank3": faiss_top3[2] if len(faiss_top3) > 2 else "",
            "llm_rerank_rank1": rerank_top3[0],
            "llm_rerank_rank2": rerank_top3[1],
            "llm_rerank_rank3": rerank_top3[2],
            "llm_rerank_raw": rerank_raw
        }
        output_rows.append(output_row)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total}")

    # Print summary
    print("\n" + "=" * 80)
    print("LLM RERANKING RESULTS (gemini-3-flash-preview)")
    print("=" * 80)
    print(f"Total competitions: {total}")
    print()
    print("FAISS Top-3:")
    print(f"  Hit@1: {faiss_hits[1]}/{total} ({100*faiss_hits[1]/total:.1f}%)")
    print(f"  Hit@3: {faiss_hits[3]}/{total} ({100*faiss_hits[3]/total:.1f}%)")
    print()
    print("LLM Reranked Top-3:")
    print(f"  Hit@1: {rerank_hits[1]}/{total} ({100*rerank_hits[1]/total:.1f}%)")
    print(f"  Hit@3: {rerank_hits[3]}/{total} ({100*rerank_hits[3]/total:.1f}%)")

    # Save results
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nResults written to: {output_csv_path}")

    # Generate statistics report
    generate_statistics_report(total, faiss_hits, rerank_hits, stats_path)


if __name__ == "__main__":
    main()
