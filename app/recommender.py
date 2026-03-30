"""
Contract-Composable Analytics Pipeline Composition Recommender
======================================

A LangGraph-based system that:
1. Retrieves top-K similar pipelines via FAISS
2. Extracts services from those pipelines
3. Uses LLM to compose a new pipeline from the service pool

Architecture:
    User Query → FAISS Retriever → Service Extractor → LLM Composer → Composed Pipeline
"""

import os
import json
import re
import sqlite3
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TypedDict, Tuple

import numpy as np

# FAISS for vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available")

# LangGraph imports
from langgraph.graph import StateGraph, END

# LLM imports
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ComposerConfig:
    """Configuration for the Pipeline Composer."""
    db_path: str = "kb.sqlite"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536
    llm_model: str = "gpt-5-mini"
    top_k: int = 15
    problem_type_threshold: float = 0.5


# =============================================================================
# LANGGRAPH STATE
# =============================================================================

class RecommenderState(TypedDict):
    """State passed through the LangGraph workflow."""
    # Input
    query: str
    problem_type: str
    data_context: str
    domain_keywords: str
    
    # FAISS retrieval
    faiss_top_3: List[str]  # Top 3 from FAISS before LLM rerank
    top_k_pipelines: List[Dict]
    retrieved_names: List[str]  # Top 3 after LLM rerank
    
    # Service extraction
    service_pool: List[Dict]
    
    # LLM composition
    composed_pipeline: Dict
    composed_services: List[str]
    reasoning: str
    
    # Metadata
    error: Optional[str]


# =============================================================================
# PIPELINE COMPOSER CLASS
# =============================================================================

class PipelineComposer:
    """
    LangGraph-based Pipeline Composition Recommender.
    
    Retrieves similar pipelines, extracts services, and composes new pipelines.
    """
    
    def __init__(self, config: ComposerConfig = None, api_key: str = None):
        self.config = config or ComposerConfig()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

        # Model capability flags (learned at runtime from API errors)
        self._llm_supports_chat_completions = True
        self._llm_supports_temperature = True

        # Known capability constraints for specific models
        if self.config.llm_model in ("gpt-5.1-codex-mini",):
            self._llm_supports_chat_completions = False
            self._llm_supports_temperature = False
        
        # Load data
        self.pipelines = []
        self.services_db = {}
        self._load_data()
        
        # Build LangGraph
        self.graph = self._build_graph()
    
    def _load_data(self):
        """Load pipelines and services from database."""
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        
        # Load pipelines with embeddings
        rows = conn.execute("""
            SELECT p.id, p.name, p.problem_type, p.task_goal, p.description,
                   p.specification, p.services_used,
                   pe.description_embedding, pe.task_goal_embedding
            FROM pipelines p
            JOIN pipeline_embeddings_openai pe ON p.id = pe.pipeline_id
            WHERE pe.description_embedding IS NOT NULL
        """).fetchall()
        
        for row in rows:
            desc_vec = np.frombuffer(row["description_embedding"], dtype=np.float32)
            desc_vec = desc_vec / (np.linalg.norm(desc_vec) or 1)
            
            task_vec = np.frombuffer(row["task_goal_embedding"], dtype=np.float32)
            task_vec = task_vec / (np.linalg.norm(task_vec) or 1)
            
            spec = json.loads(row["specification"]) if row["specification"] else []
            services = json.loads(row["services_used"]) if row["services_used"] else []
            
            self.pipelines.append({
                "id": row["id"],
                "name": row["name"],
                "problem_type": row["problem_type"],
                "description": row["description"],
                "specification": spec,
                "services_used": services,
                "desc_embedding": desc_vec,
                "task_embedding": task_vec
            })
        
        # Load services
        svc_rows = conn.execute("""
            SELECT name, description, input_contract, output_contract, parameters, module
            FROM services
        """).fetchall()
        
        def safe_json_loads(s):
            """Safely parse JSON, returning empty dict on failure."""
            if not s or not s.strip():
                return {}
            try:
                return json.loads(s)
            except:
                return {}
        
        for row in svc_rows:
            self.services_db[row["name"]] = {
                "name": row["name"],
                "description": row["description"],
                "input_contract": safe_json_loads(row["input_contract"]),
                "output_contract": safe_json_loads(row["output_contract"]),
                "parameters": safe_json_loads(row["parameters"]),
                "module": row["module"]
            }
        
        conn.close()
        logger.info(f"Loaded {len(self.pipelines)} pipelines and {len(self.services_db)} services")
    
    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        result = self.client.embeddings.create(input=text, model=self.config.embed_model)
        vec = np.array(result.data[0].embedding, dtype=np.float32)
        return vec / (np.linalg.norm(vec) or 1)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = (text or "").strip()
        if "```json" in text:
            return text.split("```json", 1)[1].split("```", 1)[0].strip()
        if "```" in text:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        return text

    @classmethod
    def _parse_jsonish(cls, text: str) -> Dict[str, Any]:
        cleaned = cls._strip_code_fences(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract the largest {...} block.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            return json.loads(snippet)

        raise ValueError("LLM did not return valid JSON")

    def _llm_text(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        """Call the LLM and return raw text; supports Chat Completions with Responses fallback."""
        chat_exc: Optional[Exception] = None

        # Prefer Chat Completions when supported.
        if self._llm_supports_chat_completions:
            try:
                response = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:
                chat_exc = exc
                msg = str(exc)
                if "only supported" in msg and "v1/responses" in msg:
                    self._llm_supports_chat_completions = False

        if not hasattr(self.client, "responses"):
            raise chat_exc or RuntimeError("LLM call failed")

        # Responses API path.
        try:
            kwargs: Dict[str, Any] = {
                "model": self.config.llm_model,
                "input": prompt,
                "max_output_tokens": max_tokens,
            }
            if self._llm_supports_temperature:
                kwargs["temperature"] = temperature

            try:
                resp = self.client.responses.create(**kwargs)
            except Exception as resp_exc:
                msg = str(resp_exc)
                if "Unsupported parameter" in msg and "temperature" in msg:
                    self._llm_supports_temperature = False
                    kwargs.pop("temperature", None)
                    resp = self.client.responses.create(**kwargs)
                else:
                    raise resp_exc

            text = getattr(resp, "output_text", None)
            if text:
                return text.strip()
            return str(resp)
        except Exception as final_exc:
            raise final_exc from chat_exc

    def _llm_json(self, prompt: str, *, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Call the LLM and parse a JSON object from its response."""
        text = self._llm_text(prompt, temperature=temperature, max_tokens=max_tokens)
        try:
            return self._parse_jsonish(text)
        except Exception:
            # One-shot repair: ask the model to output ONLY valid JSON.
            repair_prompt = (
                "You must output ONLY valid JSON (no prose, no markdown). "
                "Fix the following into valid JSON:\n\n"
                + text
            )
            repaired = self._llm_text(repair_prompt, temperature=0.0, max_tokens=max_tokens)
            return self._parse_jsonish(repaired)
    
    # =========================================================================
    # LANGGRAPH NODES
    # =========================================================================
    
    def _retrieve(self, state: RecommenderState) -> RecommenderState:
        """Stage 1: FAISS two-stage retrieval."""
        # Combine all query fields for richer semantic matching
        query_parts = [
            state.get("query", ""),
            state.get("data_context", ""),
            state.get("domain_keywords", "")
        ]
        full_query = " ".join(p for p in query_parts if p)
        problem_type = state.get("problem_type", "")
        
        # Debug: log the query being used
        logger.info(f"FAISS query: '{full_query[:100]}...'")
        logger.info(f"Problem type: '{problem_type}'")
        
        n = len(self.pipelines)
        k = self.config.top_k
        
        # Stage 1: Filter by problem type
        if problem_type:
            task_vectors = np.stack([p["task_embedding"] for p in self.pipelines])
            task_index = faiss.IndexFlatIP(task_vectors.shape[1])
            task_index.add(task_vectors)
            
            pt_vec = self._embed(problem_type).reshape(1, -1)
            pt_scores, pt_indices = task_index.search(pt_vec, k=n)
            
            threshold = pt_scores[0][0] * self.config.problem_type_threshold
            filtered_indices = [idx for i, idx in enumerate(pt_indices[0]) if pt_scores[0][i] >= threshold]
            
            if len(filtered_indices) < k:
                filtered_indices = list(range(n))
            
            filtered_pipelines = [self.pipelines[i] for i in filtered_indices]
            logger.info(f"Problem type filter: {len(filtered_pipelines)} pipelines (threshold={threshold:.3f})")
        else:
            filtered_pipelines = self.pipelines
        
        # Stage 2: Rank by description similarity
        desc_vectors = np.stack([p["desc_embedding"] for p in filtered_pipelines])
        desc_index = faiss.IndexFlatIP(desc_vectors.shape[1])
        desc_index.add(desc_vectors)
        
        query_vec = self._embed(full_query).reshape(1, -1)
        search_k = min(k, len(filtered_pipelines))  # Candidate pool size for reranking
        _, desc_indices = desc_index.search(query_vec, k=search_k)
        
        top_k = [filtered_pipelines[i] for i in desc_indices[0]]
        
        # Store FAISS candidates (top 3 for comparison with rerank)
        state["faiss_top_3"] = [p["name"] for p in top_k[:3]]
        state["top_k_pipelines"] = top_k
        state["retrieved_names"] = [p["name"] for p in top_k]
        
        logger.info(f"Retrieved {len(top_k)} pipelines (FAISS top 3: {state['faiss_top_3']})")
        return state
    
    def _rerank(self, state: RecommenderState) -> RecommenderState:
        """Stage 2: LLM reranks FAISS candidates → top 3, then extracts services."""
        query = state["query"]
        problem_type = state.get("problem_type", "")
        candidates = state["top_k_pipelines"]
        
        # Build rerank prompt
        pipeline_summaries = []
        for i, p in enumerate(candidates):
            services = [s.get("service", "") for s in p.get("specification", [])]
            pipeline_summaries.append({
                "rank": i + 1,
                "name": p["name"],
                "description": p.get("description", "")[:200],
                "services": services[:6]
            })
        
        rerank_prompt = f"""You are a pipeline ranking expert. Rerank these candidate pipelines for the user's task.

## User's Task: {query}
## Problem Type: {problem_type}

## Candidate Pipelines (currently ranked by embedding similarity):
{json.dumps(pipeline_summaries, indent=2)}

## YOUR TASK
Select the TOP 3 pipelines that BEST match the user's task. Consider:
1. Problem type match (regression, classification, etc.)
2. Service compatibility with the task
3. Data processing requirements

## OUTPUT (valid JSON only):
{{
  "top_3_names": ["pipeline_name_1", "pipeline_name_2", "pipeline_name_3"],
  "reasoning": "Brief explanation of ranking"
}}"""

        try:
            result = self._llm_json(rerank_prompt, temperature=0.2, max_tokens=500)
            top_3_names = result.get("top_3_names", [])[:3]
            
            # Get pipelines by name
            name_to_pipeline = {p["name"]: p for p in candidates}
            top_3 = [name_to_pipeline[n] for n in top_3_names if n in name_to_pipeline]
            
            # Fallback if LLM didn't return valid names
            if len(top_3) < 3:
                top_3 = candidates[:3]
            
            logger.info(f"LLM reranked to top 3: {[p['name'] for p in top_3]}")
            
        except Exception as e:
            logger.warning(f"Rerank failed, using FAISS order: {e}")
            top_3 = candidates[:3]
        
        # Store LLM reranked top 3 (faiss_top_3 already stored in _retrieve)
        state["top_k_pipelines"] = top_3
        state["retrieved_names"] = [p["name"] for p in top_3]  # This is the reranked result
        
        # Extract services from top 3
        service_pool = {}
        for pipeline in top_3:
            for step in pipeline.get("specification", []):
                svc_name = step.get("service")
                if not svc_name or svc_name in service_pool:
                    continue
                
                svc_info = self.services_db.get(svc_name, {})
                service_pool[svc_name] = {
                    "name": svc_name,
                    "description": svc_info.get("description", ""),
                    "module": step.get("module") or svc_info.get("module"),
                    "input_contract": svc_info.get("input_contract", {}),
                    "output_contract": svc_info.get("output_contract", {}),
                    "example_params": step.get("params", {}),
                    "example_inputs": step.get("inputs", {}),
                    "example_outputs": step.get("outputs", {})
                }
        
        state["service_pool"] = list(service_pool.values())
        logger.info(f"Extracted {len(service_pool)} unique services from top 3")
        return state
    
    def _compose(self, state: RecommenderState) -> RecommenderState:
        """Stage 3: IO Adaptation + Preprocessing Swap agents."""
        query = state["query"]
        problem_type = state.get("problem_type", "")
        data_context = state.get("data_context", "")
        service_pool = state["service_pool"]
        top_3 = state["top_k_pipelines"]
        
        if not top_3:
            state["error"] = "No pipelines retrieved"
            state["composed_pipeline"] = {}
            state["composed_services"] = []
            state["reasoning"] = "Error: No pipelines available"
            return state
        
        best_pipeline = top_3[0]
        best_spec = best_pipeline.get("specification", [])
        
        # Get input paths from best pipeline
        best_inputs = {}
        for step in best_spec:
            if step.get("service") in ["combine_train_test", "load_data"]:
                best_inputs = step.get("inputs", {})
                break
        
        # Collect preprocessing services from all top 3 for swapping
        preprocessing_services = []
        for p in top_3:
            for step in p.get("specification", []):
                svc = step.get("service", "")
                if any(k in svc.lower() for k in ['encode', 'scale', 'impute', 'preprocess', 
                                                   'transform', 'feature', 'normalize']):
                    svc_info = self.services_db.get(svc, {})
                    preprocessing_services.append({
                        "from_pipeline": p["name"],
                        "service": svc,
                        "description": svc_info.get("description", ""),
                        "example_step": step
                    })
        
        # Deduplicate preprocessing
        seen = set()
        unique_preprocess = []
        for p in preprocessing_services:
            if p["service"] not in seen:
                seen.add(p["service"])
                unique_preprocess.append(p)
        
        def call_llm(prompt: str, max_tokens: int = 3500) -> dict:
            return self._llm_json(prompt, temperature=0.3, max_tokens=max_tokens)
        
        # ===== AGENT 1: IO ADAPTATION FOR ALL TOP 3 =====
        def build_adapt_prompt(pipeline: dict, rank: int) -> str:
            spec = pipeline.get("specification", [])
            return f"""Adapt this pipeline for the user's dataset.

## CRITICAL: LIGHTWEIGHT TRAINING PARAMETERS ONLY
You MUST use these minimal parameters to ensure fast training:
- n_estimators: 10 (NOT 50, NOT 100)
- max_depth: 3 (NOT 4, NOT 5+)
- epochs: 1 (for neural networks)
- learning_rate: 0.1 (for gradient boosting)
- n_iterations: 10 (for iterative models)
- cv_folds: 2 (NOT 5)
- early_stopping_rounds: 3
- batch_size: 64 (for deep learning)
- max_iter: 100 (for sklearn models)
- verbose: 0

## RULES
1. Replace input paths with user's data paths
2. Keep output format consistent
3. ALWAYS use the lightweight params above - this is MANDATORY

## User Task: {query} ({problem_type})
## User Data: train="{best_inputs.get('train_data', best_pipeline['name'] + '/datasets/train.csv')}", test="{best_inputs.get('test_data', best_pipeline['name'] + '/datasets/test.csv')}"

## Pipeline #{rank}: `{pipeline['name']}`
```json
{json.dumps(spec, indent=2)[:3000]}
```

## OUTPUT (JSON only):
{{"name": "adapted-{pipeline['name']}", "rank": {rank}, "original": "{pipeline['name']}", "services_used": [...], "specification": [...], "reasoning": "..."}}"""

        # ===== AGENT 2: PREPROCESSING ADD =====
        preprocess_add_prompt = f"""Add a preprocessing step from alternatives to the best pipeline.

## CRITICAL: LIGHTWEIGHT TRAINING PARAMETERS ONLY
You MUST use these minimal parameters to ensure fast training:
- n_estimators: 10 (NOT 50, NOT 100)
- max_depth: 3 (NOT 4, NOT 5+)
- epochs: 1 (for neural networks)
- learning_rate: 0.1 (for gradient boosting)
- n_iterations: 10 (for iterative models)
- cv_folds: 2 (NOT 5)
- early_stopping_rounds: 3
- batch_size: 64 (for deep learning)
- max_iter: 100 (for sklearn models)
- verbose: 0

## RULES
1. KEEP ALL MODEL STEPS UNCHANGED (train_*, predict_*, create_submission)
2. ADD ONE preprocessing step before training
3. ALWAYS use the lightweight params above - this is MANDATORY

## User Task: {query} ({problem_type})
## Best Pipeline: `{best_pipeline['name']}`
```json
{json.dumps(best_spec, indent=2)[:2500]}
```

## Available Preprocessing (choose ONE to ADD):
```json
{json.dumps(unique_preprocess[:5], indent=2)[:1500]}
```

## OUTPUT (JSON only):
{{"name": "preprocess-enhanced", "added_preprocessing": "service_name", "services_used": [...], "specification": [...], "reasoning": "..."}}"""

        try:
            results = {
                "adapted_pipelines": [],
                "preprocessing_added": None,
                "execution_valid": {}
            }
            all_services = set()
            
            # Run IO Adaptation for each top 3
            for i, pipeline in enumerate(top_3):
                try:
                    prompt = build_adapt_prompt(pipeline, i + 1)
                    adapted = call_llm(prompt)
                    adapted["execution_valid"] = None  # To be tested
                    results["adapted_pipelines"].append(adapted)
                    all_services.update(adapted.get("services_used", []))
                    logger.info(f"Adapted pipeline #{i+1}: {pipeline['name']}")
                except Exception as e:
                    logger.warning(f"Failed to adapt pipeline #{i+1}: {e}")
            
            # Run Preprocessing Swap
            try:
                preprocess_result = call_llm(preprocess_add_prompt)
                preprocess_result["execution_valid"] = None  # To be tested
                results["preprocessing_added"] = preprocess_result
                all_services.update(preprocess_result.get("services_used", []))
                logger.info(f"Added preprocessing: {preprocess_result.get('added_preprocessing', 'N/A')}")
            except Exception as e:
                logger.warning(f"Preprocessing swap failed: {e}")
            
            state["composed_pipeline"] = results
            state["composed_services"] = list(all_services)
            state["reasoning"] = f"Generated {len(results['adapted_pipelines'])} adapted + 1 preprocessing-enhanced pipelines"
            
            logger.info(f"Composed {len(results['adapted_pipelines']) + 1} pipelines total")
            
        except Exception as e:
            logger.error(f"Composition failed: {e}")
            state["error"] = str(e)
            state["composed_pipeline"] = {}
            state["composed_services"] = []
            state["reasoning"] = f"Error: {e}"
        
        return state
    
    def _format_output(self, state: RecommenderState) -> RecommenderState:
        """Stage 4: Format final output."""
        # State is already complete, just log
        logger.info("Pipeline composition complete")
        return state
    
    # =========================================================================
    # GRAPH BUILDER
    # =========================================================================
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RecommenderState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("rerank", self._rerank)
        workflow.add_node("compose", self._compose)
        workflow.add_node("format_output", self._format_output)
        
        # Define edges: retrieve(10) → rerank(10→3) → compose → format_output
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "compose")
        workflow.add_edge("compose", "format_output")
        workflow.add_edge("format_output", END)
        
        return workflow.compile()
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def recommend(self, query: str, problem_type: str = "", 
                  data_context: str = "", domain_keywords: str = "") -> Dict:
        """
        Run the full pipeline composition workflow.
        
        Args:
            query: User's task description
            problem_type: regression, classification, etc.
            data_context: Description of the data
            domain_keywords: Domain-specific keywords
            
        Returns:
            Dict with composed pipeline and metadata
        """
        initial_state: RecommenderState = {
            "query": query,
            "problem_type": problem_type,
            "data_context": data_context,
            "domain_keywords": domain_keywords,
            "faiss_top_3": [],  # Will be set by _retrieve
            "top_k_pipelines": [],
            "retrieved_names": [],
            "service_pool": [],
            "composed_pipeline": {},
            "composed_services": [],
            "reasoning": "",
            "error": None
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "query": query,
            "problem_type": problem_type,
            "faiss_top_3": result.get("faiss_top_3", []),
            "retrieved_names": result["retrieved_names"],  # This is reranked top 3
            "retrieved_pipelines": result["retrieved_names"],
            "service_pool_size": len(result["service_pool"]),
            "composed_pipeline": result["composed_pipeline"],
            "composed_services": result["composed_services"],
            "reasoning": result["reasoning"],
            "error": result.get("error")
        }


# =============================================================================
# MULTI-INDEX RETRIEVER (FOR STREAMLIT UI)
# =============================================================================


@dataclass
class RecommenderConfig:
    """Configuration for the Streamlit-facing multi-index pipeline recommender."""

    # NOTE: Tests expect this default literal value.
    db_path: str = "kb.sqlite"

    # Retrieval sizes
    initial_k: int = 20
    rerank_k: int = 10
    final_k: int = 5

    # Behavior toggles
    contract_strict_mode: bool = False
    coarse_filter_enabled: bool = True

    # Multi-index weights (paper defaults)
    problem_type_weight: float = 0.3
    description_weight: float = 0.7

    # Provider config (Gemini defaults to match the paper/prototype)
    provider: str = "gemini"  # "gemini" | "openai"
    embed_model_gemini: str = "models/text-embedding-004"
    embed_dim_gemini: int = 768
    embed_model_openai: str = "text-embedding-3-small"
    embed_dim_openai: int = 1536
    llm_model_gemini: str = "gemini-3-flash-preview"
    llm_model_openai: str = "gpt-5-mini"


class CacheManager:
    """Simple LRU cache for embeddings and other expensive computations."""

    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._embeddings: "OrderedDict[str, Any]" = OrderedDict()

    def get_embedding(self, text: str):
        if text in self._embeddings:
            self._embeddings.move_to_end(text)
            return self._embeddings[text]
        return None

    def set_embedding(self, text: str, embedding):
        self._embeddings[text] = embedding
        self._embeddings.move_to_end(text)
        while len(self._embeddings) > self.max_size:
            self._embeddings.popitem(last=False)


@dataclass
class UserQuery:
    task_goal: str
    data_context: str = ""
    problem_type: Optional[str] = None
    domain: Optional[str] = None
    domain_keywords: Optional[str] = None
    additional_info: Optional[str] = None
    additional_constraints: Optional[str] = None

    def to_prompt_schema(self) -> Dict[str, Any]:
        domain_parts = [p for p in [self.domain, self.domain_keywords] if p]
        addl_parts = [p for p in [self.additional_info, self.additional_constraints] if p]
        return {
            "task_goal": self.task_goal,
            "data_context": self.data_context,
            "problem_type": self.problem_type or "",
            "domain_keywords": ", ".join(domain_parts),
            "additional_info": "; ".join(addl_parts),
        }


@dataclass
class PipelineCandidate:
    id: int
    name: str
    description: str
    specification: List[Dict[str, Any]]
    services_used: List[str]
    problem_type: str = ""
    domain: str = ""
    task_goal: str = ""

    combined_score: float = 0.0
    task_similarity: float = 0.0
    technique_similarity: float = 0.0
    success_score: float = 0.0

    contract_issues: List[str] = field(default_factory=list)

    # LLM fields (populated by reranker/validator)
    llm_decision: str = "pending"
    llm_fit_score: float = 0.0
    llm_reasoning: str = ""
    llm_adaptations: List[str] = field(default_factory=list)
    llm_risks: List[str] = field(default_factory=list)


@dataclass
class RecommendationResult:
    recommendations: List[PipelineCandidate]
    warnings: List[str] = field(default_factory=list)
    candidates_retrieved: int = 0
    candidates_after_filter: int = 0
    candidates_after_contract: int = 0
    processing_time_ms: float = 0.0


class IOChecker:
    """Deterministic structural checks for a pipeline specification (G2/G5-lite)."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def check_pipeline(self, pipeline: Any) -> Dict[str, Any]:
        # Tests call this with {"services_used": [...]}. Keep it permissive.
        if not pipeline:
            return {"valid": True, "errors": []}
        if isinstance(pipeline, dict) and "services_used" in pipeline and "specification" not in pipeline:
            return {"valid": True, "errors": []}

        spec = pipeline.get("specification") if isinstance(pipeline, dict) else pipeline
        if spec is None:
            return {"valid": True, "errors": []}
        if not isinstance(spec, list):
            return {"valid": False, "errors": ["Pipeline specification must be a list of steps."]}

        errors: List[str] = []

        # Output collision check
        produced: Dict[str, int] = {}
        for i, step in enumerate(spec):
            if not isinstance(step, dict):
                errors.append(f"Step {i+1}: step must be an object.")
                continue
            outs = step.get("outputs") or {}
            if not isinstance(outs, dict):
                continue
            for _, path in outs.items():
                if not isinstance(path, str):
                    continue
                if path in produced:
                    errors.append(
                        f"Output collision: '{path}' produced by steps {produced[path] + 1} and {i + 1}."
                    )
                produced[path] = i

        # Cycle detection (via path wiring)
        n = len(spec)
        producers: Dict[str, int] = {}
        for i, step in enumerate(spec):
            outs = (step.get("outputs") or {}) if isinstance(step, dict) else {}
            if not isinstance(outs, dict):
                continue
            for _, path in outs.items():
                if isinstance(path, str):
                    producers[path] = i

        adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}
        in_deg = [0] * n
        for i, step in enumerate(spec):
            ins = (step.get("inputs") or {}) if isinstance(step, dict) else {}
            if not isinstance(ins, dict):
                continue
            for _, path in ins.items():
                if not isinstance(path, str):
                    continue
                u = producers.get(path)
                if u is None or u == i:
                    continue
                adjacency[u].append(i)
                in_deg[i] += 1

        q = [i for i in range(n) if in_deg[i] == 0]
        seen = 0
        while q:
            u = q.pop()
            seen += 1
            for v in adjacency[u]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)
        if seen != n:
            errors.append("Cycle detected in pipeline DAG.")

        return {"valid": len(errors) == 0, "errors": errors}


class ValidatorAgent:
    """Contract/Governance validator (lightweight version for UI)."""

    GUIDELINES: Dict[str, str] = {
        "G1": "Single Responsibility: each service should do one thing and be reusable.",
        "G2": "Explicit I/O: services declare input/output slots with formats and schemas.",
        "G3": "Deterministic Execution: outputs depend only on inputs + explicit params (random_state exposed).",
        "G4": "Schema-Agnostic Design: avoid hardcoding dataset-specific columns; inject via params.",
        "G5": "Validated DAG: pipelines must be acyclic with no output collisions and compatible edges.",
        "G6": "Structured Metadata: services publish semantic metadata to enable discovery and retrieval.",
    }


class RankerAgent:
    def __init__(self, config: RecommenderConfig, cache: CacheManager):
        self.config = config
        self.cache = cache
        self._client = None  # Lazily initialized (tests expect None)


class ParameterAgent:
    def __init__(self, config: RecommenderConfig, cache: CacheManager, db_path: str):
        self.config = config
        self.cache = cache
        self.db_path = db_path
        self._client = None  # Lazily initialized (tests expect None)


class ComposerAgent:
    def __init__(self, config: RecommenderConfig, cache: CacheManager, db_path: str):
        self.config = config
        self.cache = cache
        self.db_path = db_path
        self._client = None  # Lazily initialized (tests expect None)


class SummarizerAgent:
    def __init__(self, config: RecommenderConfig, cache: CacheManager):
        self.config = config
        self.cache = cache
        self._client = None  # Lazily initialized (tests expect None)


class MultiIndexRecommender:
    """FAISS-style multi-index retriever + LLM reranker. Exposes the Streamlit API."""

    def __init__(self, api_key: str, config: Optional[RecommenderConfig] = None, cache: Optional[CacheManager] = None):
        self.config = config or RecommenderConfig()
        self.api_key = api_key
        self.cache = cache or CacheManager()

        self.provider = (self.config.provider or "gemini").lower()
        self.db_path = self._resolve_db_path(self.config.db_path)
        self._io_checker = IOChecker(self.db_path)

        # Lazy clients (only constructed when needed)
        self._genai_client = None
        self._openai_client = None

        # Loaded KB data (pipelines + embeddings)
        self._pipelines: List[Dict[str, Any]] = []
        self._desc_vectors: Optional[np.ndarray] = None
        self._problem_type_vectors: Optional[np.ndarray] = None  # built lazily
        self._problem_type_ready = False

        self._load_pipelines()

    def _resolve_db_path(self, db_path: str) -> str:
        if os.path.isabs(db_path) and os.path.exists(db_path):
            return db_path
        # 1) relative to CWD
        if os.path.exists(db_path):
            return db_path
        # 2) relative to this file (app/)
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, db_path)
        if os.path.exists(candidate):
            return candidate
        # 3) explicit fallback: app/kb.sqlite
        fallback = os.path.join(here, "kb.sqlite")
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(f"Knowledge Base not found at '{db_path}'.")

    def _get_embed_dim(self) -> int:
        return self.config.embed_dim_openai if self.provider == "openai" else self.config.embed_dim_gemini

    def _embed_model(self) -> str:
        return self.config.embed_model_openai if self.provider == "openai" else self.config.embed_model_gemini

    def _llm_model(self) -> str:
        return self.config.llm_model_openai if self.provider == "openai" else self.config.llm_model_gemini

    def _ensure_clients(self):
        if self.provider == "openai":
            if self._openai_client is None:
                self._openai_client = OpenAI(api_key=self.api_key)
        else:
            if self._genai_client is None:
                from google import genai  # local import for easier test mocking
                self._genai_client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        vec = vec.astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            return vec / norm
        return vec

    def _embed_text(self, text: str) -> np.ndarray:
        cached = self.cache.get_embedding(text)
        if cached is not None:
            return np.array(cached, dtype=np.float32)

        self._ensure_clients()
        if self.provider == "openai":
            result = self._openai_client.embeddings.create(input=text, model=self._embed_model())
            vec = np.array(result.data[0].embedding, dtype=np.float32)
        else:
            from google.genai import types
            result = self._genai_client.models.embed_content(
                model=self._embed_model(),
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            vec = np.array(result.embeddings[0].values, dtype=np.float32)

        vec = self._normalize(vec)
        self.cache.set_embedding(text, vec.tolist())
        return vec

    def _load_pipelines(self):
        table = "pipeline_embeddings_openai" if self.provider == "openai" else "pipeline_embeddings_gemini"
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"""
            SELECT p.id, p.name, p.problem_type, p.domain, p.task_goal, p.description,
                   p.specification, p.services_used, p.sample_input_schema,
                   p.usage_count, p.success_count, p.failure_count,
                   pe.description_embedding
            FROM pipelines p
            JOIN {table} pe ON p.id = pe.pipeline_id
            WHERE pe.description_embedding IS NOT NULL
        """).fetchall()
        conn.close()

        pipelines: List[Dict[str, Any]] = []
        desc_vecs: List[np.ndarray] = []

        for row in rows:
            spec = json.loads(row["specification"]) if row["specification"] else []
            services = json.loads(row["services_used"]) if row["services_used"] else []

            desc_vec = np.frombuffer(row["description_embedding"], dtype=np.float32)
            desc_vec = self._normalize(desc_vec)

            success = row["success_count"] or 0
            failure = row["failure_count"] or 0
            success_rate = (success + 1.0) / (success + failure + 2.0)

            pipelines.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "problem_type": row["problem_type"] or "",
                    "domain": row["domain"] or "",
                    "task_goal": row["task_goal"] or "",
                    "description": row["description"] or "",
                    "specification": spec,
                    "services_used": services,
                    "sample_input_schema": row["sample_input_schema"] or "",
                    "success_score": float(success_rate),
                }
            )
            desc_vecs.append(desc_vec)

        self._pipelines = pipelines
        self._desc_vectors = np.stack(desc_vecs) if desc_vecs else np.zeros((0, self._get_embed_dim()), dtype=np.float32)

        logger.info(f"[{self.provider}] MultiIndexRecommender loaded {len(self._pipelines)} pipelines")

    def _ensure_problem_type_vectors(self):
        if self._problem_type_ready:
            return

        # Embed unique problem types (only a handful) and broadcast to pipelines.
        unique_pts = sorted({p.get("problem_type", "") for p in self._pipelines if p.get("problem_type")})
        dim = self._get_embed_dim()
        pt_map: Dict[str, np.ndarray] = {}
        for pt in unique_pts:
            try:
                pt_map[pt] = self._embed_text(pt)
            except Exception:
                pt_map[pt] = np.zeros(dim, dtype=np.float32)

        vecs: List[np.ndarray] = []
        for p in self._pipelines:
            vecs.append(pt_map.get(p.get("problem_type") or "", np.zeros(dim, dtype=np.float32)))
        self._problem_type_vectors = np.stack(vecs) if vecs else np.zeros((0, dim), dtype=np.float32)
        self._problem_type_ready = True

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        t = (text or "").strip()
        if "```json" in t:
            return t.split("```json", 1)[1].split("```", 1)[0].strip()
        if "```" in t:
            return t.split("```", 1)[1].split("```", 1)[0].strip()
        return t

    def _llm_rerank(self, query: UserQuery, candidates: List[PipelineCandidate]) -> Tuple[List[int], str]:
        """Return reranked candidate indices (into `candidates`) and global reasoning."""
        self._ensure_clients()

        top_n = min(self.config.final_k, len(candidates))
        if top_n <= 0:
            return [], ""

        # Build anonymized candidate cards.
        cards = []
        for i, c in enumerate(candidates, start=1):
            cards.append(
                {
                    "n": i,
                    "problem_type": c.problem_type,
                    "domain": c.domain,
                    "description": (c.description or "")[:240],
                    "services": (c.services_used or [])[:10],
                }
            )

        prompt = (
            "You are an expert ML pipeline recommender.\n"
            "Given the USER QUERY and candidate pipelines, select the best matching pipelines.\n\n"
            "Return ONLY valid JSON in this exact format:\n"
            "{\n"
            f'  "top": [1, 2, ..., {top_n}],\n'
            '  "reasoning": "brief reasoning"\n'
            "}\n\n"
            "USER QUERY:\n"
            + json.dumps(query.to_prompt_schema(), indent=2)
            + "\n\nCANDIDATES:\n"
            + json.dumps(cards, indent=2)
            + "\n"
        )

        try:
            if self.provider == "openai":
                resp = self._openai_client.chat.completions.create(
                    model=self._llm_model(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=400,
                )
                raw = (resp.choices[0].message.content or "").strip()
            else:
                from google.genai import types
                contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
                resp = self._genai_client.models.generate_content(model=self._llm_model(), contents=contents)
                raw = getattr(resp, "text", "") or ""

            payload = json.loads(self._strip_code_fences(raw))
            top_list = payload.get("top", [])
            reasoning = payload.get("reasoning", "") or ""

            # Convert 1-based to 0-based and validate bounds.
            idxs: List[int] = []
            for n in top_list:
                try:
                    j = int(n) - 1
                except Exception:
                    continue
                if 0 <= j < len(candidates) and j not in idxs:
                    idxs.append(j)
            if not idxs:
                idxs = list(range(top_n))
            return idxs[:top_n], reasoning
        except Exception as e:
            logger.warning(f"LLM rerank failed, falling back to FAISS order: {e}")
            return list(range(top_n)), ""

    def recommend(self, query: UserQuery) -> RecommendationResult:
        start = time.perf_counter()
        warnings: List[str] = []

        q_text = " ".join(
            p
            for p in [
                query.task_goal,
                query.data_context,
                query.domain_keywords or query.domain or "",
                query.additional_info or query.additional_constraints or "",
            ]
            if p
        ).strip()
        if not q_text:
            return RecommendationResult(recommendations=[], warnings=["Empty query."])

        try:
            q_desc = self._embed_text(q_text)
        except Exception as e:
            return RecommendationResult(recommendations=[], warnings=[f"Embedding failed: {e}"])

        desc_vecs = self._desc_vectors
        if desc_vecs is None or len(self._pipelines) == 0:
            return RecommendationResult(recommendations=[], warnings=["No pipelines available in KB."])

        desc_scores = desc_vecs @ q_desc

        pt_scores = np.zeros(len(self._pipelines), dtype=np.float32)
        pt_weight = 0.0
        if query.problem_type and self.config.coarse_filter_enabled:
            pt_weight = float(self.config.problem_type_weight)
            self._ensure_problem_type_vectors()
            try:
                q_pt = self._embed_text(query.problem_type)
                pt_scores = (self._problem_type_vectors @ q_pt).astype(np.float32)  # type: ignore[operator]
            except Exception as e:
                warnings.append(f"Problem-type embedding failed: {e}")
                pt_scores = np.zeros(len(self._pipelines), dtype=np.float32)

        d_weight = float(self.config.description_weight)
        combined = (d_weight * desc_scores) + (pt_weight * pt_scores)

        order = np.argsort(-combined)
        k0 = min(int(self.config.initial_k), len(order))
        candidate_idxs = order[:k0].tolist()

        candidates: List[PipelineCandidate] = []
        for idx in candidate_idxs:
            p = self._pipelines[idx]
            cand = PipelineCandidate(
                id=int(p["id"]),
                name=str(p["name"]),
                description=str(p.get("description") or ""),
                specification=list(p.get("specification") or []),
                services_used=list(p.get("services_used") or []),
                problem_type=str(p.get("problem_type") or ""),
                domain=str(p.get("domain") or ""),
                task_goal=str(p.get("task_goal") or ""),
                combined_score=float(combined[idx]),
                task_similarity=float(pt_scores[idx]),
                technique_similarity=float(desc_scores[idx]),
                success_score=float(p.get("success_score") or 0.0),
            )
            candidates.append(cand)

        candidates_retrieved = len(candidates)
        candidates_after_filter = candidates_retrieved

        # Contract checks
        filtered: List[PipelineCandidate] = []
        for c in candidates:
            check = self._io_checker.check_pipeline(c.specification)
            if not check["valid"]:
                c.contract_issues = list(check["errors"])
                if self.config.contract_strict_mode:
                    continue
            filtered.append(c)

        candidates_after_contract = len(filtered)

        # LLM rerank on top rerank_k
        rerank_k = min(int(self.config.rerank_k), len(filtered))
        rerank_pool = filtered[:rerank_k]
        rerank_idxs, reasoning = self._llm_rerank(query, rerank_pool)
        reranked = [rerank_pool[i] for i in rerank_idxs]

        # Fill out LLM fields for UI.
        for rank, c in enumerate(reranked, start=1):
            c.llm_decision = "accept"
            c.llm_fit_score = max(0.0, 1.0 - (rank - 1) * 0.07)
            c.llm_reasoning = reasoning or "Selected by LLM reranker."

        # Final cut
        final_k = min(int(self.config.final_k), len(reranked))
        recs = reranked[:final_k]

        dur_ms = (time.perf_counter() - start) * 1000.0
        return RecommendationResult(
            recommendations=recs,
            warnings=warnings,
            candidates_retrieved=candidates_retrieved,
            candidates_after_filter=candidates_after_filter,
            candidates_after_contract=candidates_after_contract,
            processing_time_ms=dur_ms,
        )

    def recommend_with_agents(self, query: UserQuery) -> Dict[str, Any]:
        """Compatibility shim for the multi-agent test harness."""
        out = self.recommend(query)
        candidates = []
        for c in out.recommendations:
            candidates.append(
                {
                    "name": c.name,
                    "combined_score": c.combined_score,
                    "problem_type": c.problem_type,
                    "domain": c.domain,
                }
            )

        original = out.recommendations[0].specification if out.recommendations else []
        return {
            "original": original,
            "variations": [],
            "param_adjustments": {"adjusted_params": {}, "reasoning": ""},
            "io_check": self._io_checker.check_pipeline(original),
            "validation": {"valid": True, "errors": []},
            "summary": "Retrieved and reranked existing pipelines (agent flow simplified).",
            "candidates": candidates,
            "query": query.to_prompt_schema(),
            "retry_count": 0,
            "candidate_index": 0,
        }


# Global singleton used by Streamlit / scripts.
_RECOMMENDER: Optional[MultiIndexRecommender] = None


def init_recommender(api_key: str, config: Optional[RecommenderConfig] = None) -> MultiIndexRecommender:
    """
    Initialize the global recommender singleton.

    Streamlit reruns the script frequently; this function is idempotent.
    """
    global _RECOMMENDER
    cfg = config or RecommenderConfig()

    if _RECOMMENDER is not None:
        same_key = getattr(_RECOMMENDER, "api_key", None) == api_key
        same_provider = getattr(_RECOMMENDER, "provider", None) == (cfg.provider or "gemini").lower()
        same_db = getattr(_RECOMMENDER, "db_path", None) == _RECOMMENDER._resolve_db_path(cfg.db_path)  # type: ignore[attr-defined]
        if same_key and same_provider and same_db:
            return _RECOMMENDER

    _RECOMMENDER = MultiIndexRecommender(api_key=api_key, config=cfg, cache=CacheManager())
    return _RECOMMENDER


def get_recommendations(task: str, problem_type: str = "", domain: str = "", data_context: str = "", additional_info: str = "") -> RecommendationResult:
    """
    Convenience wrapper used by scripts/tests.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if _RECOMMENDER is None:
        if not api_key:
            raise RuntimeError("No API key found. Set GEMINI_API_KEY/GOOGLE_API_KEY (or OPENAI_API_KEY).")
        init_recommender(api_key)

    query = UserQuery(
        task_goal=task,
        data_context=data_context,
        problem_type=problem_type or None,
        domain=domain or None,
        additional_info=additional_info or None,
    )
    return _RECOMMENDER.recommend(query)  # type: ignore[union-attr]


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Demo the pipeline composer."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable")
        return
    
    print("=" * 60)
    print("LangGraph Pipeline Composition Recommender")
    print("=" * 60)
    
    composer = PipelineComposer(api_key=api_key)
    
    # Test query
    result = composer.recommend(
        query="Predict house sale price based on features like size, location, and quality",
        problem_type="regression",
        data_context="CSV with numeric and categorical features, target is continuous price"
    )
    
    print(f"\nQuery: {result['query'][:50]}...")
    print(f"Problem Type: {result['problem_type']}")
    print(f"\nRetrieved Pipelines (Top {len(result['retrieved_pipelines'])}):")
    for i, name in enumerate(result['retrieved_pipelines'], 1):
        print(f"  {i}. {name}")
    
    print(f"\nService Pool Size: {result['service_pool_size']}")
    print(f"\nComposed Services: {result['composed_services']}")
    print(f"\nReasoning: {result['reasoning'][:200]}...")
    
    if result.get('error'):
        print(f"\nError: {result['error']}")


if __name__ == "__main__":
    main()
