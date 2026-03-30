"""
Knowledge Base v2 - Optimized for LLM RAG
================================================

Designed for Multi-Agent RAG pipeline recommendation system.

Key Design Decisions:
1. Store FULL source code - LLM can understand implementation details
2. Store RICH metadata - enables semantic search and filtering
3. Track execution history - learn from successful/failed runs
4. Embeddings support - for vector similarity search

The KB serves two purposes:
- RETRIEVAL: Find relevant services/pipelines for a given task
- GENERATION: Provide context for LLM to generate new pipelines

Schema Overview:
================
┌─────────────────────────────────────────────────────────────┐
│                    KB = (P, M, E)                           │
├─────────────────────────────────────────────────────────────┤
│  M: Service Registry                                        │
│     - services (metadata + source_code)                     │
│     - service_embeddings (for vector search)                │
│                                                             │
│  P: Pipeline Corpus                                         │
│     - pipelines (specification + metadata)                  │
│     - pipeline_embeddings (for RAG retrieval)               │
│                                                             │
│  E: Execution History                                       │
│     - execution_runs (success/failure tracking)             │
│     - execution_metrics (model performance)                 │
│     - artifacts (produced files with lineage)               │
└─────────────────────────────────────────────────────────────┘

Author: Framework
Version: 2.0.0
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


# =============================================================================
# DATABASE SCHEMA v2
# =============================================================================

SCHEMA_V2_SQL = """
-- ===========================================================================
-- SERVICE REGISTRY (M) - Core service definitions with full source code
-- ===========================================================================

CREATE TABLE IF NOT EXISTS services (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    version TEXT DEFAULT '1.0.0',
    module TEXT,                          -- Module file this service belongs to (e.g., "house_prices_services")

    -- G6: Rich Semantic Metadata (for LLM understanding)
    description TEXT,                    -- Human-readable description
    docstring TEXT,                      -- Full Python docstring
    tags TEXT,                           -- JSON array of tags
    category TEXT,                       -- preprocessing, modeling, etc.

    -- G2: Explicit Data Interface
    input_contract TEXT NOT NULL,        -- JSON: {slot: {format, schema, required}}
    output_contract TEXT NOT NULL,       -- JSON: {slot: {format, schema}}
    parameters TEXT,                     -- JSON: {param: {type, default, description}}

    -- G4: Schema-Agnostic markers
    dynamic_columns TEXT,                -- JSON: params that inject column names

    -- FULL SOURCE CODE (key for LLM understanding!)
    source_code TEXT NOT NULL,           -- Complete function source
    source_hash TEXT,                    -- SHA256 for change detection

    -- Dependencies (for environment setup)
    imports TEXT,                        -- JSON: ["pandas", "sklearn.ensemble"]

    -- Usage Statistics (for ranking)
    usage_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 1.0,
    avg_runtime_seconds REAL,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================================================
-- SERVICE EMBEDDINGS - Vector representations for semantic search
-- ===========================================================================

CREATE TABLE IF NOT EXISTS service_embeddings (
    service_id INTEGER PRIMARY KEY REFERENCES services(id) ON DELETE CASCADE,

    -- Different embedding types for different search strategies
    description_embedding BLOB,          -- Embed description + docstring
    code_embedding BLOB,                 -- Embed source code
    contract_embedding BLOB,             -- Embed I/O contract JSON

    -- Metadata
    embedding_model TEXT,                -- e.g., "text-embedding-3-small"
    embedding_dim INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================================================
-- PIPELINE CORPUS (P) - Successful workflow specifications
-- ===========================================================================

CREATE TABLE IF NOT EXISTS pipelines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    version TEXT DEFAULT '1.0.0',

    -- Semantic Metadata (for LLM retrieval)
    description TEXT,                    -- What does this pipeline do?
    task_goal TEXT,                      -- "Predict house prices from features"
    problem_type TEXT,                   -- regression, classification, clustering
    domain TEXT,                         -- real_estate, finance, healthcare

    -- The Pipeline Specification (DAG)
    specification TEXT NOT NULL,         -- Full JSON: [{service, inputs, outputs, params}]

    -- Derived Data (for quick filtering)
    services_used TEXT,                  -- JSON: ["service1", "service2"]
    input_files TEXT,                    -- JSON: initial input paths
    output_files TEXT,                   -- JSON: final output paths
    step_count INTEGER,

    -- Sample Data Context (helps LLM understand use case)
    sample_input_schema TEXT,            -- JSON: {column: dtype} of input data
    sample_output_schema TEXT,           -- JSON: expected output format

    -- Quality Metrics (from successful runs)
    best_score REAL,                     -- Best achieved metric
    score_metric TEXT,                   -- "rmse", "accuracy", "f1"
    avg_runtime_seconds REAL,

    -- Usage Statistics
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================================================
-- PIPELINE EMBEDDINGS - For RAG retrieval
-- ===========================================================================

CREATE TABLE IF NOT EXISTS pipeline_embeddings (
    pipeline_id INTEGER PRIMARY KEY REFERENCES pipelines(id) ON DELETE CASCADE,

    -- Semantic embeddings
    task_embedding BLOB,                 -- Embed task_goal + description
    spec_embedding BLOB,                 -- Embed full specification

    -- Metadata
    embedding_model TEXT,
    embedding_dim INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================================================
-- EXECUTION HISTORY (E) - Learn from past runs
-- ===========================================================================

CREATE TABLE IF NOT EXISTS execution_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_id INTEGER REFERENCES pipelines(id),
    pipeline_name TEXT,                  -- Denormalized for quick access

    -- Execution Status
    status TEXT NOT NULL,                -- 'running', 'success', 'failed'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL,

    -- Environment (for reproducibility)
    working_directory TEXT,
    python_version TEXT,

    -- Results
    steps_completed INTEGER,
    total_steps INTEGER,

    -- Error Tracking (helps LLM learn from failures)
    failed_step INTEGER,
    failed_service TEXT,
    error_type TEXT,                     -- "ValueError", "FileNotFoundError"
    error_message TEXT,
    error_traceback TEXT,

    -- Metrics (for model evaluation)
    metrics TEXT,                        -- JSON: {rmse: 0.5, r2: 0.9}

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================================================
-- EXECUTION STEP DETAILS - Per-step tracking
-- ===========================================================================

CREATE TABLE IF NOT EXISTS execution_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES execution_runs(id) ON DELETE CASCADE,

    step_order INTEGER,
    service_name TEXT,

    status TEXT,                         -- 'success', 'failed', 'skipped'
    started_at TIMESTAMP,
    duration_seconds REAL,

    -- I/O Tracking
    input_paths TEXT,                    -- JSON
    output_paths TEXT,                   -- JSON
    params_used TEXT,                    -- JSON

    -- Result
    result_message TEXT,
    error_message TEXT
);

-- ===========================================================================
-- ARTIFACTS - Track produced files with lineage
-- ===========================================================================

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    path TEXT NOT NULL,
    format TEXT,                         -- csv, pickle, json

    -- Lineage
    produced_by_run_id INTEGER REFERENCES execution_runs(id),
    produced_by_service TEXT,
    produced_at TIMESTAMP,

    -- For tabular data
    row_count INTEGER,
    column_count INTEGER,
    columns TEXT,                        -- JSON: {col: dtype}

    -- Integrity
    file_hash TEXT,                      -- SHA256
    file_size_bytes INTEGER,

    UNIQUE(path, produced_by_run_id)
);

-- ===========================================================================
-- TAGS - For discovery and filtering
-- ===========================================================================

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT,                       -- 'task', 'domain', 'technique'
    usage_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS service_tags (
    service_id INTEGER REFERENCES services(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY(service_id, tag_id)
);

CREATE TABLE IF NOT EXISTS pipeline_tags (
    pipeline_id INTEGER REFERENCES pipelines(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY(pipeline_id, tag_id)
);

-- ===========================================================================
-- FORMATS - I/O format registry
-- ===========================================================================

CREATE TABLE IF NOT EXISTS formats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    produces TEXT,                       -- DataFrame, dict, Any
    file_extensions TEXT,                -- JSON: [".csv", ".tsv"]
    description TEXT
);

-- ===========================================================================
-- RAG CONTEXT CACHE - Pre-computed contexts for common queries
-- ===========================================================================

CREATE TABLE IF NOT EXISTS rag_context_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT UNIQUE,              -- SHA256 of query
    query_text TEXT,

    -- Retrieved context
    retrieved_services TEXT,             -- JSON: [service_ids]
    retrieved_pipelines TEXT,            -- JSON: [pipeline_ids]
    context_text TEXT,                   -- Pre-formatted context for LLM

    -- Quality feedback
    was_helpful BOOLEAN,
    user_rating INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP                 -- Cache expiry
);

-- ===========================================================================
-- SERVICE GRAPH - Which services can follow which (for LLM composition)
-- ===========================================================================

CREATE TABLE IF NOT EXISTS service_graph (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_service TEXT NOT NULL,          -- Source service name
    to_service TEXT NOT NULL,            -- Target service name
    from_output_slot TEXT,               -- Output slot that connects
    to_input_slot TEXT,                  -- Input slot that receives
    frequency INTEGER DEFAULT 1,         -- How often this edge appears in pipelines
    UNIQUE(from_service, to_service, from_output_slot, to_input_slot)
);

-- ===========================================================================
-- PIPELINE PATTERNS - Common patterns for LLM to learn
-- ===========================================================================

CREATE TABLE IF NOT EXISTS pipeline_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,           -- e.g., "unified_preprocessing"
    description TEXT,                    -- What this pattern does
    problem_types TEXT,                  -- JSON: ["regression", "classification"]

    -- Pattern structure
    services_sequence TEXT NOT NULL,     -- JSON: ordered list of service names
    template_spec TEXT,                  -- JSON: template pipeline spec with placeholders

    -- Usage
    usage_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 1.0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================================================
-- COMPOSITION RULES - Constraints for valid pipelines
-- ===========================================================================

CREATE TABLE IF NOT EXISTS composition_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type TEXT NOT NULL,             -- "requires", "conflicts", "suggests"
    service_a TEXT NOT NULL,             -- First service
    service_b TEXT,                      -- Second service (optional)
    condition TEXT,                      -- JSON: when this rule applies
    message TEXT,                        -- Human-readable explanation

    UNIQUE(rule_type, service_a, service_b)
);

-- ===========================================================================
-- INDEXES
-- ===========================================================================

CREATE INDEX IF NOT EXISTS idx_service_graph_from ON service_graph(from_service);
CREATE INDEX IF NOT EXISTS idx_service_graph_to ON service_graph(to_service);
CREATE INDEX IF NOT EXISTS idx_services_category ON services(category);
CREATE INDEX IF NOT EXISTS idx_services_usage ON services(usage_count DESC);
CREATE INDEX IF NOT EXISTS idx_pipelines_problem_type ON pipelines(problem_type);
CREATE INDEX IF NOT EXISTS idx_pipelines_domain ON pipelines(domain);
CREATE INDEX IF NOT EXISTS idx_execution_runs_status ON execution_runs(status);
CREATE INDEX IF NOT EXISTS idx_execution_runs_pipeline ON execution_runs(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_path ON artifacts(path);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);

-- ===========================================================================
-- VIEWS - Common queries for RAG
-- ===========================================================================

-- View: Services with full context for LLM
CREATE VIEW IF NOT EXISTS v_service_context AS
SELECT
    s.id,
    s.name,
    s.description,
    s.docstring,
    s.category,
    s.tags,
    s.input_contract,
    s.output_contract,
    s.parameters,
    s.source_code,
    s.imports,
    s.usage_count,
    s.success_rate
FROM services s
ORDER BY s.usage_count DESC;

-- View: Pipeline context with services
CREATE VIEW IF NOT EXISTS v_pipeline_context AS
SELECT
    p.id,
    p.name,
    p.description,
    p.task_goal,
    p.problem_type,
    p.domain,
    p.specification,
    p.services_used,
    p.best_score,
    p.score_metric,
    p.usage_count,
    p.success_count,
    p.failure_count
FROM pipelines p
ORDER BY p.success_count DESC;

-- View: Recent failures (for LLM to learn from)
CREATE VIEW IF NOT EXISTS v_recent_failures AS
SELECT
    er.pipeline_name,
    er.failed_service,
    er.error_type,
    er.error_message,
    er.created_at
FROM execution_runs er
WHERE er.status = 'failed'
ORDER BY er.created_at DESC
LIMIT 100;
"""


# =============================================================================
# KNOWLEDGE BASE CLASS v2
# =============================================================================

class KnowledgeBase:
    """
    Knowledge Base v2 - Optimized for LLM RAG.

    Key Features:
    - Full source code storage for LLM understanding
    - Embedding support for semantic search
    - Execution history for learning from successes/failures
    - Pre-computed RAG context caching

    Usage:
        kb = KnowledgeBase("kb.sqlite")

        # Register service with full source
        kb.register_service(
            name="impute_missing",
            source_code=inspect.getsource(impute_missing),
            input_contract={...},
            output_contract={...}
        )

        # Get RAG context for LLM
        context = kb.get_rag_context(
            task="predict house prices",
            problem_type="regression"
        )
    """

    def __init__(self, db_path: str = "kb.sqlite"):
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._init_schema()
        self._init_formats()

    def _connect(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def _init_schema(self):
        self.conn.executescript(SCHEMA_V2_SQL)
        self.conn.commit()

    def _init_formats(self):
        formats = [
            ("csv", "DataFrame", [".csv"], "Comma-separated values"),
            ("parquet", "DataFrame", [".parquet"], "Apache Parquet"),
            ("pickle", "Any", [".pkl", ".joblib"], "Python serialized object"),
            ("json", "dict", [".json"], "JSON object"),
        ]
        for name, produces, exts, desc in formats:
            self.conn.execute("""
                INSERT OR IGNORE INTO formats (name, produces, file_extensions, description)
                VALUES (?, ?, ?, ?)
            """, (name, produces, json.dumps(exts), desc))
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()

    # =========================================================================
    # SERVICE REGISTRY OPERATIONS
    # =========================================================================

    def register_service(
        self,
        name: str,
        source_code: str,
        input_contract: Dict,
        output_contract: Dict,
        description: str = "",
        docstring: str = "",
        tags: List[str] = None,
        category: str = None,
        parameters: Dict = None,
        imports: List[str] = None,
        dynamic_columns: List[str] = None,
        version: str = "1.0.0",
        module: str = None
    ) -> int:
        """
        Register a service with full source code.

        The source code is essential for LLM to understand:
        - Implementation details
        - Error handling patterns
        - Data transformations
        """
        source_hash = hashlib.sha256(source_code.encode()).hexdigest()[:16]

        # Extract docstring if not provided
        if not docstring and '"""' in source_code:
            try:
                start = source_code.index('"""') + 3
                end = source_code.index('"""', start)
                docstring = source_code[start:end].strip()
            except:
                pass

        cursor = self.conn.execute("""
            INSERT INTO services (
                name, version, description, docstring, tags, category,
                input_contract, output_contract, parameters,
                dynamic_columns, source_code, source_hash, imports, module
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                version = excluded.version,
                description = excluded.description,
                docstring = excluded.docstring,
                tags = excluded.tags,
                category = excluded.category,
                input_contract = excluded.input_contract,
                output_contract = excluded.output_contract,
                parameters = excluded.parameters,
                dynamic_columns = excluded.dynamic_columns,
                source_code = excluded.source_code,
                source_hash = excluded.source_hash,
                imports = excluded.imports,
                module = excluded.module,
                updated_at = CURRENT_TIMESTAMP
        """, (
            name, version, description, docstring,
            json.dumps(tags or []), category,
            json.dumps(input_contract), json.dumps(output_contract),
            json.dumps(parameters or {}),
            json.dumps(dynamic_columns or []),
            source_code, source_hash,
            json.dumps(imports or []),
            module
        ))
        self.conn.commit()

        # Always get the service ID by querying (lastrowid is unreliable with ON CONFLICT)
        service_id = self._get_id("services", name)

        # Clear existing tags before adding new ones (for updates)
        if service_id:
            self.conn.execute("DELETE FROM service_tags WHERE service_id = ?", (service_id,))
            self.conn.commit()

            # Register tags
            if tags:
                self._register_tags(tags, "service", service_id)

        return service_id

    def get_service(self, name: str) -> Optional[Dict]:
        """Get full service details including source code."""
        row = self.conn.execute(
            "SELECT * FROM services WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_service_source(self, name: str) -> Optional[str]:
        """Get just the source code."""
        row = self.conn.execute(
            "SELECT source_code FROM services WHERE name = ?", (name,)
        ).fetchone()
        return row["source_code"] if row else None

    def get_service_contract(self, name: str) -> Optional[Dict]:
        """Get I/O contract for validation."""
        row = self.conn.execute(
            "SELECT input_contract, output_contract FROM services WHERE name = ?",
            (name,)
        ).fetchone()
        if row:
            return {
                "input": json.loads(row["input_contract"]),
                "output": json.loads(row["output_contract"]),
            }
        return None

    def list_services(self, category: str = None, tag: str = None) -> List[Dict]:
        """List services with optional filters."""
        query = "SELECT * FROM services WHERE 1=1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)
        if tag:
            query += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')

        query += " ORDER BY usage_count DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # PIPELINE CORPUS OPERATIONS
    # =========================================================================

    def register_pipeline(
        self,
        name: str,
        specification: List[Dict],
        description: str = "",
        task_goal: str = "",
        problem_type: str = None,
        domain: str = None,
        tags: List[str] = None,
        sample_input_schema: Dict = None,
        version: str = "1.0.0"
    ) -> int:
        """Register a pipeline with full specification."""

        # Extract metadata from specification
        services_used = [step.get("service") for step in specification]
        input_files = []
        output_files = []
        all_outputs = set()

        for step in specification:
            all_outputs.update(step.get("outputs", {}).values())

        for step in specification:
            for path in step.get("inputs", {}).values():
                if path not in all_outputs:
                    input_files.append(path)
            for path in step.get("outputs", {}).values():
                # Check if consumed by later step
                is_final = True
                for later_step in specification:
                    if path in later_step.get("inputs", {}).values():
                        is_final = False
                        break
                if is_final:
                    output_files.append(path)

        cursor = self.conn.execute("""
            INSERT INTO pipelines (
                name, version, description, task_goal, problem_type, domain,
                specification, services_used, input_files, output_files,
                step_count, sample_input_schema
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                version = excluded.version,
                description = excluded.description,
                task_goal = excluded.task_goal,
                problem_type = excluded.problem_type,
                domain = excluded.domain,
                specification = excluded.specification,
                services_used = excluded.services_used,
                input_files = excluded.input_files,
                output_files = excluded.output_files,
                step_count = excluded.step_count,
                sample_input_schema = excluded.sample_input_schema,
                updated_at = CURRENT_TIMESTAMP
        """, (
            name, version, description, task_goal, problem_type, domain,
            json.dumps(specification),
            json.dumps(services_used),
            json.dumps(list(set(input_files))),
            json.dumps(list(set(output_files))),
            len(specification),
            json.dumps(sample_input_schema or {})
        ))
        self.conn.commit()

        pipeline_id = cursor.lastrowid or self._get_id("pipelines", name)

        if tags:
            self._register_tags(tags, "pipeline", pipeline_id)

        return pipeline_id

    def get_pipeline(self, name: str) -> Optional[Dict]:
        """Get full pipeline details."""
        row = self.conn.execute(
            "SELECT * FROM pipelines WHERE name = ?", (name,)
        ).fetchone()
        if row:
            result = dict(row)
            result["specification"] = json.loads(result["specification"])
            result["services_used"] = json.loads(result["services_used"] or "[]")
            return result
        return None

    def list_pipelines(self, problem_type: str = None, domain: str = None) -> List[Dict]:
        """List pipelines with optional filters."""
        query = "SELECT * FROM pipelines WHERE 1=1"
        params = []

        if problem_type:
            query += " AND problem_type = ?"
            params.append(problem_type)
        if domain:
            query += " AND domain = ?"
            params.append(domain)

        query += " ORDER BY success_count DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_pipelines_without_embeddings(self) -> List[Dict]:
        """Get pipelines that are missing embeddings."""
        return [dict(row) for row in self.conn.execute("""
            SELECT p.id, p.name, p.task_goal, p.description
            FROM pipelines p
            LEFT JOIN pipeline_embeddings pe ON p.id = pe.pipeline_id
            WHERE pe.task_embedding IS NULL
        """).fetchall()]

    def update_pipeline_embedding(self, pipeline_id: int, embedding: List[float], model: str):
        """Update the task embedding for a pipeline."""
        import json
        self.conn.execute("""
            INSERT INTO pipeline_embeddings (pipeline_id, task_embedding, embedding_model, embedding_dim)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(pipeline_id) DO UPDATE SET
                task_embedding = excluded.task_embedding,
                embedding_model = excluded.embedding_model,
                embedding_dim = excluded.embedding_dim,
                updated_at = CURRENT_TIMESTAMP
        """, (pipeline_id, json.dumps(embedding), model, len(embedding)))
        self.conn.commit()

    # =========================================================================
    # EXECUTION TRACKING
    # =========================================================================

    def start_execution(self, pipeline_name: str, working_dir: str = None) -> int:
        """Record start of pipeline execution."""
        import sys

        cursor = self.conn.execute("""
            INSERT INTO execution_runs (
                pipeline_name, status, started_at, working_directory, python_version
            ) VALUES (?, 'running', CURRENT_TIMESTAMP, ?, ?)
        """, (pipeline_name, working_dir, f"{sys.version_info.major}.{sys.version_info.minor}"))
        self.conn.commit()
        return cursor.lastrowid

    def record_step(
        self,
        run_id: int,
        step_order: int,
        service_name: str,
        status: str,
        duration: float,
        input_paths: Dict = None,
        output_paths: Dict = None,
        params: Dict = None,
        result_message: str = None,
        error_message: str = None
    ):
        """Record execution of a single step."""
        self.conn.execute("""
            INSERT INTO execution_steps (
                run_id, step_order, service_name, status, duration_seconds,
                input_paths, output_paths, params_used, result_message, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, step_order, service_name, status, duration,
            json.dumps(input_paths or {}),
            json.dumps(output_paths or {}),
            json.dumps(params or {}),
            result_message, error_message
        ))
        self.conn.commit()

    def complete_execution(
        self,
        run_id: int,
        status: str,
        steps_completed: int,
        total_steps: int,
        metrics: Dict = None,
        error_info: Dict = None
    ):
        """Record completion of pipeline execution."""

        # Calculate duration
        row = self.conn.execute(
            "SELECT started_at FROM execution_runs WHERE id = ?", (run_id,)
        ).fetchone()

        update_sql = """
            UPDATE execution_runs SET
                status = ?,
                completed_at = CURRENT_TIMESTAMP,
                steps_completed = ?,
                total_steps = ?,
                metrics = ?
        """
        params = [status, steps_completed, total_steps, json.dumps(metrics or {})]

        if error_info:
            update_sql += """,
                failed_step = ?,
                failed_service = ?,
                error_type = ?,
                error_message = ?,
                error_traceback = ?
            """
            params.extend([
                error_info.get("step"),
                error_info.get("service"),
                error_info.get("type"),
                error_info.get("message"),
                error_info.get("traceback")
            ])

        update_sql += " WHERE id = ?"
        params.append(run_id)

        self.conn.execute(update_sql, params)

        # Update pipeline success/failure counts
        pipeline_name = self.conn.execute(
            "SELECT pipeline_name FROM execution_runs WHERE id = ?", (run_id,)
        ).fetchone()["pipeline_name"]

        if status == "success":
            self.conn.execute("""
                UPDATE pipelines SET
                    success_count = success_count + 1,
                    usage_count = usage_count + 1
                WHERE name = ?
            """, (pipeline_name,))
        else:
            self.conn.execute("""
                UPDATE pipelines SET
                    failure_count = failure_count + 1,
                    usage_count = usage_count + 1
                WHERE name = ?
            """, (pipeline_name,))

        self.conn.commit()

    def record_artifact(
        self,
        run_id: int,
        path: str,
        format: str,
        service_name: str,
        row_count: int = None,
        column_count: int = None,
        columns: Dict = None
    ):
        """Record a produced artifact with lineage."""
        self.conn.execute("""
            INSERT OR REPLACE INTO artifacts (
                path, format, produced_by_run_id, produced_by_service,
                produced_at, row_count, column_count, columns
            ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
        """, (
            path, format, run_id, service_name,
            row_count, column_count, json.dumps(columns or {})
        ))
        self.conn.commit()

    # =========================================================================
    # RAG CONTEXT GENERATION (Key for LLM!)
    # =========================================================================

    def get_rag_context(
        self,
        task: str = None,
        problem_type: str = None,
        domain: str = None,
        required_inputs: List[str] = None,
        required_outputs: List[str] = None,
        max_services: int = 10,
        max_pipelines: int = 5,
        include_source: bool = True,
        include_failures: bool = True
    ) -> str:
        """
        Generate rich context for LLM pipeline recommendation.

        This is the main method for RAG - it assembles relevant
        services and pipelines into a context string that helps
        the LLM generate appropriate pipeline specifications.

        Parameters
        ----------
        task : str
            Natural language task description
        problem_type : str
            regression, classification, etc.
        domain : str
            Domain filter
        required_inputs : List[str]
            Required input formats (e.g., ["csv"])
        required_outputs : List[str]
            Required output formats
        max_services : int
            Maximum services to include
        max_pipelines : int
            Maximum pipelines to include
        include_source : bool
            Include full source code
        include_failures : bool
            Include recent failure examples

        Returns
        -------
        str
            Formatted context for LLM prompt
        """
        context_parts = []

        # 1. RELEVANT SERVICES
        context_parts.append("=" * 60)
        context_parts.append("AVAILABLE SERVICES (M)")
        context_parts.append("=" * 60)

        services = self._get_relevant_services(
            problem_type=problem_type,
            max_count=max_services
        )

        for svc in services:
            context_parts.append(f"\n### {svc['name']}")
            context_parts.append(f"Category: {svc.get('category', 'unknown')}")
            context_parts.append(f"Description: {svc.get('description', '')}")

            # Input/Output contracts
            input_contract = json.loads(svc['input_contract'])
            output_contract = json.loads(svc['output_contract'])

            context_parts.append("\nInputs:")
            for slot, spec in input_contract.items():
                req = "required" if spec.get("required", True) else "optional"
                context_parts.append(f"  - {slot}: <{spec.get('format')}> ({req})")

            context_parts.append("\nOutputs:")
            for slot, spec in output_contract.items():
                context_parts.append(f"  - {slot}: <{spec.get('format')}>")

            # Parameters
            params = json.loads(svc.get('parameters') or '{}')
            if params:
                context_parts.append("\nParameters:")
                for param, spec in params.items():
                    default = spec.get('default', 'none')
                    context_parts.append(f"  - {param}: {spec.get('type', 'any')} (default: {default})")

            # Source code (if requested)
            if include_source and svc.get('source_code'):
                context_parts.append("\nSource Code:")
                context_parts.append("```python")
                context_parts.append(svc['source_code'])
                context_parts.append("```")

        # 2. EXAMPLE PIPELINES
        context_parts.append("\n" + "=" * 60)
        context_parts.append("EXAMPLE PIPELINES (P)")
        context_parts.append("=" * 60)

        pipelines = self._get_relevant_pipelines(
            problem_type=problem_type,
            domain=domain,
            max_count=max_pipelines
        )

        for p in pipelines:
            context_parts.append(f"\n### {p['name']}")
            context_parts.append(f"Task: {p.get('task_goal', p.get('description', ''))}")
            context_parts.append(f"Problem Type: {p.get('problem_type', 'unknown')}")
            context_parts.append(f"Success Rate: {p.get('success_count', 0)}/{p.get('usage_count', 0)}")

            if p.get('best_score'):
                context_parts.append(f"Best Score: {p['best_score']} ({p.get('score_metric', 'unknown')})")

            context_parts.append("\nSpecification:")
            context_parts.append("```json")
            spec = json.loads(p['specification']) if isinstance(p['specification'], str) else p['specification']
            context_parts.append(json.dumps(spec, indent=2))
            context_parts.append("```")

        # 3. RECENT FAILURES (helps LLM avoid common mistakes)
        if include_failures:
            failures = self._get_recent_failures(limit=5)
            if failures:
                context_parts.append("\n" + "=" * 60)
                context_parts.append("RECENT FAILURES (Learn from these!)")
                context_parts.append("=" * 60)

                for f in failures:
                    context_parts.append(f"\n- Pipeline: {f['pipeline_name']}")
                    context_parts.append(f"  Failed Service: {f['failed_service']}")
                    context_parts.append(f"  Error: {f['error_type']}: {f['error_message']}")

        return "\n".join(context_parts)

    def get_service_context_for_llm(self, service_name: str) -> str:
        """Get full context for a single service (for LLM understanding)."""
        svc = self.get_service(service_name)
        if not svc:
            return f"Service '{service_name}' not found."

        parts = [
            f"# Service: {service_name}",
            f"\n## Description\n{svc.get('description', '')}",
            f"\n## Docstring\n{svc.get('docstring', '')}",
            f"\n## Category\n{svc.get('category', 'unknown')}",
            f"\n## Tags\n{svc.get('tags', '[]')}",
        ]

        # Contracts
        parts.append("\n## Input Contract")
        parts.append(f"```json\n{svc['input_contract']}\n```")

        parts.append("\n## Output Contract")
        parts.append(f"```json\n{svc['output_contract']}\n```")

        # Full source
        parts.append("\n## Source Code")
        parts.append(f"```python\n{svc['source_code']}\n```")

        return "\n".join(parts)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_id(self, table: str, name: str) -> Optional[int]:
        row = self.conn.execute(
            f"SELECT id FROM {table} WHERE name = ?", (name,)
        ).fetchone()
        return row["id"] if row else None

    def _register_tags(self, tags: List[str], entity_type: str, entity_id: int):
        for tag_name in tags:
            self.conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
            tag_row = self.conn.execute(
                "SELECT id FROM tags WHERE name = ?", (tag_name,)
            ).fetchone()
            if tag_row:
                tag_id = tag_row["id"]
                if entity_type == "service":
                    self.conn.execute(
                        "INSERT OR IGNORE INTO service_tags (service_id, tag_id) VALUES (?, ?)",
                        (entity_id, tag_id)
                    )
                elif entity_type == "pipeline":
                    self.conn.execute(
                        "INSERT OR IGNORE INTO pipeline_tags (pipeline_id, tag_id) VALUES (?, ?)",
                        (entity_id, tag_id)
                    )
        self.conn.commit()

    def _get_relevant_services(
        self,
        problem_type: str = None,
        max_count: int = 10
    ) -> List[Dict]:
        """Get services relevant to the problem type."""
        query = "SELECT * FROM services ORDER BY usage_count DESC LIMIT ?"
        rows = self.conn.execute(query, (max_count,)).fetchall()
        return [dict(row) for row in rows]

    def _get_relevant_pipelines(
        self,
        problem_type: str = None,
        domain: str = None,
        max_count: int = 5
    ) -> List[Dict]:
        """Get pipelines relevant to problem type and domain."""
        query = "SELECT * FROM pipelines WHERE 1=1"
        params = []

        if problem_type:
            query += " AND problem_type = ?"
            params.append(problem_type)
        if domain:
            query += " AND domain = ?"
            params.append(domain)

        query += " ORDER BY success_count DESC LIMIT ?"
        params.append(max_count)

        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def _get_recent_failures(self, limit: int = 5) -> List[Dict]:
        """Get recent failures for learning."""
        rows = self.conn.execute("""
            SELECT pipeline_name, failed_service, error_type, error_message
            FROM execution_runs
            WHERE status = 'failed' AND error_message IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # SERVICE GRAPH & PATTERNS (for LLM composition)
    # =========================================================================

    def build_service_graph_from_pipelines(self):
        """
        Analyze all pipelines to build service dependency graph.
        This helps LLM understand which services can follow which.
        """
        # Clear existing graph
        self.conn.execute("DELETE FROM service_graph")

        # Get all pipelines
        pipelines = self.list_pipelines()

        for p in pipelines:
            spec = p.get("specification", [])
            if isinstance(spec, str):
                spec = json.loads(spec)

            # Track outputs -> path mapping
            output_registry = {}

            for i, step in enumerate(spec):
                service = step.get("service")
                inputs = step.get("inputs", {})
                outputs = step.get("outputs", {})

                # Skip steps without service name or with invalid inputs/outputs
                if not service:
                    continue
                if not isinstance(inputs, dict) or not isinstance(outputs, dict):
                    continue

                # Find which previous service produced each input
                for in_slot, in_path in inputs.items():
                    if in_path in output_registry:
                        from_service, from_slot = output_registry[in_path]
                        # Add edge
                        self.conn.execute("""
                            INSERT INTO service_graph (from_service, to_service, from_output_slot, to_input_slot, frequency)
                            VALUES (?, ?, ?, ?, 1)
                            ON CONFLICT(from_service, to_service, from_output_slot, to_input_slot)
                            DO UPDATE SET frequency = frequency + 1
                        """, (from_service, service, from_slot, in_slot))

                # Register outputs
                for out_slot, out_path in outputs.items():
                    output_registry[out_path] = (service, out_slot)

        self.conn.commit()

    def get_service_graph(self) -> List[Dict]:
        """Get the service dependency graph."""
        rows = self.conn.execute("""
            SELECT from_service, to_service, from_output_slot, to_input_slot, frequency
            FROM service_graph
            ORDER BY frequency DESC
        """).fetchall()
        return [dict(row) for row in rows]

    def get_next_services(self, service_name: str) -> List[Dict]:
        """Get services that commonly follow the given service."""
        rows = self.conn.execute("""
            SELECT to_service, to_input_slot, from_output_slot, frequency
            FROM service_graph
            WHERE from_service = ?
            ORDER BY frequency DESC
        """, (service_name,)).fetchall()
        return [dict(row) for row in rows]

    def get_previous_services(self, service_name: str) -> List[Dict]:
        """Get services that commonly precede the given service."""
        rows = self.conn.execute("""
            SELECT from_service, from_output_slot, to_input_slot, frequency
            FROM service_graph
            WHERE to_service = ?
            ORDER BY frequency DESC
        """, (service_name,)).fetchall()
        return [dict(row) for row in rows]

    def register_pattern(
        self,
        name: str,
        description: str,
        services_sequence: List[str],
        problem_types: List[str] = None,
        template_spec: List[Dict] = None
    ):
        """Register a common pipeline pattern."""
        self.conn.execute("""
            INSERT INTO pipeline_patterns (name, description, problem_types, services_sequence, template_spec)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                description = excluded.description,
                problem_types = excluded.problem_types,
                services_sequence = excluded.services_sequence,
                template_spec = excluded.template_spec
        """, (
            name, description,
            json.dumps(problem_types or []),
            json.dumps(services_sequence),
            json.dumps(template_spec or [])
        ))
        self.conn.commit()

    def get_patterns(self, problem_type: str = None) -> List[Dict]:
        """Get pipeline patterns, optionally filtered by problem type."""
        if problem_type:
            rows = self.conn.execute("""
                SELECT * FROM pipeline_patterns
                WHERE problem_types LIKE ?
                ORDER BY usage_count DESC
            """, (f'%"{problem_type}"%',)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM pipeline_patterns ORDER BY usage_count DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def register_composition_rule(
        self,
        rule_type: str,
        service_a: str,
        service_b: str = None,
        condition: Dict = None,
        message: str = ""
    ):
        """
        Register a composition rule.

        rule_type:
        - "requires": service_a requires service_b before it
        - "conflicts": service_a conflicts with service_b
        - "suggests": service_a suggests using service_b after
        """
        self.conn.execute("""
            INSERT INTO composition_rules (rule_type, service_a, service_b, condition, message)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(rule_type, service_a, service_b) DO UPDATE SET
                condition = excluded.condition,
                message = excluded.message
        """, (rule_type, service_a, service_b, json.dumps(condition or {}), message))
        self.conn.commit()

    def get_composition_rules(self, service_name: str = None) -> List[Dict]:
        """Get composition rules, optionally for a specific service."""
        if service_name:
            rows = self.conn.execute("""
                SELECT * FROM composition_rules
                WHERE service_a = ? OR service_b = ?
            """, (service_name, service_name)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM composition_rules").fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # ENHANCED RAG CONTEXT (for LLM composition)
    # =========================================================================

    def get_composition_context(self, problem_type: str = None) -> str:
        """
        Generate context specifically for LLM pipeline COMPOSITION.

        This includes:
        1. Service graph (what follows what)
        2. Common patterns
        3. Composition rules
        4. Category-based service grouping
        """
        parts = []

        # 1. SERVICE CATEGORIES
        parts.append("=" * 60)
        parts.append("SERVICE CATEGORIES")
        parts.append("=" * 60)
        parts.append("""
Services are organized into categories that typically flow in this order:
1. data-handling: Load and combine data
2. preprocessing: Clean, impute, encode, scale
3. modeling: Train models
4. inference: Make predictions, format output
""")

        # Group services by category
        services = self.list_services()
        by_category = {}
        for svc in services:
            cat = svc.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(svc["name"])

        for cat, names in by_category.items():
            parts.append(f"\n{cat}: {', '.join(names)}")

        # 2. SERVICE FLOW GRAPH
        parts.append("\n" + "=" * 60)
        parts.append("SERVICE FLOW GRAPH (common sequences)")
        parts.append("=" * 60)

        graph = self.get_service_graph()
        if graph:
            for edge in graph[:15]:  # Top 15 edges
                parts.append(
                    f"  {edge['from_service']}.{edge['from_output_slot']} "
                    f"-> {edge['to_service']}.{edge['to_input_slot']} "
                    f"(used {edge['frequency']}x)"
                )
        else:
            parts.append("  (No graph data yet - run build_service_graph_from_pipelines())")

        # 3. COMMON PATTERNS
        parts.append("\n" + "=" * 60)
        parts.append("COMMON PIPELINE PATTERNS")
        parts.append("=" * 60)

        patterns = self.get_patterns(problem_type)
        if patterns:
            for p in patterns:
                parts.append(f"\n### {p['name']}")
                parts.append(f"  {p['description']}")
                seq = json.loads(p['services_sequence']) if isinstance(p['services_sequence'], str) else p['services_sequence']
                parts.append(f"  Services: {' -> '.join(seq)}")
        else:
            # Add default patterns
            parts.append("""
### unified_preprocessing (Recommended for tabular data)
  Combine train+test, preprocess together, split back
  Services: combine_train_test -> preprocess_steps -> split_train_test -> train_model -> predict

### fit_transform_pattern
  Fit transformers on train, apply to test
  Services: impute_missing (fit) -> apply_imputers (transform)
  Services: encode_onehot (fit) -> apply_encoder (transform)
""")

        # 4. COMPOSITION RULES
        parts.append("\n" + "=" * 60)
        parts.append("COMPOSITION RULES")
        parts.append("=" * 60)

        rules = self.get_composition_rules()
        if rules:
            for r in rules:
                parts.append(f"  [{r['rule_type']}] {r['service_a']} <-> {r['service_b']}: {r['message']}")
        else:
            parts.append("""
  [requires] train_model requires preprocessed data (no missing values, all numeric)
  [requires] predict requires a trained model from train_model
  [requires] format_submission requires predictions from predict
  [suggests] After impute_missing, use encode_onehot for categorical columns
  [suggests] After encode_onehot, consider scale_features for numeric columns
  [conflicts] Don't use apply_imputers if impute_missing wasn't used earlier
""")

        # 5. PIPELINE STRUCTURE TEMPLATE
        parts.append("\n" + "=" * 60)
        parts.append("PIPELINE JSON STRUCTURE")
        parts.append("=" * 60)
        parts.append('''
A pipeline is a JSON array of steps:
```json
[
  {
    "service": "service_name",
    "inputs": {
      "input_slot": "path/to/input.csv"
    },
    "outputs": {
      "output_slot": "path/to/output.csv"
    },
    "params": {
      "param_name": "value"
    }
  },
  ...
]
```

Rules:
- Each step's input paths must match a previous step's output path (or be initial data)
- Use "artifacts/" prefix for intermediate files
- Use "datasets/" prefix for source data
- Final submission should be "submission.csv"
''')

        return "\n".join(parts)

    # =========================================================================
    # FORMAT COMPATIBILITY (for pipeline validation)
    # =========================================================================

    def check_format_compatibility(self, output_fmt: str, input_fmt: str) -> bool:
        """Check if output format is compatible with input format."""
        # Get format info from DB
        out_row = self.conn.execute(
            "SELECT produces FROM formats WHERE name = ?", (output_fmt,)
        ).fetchone()
        in_row = self.conn.execute(
            "SELECT produces FROM formats WHERE name = ?", (input_fmt,)
        ).fetchone()

        if not out_row or not in_row:
            # If format not in DB, check exact match
            return output_fmt == input_fmt

        out_produces = out_row["produces"]
        in_produces = in_row["produces"]

        return (
            out_produces == in_produces or
            out_produces == "Any" or
            in_produces == "Any"
        )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get KB statistics."""
        return {
            "services": self.conn.execute("SELECT COUNT(*) FROM services").fetchone()[0],
            "pipelines": self.conn.execute("SELECT COUNT(*) FROM pipelines").fetchone()[0],
            "executions": self.conn.execute("SELECT COUNT(*) FROM execution_runs").fetchone()[0],
            "successful_runs": self.conn.execute(
                "SELECT COUNT(*) FROM execution_runs WHERE status = 'success'"
            ).fetchone()[0],
            "failed_runs": self.conn.execute(
                "SELECT COUNT(*) FROM execution_runs WHERE status = 'failed'"
            ).fetchone()[0],
            "artifacts": self.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
        }

    def describe(self):
        """Print KB overview."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("KNOWLEDGE BASE v2")
        print("=" * 60)
        print(f"Database: {self.db_path}")
        print(f"\nServices (M):    {stats['services']}")
        print(f"Pipelines (P):   {stats['pipelines']}")
        print(f"Executions (E):  {stats['executions']}")
        print(f"  - Successful:  {stats['successful_runs']}")
        print(f"  - Failed:      {stats['failed_runs']}")
        print(f"Artifacts:       {stats['artifacts']}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    kb = KnowledgeBase("test_kb_v2.sqlite")

    # Register a sample service with full source
    sample_source = '''
@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Fill missing values"
)
def fill_missing(inputs, outputs, strategy="median"):
    """Fill missing values using specified strategy."""
    df = IOManager.load(inputs["data"], "csv")
    df = df.fillna(df.median() if strategy == "median" else df.mean())
    IOManager.save(df, outputs["data"], "csv")
    return f"Filled {df.isnull().sum().sum()} missing values"
'''

    kb.register_service(
        name="fill_missing",
        source_code=sample_source,
        input_contract={"data": {"format": "csv", "required": True}},
        output_contract={"data": {"format": "csv"}},
        description="Fill missing values using median or mean",
        tags=["preprocessing", "imputation"],
        category="preprocessing"
    )

    # Get RAG context
    print("\n" + "=" * 60)
    print("RAG CONTEXT FOR LLM")
    print("=" * 60)
    context = kb.get_rag_context(problem_type="regression", include_source=True)
    print(context[:2000] + "..." if len(context) > 2000 else context)

    kb.describe()
    kb.close()
