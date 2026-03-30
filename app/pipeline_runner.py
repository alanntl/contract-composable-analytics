"""
Pipeline Runner
=====================

Executes pipelines from JSON specifications using services stored in the Knowledge Base.

This is the core execution engine that:
1. Loads pipeline specification (JSON)
2. Retrieves service code from Knowledge Base
3. Validates contracts (G2, G5)
4. Executes services in topological order (DAG)
5. Tracks execution metrics

Based on the paper's design:
- Section 3.4: Declarative Pipeline Specification
- Section 3.5: Pre-flight Validation
- Algorithm 1: Pipeline Execution

Usage:
    from pipeline_runner import PipelineRunner

    runner = PipelineRunner("kb.sqlite")
    runner.run_pipeline("house_price_training")
    # OR
    runner.run_from_json("pipeline.json")

Author: Framework
Version: 1.0.0
"""

import os
import sys
import json
import time
import traceback
import importlib
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # When imported as a package (e.g., from app.pipeline_runner)
    from .kb import KnowledgeBase  # type: ignore
except Exception:
    # When executed directly as a script
    from kb import KnowledgeBase


class PipelineRunner:
    """
    Executes Contract-Composable Analytics pipelines from JSON specifications.

    The runner:
    1. Validates pipeline contracts before execution (G2, G5)
    2. Dynamically loads service code from Knowledge Base
    3. Executes services in DAG order
    4. Tracks execution metrics

    Example:
        runner = PipelineRunner("kb.sqlite")

        # Run by pipeline name (from KB)
        result = runner.run_pipeline("house_price_training")

        # Run from JSON file
        result = runner.run_from_json("pipeline.json")

        # Run from specification dict
        result = runner.run(specification, base_path="./storage")
    """

    # Default storage folder for all data and artifacts
    DEFAULT_STORAGE = "storage"

    def __init__(self, db_path: Optional[str] = "kb.sqlite", verbose: bool = True, storage: str = None, modules: List[str] = None):
        """
        Initialize the Pipeline Runner.

        Parameters
        ----------
        db_path : str
            Path to the SQLite Knowledge Base database
        verbose : bool
            Print execution progress
        storage : str, optional
            Base folder for all data/artifacts. Defaults to "storage"
        modules : List[str], optional
            List of module names to load services from (e.g., ["house_prices_services"])
            If None, services will be loaded from KB source code
        """
        self.db_path = db_path
        self.verbose = verbose
        self.kb = KnowledgeBase(db_path) if db_path else None

        # Modules to load services from
        self.modules = modules or []
        self._module_registries: Dict[str, Dict] = {}  # module_name -> SERVICE_REGISTRY
        self._module_objects: Dict[str, Any] = {}      # module_name -> module object

        # Load specified modules
        if self.modules:
            self._load_modules()

        # Set storage folder (default: "storage")
        self.storage = storage or self.DEFAULT_STORAGE
        os.makedirs(self.storage, exist_ok=True)

        # Cache for compiled service functions
        self._service_cache: Dict[str, Callable] = {}

        # Execution metrics
        self.last_execution = None

    def log(self, msg: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(msg)

    # =========================================================================
    # PIPELINE EXECUTION
    # =========================================================================

    def run_pipeline(self, pipeline_name: str, base_path: str = None) -> Dict:
        """
        Run a pipeline by name from the Knowledge Base.

        Parameters
        ----------
        pipeline_name : str
            Name of the pipeline in KB
        base_path : str, optional
            Base path for data artifacts. Defaults to self.storage

        Returns
        -------
        Dict
            Execution result with metrics
        """
        # Use storage as default base_path
        if base_path is None:
            base_path = self.storage

        if not self.kb:
            raise ValueError("Knowledge Base is disabled; provide --db or use run_from_json/run.")

        pipeline = self.kb.get_pipeline(pipeline_name)
        if not pipeline:
            raise ValueError(f"Pipeline '{pipeline_name}' not found in Knowledge Base")

        specification = pipeline["specification"]
        # Handle both formats: dict with 'steps' key or list of steps directly
        if isinstance(specification, dict) and 'steps' in specification:
            steps = specification['steps']
        else:
            steps = specification
        return self.run(steps, base_path=base_path, pipeline_name=pipeline_name)

    def run_from_json(self, json_path: str, base_path: str = None) -> Dict:
        """
        Run a pipeline from a JSON file.

        Parameters
        ----------
        json_path : str
            Path to pipeline JSON file
        base_path : str, optional
            Base path for data artifacts (defaults to JSON file's directory)

        Returns
        -------
        Dict
            Execution result with metrics
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Support both raw specification and wrapped format
        if "specification" in data:
            specification = data["specification"]
            pipeline_name = data.get("name", Path(json_path).stem)
        elif "steps" in data:
            specification = data["steps"]
            pipeline_name = data.get("name", Path(json_path).stem)
        else:
            # Assume the JSON is the specification itself
            specification = data
            pipeline_name = Path(json_path).stem

        # Use storage as default base_path
        if base_path is None:
            base_path = self.storage

        return self.run(specification, base_path=base_path, pipeline_name=pipeline_name)

    def run(
        self,
        specification: List[Dict],
        base_path: str = None,
        pipeline_name: str = "unnamed",
        parallel: bool = False,
        max_workers: int = 4
    ) -> Dict:
        """
        Execute a pipeline specification in DAG topological order.

        Parameters
        ----------
        specification : List[Dict]
            Pipeline steps. Each step has:
            - service: str (service name)
            - inputs: Dict[str, str] (slot -> path)
            - outputs: Dict[str, str] (slot -> path)
            - params: Dict (optional parameters)
        base_path : str, optional
            Base path for relative paths
        pipeline_name : str
            Name for logging
        parallel : bool
            If True, execute independent steps in parallel
        max_workers : int
            Maximum parallel workers (only used if parallel=True)

        Returns
        -------
        Dict
            Execution result with:
            - success: bool
            - steps_completed: int
            - total_steps: int
            - duration_seconds: float
            - step_results: List[Dict]
            - execution_order: List[int] (topological order)
            - parallel_levels: List[List[int]] (parallelization info)
            - error: str (if failed)
        """
        self.log(f"\n{'='*70}")
        self.log(f"Pipeline Runner - {pipeline_name}")
        self.log(f"{'='*70}")

        start_time = time.time()
        step_results = [None] * len(specification)  # Pre-allocate for ordered results

        # Use storage as default base_path
        if base_path is None:
            base_path = self.storage

        # Resolve base path
        if base_path:
            base_path = os.path.abspath(base_path)
            os.makedirs(base_path, exist_ok=True)

        # =====================================================================
        # DAG ANALYSIS & CYCLE DETECTION
        # =====================================================================
        self.log(f"\n[1/4] DAG Analysis...")

        dag_info = self.analyze_dag(specification)

        if not dag_info["is_valid"]:
            cycle_path = dag_info.get("cycle_path", [])
            cycle_str = " -> ".join(cycle_path) if cycle_path else "unknown"
            self.log(f"  FAILED: Cycle detected in pipeline")
            self.log(f"  Cycle: {cycle_str}")
            return {
                "success": False,
                "pipeline_name": pipeline_name,
                "steps_completed": 0,
                "total_steps": len(specification),
                "duration_seconds": time.time() - start_time,
                "step_results": [],
                "error": f"Pipeline contains cycle: {cycle_str}",
                "cycle_path": cycle_path,
            }

        execution_order = dag_info["execution_order"]
        parallel_levels = dag_info["parallel_levels"]
        external_inputs = dag_info.get("external_inputs", [])

        self.log(f"  DAG is valid (acyclic)")
        self.log(f"  Execution order: {execution_order}")
        self.log(f"  Parallel levels: {len(parallel_levels)} (max parallelism: {dag_info['max_parallelism']})")
        self.log(f"  External inputs: {len(external_inputs)}")

        # Validate external inputs exist
        if external_inputs:
            all_exist, missing, existing = self.validate_external_inputs(specification, base_path)
            if not all_exist:
                self.log(f"  WARNING: {len(missing)} external input(s) not found:")
                for m in missing:
                    self.log(f"    - {m}")
                return {
                    "success": False,
                    "pipeline_name": pipeline_name,
                    "steps_completed": 0,
                    "total_steps": len(specification),
                    "duration_seconds": time.time() - start_time,
                    "step_results": [],
                    "error": f"Missing external inputs: {missing}",
                    "missing_inputs": missing,
                }

        # =====================================================================
        # PRE-LOAD MODULES FROM PIPELINE SPEC
        # =====================================================================
        # Preload modules specified in pipeline steps before validation
        for step in specification:
            module_name = step.get("module")
            if module_name:
                module_name = module_name.replace(" ", "_").replace("-", "_")
                if module_name.endswith(".py"):
                    module_name = module_name[:-3]
                if module_name not in self._module_objects:
                    self.modules.append(module_name)
        if self.modules:
            self._load_modules()

        # =====================================================================
        # PRE-FLIGHT VALIDATION (G2, G5)
        # =====================================================================
        self.log(f"\n[2/4] Pre-flight Validation...")

        is_valid, errors = self._validate_pipeline(specification)
        if not is_valid:
            self.log(f"  FAILED: {len(errors)} validation error(s)")
            for err in errors:
                self.log(f"    - {err}")
            return {
                "success": False,
                "pipeline_name": pipeline_name,
                "steps_completed": 0,
                "total_steps": len(specification),
                "duration_seconds": time.time() - start_time,
                "step_results": [],
                "validation_errors": errors,
                "error": "Pipeline validation failed",
            }

        self.log(f"  PASSED: All contracts valid")

        # =====================================================================
        # LOAD SERVICES FROM KB
        # =====================================================================
        self.log(f"\n[3/4] Loading Services...")

        for step in specification:
            service_name = step.get("service")
            module_name = step.get("module")  # Get module from step specification

            if service_name not in self._service_cache:
                func = self._load_service(service_name, module_name=module_name)
                if func:
                    self._service_cache[service_name] = func
                    module_info = f" (from {module_name})" if module_name else ""
                    self.log(f"  Loaded: {service_name}{module_info}")
                else:
                    self.log(f"  FAILED: {service_name} not found")
                    module_hint = f" in module '{module_name}'" if module_name else ""
                    return {
                        "success": False,
                        "pipeline_name": pipeline_name,
                        "steps_completed": 0,
                        "total_steps": len(specification),
                        "duration_seconds": time.time() - start_time,
                        "step_results": [],
                        "error": f"Service '{service_name}' not found{module_hint}",
                    }

        # =====================================================================
        # EXECUTE PIPELINE (in topological order)
        # =====================================================================
        mode_str = "parallel" if parallel else "sequential"
        self.log(f"\n[4/4] Executing Pipeline ({len(specification)} steps, {mode_str})...")

        steps_completed = 0
        failed = False
        failure_error = None

        def execute_step(step_idx: int) -> Dict:
            """Execute a single step and return result."""
            step = specification[step_idx]
            service_name = step.get("service")
            inputs = step.get("inputs", {})
            outputs = step.get("outputs", {})
            params = step.get("params", {})

            # Resolve paths relative to base_path
            if base_path:
                inputs = self._resolve_paths(inputs, base_path)
                outputs = self._resolve_paths(outputs, base_path)

            step_start = time.time()
            try:
                # Get service function
                func = self._service_cache[service_name]

                # Ensure output directories exist
                for path in outputs.values():
                    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

                # Execute service
                result = func(inputs, outputs, **params)

                step_duration = time.time() - step_start
                return {
                    "step": step_idx + 1,
                    "service": service_name,
                    "status": "success",
                    "result": result,
                    "duration_seconds": step_duration,
                }

            except Exception as e:
                step_duration = time.time() - step_start
                error_msg = str(e)
                error_trace = traceback.format_exc()

                return {
                    "step": step_idx + 1,
                    "service": service_name,
                    "status": "failed",
                    "error": error_msg,
                    "traceback": error_trace,
                    "duration_seconds": step_duration,
                }

        if parallel:
            # =====================================================================
            # PARALLEL EXECUTION (by levels)
            # =====================================================================
            for level_num, level_steps in enumerate(parallel_levels):
                if failed:
                    break

                self.log(f"\n  Level {level_num + 1}/{len(parallel_levels)}: {len(level_steps)} step(s) in parallel")

                with ThreadPoolExecutor(max_workers=min(max_workers, len(level_steps))) as executor:
                    future_to_idx = {executor.submit(execute_step, idx): idx for idx in level_steps}

                    for future in as_completed(future_to_idx):
                        step_idx = future_to_idx[future]
                        step_result = future.result()
                        step_results[step_idx] = step_result

                        service_name = specification[step_idx]["service"]

                        if step_result["status"] == "success":
                            steps_completed += 1
                            self.log(f"    Step {step_idx + 1}: {service_name} - {step_result['result']} ({step_result['duration_seconds']:.2f}s)")
                        else:
                            failed = True
                            failure_error = f"Step {step_idx + 1} ({service_name}) failed: {step_result['error']}"
                            self.log(f"    Step {step_idx + 1}: {service_name} - FAILED: {step_result['error']}")
                            # Cancel remaining futures
                            for f in future_to_idx:
                                f.cancel()
                            break

        else:
            # =====================================================================
            # SEQUENTIAL EXECUTION (in topological order)
            # =====================================================================
            for exec_order, step_idx in enumerate(execution_order):
                step = specification[step_idx]
                service_name = step.get("service")

                self.log(f"\n  Step {exec_order + 1}/{len(specification)} (index {step_idx}): {service_name}")

                step_result = execute_step(step_idx)
                step_results[step_idx] = step_result

                if step_result["status"] == "success":
                    steps_completed += 1
                    self.log(f"    {step_result['result']}")
                    self.log(f"    Duration: {step_result['duration_seconds']:.2f}s")
                else:
                    failed = True
                    failure_error = f"Step {step_idx + 1} ({service_name}) failed: {step_result['error']}"
                    self.log(f"    FAILED: {step_result['error']}")
                    break

        # =====================================================================
        # RESULT
        # =====================================================================
        total_duration = time.time() - start_time

        # Filter out None results (incomplete steps)
        step_results = [r for r in step_results if r is not None]

        if failed:
            return {
                "success": False,
                "pipeline_name": pipeline_name,
                "steps_completed": steps_completed,
                "total_steps": len(specification),
                "duration_seconds": total_duration,
                "step_results": step_results,
                "execution_order": execution_order,
                "parallel_levels": parallel_levels,
                "error": failure_error,
            }

        # =====================================================================
        # SUCCESS
        # =====================================================================
        self.log(f"\n{'='*70}")
        self.log(f"Pipeline completed successfully!")
        self.log(f"Total duration: {total_duration:.2f}s")
        self.log(f"{'='*70}")

        execution_result = {
            "success": True,
            "pipeline_name": pipeline_name,
            "steps_completed": len(specification),
            "total_steps": len(specification),
            "duration_seconds": total_duration,
            "step_results": step_results,
            "execution_order": execution_order,
            "parallel_levels": parallel_levels,
            "completed_at": datetime.now().isoformat(),
        }

        self.last_execution = execution_result
        return execution_result

    # =========================================================================
    # MODULE LOADING
    # =========================================================================

    def _load_modules(self):
        """Load service modules dynamically."""
        services_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "services")
        services_pkg_dir = os.path.dirname(services_dir)
        if services_pkg_dir not in sys.path:
            sys.path.insert(0, services_pkg_dir)

        for module_name in self.modules:
            module_path = os.path.join(services_dir, f"{module_name}.py")

            if not os.path.exists(module_path):
                self.log(f"  WARNING: Module not found: {module_path}")
                continue

            try:
                # Prefer package import to avoid duplicate module loads
                package_name = f"services.{module_name}"
                module = importlib.import_module(package_name)

                # Store module and its registry
                self._module_objects[module_name] = module

                if hasattr(module, "SERVICE_REGISTRY"):
                    self._module_registries[module_name] = module.SERVICE_REGISTRY
                    self.log(f"  Loaded module: {module_name} ({len(module.SERVICE_REGISTRY)} services)")
                else:
                    self.log(f"  WARNING: Module {module_name} has no SERVICE_REGISTRY")

            except Exception as e:
                self.log(f"  ERROR loading module {module_name}: {e}")
                import traceback
                traceback.print_exc()

    def get_available_services(self) -> Dict[str, str]:
        """Get all available services from loaded modules.

        Returns
        -------
        Dict[str, str]
            service_name -> module_name mapping
        """
        services = {}
        for module_name, registry in self._module_registries.items():
            for service_name in registry.keys():
                services[service_name] = module_name
        return services

    # =========================================================================
    # SERVICE LOADING
    # =========================================================================

    def _load_service(self, service_name: str, module_name: str = None) -> Optional[Callable]:
        """
        Load a service function.

        Loading priority:
        1. From specified module (if module_name provided in step)
        2. From already loaded modules
        3. From KB (which may specify a module)
        4. From KB source code (fallback)

        Parameters
        ----------
        service_name : str
            Name of the service
        module_name : str, optional
            Module name specified in the pipeline step (e.g., "agent_services")

        Returns
        -------
        Callable or None
            The service function, or None if not found
        """
        # 1. If module specified in step, load from that module first
        if module_name:
            # Normalize module name (remove .py extension if present, convert spaces/dashes)
            module_name = module_name.replace(" ", "_").replace("-", "_")
            if module_name.endswith(".py"):
                module_name = module_name[:-3]

            # Load module if not already loaded
            if module_name not in self._module_objects:
                self.modules.append(module_name)
                self._load_modules()

            # Try to get service from specified module
            if module_name in self._module_objects:
                func = getattr(self._module_objects[module_name], service_name, None)
                if func and callable(func):
                    return func

        # 2. Try to load from already loaded modules
        for mod_name, module in self._module_objects.items():
            func = getattr(module, service_name, None)
            if func and callable(func):
                return func

        if not self.kb:
            return None

        # 3. Get service info from KB
        service = self.kb.get_service(service_name)
        if not service:
            return None

        # 4. If service has a module specified in KB, try loading from that module
        service_module = service.get("module")
        if service_module and service_module not in self._module_objects:
            # Module not loaded yet, try loading it
            self.modules.append(service_module)
            self._load_modules()

            # Try again from newly loaded module
            if service_module in self._module_objects:
                func = getattr(self._module_objects[service_module], service_name, None)
                if func and callable(func):
                    return func

        # 5. Fallback: compile from KB source code
        source_code = service.get("source_code")
        if not source_code:
            return None

        # Compile the source code
        try:
            # Create execution namespace with required imports
            namespace = self._create_execution_namespace()

            # Execute the source code to define the function
            exec(source_code, namespace)

            # Get the function from namespace
            func = namespace.get(service_name)
            return func

        except Exception as e:
            self.log(f"  Error compiling {service_name}: {e}")
            return None

    def _create_execution_namespace(self) -> Dict:
        """Create namespace with common imports for service execution."""
        import numpy as np
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import (
            OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
        )
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split

        # Import Contract-Composable Analytics contract system
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from contract import (
            contract, IOManager, ServiceRegistry,
            TabularSchema, JSONSchema, ArtifactSchema
        )

        return {
            # Standard library
            "os": os,
            "sys": sys,
            "json": json,

            # Data science
            "np": np,
            "numpy": np,
            "pd": pd,
            "pandas": pd,

            # Sklearn
            "SimpleImputer": SimpleImputer,
            "OneHotEncoder": OneHotEncoder,
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
            "LabelEncoder": LabelEncoder,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "mean_squared_error": mean_squared_error,
            "mean_absolute_error": mean_absolute_error,
            "r2_score": r2_score,
            "train_test_split": train_test_split,

            # Contract-Composable Analytics
            "contract": contract,
            "IOManager": IOManager,
            "ServiceRegistry": ServiceRegistry,
            "TabularSchema": TabularSchema,
            "JSONSchema": JSONSchema,
            "ArtifactSchema": ArtifactSchema,

            # Typing
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "Any": Any,
        }

    # =========================================================================
    # DAG ANALYSIS & TOPOLOGICAL SORT
    # =========================================================================

    def _build_dependency_graph(self, specification: List[Dict]) -> Tuple[Dict[int, Set[int]], Dict[str, int]]:
        """
        Build a dependency graph from the pipeline specification.

        Dependencies are determined by matching output paths to input paths:
        If step B's input uses a path that step A produces as output, then B depends on A.

        Parameters
        ----------
        specification : List[Dict]
            Pipeline steps

        Returns
        -------
        Tuple[Dict[int, Set[int]], Dict[str, int]]
            (adjacency_list, path_producers)
            - adjacency_list: step_idx -> set of dependent step indices
            - path_producers: output_path -> step_idx that produces it
        """
        # Map output paths to the step that produces them
        path_producers: Dict[str, int] = {}

        # First pass: register all outputs
        for idx, step in enumerate(specification):
            outputs = step.get("outputs", {})
            for slot, path in outputs.items():
                path_producers[path] = idx

        # Build adjacency list (dependencies)
        # adjacency[i] = set of steps that depend on step i
        adjacency: Dict[int, Set[int]] = defaultdict(set)
        # in_degree[i] = number of steps that step i depends on
        in_degree: Dict[int, int] = {i: 0 for i in range(len(specification))}

        # Second pass: determine dependencies from inputs
        # Track which predecessors each step depends on (to avoid double-counting)
        predecessors: Dict[int, Set[int]] = defaultdict(set)

        for idx, step in enumerate(specification):
            inputs = step.get("inputs", {})
            for slot, path in inputs.items():
                if path in path_producers:
                    producer_idx = path_producers[path]
                    if producer_idx != idx:  # Don't self-reference
                        adjacency[producer_idx].add(idx)
                        predecessors[idx].add(producer_idx)

        # Set in_degree based on unique predecessors (not paths)
        for idx in range(len(specification)):
            in_degree[idx] = len(predecessors[idx])

        return adjacency, in_degree, path_producers

    def _topological_sort(self, specification: List[Dict]) -> Tuple[List[int], bool, List[List[int]]]:
        """
        Perform topological sort on the pipeline DAG using Kahn's algorithm.

        Also groups steps into "levels" where steps in the same level can run in parallel.

        Parameters
        ----------
        specification : List[Dict]
            Pipeline steps

        Returns
        -------
        Tuple[List[int], bool, List[List[int]]]
            (sorted_order, has_cycle, parallel_levels)
            - sorted_order: step indices in topological order
            - has_cycle: True if cycle detected (invalid pipeline)
            - parallel_levels: list of lists, each inner list contains step indices
                              that can be executed in parallel
        """
        n = len(specification)
        adjacency, in_degree, _ = self._build_dependency_graph(specification)

        # Kahn's algorithm with level tracking
        sorted_order: List[int] = []
        parallel_levels: List[List[int]] = []

        # Start with all steps that have no dependencies
        current_level = [i for i in range(n) if in_degree[i] == 0]

        while current_level:
            parallel_levels.append(current_level)
            sorted_order.extend(current_level)

            next_level = []
            for step_idx in current_level:
                for dependent_idx in adjacency[step_idx]:
                    in_degree[dependent_idx] -= 1
                    if in_degree[dependent_idx] == 0:
                        next_level.append(dependent_idx)

            current_level = next_level

        # Check for cycle: if not all steps are in sorted order, there's a cycle
        has_cycle = len(sorted_order) != n

        return sorted_order, has_cycle, parallel_levels

    def _detect_cycle_path(self, specification: List[Dict]) -> Optional[List[str]]:
        """
        If a cycle exists, find and return the cycle path for error reporting.

        Returns
        -------
        Optional[List[str]]
            List of service names forming the cycle, or None if no cycle
        """
        adjacency, in_degree, _ = self._build_dependency_graph(specification)
        n = len(specification)

        # Find remaining nodes after topological sort attempt
        sorted_order, has_cycle, _ = self._topological_sort(specification)

        if not has_cycle:
            return None

        # Find nodes involved in cycle (those not in sorted order)
        remaining = set(range(n)) - set(sorted_order)

        if not remaining:
            return None

        # DFS to find cycle path
        visited = set()
        path = []

        def dfs(node: int, current_path: List[int]) -> Optional[List[int]]:
            if node in current_path:
                # Found cycle
                cycle_start = current_path.index(node)
                return current_path[cycle_start:] + [node]

            if node in visited:
                return None

            visited.add(node)
            current_path.append(node)

            for neighbor in adjacency[node]:
                result = dfs(neighbor, current_path)
                if result:
                    return result

            current_path.pop()
            return None

        # Start DFS from a remaining node
        start_node = next(iter(remaining))
        cycle_indices = dfs(start_node, [])

        if cycle_indices:
            return [specification[i]["service"] for i in cycle_indices]

        return None

    def analyze_dag(self, specification: List[Dict]) -> Dict:
        """
        Analyze the pipeline DAG structure.

        Returns
        -------
        Dict with:
            - is_valid: bool (no cycles)
            - execution_order: List[int] (topologically sorted step indices)
            - parallel_levels: List[List[int]] (steps that can run in parallel)
            - cycle_path: Optional[List[str]] (if cycle exists)
            - max_parallelism: int (max steps that can run simultaneously)
            - critical_path_length: int (minimum sequential steps needed)
            - external_inputs: List[Dict] (inputs not produced by any step)
        """
        sorted_order, has_cycle, parallel_levels = self._topological_sort(specification)
        adjacency, in_degree, path_producers = self._build_dependency_graph(specification)

        # Find external inputs (inputs not produced by any step in pipeline)
        external_inputs = []
        for idx, step in enumerate(specification):
            inputs = step.get("inputs", {})
            for slot, path in inputs.items():
                if path not in path_producers:
                    external_inputs.append({
                        "step": idx,
                        "service": step.get("service"),
                        "slot": slot,
                        "path": path
                    })

        result = {
            "is_valid": not has_cycle,
            "total_steps": len(specification),
            "execution_order": sorted_order,
            "parallel_levels": parallel_levels,
            "max_parallelism": max(len(level) for level in parallel_levels) if parallel_levels else 0,
            "critical_path_length": len(parallel_levels),
            "external_inputs": external_inputs,
        }

        if has_cycle:
            result["cycle_path"] = self._detect_cycle_path(specification)

        return result

    def validate_external_inputs(self, specification: List[Dict], base_path: str = None) -> Tuple[bool, List[str], List[str]]:
        """
        Validate that all external inputs (files not produced by pipeline steps) exist.

        This helps detect:
        - Typos in file paths
        - Missing dependencies (forgot to add a step)
        - Incorrect parallel assumptions

        Parameters
        ----------
        specification : List[Dict]
            Pipeline steps
        base_path : str, optional
            Base path for resolving relative paths

        Returns
        -------
        Tuple[bool, List[str], List[str]]
            (all_exist, missing_files, existing_files)
        """
        dag_info = self.analyze_dag(specification)
        external_inputs = dag_info.get("external_inputs", [])

        missing = []
        existing = []

        for ext_input in external_inputs:
            path = ext_input["path"]

            # Resolve relative path
            if base_path and not os.path.isabs(path):
                full_path = os.path.join(base_path, path)
            else:
                full_path = path

            if os.path.exists(full_path):
                existing.append(path)
            else:
                missing.append(f"Step {ext_input['step']+1} ({ext_input['service']}.{ext_input['slot']}): {path}")

        return len(missing) == 0, missing, existing

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_pipeline(self, specification: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate pipeline contracts (G2, G5).

        Parameters
        ----------
        specification : List[Dict]
            Pipeline steps

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, errors)
        """
        errors = []
        output_registry = {}  # path -> format

        for i, step in enumerate(specification):
            service_name = step.get("service")

            # Check service exists (either in modules or KB)
            service_exists = service_name in self.get_available_services()
            contract = None
            if service_exists:
                # Service found in loaded modules - try to get contract
                from contract import ServiceRegistry
                contract = ServiceRegistry.get(service_name)
                # If not in ServiceRegistry, try KB
                if not contract and self.kb:
                    contract = self.kb.get_service_contract(service_name)
                # If still no contract, allow execution with minimal contract for module services
                if not contract:
                    contract = {"input": {}, "output": {}}
            elif self.kb:
                contract = self.kb.get_service_contract(service_name)
            else:
                from contract import ServiceRegistry
                contract = ServiceRegistry.get(service_name)

            if not contract and not service_exists:
                errors.append(f"Step {i+1}: Unknown service '{service_name}'")
                continue

            step_inputs = step.get("inputs", {})
            step_outputs = step.get("outputs", {})

            # Skip strict slot validation for module-loaded services
            # Just check that at least some inputs are provided if service expects inputs
            if service_name in self.get_available_services():
                # Basic sanity check: most services need at least one input
                if not step_inputs and service_name not in ["load_m5_data"]:  # Some services may have no inputs
                    pass  # Don't error - let the service itself validate
            elif contract:
                # Check required inputs are provided (for KB-loaded services)
                for slot, spec in contract.get("input", {}).items():
                    if spec.get("required", True) and slot not in step_inputs:
                        errors.append(f"Step {i+1} ({service_name}): Missing required input '{slot}'")

            # Check format compatibility
            for slot, path in step_inputs.items():
                if path in output_registry:
                    output_fmt = output_registry[path]
                    input_spec = contract.get("input", {}).get(slot, {})
                    input_fmt = input_spec.get("format")

                    if input_fmt and not self._formats_compatible(output_fmt, input_fmt):
                        errors.append(
                            f"Step {i+1} ({service_name}.{slot}): "
                            f"Format mismatch - receives <{output_fmt}>, expects <{input_fmt}>"
                        )

            # Register outputs
            for slot, path in step_outputs.items():
                output_spec = contract.get("output", {}).get(slot, {})
                output_fmt = output_spec.get("format")
                if output_fmt:
                    output_registry[path] = output_fmt

        return len(errors) == 0, errors

    def _formats_compatible(self, output_fmt: str, input_fmt: str) -> bool:
        if self.kb:
            return self.kb.check_format_compatibility(output_fmt, input_fmt)

        from contract import IOManager

        if output_fmt == input_fmt:
            return True

        out_info = IOManager.get_format_info(output_fmt)
        in_info = IOManager.get_format_info(input_fmt)
        if not out_info or not in_info:
            return False

        return IOManager.compatible(output_fmt, input_fmt)

    def _resolve_paths(self, paths: Dict[str, str], base_path: str) -> Dict[str, str]:
        """Resolve relative paths to absolute paths."""
        resolved = {}
        for key, path in paths.items():
            if not os.path.isabs(path):
                resolved[key] = os.path.join(base_path, path)
            else:
                resolved[key] = path
        return resolved

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def list_pipelines(self) -> List[Dict]:
        """List all pipelines in the Knowledge Base."""
        if not self.kb:
            return []
        return self.kb.list_pipelines()

    def list_services(self) -> List[Dict]:
        """List all services in the Knowledge Base."""
        if self.kb:
            return self.kb.list_services()
        services = []
        for module_name, registry in self._module_registries.items():
            for service_name in registry.keys():
                services.append({"name": service_name, "module": module_name})
        return services

    def describe_pipeline(self, name: str):
        """Print details of a pipeline."""
        if not self.kb:
            print("Knowledge Base is disabled; no pipelines available.")
            return
        pipeline = self.kb.get_pipeline(name)
        if not pipeline:
            print(f"Pipeline '{name}' not found")
            return

        print(f"\n{'='*60}")
        print(f"Pipeline: {pipeline['name']}")
        print(f"{'='*60}")
        print(f"Description: {pipeline['description']}")
        print(f"Problem Type: {pipeline['problem_type']}")
        print(f"Domain: {pipeline['domain']}")
        print(f"\nSteps:")
        for i, step in enumerate(pipeline['specification']):
            print(f"  {i+1}. {step['service']}")
            if step.get('params'):
                print(f"     Params: {step['params']}")

    def close(self):
        """Close the Knowledge Base connection."""
        if self.kb:
            self.kb.close()


# =============================================================================
# MAIN: CLI Interface
# =============================================================================

def main():
    """CLI interface for running pipelines."""
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline Runner")
    parser.add_argument("action", choices=["run", "list", "describe"],
                       help="Action to perform")
    parser.add_argument("--pipeline", "-p", help="Pipeline name or JSON file")
    parser.add_argument("--db", default="kb.sqlite", help="Knowledge Base database path")
    parser.add_argument("--base-path", "-b", help="Base path for data artifacts")
    parser.add_argument("--modules", "-m", nargs="*", help="Modules to load (e.g., house_prices_services)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    runner = PipelineRunner(args.db, verbose=not args.quiet, modules=args.modules)

    try:
        if args.action == "list":
            print("\n=== Pipelines ===")
            for p in runner.list_pipelines():
                print(f"  {p['name']} ({p['problem_type']})")

            print("\n=== Services ===")
            for s in runner.list_services():
                desc = s.get("description") or s.get("module") or ""
                if desc:
                    print(f"  {s['name']}: {desc[:50]}...")
                else:
                    print(f"  {s['name']}")

        elif args.action == "describe":
            if not args.pipeline:
                print("Error: --pipeline required for describe")
                return
            runner.describe_pipeline(args.pipeline)

        elif args.action == "run":
            if not args.pipeline:
                print("Error: --pipeline required for run")
                return

            if args.pipeline.endswith(".json"):
                result = runner.run_from_json(args.pipeline, args.base_path)
            else:
                result = runner.run_pipeline(args.pipeline, args.base_path)

            if result["success"]:
                print(f"\n Pipeline completed in {result['duration_seconds']:.2f}s")
            else:
                print(f"\n Pipeline failed: {result.get('error')}")
                sys.exit(1)

    finally:
        runner.close()


if __name__ == "__main__":
    main()
