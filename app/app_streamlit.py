"""
Pipeline Workbench
=====================
Redesigned layout matching the 'Blue Enterprise' screenshot.
Features: 5-Tab Navigation, 3-Column Studio, Custom Header.
"""

import ast
import json
import os
import sys
import time
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from contract import ServiceRegistry
from pipeline_runner import PipelineRunner
import recommender
# from snake_page import render_snake_game  # Easter egg, removed for publication

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get the directory where this script lives (app folder)
WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure local imports work regardless of working directory
if WEBAPP_DIR not in sys.path:
    sys.path.insert(0, WEBAPP_DIR)

SERVICES_DIR = os.path.join(WEBAPP_DIR, "services")
DATASETS_DIR = os.path.join(WEBAPP_DIR, "datasets")
ARTIFACTS_DIR = os.path.join(WEBAPP_DIR, "artifacts")
STORAGE_DIR = os.path.join(WEBAPP_DIR, "storage")
PIPELINE_SAVE_DIR = os.path.join(STORAGE_DIR, "pipelines")
KB_PATH = os.path.join(WEBAPP_DIR, "kb.sqlite")

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(PIPELINE_SAVE_DIR, exist_ok=True)

st.set_page_config(
    page_title="Contract-Composable Analytics",
    page_icon="🧱",
    layout="wide",
)

# Custom CSS for fixed viewport height with scrollable panels
st.markdown("""
<style>
    /* Force page to fit viewport exactly */
    html, body, [data-testid="stAppViewContainer"] {
        height: 100vh !important;
        overflow: hidden !important;
    }
    
    /* Hide default streamlit padding/margin for tighter fit */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0;
        max-width: 100%;
        height: calc(100vh - 80px);
        overflow: hidden;
    }
    
    /* Fix main content area height to viewport */
    section[data-testid="stMain"] {
        height: 100vh;
        overflow: hidden;
    }
    
    /* Make columns scrollable only when needed */
    div[data-testid="column"] {
        height: calc(100vh - 140px);
        overflow: auto;
        padding-right: 8px;
    }
    
    /* Style scrollbars globally - thin and subtle */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(136, 136, 136, 0.4);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(85, 85, 85, 0.7);
    }
    
    /* Expander content scrollable */
    .streamlit-expanderContent {
        max-height: calc(100vh - 250px);
        overflow: auto;
    }
    
    /* Code blocks scrollable */
    pre, code {
        max-height: 400px;
        overflow: auto !important;
    }
    
    /* JSON viewer scrollable */
    [data-testid="stJson"] {
        max-height: 450px;
        overflow: auto !important;
    }
    
    /* Text areas with proper scrolling */
    textarea {
        max-height: 500px !important;
    }
    
    /* Tab content panels scrollable */
    [data-testid="stVerticalBlockBorderWrapper"] {
        max-height: calc(100vh - 200px);
        overflow: auto;
    }
    
    /* Log/output containers */
    .stCodeBlock {
        max-height: 350px;
        overflow: auto !important;
    }
    
    /* Service list in sidebar */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        max-height: calc(100vh - 150px);
        overflow: auto;
    }
    
    /* Dataframe containers */
    [data-testid="stDataFrame"] {
        max-height: 400px;
        overflow: auto !important;
    }
    
    /* Reduce title size */
    h1 {
        font-size: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Microservice Editor - Module browser panel (scrollable independently) */
    .module-browser-panel {
        max-height: calc(100vh - 200px);
        overflow-y: auto;
        overflow-x: hidden;
        padding-right: 5px;
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Microservice Editor - Code editor panel (scrollable independently) */
    .code-editor-panel {
        max-height: calc(100vh - 180px);
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    /* Services list within module browser */
    .services-list {
        max-height: 300px;
        overflow-y: auto;
        padding: 5px;
        background: rgba(128, 128, 128, 0.05);
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# STATE & HELPERS
# =============================================================================

st.title("Contract-Composable Analytics: A Collaborative and Modular Architecture for Data Analytics")

def init_session_state():
    if 'pipeline_steps' not in st.session_state:
        st.session_state.pipeline_steps = []
    if 'execution_log' not in st.session_state:
        st.session_state.execution_log = []
    if 'selected_service_module' not in st.session_state:
        st.session_state.selected_service_module = "All"
    if 'wb_selected_module' not in st.session_state:
        st.session_state.wb_selected_module = None
    if 'wb_selected_service' not in st.session_state:
        st.session_state.wb_selected_service = None
    if 'view_file' not in st.session_state:
        st.session_state.view_file = None
    if 'storage_current_dir' not in st.session_state:
        st.session_state.storage_current_dir = STORAGE_DIR
    if 'pipeline_base_dir' not in st.session_state:
        st.session_state.pipeline_base_dir = STORAGE_DIR
    if 'pipeline_json_editor' not in st.session_state:
        st.session_state.pipeline_json_editor = ""
    # Center-panel JSON input (lets users paste a full pipeline spec without needing the right panel).
    if 'pipeline_json_input_main' not in st.session_state:
        st.session_state.pipeline_json_input_main = ""
    if 'pipeline_input_mode_main' not in st.session_state:
        st.session_state.pipeline_input_mode_main = "Builder"
    if 'pipeline_json_last_synced' not in st.session_state:
        st.session_state.pipeline_json_last_synced = ""
    if 'pipeline_steps_last_json' not in st.session_state:
        st.session_state.pipeline_steps_last_json = ""
    if '_pipeline_json_sync_value' not in st.session_state:
        st.session_state._pipeline_json_sync_value = None
    if '_pipeline_json_sync_force' not in st.session_state:
        st.session_state._pipeline_json_sync_force = False
    if '_pipeline_toast' not in st.session_state:
        st.session_state._pipeline_toast = None
    if 'copied_path_scope' not in st.session_state:
        st.session_state.copied_path_scope = "storage"

def list_service_modules(services_dir: str) -> List[str]:
    services_path = Path(services_dir)
    modules = []
    for path in services_path.glob("*.py"):
        if path.name in {"__init__.py"} or path.name.startswith("_"):
            continue
        modules.append(path.stem)
    return sorted(modules)

@st.cache_resource
def load_services_from_modules(services_dir: str) -> List[Dict[str, Any]]:
    ServiceRegistry._services = {}
    for module_name in list_service_modules(services_dir):
        try:
            importlib.import_module(f"services.{module_name}")
        except Exception:
            continue

    services = []
    for name, contract in ServiceRegistry.list_all().items():
        func = contract.get("function")
        module_name = None
        if func and getattr(func, "__module__", None):
            module_name = func.__module__.split(".")[-1]
        services.append({
            "name": name,
            "module": module_name or "unknown",
            "tags": json.dumps(contract.get("tags", [])),
            "input_contract": json.dumps(contract.get("input", {})),
            "output_contract": json.dumps(contract.get("output", {})),
            "parameters": json.dumps({}),
        })

    return services

init_session_state()
SERVICE_MODULES = list_service_modules(SERVICES_DIR)
all_services = load_services_from_modules(SERVICES_DIR)

# Categorize Services by Tags (Virtual Categories)
# Categorize Services by Tags (Dynamic)
categories = {"All": []}

for s in all_services:
    categories["All"].append(s['name'])
    
    # Extract tags
    tags = s.get('tags', [])
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except:
            tags = []
    
    # Use tags as categories
    if not tags:
        if "Uncategorized" not in categories:
            categories["Uncategorized"] = []
        categories["Uncategorized"].append(s['name'])
    else:
        for tag in tags:
            # Format tag: "data-loading" -> "Data Loading"
            cat_name = tag.replace("-", " ").title()
            if cat_name not in categories:
                categories[cat_name] = []
            categories[cat_name].append(s['name'])

# Deduplicate
for k in categories:
    categories[k] = sorted(list(set(categories[k])))

# =============================================================================
# COMPONENT RENDERERS
# =============================================================================

def _has_contract_decorator(node: ast.FunctionDef) -> bool:
    for deco in node.decorator_list:
        if isinstance(deco, ast.Name) and deco.id == "contract":
            return True
        if isinstance(deco, ast.Call):
            if isinstance(deco.func, ast.Name) and deco.func.id == "contract":
                return True
    return False

def _safe_literal_eval(node: ast.AST) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return None

def load_service_param_overrides(services_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    overrides: Dict[str, Dict[str, Dict[str, Any]]] = {}
    services_path = Path(services_dir)
    for py_file in services_path.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not _has_contract_decorator(node):
                continue
            params: Dict[str, Dict[str, Any]] = {}
            args = node.args.args
            defaults = node.args.defaults
            default_map: Dict[str, Any] = {}
            if defaults:
                for arg, default in zip(args[-len(defaults):], defaults):
                    default_map[arg.arg] = _safe_literal_eval(default)
            if node.args.kwonlyargs:
                for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                    default_map[arg.arg] = _safe_literal_eval(default)
            all_args = list(args) + list(node.args.kwonlyargs)
            for arg in all_args:
                if arg.arg in ("inputs", "outputs"):
                    continue
                info: Dict[str, Any] = {"type": "any"}
                if arg.annotation is not None:
                    try:
                        info["type"] = ast.unparse(arg.annotation)
                    except Exception:
                        info["type"] = "any"
                if arg.arg in default_map:
                    info["default"] = default_map[arg.arg]
                params[arg.arg] = info
            if params:
                overrides[node.name] = params
    return overrides

SERVICE_PARAM_OVERRIDES = load_service_param_overrides(SERVICES_DIR)

def get_workspace_dir() -> str:
    """Return a safe workspace directory under storage."""
    storage_root = Path(STORAGE_DIR).resolve()
    try:
        workspace = Path(st.session_state.pipeline_base_dir).resolve()
    except Exception:
        workspace = storage_root
    if not str(workspace).startswith(str(storage_root)):
        workspace = storage_root
        st.session_state.pipeline_base_dir = str(storage_root)
    return str(workspace)

def get_service_param_meta(service_name: str) -> Dict[str, Dict[str, Any]]:
    svc_def = next((s for s in all_services if s['name'] == service_name), None)
    base_meta: Dict[str, Dict[str, Any]] = {}
    if svc_def:
        try:
            base_meta = json.loads(svc_def.get('parameters') or "{}")
        except Exception:
            base_meta = {}
    overrides = SERVICE_PARAM_OVERRIDES.get(service_name, {})
    if not overrides:
        return base_meta
    merged: Dict[str, Dict[str, Any]] = {name: dict(info) for name, info in overrides.items()}
    for name, info in base_meta.items():
        if name not in merged:
            continue
        if "default" not in merged[name] and "default" in info:
            merged[name]["default"] = info["default"]
        if (not merged[name].get("type") or merged[name].get("type") == "any") and info.get("type"):
            merged[name]["type"] = info["type"]
        if "description" not in merged[name] and info.get("description"):
            merged[name]["description"] = info.get("description")
    return merged

def infer_param_type(param_info: Dict[str, Any], current_value: Any) -> str:
    raw_type = param_info.get("type") if param_info else None
    type_str = str(raw_type).lower() if raw_type else ""
    if not type_str and current_value is not None:
        if isinstance(current_value, bool):
            return "bool"
        if isinstance(current_value, int):
            return "int"
        if isinstance(current_value, float):
            return "float"
        if isinstance(current_value, (list, tuple, set)):
            return "list"
        if isinstance(current_value, dict):
            return "dict"
        if isinstance(current_value, str):
            return "str"
    if "bool" in type_str:
        return "bool"
    if "int" in type_str:
        return "int"
    if "float" in type_str or "double" in type_str:
        return "float"
    if "list" in type_str or "tuple" in type_str or "set" in type_str:
        return "list"
    if "dict" in type_str or "map" in type_str:
        return "dict"
    if "str" in type_str or "string" in type_str:
        return "str"
    return "any"

def coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None

def format_json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (tuple, set)):
        value = list(value)
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, indent=2)
        except Exception:
            return str(value)
    return str(value)

def pipeline_steps_to_json_text(steps: Any) -> str:
    """Serialize steps for UI editing. Uses default=str to avoid hard crashes on odd types."""
    return json.dumps(steps, indent=2, default=str)

def request_pipeline_json_sync(json_text: str, *, force: bool = False, toast: Optional[str] = None, icon: str = "✅"):
    """
    Queue a JSON editor sync for the *next rerun*.

    Streamlit doesn't allow mutating widget-backed session_state keys after the widget
    is instantiated in the current run. So we store a pending value and apply it
    early in the next run.
    """
    st.session_state._pipeline_json_sync_value = json_text
    st.session_state._pipeline_json_sync_force = force
    if toast:
        st.session_state._pipeline_toast = {"msg": toast, "icon": icon}

def maybe_sync_pipeline_json_views():
    """Keep main/side JSON editors in sync with `pipeline_steps` when safe."""
    # 1) Apply explicit pending sync request.
    pending = st.session_state.get("_pipeline_json_sync_value")
    if pending is not None:
        force = bool(st.session_state.get("_pipeline_json_sync_force"))
        st.session_state._pipeline_json_sync_value = None
        st.session_state._pipeline_json_sync_force = False

        if force:
            st.session_state.pipeline_json_editor = pending
            st.session_state.pipeline_json_input_main = pending
            st.session_state.pipeline_json_last_synced = pending
        else:
            last_synced = st.session_state.pipeline_json_last_synced
            editors_clean = (
                st.session_state.pipeline_json_editor in ("", last_synced)
                and st.session_state.pipeline_json_input_main in ("", last_synced)
            )
            if editors_clean:
                st.session_state.pipeline_json_editor = pending
                st.session_state.pipeline_json_input_main = pending
                st.session_state.pipeline_json_last_synced = pending

    # 2) Auto-sync from pipeline_steps when editors are clean.
    steps_json = pipeline_steps_to_json_text(st.session_state.pipeline_steps)
    st.session_state.pipeline_steps_last_json = steps_json

    last_synced = st.session_state.pipeline_json_last_synced
    editors_clean = (
        st.session_state.pipeline_json_editor in ("", last_synced)
        and st.session_state.pipeline_json_input_main in ("", last_synced)
    )
    if editors_clean and steps_json != last_synced:
        st.session_state.pipeline_json_editor = steps_json
        st.session_state.pipeline_json_input_main = steps_json
        st.session_state.pipeline_json_last_synced = steps_json

    # 3) Show any queued toast.
    toast_payload = st.session_state.get("_pipeline_toast")
    if toast_payload:
        try:
            st.toast(toast_payload.get("msg", ""), icon=toast_payload.get("icon", "✅"))
        finally:
            st.session_state._pipeline_toast = None

def extract_and_normalize_pipeline_steps(data: Any) -> List[Dict[str, Any]]:
    """
    Accept common pipeline JSON shapes and return a normalized list[step].

    Supported inputs:
    - [ {service, module?, inputs?, outputs?, params?}, ... ]
    - { "steps": [...] }
    - { "specification": [...] }  (save format in this app)
    - { "specification": { "steps": [...] } }
    """
    steps: Any = None
    if isinstance(data, list):
        steps = data
    elif isinstance(data, dict):
        if "specification" in data:
            spec = data.get("specification")
            if isinstance(spec, dict) and "steps" in spec:
                steps = spec.get("steps")
            else:
                steps = spec
        elif "steps" in data:
            steps = data.get("steps")

    if not isinstance(steps, list):
        raise ValueError(
            "Expected a JSON list of steps, or an object with 'steps'/'specification'."
        )

    normalized: List[Dict[str, Any]] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"Step #{idx + 1} must be a JSON object.")
        if not step.get("service"):
            raise ValueError(f"Step #{idx + 1} is missing required key 'service'.")

        s = dict(step)
        s.setdefault("inputs", {})
        s.setdefault("outputs", {})
        s.setdefault("params", {})

        # Ensure the board UI won't crash on missing/invalid shapes.
        if s["inputs"] is None:
            s["inputs"] = {}
        if s["outputs"] is None:
            s["outputs"] = {}
        if s["params"] is None:
            s["params"] = {}

        if not isinstance(s["inputs"], dict):
            raise ValueError(f"Step #{idx + 1} key 'inputs' must be an object.")
        if not isinstance(s["outputs"], dict):
            raise ValueError(f"Step #{idx + 1} key 'outputs' must be an object.")
        if not isinstance(s["params"], dict):
            raise ValueError(f"Step #{idx + 1} key 'params' must be an object.")

        normalized.append(s)

    return normalized

def add_service_to_pipeline(service_name: str, insert_at: int = None, module_name: str = None):
    """Add a service to the pipeline state with default contract values.
    
    Args:
        service_name: Name of the service to add
        insert_at: Optional position to insert at
        module_name: Optional module to use for contract lookup (important when 
                     same service name exists in multiple modules)
    """
    # 1. Fetch Contract - prefer matching by module if provided
    svc_def = None
    if module_name:
        svc_def = next((s for s in all_services if s['name'] == service_name and s.get('module') == module_name), None)
    if not svc_def:
        svc_def = next((s for s in all_services if s['name'] == service_name), None)
    if not svc_def:
        st.error(f"Service {service_name} not found")
        return

    # Get module name from service definition if not provided
    if not module_name:
        module_name = svc_def.get('module', 'unknown')

    try:
        input_c = json.loads(svc_def['input_contract']) if svc_def.get('input_contract') else {}
        output_c = json.loads(svc_def['output_contract']) if svc_def.get('output_contract') else {}
        params_c = get_service_param_meta(service_name)
    except Exception:
        input_c, output_c, params_c = {}, {}, {}

    # 2. Auto-Wire Inputs (Try to match previous outputs)
    inputs = {}
    step_idx = len(st.session_state.pipeline_steps)

    # Default input paths (favor datasets/)
    for slot in input_c:
        slot_lower = slot.lower()
        if "train" in slot_lower:
            val = "datasets/train.csv"
        elif "test" in slot_lower:
            val = "datasets/test.csv"
        elif "valid" in slot_lower or "val" in slot_lower:
            val = "datasets/valid.csv"
        else:
            val = f"datasets/{slot}.csv"

        # Try to find matching output from previous steps
        if st.session_state.pipeline_steps:
            prev_step = st.session_state.pipeline_steps[-1]
            for p_out_slot, p_out_path in prev_step.get('outputs', {}).items():
                inputs[slot] = p_out_path
                break

        # If nothing wired, use default
        if slot not in inputs:
            inputs[slot] = val

    # 3. Default Outputs
    outputs = {}
    for slot in output_c:
        outputs[slot] = f"artifacts/{service_name}_{step_idx}_{slot}.csv"
        
    # 4. Default Params
    params = {}
    for p, details in params_c.items():
        params[p] = details.get('default', None)

    # 5. Append step with module field
    step = {
        "module": module_name,
        "service": service_name,
        "inputs": inputs,
        "outputs": outputs,
        "params": params
    }
    if insert_at is not None:
        st.session_state.pipeline_steps.insert(insert_at, step)
    else:
        st.session_state.pipeline_steps.append(step)


def render_resources_panel():
    """Left Column: Combined Resources (Files, Services & Pipelines)"""
    
    tab_files, tab_services, tab_pipelines = st.tabs(["📂 Files", "🧱 Services", "📋 Pipelines"])
    
    with tab_files:
        render_file_explorer()
        
    with tab_services:
        st.markdown("##### Service Catalog")
        
        # 1. Category Filter
        cat_names = sorted([k for k in categories.keys() if k != "All"])
        sel_cats = st.multiselect("Filter:", cat_names, placeholder="All Categories")
        
        # 2. Filter logic
        if sel_cats:
            filtered = set()
            for c in sel_cats:
                filtered.update(categories.get(c, []))
            available = sorted(list(filtered))
        else:
            available = sorted([s['name'] for s in all_services])

        # 3. Service List (Click to Add)
        st.caption("Click to add to pipeline workspace →")
        for i, svc_name in enumerate(available):
            col_add, col_name = st.columns([0.5, 4])
            if col_add.button("➕", key=f"add_svc_{i}", help=f"Add {svc_name}", use_container_width=True):
                # Get module from service definition
                svc_entry = next((s for s in all_services if s['name'] == svc_name), None)
                svc_module = svc_entry.get('module') if svc_entry else None
                add_service_to_pipeline(svc_name, module_name=svc_module)
                request_pipeline_json_sync(
                    pipeline_steps_to_json_text(st.session_state.pipeline_steps),
                    force=True,
                    toast=f"Added {svc_name}",
                    icon="🧩",
                )
                st.rerun()
            col_name.text(svc_name)
            
        if not available:
            st.info("No services found.")
    
    with tab_pipelines:
        st.markdown("##### Saved Pipelines")
        
        # Display feedback message if any
        if 'resource_msg' in st.session_state and st.session_state.resource_msg:
             msg, icon = st.session_state.resource_msg
             if icon == "✅": st.success(msg)
             elif icon == "➕": st.success(msg)
             elif icon == "🗑️": st.warning(msg)
             else: st.info(msg)
             # Clear after showing
             st.session_state.resource_msg = None
        
        # Read pipelines from database
        import sqlite3
        conn = sqlite3.connect(KB_PATH)
        conn.row_factory = sqlite3.Row
        
        try:
            pipelines = conn.execute("""
                SELECT id, name, step_count, description, specification 
                FROM pipelines 
                ORDER BY updated_at DESC, name
            """).fetchall()
            
            if pipelines:
                # Search filter
                search = st.text_input("🔍 Search:", placeholder="Filter pipelines...", key="pipe_search")
                
                st.caption("Click to load into workspace")
                
                for pipe in pipelines:
                    # Filter by search
                    if search and search.lower() not in pipe['name'].lower():
                        continue
                    
                    name = pipe['name']
                    step_count = pipe['step_count'] or 0
                    desc = pipe['description'] or ""
                    
                    col_load, col_name, col_del = st.columns([0.5, 3, 0.5])
                    
                    if col_load.button("➕", key=f"load_pipe_{pipe['id']}", help=f"Add steps from: {name}", use_container_width=True):
                        try:
                            spec = json.loads(pipe['specification']) if pipe['specification'] else []
                            # Handle both list format and dict with 'steps' key
                            if isinstance(spec, dict):
                                steps = spec.get('steps', [])
                            else:
                                steps = spec
                            
                            # Append steps instead of replacing
                            if steps:
                                st.session_state.pipeline_steps.extend(steps)
                                request_pipeline_json_sync(
                                    pipeline_steps_to_json_text(st.session_state.pipeline_steps),
                                    force=True,
                                )
                                st.session_state.resource_msg = (f"Added {len(steps)} steps from {name}", "➕")
                                st.rerun()
                            else:
                                st.warning("Pipeline is empty.")
                        except Exception as e:
                            st.error(f"Error parsing: {e}")
                    
                    col_name.text(f"{name[:20]} ({step_count})")
                    
                    if col_del.button("🗑️", key=f"del_pipe_{pipe['id']}", help="Delete from KB"):
                        try:
                            conn.execute("DELETE FROM pipelines WHERE id = ?", [pipe['id']])
                            conn.commit()
                            st.session_state.resource_msg = (f"Deleted {name}", "🗑️")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
            else:
                st.info("No pipelines in Knowledge Base.")
        finally:
            conn.close()

def render_file_explorer():
    """Left Column: simplified file explorer (Helper for Resources Panel)"""
    # Root folder: storage only
    root_dir = Path(STORAGE_DIR).resolve()
    current_dir = Path(st.session_state.storage_current_dir).resolve()
    workspace_dir = Path(get_workspace_dir()).resolve()
    if not str(current_dir).startswith(str(root_dir)):
        current_dir = root_dir
        st.session_state.storage_current_dir = str(root_dir)
    
    # Breadcrumbs & Navigation
    rel_path = "." if current_dir == root_dir else str(current_dir.relative_to(root_dir))
    
    # Path navigation
    nav_cols = st.columns([1, 4])
    if nav_cols[0].button("⬆️", help="Go up one level", use_container_width=True, disabled=current_dir == root_dir):
        st.session_state.storage_current_dir = str(current_dir.parent)
        st.rerun()
    nav_cols[1].caption(f"📁 **{rel_path}**")
    
    # Workspace controls
    # Workspace controls
    ws_rel_path = "." if workspace_dir == root_dir else str(workspace_dir.relative_to(root_dir))
    
    st.markdown("---")
    
    ws_cols = st.columns([3, 1, 1])
    ws_cols[0].caption(f"**Workspace:** `{ws_rel_path}`")
    
    if ws_cols[1].button("📍", use_container_width=True, disabled=current_dir == workspace_dir, help="Set as workspace"):
        st.session_state.pipeline_base_dir = str(current_dir)
        st.toast("Updated", icon="🧭")
        st.rerun()
        
    if ws_cols[2].button("➡️", use_container_width=True, disabled=current_dir == workspace_dir, help="Go to workspace"):
        st.session_state.storage_current_dir = str(workspace_dir)
        st.rerun()



    # Filter
    filter_txt = st.text_input("Search Files:", placeholder="Search...")
    
    # File List Table
    entries = list(current_dir.iterdir()) if current_dir.exists() else []
    dirs = sorted([e for e in entries if e.is_dir()], key=lambda p: p.name.lower())
    files = sorted([e for e in entries if e.is_file()], key=lambda p: p.name.lower())

    if not dirs and not files:
        st.caption("No files found.")

    # Folders first
    for i, d in enumerate(dirs):
        if filter_txt and filter_txt.lower() not in d.name.lower():
            continue
        col_type, col_name = st.columns([1, 4])
        col_type.caption("📁")
        if col_name.button(d.name, key=f"dir_{i}", use_container_width=True, type="secondary"):
            st.session_state.storage_current_dir = str(d.resolve())
            st.rerun()

    # Files
    for j, f in enumerate(files):
        if filter_txt and filter_txt.lower() not in f.name.lower():
            continue
        col_type, col_name = st.columns([1, 4])
        col_type.caption("📄")
        if col_name.button(f.name, key=f"file_{j}", use_container_width=True, type="secondary"):
            full_path = f.resolve()
            st.session_state['view_file'] = str(full_path)
            # Store relative path for copying (prefer workspace-relative)
            try:
                rel_file_path = str(full_path.relative_to(workspace_dir))
                st.session_state['copied_path_scope'] = "workspace"
            except ValueError:
                rel_file_path = str(full_path.relative_to(root_dir))
                st.session_state['copied_path_scope'] = "storage"
            st.session_state['copied_path'] = rel_file_path
            
    st.markdown("---")
    
    # Show copied path prominently
    copied_path = st.session_state.get('copied_path')
    if copied_path:
        scope = st.session_state.get('copied_path_scope', "storage")
        st.success(f"📋 Copied ({scope}): `{copied_path}`")
    
    open_file = st.checkbox("Preview below", key="open_file_checkbox")
    
    b1, b2 = st.columns(2)
    view_clicked = b1.button("Preview", use_container_width=True)

    selected_file = st.session_state.get('view_file')
    if selected_file and os.path.isfile(selected_file):
        with open(selected_file, "rb") as f:
            file_bytes = f.read()
        b2.download_button(
            "Download",
            data=file_bytes,
            file_name=os.path.basename(selected_file),
            use_container_width=True
        )
    else:
        b2.button("Download", use_container_width=True, disabled=True)
        
    if selected_file and os.path.isfile(selected_file) and (view_clicked or open_file):
        st.markdown("---")
        ext = Path(selected_file).suffix.lower()
        try:
            if ext in [".csv"]:
                df = pd.read_csv(selected_file)
                st.dataframe(df.head(50), use_container_width=True)
            elif ext in [".json"]:
                with open(selected_file, "r", encoding="utf-8") as f:
                    st.json(json.load(f))
            elif ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
                st.image(selected_file, use_container_width=True)
            else:
                with open(selected_file, "r", encoding="utf-8", errors="ignore") as f:
                    st.code(f.read(2000))
        except Exception as e:
            st.error(f"Error: {e}")

def render_pipeline_builder():
    """Center Column: Pipeline Composition Workspace"""
    st.markdown("##### 🚧 Pipeline Board")
    workspace_dir = Path(get_workspace_dir()).resolve()
    storage_root = Path(STORAGE_DIR).resolve()
    ws_rel = "." if workspace_dir == storage_root else str(workspace_dir.relative_to(storage_root))
    st.caption(f"Workspace / {ws_rel}")

    mode = st.segmented_control(
        "Pipeline Input Mode",
        ["Builder", "JSON"],
        default=st.session_state.pipeline_input_mode_main,
        key="pipeline_input_mode_main",
        help="Use Builder to add/configure services visually, or JSON to paste/edit a pipeline specification directly.",
        label_visibility="collapsed",
    )

    # Keep the "Add service" controls available in both modes, so edits are always
    # reflected back into a valid JSON pipeline spec.
    with st.container():
        # Get unique modules from all_services
        modules = sorted(list(set(s['module'] for s in all_services if s['module'])))
        
        c1, c2, c3 = st.columns([2, 2, 1])
        
        selected_mod = c1.selectbox(
            "Select Module:", 
            options=modules, 
            index=0 if modules else None,
            key="wb_module_selector",
            label_visibility="collapsed"
        )
        
        # Filter services by selected module
        mod_services = sorted([s['name'] for s in all_services if s['module'] == selected_mod])
        
        selected_svc = c2.selectbox(
            "Select Service:", 
            options=mod_services, 
            index=0 if mod_services else None,
            key="wb_service_selector",
            label_visibility="collapsed"
        )
        
        if c3.button("Add", type="primary", use_container_width=True, help="Add selected service to pipeline"):
            if selected_svc:
                add_service_to_pipeline(selected_svc, module_name=selected_mod)
                request_pipeline_json_sync(
                    pipeline_steps_to_json_text(st.session_state.pipeline_steps),
                    force=True,
                    toast=f"Added {selected_svc}",
                    icon="🧩",
                )
                st.rerun()

    # Action Bar: DAG, Run, Save, Clear (all on one row)
    c_run, c_save, c_clear = st.columns([2, 1, 0.5])
    
    # Clear Button
    if c_clear.button("🗑️", help="Clear Pipeline Board"):
        st.session_state.pipeline_steps = []
        request_pipeline_json_sync(
            pipeline_steps_to_json_text(st.session_state.pipeline_steps),
            force=True,
            toast="Pipeline cleared",
            icon="🧹",
        )
        st.rerun()
    
    runner_col, parallel_col = c_run.columns([3, 1.5])
    parallel = parallel_col.checkbox("DAG", value=True, help="Execute independent steps in parallel (DAG mode)")
    
    if runner_col.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        if not st.session_state.pipeline_steps:
            st.warning("Empty pipeline!")
        else:
            st.session_state.execution_log = ["🚀 Starting Pipeline..."]
            workspace_dir = get_workspace_dir()
            runner = PipelineRunner(None, verbose=False, storage=workspace_dir, modules=SERVICE_MODULES)
            with st.spinner("Running..."):
                try:
                    res = runner.run(st.session_state.pipeline_steps, base_path=workspace_dir, parallel=parallel)
                    if res['success']:
                        st.session_state.execution_log.append("✅ Done!")
                        st.success("Pipeline completed successfully!")
                        
                        # Show output file paths
                        st.markdown("##### 📁 Output Files")
                        outputs_found = []
                        for step in st.session_state.pipeline_steps:
                            for out_slot, out_path in step.get('outputs', {}).items():
                                full_path = os.path.join(workspace_dir, out_path)
                                if os.path.exists(full_path):
                                    size_kb = os.path.getsize(full_path) / 1024
                                    outputs_found.append(f"✅ `{out_path}` ({size_kb:.1f} KB)")
                                else:
                                    outputs_found.append(f"⚠️ `{out_path}` (not found)")
                        
                        if outputs_found:
                            for out_msg in outputs_found:
                                st.markdown(out_msg)
                        
                        # Show workspace path
                        st.info(f"📂 Workspace: `{workspace_dir}`")
                    else:
                        st.session_state.execution_log.append(f"❌ Failed: {res.get('error')}")
                        st.error(f"Failed: {res.get('error')}")
                except Exception as e:
                    st.error(f"Error: {e}")

    if c_save.button("💾 Save", use_container_width=True):
        name = f"pipeline_{int(time.time())}"
        path = os.path.join(PIPELINE_SAVE_DIR, f"{name}.json")
        payload = {
            "name": name,
            "specification": st.session_state.pipeline_steps,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            st.toast(f"Saved: {path}")
        except Exception as e:
            st.error(str(e))

    st.markdown("---")

    if mode == "Builder":
        if not st.session_state.pipeline_steps:
            st.info("👈 Add services from the selectors above or the Left Panel to start building.")
            
        # Pipeline Steps Display
        for i, step in enumerate(st.session_state.pipeline_steps):
            with st.container():
                # Card-like header with module info
                col_idx, col_name, col_up, col_down, col_ins, col_del = st.columns([0.5, 3, 0.5, 0.5, 0.5, 0.5])
                
                # 1. Index
                col_idx.markdown(f"**#{i+1}**")
                
                # 2. Service Name
                module_name = step.get('module', '')
                if module_name:
                    col_name.markdown(f"**{step['service']}** `({module_name})`")
                else:
                    col_name.markdown(f"**{step['service']}**")
                
                # 3. Move Up
                if i > 0:
                    if col_up.button("⬆", key=f"up_{i}", help="Move Up", use_container_width=True):
                        st.session_state.pipeline_steps[i], st.session_state.pipeline_steps[i-1] = st.session_state.pipeline_steps[i-1], st.session_state.pipeline_steps[i]
                        request_pipeline_json_sync(
                            pipeline_steps_to_json_text(st.session_state.pipeline_steps),
                            force=True,
                        )
                        st.rerun()
                
                # 4. Move Down
                if i < len(st.session_state.pipeline_steps) - 1:
                    if col_down.button("⬇", key=f"down_{i}", help="Move Down", use_container_width=True):
                        st.session_state.pipeline_steps[i], st.session_state.pipeline_steps[i+1] = st.session_state.pipeline_steps[i+1], st.session_state.pipeline_steps[i]
                        request_pipeline_json_sync(
                            pipeline_steps_to_json_text(st.session_state.pipeline_steps),
                            force=True,
                        )
                        st.rerun()
                
                # 5. Insert Below
                # Use the currently selected service from the top selector
                current_selected_svc = st.session_state.get("wb_service_selector")
                btn_disabled = not current_selected_svc
                help_text = f"Insert '{current_selected_svc}' after this step" if current_selected_svc else "Select a service above to insert"
                
                if col_ins.button("➕", key=f"ins_{i}", help=help_text, disabled=btn_disabled, use_container_width=True):
                    if current_selected_svc:
                        current_selected_mod = st.session_state.get("wb_module_selector")
                        add_service_to_pipeline(current_selected_svc, insert_at=i+1, module_name=current_selected_mod)
                        request_pipeline_json_sync(
                            pipeline_steps_to_json_text(st.session_state.pipeline_steps),
                            force=True,
                            toast=f"Inserted {current_selected_svc} at step {i+2}",
                            icon="🧩",
                        )
                        st.rerun()
    
                # 6. Delete
                if col_del.button("🗑️", key=f"del_step_{i}", help="Remove Step", use_container_width=True):
                     st.session_state.pipeline_steps.pop(i)
                     request_pipeline_json_sync(
                         pipeline_steps_to_json_text(st.session_state.pipeline_steps),
                         force=True,
                     )
                     st.rerun()
                
                # Simple Expanders for Config (keep canvas clean)
                with st.expander("⚙️ Configuration", expanded=False):
                    # Inputs
                    st.caption("Inputs")
                    for k, v in step['inputs'].items():
                        input_key = f"in_{i}_{k}"
                        new_val = st.text_input(f"In: {k}", value=v, key=input_key)
                        # Update session state if value changed
                        if new_val != v:
                            st.session_state.pipeline_steps[i]['inputs'][k] = new_val
                    
                    # Outputs
                    st.caption("Outputs")
                    for k, v in step['outputs'].items():
                        output_key = f"out_{i}_{k}"
                        new_val = st.text_input(f"Out: {k}", value=v, key=output_key)
                        if new_val != v:
                            st.session_state.pipeline_steps[i]['outputs'][k] = new_val
                        
                    # Params
                    params_dict = st.session_state.pipeline_steps[i].get('params') or {}
                    param_meta = get_service_param_meta(step['service'])
                    for param_name, info in param_meta.items():
                        if param_name not in params_dict:
                            params_dict[param_name] = info.get("default", None)
                    st.session_state.pipeline_steps[i]['params'] = params_dict
                    if params_dict:
                        st.caption("Parameters")
                        ordered_keys = list(param_meta.keys()) + [k for k in params_dict.keys() if k not in param_meta]
                        for k in ordered_keys:
                            v = params_dict.get(k)
                            param_key = f"p_{i}_{k}"
                            param_info = param_meta.get(k, {})
                            param_type = infer_param_type(param_info, v)
                            help_text = param_info.get("description")
                            if param_type in {"list", "dict"}:
                                new_val = st.text_area(
                                    f"Param: {k} (JSON {param_type})",
                                    value=format_json_value(v),
                                    key=param_key,
                                    height=80,
                                    help=help_text,
                                    placeholder="[]" if param_type == "list" else "{}",
                                )
                                parsed_val = v
                                if new_val.strip() == "":
                                    parsed_val = None
                                else:
                                    try:
                                        parsed_val = json.loads(new_val)
                                        if param_type == "list" and not isinstance(parsed_val, list):
                                            raise ValueError("Expected a JSON list.")
                                        if param_type == "dict" and not isinstance(parsed_val, dict):
                                            raise ValueError("Expected a JSON object.")
                                    except Exception:
                                        st.error(f"Param '{k}' expects {param_type} JSON.")
                                        parsed_val = v
                            else:
                                new_val = st.text_input(
                                    f"Param: {k}",
                                    value=str(v) if v is not None else "",
                                    key=param_key,
                                    help=help_text
                                )
                                parsed_val = new_val
                                if new_val == "None":
                                    parsed_val = None
                                elif new_val.lower() == "true":
                                    parsed_val = True
                                elif new_val.lower() == "false":
                                    parsed_val = False
                                elif param_type == "int":
                                    if new_val.strip() == "":
                                        parsed_val = None
                                    else:
                                        try:
                                            parsed_val = int(new_val)
                                        except ValueError:
                                            parsed_val = new_val
                                elif param_type == "float":
                                    if new_val.strip() == "":
                                        parsed_val = None
                                    else:
                                        try:
                                            parsed_val = float(new_val)
                                        except ValueError:
                                            parsed_val = new_val
                                elif param_type == "any":
                                    try:
                                        if "." in new_val:
                                            parsed_val = float(new_val)
                                        else:
                                            parsed_val = int(new_val)
                                    except Exception:
                                        parsed_val = new_val
                            
                            if parsed_val != v:
                                st.session_state.pipeline_steps[i]['params'][k] = parsed_val
    
                st.markdown("---")
    else:
        st.caption("Paste/edit the pipeline JSON below. You can also add services using the selectors above.")
        st.text_area(
            "Pipeline JSON Spec",
            key="pipeline_json_input_main",
            height=420,
            placeholder='[\n  {\n    "module": "my_services",\n    "service": "load_csv",\n    "inputs": {"path": "datasets/train.csv"},\n    "outputs": {"data": "artifacts/train.csv"},\n    "params": {}\n  }\n]\n',
        )
        apply_col, sync_col = st.columns([1, 1])
        if apply_col.button("Apply JSON", use_container_width=True, type="primary"):
            try:
                raw = st.session_state.pipeline_json_input_main or "[]"
                data = json.loads(raw)
                steps = extract_and_normalize_pipeline_steps(data)
                st.session_state.pipeline_steps = steps
                request_pipeline_json_sync(
                    pipeline_steps_to_json_text(steps),
                    force=True,
                    toast="Pipeline updated from JSON",
                )
                st.rerun()
            except Exception as e:
                st.error(f"Invalid pipeline JSON: {e}")
        if sync_col.button("Sync From Current Pipeline", use_container_width=True):
            request_pipeline_json_sync(
                pipeline_steps_to_json_text(st.session_state.pipeline_steps),
                force=True,
                toast="JSON synced from pipeline",
                icon="🔄",
            )
            st.rerun()

        if not st.session_state.pipeline_steps:
            st.info("Paste JSON above, or load a pipeline/service from the left panel or AI recommendations.")
        else:
            st.caption(f"{len(st.session_state.pipeline_steps)} step(s) loaded. Switch to Builder to edit visually.")


    
    st.divider()
    
    # --- System Status & AI Results (Center-Bottom) ---
    c_status, c_ai = st.columns([1, 1], gap="medium")
    
    with c_status:
        st.markdown("##### 📜 System Status")
        if st.session_state.execution_log:
            log_text = "\n".join(st.session_state.execution_log)
            st.code(log_text, language="text")
            if st.session_state.execution_log[-1].startswith("✅"):
                st.success("Ready")
            elif st.session_state.execution_log[-1].startswith("❌"):
                st.error("Failed")
        else:
            st.info("System ready.")

    with c_ai:
        st.markdown("##### 🤖 AI Recommendations")
        if 'recom_results' in st.session_state and st.session_state.recom_results:
            res = st.session_state.recom_results
            pipelines = res.get("pipelines", [])
            verdicts = {v["name"]: v for v in res.get("verdict", [])}
            
            if not pipelines:
                st.info("No relevant pipelines found.")
            
            for i, p in enumerate(pipelines):
                name = p["name"]
                verdict = verdicts.get(name, {})
                decision = verdict.get("decision", "unknown").lower()
                explanation = verdict.get("explanation", "No explanation provided.")
                
                icon = "❓"
                color = "gray"
                if decision == "accept":
                    icon = "✅"
                    color = "green"
                elif decision == "reject":
                    icon = "❌" 
                    color = "red"
                
                with st.expander(f"{icon} {name}", expanded=(decision=="accept")):
                    st.caption(f"**Verdict:** :{color}[{decision.upper()}]")
                    st.caption(explanation)
                    
                    if st.button("📥 Load", key=f"cnt_load_{name}_{i}", use_container_width=True):
                        spec = p.get("details", [])
                        if isinstance(spec, dict) and "steps" in spec:
                            steps = spec["steps"]
                        else:
                            steps = spec
                            
                        if steps:
                            st.session_state.pipeline_steps = steps
                            request_pipeline_json_sync(
                                pipeline_steps_to_json_text(steps),
                                force=True,
                                toast=f"Loaded {name}",
                                icon="📥",
                            )
                            st.rerun()
                        else:
                            st.error("Empty pipeline.")
        else:
            st.caption("No recommendations yet. Use the **AI Recommender** panel to generate suggestions.")


def render_right_panel():
    """Right Column: JSON/Graph Viewer & Logs"""
    
    # Tabs
    tab_json, tab_log, tab_ai = st.tabs(["🔍 Inspect", "📜 Logs", "🤖 AI"])
    
    with tab_json:
        # ... (keep existing JSON content, but remove 'Show Graph Visualizer' button)
        st.markdown("**Live Pipeline State**")
        st.json(st.session_state.pipeline_steps, expanded=False)
        
        st.markdown("**Pipeline Configuration**")
        st.caption("Edit the JSON list of steps (or an object with a 'steps'/'specification' key).")
        if not st.session_state.pipeline_json_editor:
            st.session_state.pipeline_json_editor = pipeline_steps_to_json_text(st.session_state.pipeline_steps)
        st.text_area(
            "Pipeline JSON",
            key="pipeline_json_editor",
            height=320
        )
        apply_col, sync_col = st.columns(2)
        if apply_col.button("Apply JSON", use_container_width=True):
            try:
                raw = st.session_state.pipeline_json_editor or "[]"
                data = json.loads(raw)
                steps = extract_and_normalize_pipeline_steps(data)
                st.session_state.pipeline_steps = steps
                request_pipeline_json_sync(
                    pipeline_steps_to_json_text(steps),
                    force=True,
                    toast="Pipeline updated from JSON",
                )
                st.rerun()
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
        if sync_col.button("Sync From Pipeline", use_container_width=True):
            request_pipeline_json_sync(
                pipeline_steps_to_json_text(st.session_state.pipeline_steps),
                force=True,
                toast="JSON synced from pipeline",
                icon="🔄",
            )
            st.rerun()


    
    with tab_log:
        st.markdown("**Execution Logs**")
        if st.session_state.execution_log:
            log_text = "\n".join(st.session_state.execution_log)
            st.code(log_text, language="text")
        else:
            st.caption("No logs yet.")

    # AI Assistant Tab Content
    # 1. Init State for Form Fields
    if "ai_task_goal" not in st.session_state: st.session_state.ai_task_goal = ""
    if "ai_data_context" not in st.session_state: st.session_state.ai_data_context = ""
    if "ai_problem_type" not in st.session_state: st.session_state.ai_problem_type = ""
    if "ai_domain_keywords" not in st.session_state: st.session_state.ai_domain_keywords = ""
    if "ai_additional_info" not in st.session_state: st.session_state.ai_additional_info = ""

    def prefill_house_prices():
        st.session_state.ai_task_goal = "Predict house prices in Ames, Iowa based on physical and location features."
        st.session_state.ai_data_context = "Input: CSV with 79 features (zoning, lot size, etc.). Output: SalePrice (numeric)."
        st.session_state.ai_problem_type = "Regression"
        st.session_state.ai_domain_keywords = "Real Estate, Housing, Economics"
        st.session_state.ai_additional_info = "Prefer tree-based models (RandomForest, XGBoost). Metric: RMSE."

    with tab_ai:
        st.markdown("### 🧠 Multi-Index Recommender")
        st.caption("Describe your task constraints to find multi-signal, verified pipelines.")
        
        # API Key Handling (Priority: Env -> Secrets -> UI Input)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
             try:
                 api_key = st.secrets["GEMINI_API_KEY"]
             except:
                 pass
        
        if not api_key:
            api_key = st.text_input("Enter Gemini API Key:", type="password", help="Get a key from Google AI Studio", value="test-key-default")
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key

        if not api_key:
            st.warning("⚠️ Please provide a Gemini API Key to use the Recommender.")
        else:
            try:
                recommender.init_recommender(api_key)
            except Exception as e:
                st.error(f"Failed to init recommender: {e}")
            
            # Prefill Button
            if st.button("🏠 Prefill: House Price Prediction", help="Load example values from paper", on_click=prefill_house_prices):
                pass

            # Form Inputs
            with st.form("recommender_form"):
                st.text_area("Task Goal", key="ai_task_goal", height=70, placeholder="What should the pipeline achieve?")
                c1, c2 = st.columns(2)
                c1.text_input("Problem Type", key="ai_problem_type", placeholder="Regression, Classification...")
                c2.text_input("Domain Keywords", key="ai_domain_keywords", placeholder="Finance, Medical...")
                st.text_area("Data Context", key="ai_data_context", height=70, placeholder="Dataset details, size, format...")
                st.text_input("Additional Info", key="ai_additional_info", placeholder="Constraints, metrics...")
                
                submitted = st.form_submit_button("🔍 Get Recommendations", type="primary", use_container_width=True)
            
            if submitted:
                query = recommender.UserQuery(
                    task_goal=st.session_state.ai_task_goal,
                    data_context=st.session_state.ai_data_context,
                    problem_type=st.session_state.ai_problem_type or None,
                    domain_keywords=st.session_state.ai_domain_keywords or None,
                    additional_info=st.session_state.ai_additional_info or None
                )

                with st.spinner("Analyzing Knowledge Base & Verifying Candidates..."):
                    try:
                        result = recommender._RECOMMENDER.recommend(query)
                        pipelines = []
                        verdicts = []

                        for rank, rec in enumerate(result.recommendations, start=1):
                            pipelines.append({
                                "name": rec.name,
                                "description": rec.description,
                                "details": rec.specification,
                                "services_used": rec.services_used,
                                "scores": {
                                    "combined": rec.combined_score,
                                    "task": rec.task_similarity,
                                    "technique": rec.technique_similarity,
                                    "success": rec.success_score,
                                    "fit": rec.llm_fit_score,
                                },
                                "contract_issues": rec.contract_issues,
                                "problem_type": rec.problem_type,
                                "domain": rec.domain,
                            })
                            verdicts.append({
                                "name": rec.name,
                                "rank": rank,
                                "decision": rec.llm_decision or "pending",
                                "explanation": rec.llm_reasoning or "No reasoning provided.",
                                "adaptations": rec.llm_adaptations,
                                "risks": rec.llm_risks,
                                "fit_score": rec.llm_fit_score,
                            })

                        st.session_state.recom_results = {
                            "pipelines": pipelines,
                            "verdict": verdicts,
                            "warnings": result.warnings,
                            "diagnostics": {
                                "candidates_retrieved": result.candidates_retrieved,
                                "candidates_after_filter": result.candidates_after_filter,
                                "candidates_after_contract": result.candidates_after_contract,
                                "processing_time_ms": result.processing_time_ms,
                            },
                        }
                    except Exception as e:
                        st.error(f"Recommendation failed: {e}")

            # Display Results
            if 'recom_results' in st.session_state and st.session_state.recom_results:
                res = st.session_state.recom_results
                warnings = res.get("warnings", [])
                diagnostics = res.get("diagnostics", {})
                pipelines = res.get("pipelines", [])
                verdicts = {v["name"]: v for v in res.get("verdict", [])}

                if warnings:
                    for warning in warnings:
                        st.warning(warning)

                if diagnostics:
                    with st.expander("📊 Retrieval Diagnostics"):
                        st.json(diagnostics)
                
                if not pipelines:
                    st.info("No relevant pipelines found.")
                
                for p in pipelines:
                    name = p["name"]
                    verdict = verdicts.get(name, {})
                    decision = verdict.get("decision", "unknown").lower()
                    explanation = verdict.get("explanation", "No explanation provided.")
                    
                    # Color code header
                    icon = "❓"
                    color = "gray"
                    if decision == "accept":
                        icon = "✅"
                        color = "green"
                    elif decision == "adapt":
                        icon = "🛠️"
                        color = "orange"
                    elif decision == "reject":
                        icon = "❌" 
                        color = "red"
                    
                    with st.expander(f"{icon} {name}", expanded=(decision=="accept")):
                        st.markdown(f"**Verdict:** :{color}[{decision.upper()}]")
                        st.markdown(f"**Reasoning:** {explanation}")
                        st.caption(p.get("description", ""))
                        scores = p.get("scores", {})
                        if scores:
                            st.markdown(
                                f"**Scores:** combined {scores.get('combined', 0):.3f} | "
                                f"task {scores.get('task', 0):.3f} | "
                                f"technique {scores.get('technique', 0):.3f} | "
                                f"success {scores.get('success', 0):.3f} | "
                                f"fit {scores.get('fit', 0):.3f}"
                            )
                        if verdict.get("adaptations"):
                            st.markdown("**Adaptations:** " + "; ".join(verdict.get("adaptations", [])))
                        if verdict.get("risks"):
                            st.markdown("**Risks:** " + "; ".join(verdict.get("risks", [])))
                        if p.get("contract_issues"):
                            st.markdown("**Contract Issues:** " + "; ".join(p.get("contract_issues", [])))
                        
                        # Load Button
                        if st.button("📥 Load to Workspace", key=f"rec_load_{name}"):
                            spec = p.get("details", [])
                            # Normalize spec
                            if isinstance(spec, dict) and "steps" in spec:
                                steps = spec["steps"]
                            else:
                                steps = spec
                                
                            if steps:
                                st.session_state.pipeline_steps = steps
                                request_pipeline_json_sync(
                                    pipeline_steps_to_json_text(steps),
                                    force=True,
                                    toast=f"Loaded {name}",
                                    icon="📥",
                                )
                                st.rerun()
                            else:
                                st.error("This pipeline specification is empty.")
                        
                        # Show raw spec
                        st.json(p.get("details", {}))


def render_microservice_editor():
    """Enhanced Microservice Editor with module browser and code editor."""
    st.markdown("### 🛠️ Microservice Editor")
    
    services_dir = SERVICES_DIR
    if not os.path.exists(services_dir):
        st.error(f"Services directory not found: {services_dir}")
        return

    # Get all Python module files
    module_files = sorted([f for f in os.listdir(services_dir) if f.endswith(".py") and f != "__init__.py"])
    if not module_files:
        st.info("No service module files found.")
        return

    # Initialize session state for editor
    if 'editor_current_file' not in st.session_state:
        st.session_state.editor_current_file = module_files[0] if module_files else None
    if 'editor_content' not in st.session_state:
        st.session_state.editor_content = ""

    # Group services by module from KB
    module_services = {}
    for svc in all_services:
        mod = svc.get('module', 'unknown')
        if mod not in module_services:
            module_services[mod] = []
        module_services[mod].append(svc['name'])

    # Two-column layout: Module browser | Code editor
    col_browser, col_editor = st.columns([1, 3])
    
    with col_browser:
        # Start scrollable module browser panel
        st.markdown('<div class="module-browser-panel">', unsafe_allow_html=True)
        
        st.markdown("##### 📁 Service Modules")
        
        # Show module list with service counts
        for module_file in module_files:
            module_name = module_file.replace(".py", "")
            svc_count = len(module_services.get(module_name, []))
            
            is_selected = st.session_state.editor_current_file == module_file
            button_type = "primary" if is_selected else "secondary"
            
            if st.button(
                f"📦 {module_name} ({svc_count})",
                key=f"mod_{module_name}",
                use_container_width=True,
                type=button_type
            ):
                # Load this module file
                file_path = os.path.join(services_dir, module_file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        st.session_state.editor_content = f.read()
                    st.session_state.editor_current_file = module_file
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading {module_file}: {e}")
        
        st.markdown("---")
        
        # Show services in selected module (in a scrollable container)
        if st.session_state.editor_current_file:
            selected_module = st.session_state.editor_current_file.replace(".py", "")
            services_in_module = module_services.get(selected_module, [])
            
            if services_in_module:
                st.markdown(f"##### Services ({len(services_in_module)})")
                st.markdown('<div class="services-list">', unsafe_allow_html=True)
                for svc_name in sorted(services_in_module):
                    st.caption(f"• {svc_name}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.caption("No registered services.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_editor:
        # Start scrollable code editor panel
        st.markdown('<div class="code-editor-panel">', unsafe_allow_html=True)
        
        if st.session_state.editor_current_file:
            file_path = os.path.join(services_dir, st.session_state.editor_current_file)
            
            # Header with file name and actions
            header_cols = st.columns([2.5, 1, 1, 1])
            header_cols[0].markdown(f"**Editing:** `{st.session_state.editor_current_file}`")
            
            # Reload button
            if header_cols[1].button("🔄 Reload", use_container_width=True, help="Reload from disk"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        st.session_state.editor_content = f.read()
                    st.toast("Reloaded from disk", icon="🔄")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reloading: {e}")
            
            # Save button
            if header_cols[2].button("💾 Save", type="primary", use_container_width=True):
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(st.session_state.editor_content)
                    st.toast(f"Saved {st.session_state.editor_current_file}!", icon="✅")
                    st.success("File saved successfully!")
                except Exception as e:
                    st.error(f"Error saving: {e}")
            
            # Delete button with confirmation
            if 'confirm_delete' not in st.session_state:
                st.session_state.confirm_delete = False
            
            if header_cols[3].button("🗑️ Delete", use_container_width=True, help="Delete this module"):
                st.session_state.confirm_delete = True
            
            # Show confirmation dialog
            if st.session_state.confirm_delete:
                st.warning(f"⚠️ Are you sure you want to delete `{st.session_state.editor_current_file}`? This cannot be undone!")
                confirm_cols = st.columns([1, 1, 2])
                if confirm_cols[0].button("✅ Yes, Delete", type="primary"):
                    try:
                        os.remove(file_path)
                        st.toast(f"Deleted {st.session_state.editor_current_file}", icon="🗑️")
                        # Reset editor state
                        st.session_state.editor_current_file = None
                        st.session_state.editor_content = ""
                        st.session_state.confirm_delete = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting: {e}")
                if confirm_cols[1].button("❌ Cancel"):
                    st.session_state.confirm_delete = False
                    st.rerun()
            
            # Load content if not already loaded
            if not st.session_state.editor_content:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        st.session_state.editor_content = f.read()
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    st.session_state.editor_content = f"# Error loading {st.session_state.editor_current_file}"
            
            # Code editor
            new_content = st.text_area(
                "Python Code",
                value=st.session_state.editor_content,
                height=500,
                key="code_editor_area",
                label_visibility="collapsed"
            )
            
            # Update session state if content changed
            if new_content != st.session_state.editor_content:
                st.session_state.editor_content = new_content
            
            # Show file stats
            line_count = len(st.session_state.editor_content.split('\n'))
            char_count = len(st.session_state.editor_content)
            st.caption(f"📊 {line_count} lines | {char_count:,} characters")
        else:
            st.info("Select a module from the left panel to edit.")
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_kb_editor():
    """Knowledge Base Editor - View and edit database tables."""
    import sqlite3
    
    st.markdown("### 📚 Knowledge Base Editor")
    
    db_path = KB_PATH
    if not os.path.exists(db_path):
        st.error(f"Database not found: {db_path}")
        return
    
    # Initialize session state
    if 'kb_selected_table' not in st.session_state:
        st.session_state.kb_selected_table = None
    if 'kb_edit_row' not in st.session_state:
        st.session_state.kb_edit_row = None
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get all tables
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    table_names = [t['name'] for t in tables if not t['name'].startswith('sqlite_')]
    
    # Two column layout
    col_tables, col_data = st.columns([1, 3])
    
    with col_tables:
        st.markdown('<div class="module-browser-panel">', unsafe_allow_html=True)
        st.markdown("##### 📋 Tables")
        
        for table_name in table_names:
            # Get row count
            count = conn.execute(f"SELECT COUNT(*) as cnt FROM {table_name}").fetchone()['cnt']
            
            is_selected = st.session_state.kb_selected_table == table_name
            btn_type = "primary" if is_selected else "secondary"
            
            if st.button(f"📊 {table_name} ({count})", key=f"tbl_{table_name}", 
                        use_container_width=True, type=btn_type):
                st.session_state.kb_selected_table = table_name
                st.session_state.kb_edit_row = None
                st.rerun()
        
        st.markdown("---")
        
        # Re-sync button
        if st.button("🔄 Re-sync KB", use_container_width=True, help="Re-sync services from modules"):
            st.info("Re-syncing... Please wait.")
            try:
                import subprocess
                result = subprocess.run(
                    ["/Users/an/anaconda3/bin/python", "sync_kb.py"],
                    cwd=WEBAPP_DIR,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    st.success("KB re-synced successfully!")
                    st.toast("Knowledge Base updated", icon="✅")
                else:
                    st.error(f"Sync failed: {result.stderr}")
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_data:
        st.markdown('<div class="code-editor-panel">', unsafe_allow_html=True)
        
        if st.session_state.kb_selected_table:
            table_name = st.session_state.kb_selected_table
            
            # Header
            st.markdown(f"##### Table: `{table_name}`")
            
            # Get columns
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = [row['name'] for row in cursor.fetchall()]
            
            # Get data
            rows = conn.execute(f"SELECT * FROM {table_name} LIMIT 100").fetchall()
            
            if rows:
                # Convert to dataframe
                df = pd.DataFrame([dict(row) for row in rows])
                
                # Show dataframe
                st.dataframe(df, use_container_width=True, height=300)
                
                st.markdown("---")
                
                # Row editor section
                st.markdown("##### ✏️ Edit Row")
                
                # Select row to edit
                if 'rowid' in columns or 'id' in columns:
                    id_col = 'id' if 'id' in columns else 'rowid'
                    row_ids = df[id_col].tolist() if id_col in df.columns else list(range(len(df)))
                else:
                    row_ids = list(range(len(df)))
                    id_col = None
                
                selected_idx = st.selectbox("Select row to edit:", range(len(df)), 
                                           format_func=lambda i: f"Row {i+1}" + (f" (id={df.iloc[i].get('id', df.iloc[i].get('name', i))})" if len(df) > 0 else ""))
                
                if selected_idx is not None and len(df) > 0:
                    row_data = df.iloc[selected_idx].to_dict()
                    
                    # Edit form
                    with st.form(key="edit_row_form"):
                        edited_data = {}
                        for col in columns:
                            current_val = row_data.get(col, "")
                            if current_val is None:
                                current_val = ""
                            
                            # Use text_area for long fields
                            if isinstance(current_val, str) and len(str(current_val)) > 100:
                                edited_data[col] = st.text_area(f"{col}:", value=str(current_val), height=100)
                            else:
                                edited_data[col] = st.text_input(f"{col}:", value=str(current_val))
                        
                        col1, col2, col3 = st.columns(3)
                        
                        if col1.form_submit_button("💾 Save Changes", type="primary"):
                            try:
                                # Build UPDATE query
                                set_clause = ", ".join([f"{col} = ?" for col in columns if col != id_col])
                                values = [edited_data[col] for col in columns if col != id_col]
                                
                                if id_col and id_col in row_data:
                                    where_clause = f"WHERE {id_col} = ?"
                                    values.append(row_data[id_col])
                                else:
                                    # Use ROWID
                                    where_clause = f"WHERE rowid = ?"
                                    values.append(selected_idx + 1)
                                
                                conn.execute(f"UPDATE {table_name} SET {set_clause} {where_clause}", values)
                                conn.commit()
                                st.success("Row updated!")
                                st.toast("Changes saved", icon="✅")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error updating: {e}")
                        
                        if col2.form_submit_button("🗑️ Delete Row"):
                            try:
                                if id_col and id_col in row_data:
                                    conn.execute(f"DELETE FROM {table_name} WHERE {id_col} = ?", [row_data[id_col]])
                                else:
                                    conn.execute(f"DELETE FROM {table_name} WHERE rowid = ?", [selected_idx + 1])
                                conn.commit()
                                st.success("Row deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting: {e}")
                
                # Add new row section
                st.markdown("---")
                with st.expander("➕ Add New Row"):
                    with st.form(key="add_row_form"):
                        new_data = {}
                        for col in columns:
                            if col in ['id', 'rowid']:
                                continue  # Skip auto-increment columns
                            new_data[col] = st.text_input(f"{col}:", key=f"new_{col}")
                        
                        if st.form_submit_button("➕ Add Row", type="primary"):
                            try:
                                cols = [c for c in columns if c not in ['id', 'rowid']]
                                placeholders = ", ".join(["?" for _ in cols])
                                col_names = ", ".join(cols)
                                values = [new_data.get(c, "") for c in cols]
                                
                                conn.execute(f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})", values)
                                conn.commit()
                                st.success("Row added!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error adding: {e}")
            else:
                st.info("This table is empty.")
                
            # Show row count
            total_count = conn.execute(f"SELECT COUNT(*) as cnt FROM {table_name}").fetchone()['cnt']
            st.caption(f"Showing {min(100, total_count)} of {total_count} rows")
        else:
            st.info("👈 Select a table from the left panel to view and edit data.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    conn.close()


DATASET_SUFFIXES = {".csv", ".tsv", ".parquet"}
DATASET_FAST_EXCLUDE_DIRS = {
    "artifacts", "models", "checkpoints", "outputs", "predictions",
    "logs", "cache", "__pycache__"
}
DATASET_ALWAYS_EXCLUDE_DIRS = {".git", ".idea", ".vscode", ".mypy_cache", ".pytest_cache"}


def _format_dataset_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


@st.cache_data(show_spinner=False, ttl=600)
def _scan_dataset_inventory(storage_dir: str, scope: str) -> List[Dict[str, Any]]:
    """Scan tabular files in storage and cache results for faster UI reruns."""
    storage_path = Path(storage_dir)
    if not storage_path.exists():
        return []

    inventory = []
    dataset_roots = []
    if scope == "datasets_only":
        # First pass: find directories named "datasets" without scanning every file.
        for dirpath, dirnames, _ in os.walk(storage_path, topdown=True):
            dirnames[:] = [
                d for d in dirnames
                if d not in DATASET_ALWAYS_EXCLUDE_DIRS and not d.startswith(".")
            ]
            current_dir = Path(dirpath)
            for d in list(dirnames):
                if d == "datasets":
                    dataset_roots.append(current_dir / d)
            # Skip heavy folders while searching for dataset roots.
            dirnames[:] = [d for d in dirnames if d not in DATASET_FAST_EXCLUDE_DIRS and d != "datasets"]
    else:
        dataset_roots = [storage_path]

    seen_roots = set()
    for root_dir in dataset_roots:
        root_dir = Path(root_dir)
        if not root_dir.exists() or not root_dir.is_dir():
            continue
        root_key = str(root_dir.resolve())
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)

        for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
            dirnames[:] = [
                d for d in dirnames
                if d not in DATASET_ALWAYS_EXCLUDE_DIRS and not d.startswith(".")
            ]
            dir_path_obj = Path(dirpath)
            for file_name in filenames:
                suffix = Path(file_name).suffix.lower()
                if suffix not in DATASET_SUFFIXES:
                    continue
                file_path = dir_path_obj / file_name
                try:
                    stats = file_path.stat()
                    rel_path = file_path.relative_to(storage_path)
                    folder = str(rel_path.parent) if rel_path.parent != Path(".") else "(root)"
                    folder_layer = 0 if rel_path.parent == Path(".") else len(rel_path.parent.parts)
                    inventory.append(
                        {
                            "folder": folder,
                            "layer": folder_layer,
                            "file": file_name,
                            "ext": suffix.lstrip("."),
                            "path": str(rel_path),
                            "full_path": str(file_path),
                            "size_bytes": stats.st_size,
                            "size_label": _format_dataset_size(stats.st_size),
                            "modified_ts": stats.st_mtime,
                            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stats.st_mtime)),
                        }
                    )
                except OSError:
                    continue

    return inventory


@st.cache_data(show_spinner=False, ttl=300)
def _load_dataset_preview_cached(file_path: str, nrows: int, file_mtime: float) -> pd.DataFrame:
    """Load a preview frame with caching keyed by path + mtime + row limit."""
    _ = file_mtime  # Part of cache key for invalidation when file changes.
    path_obj = Path(file_path)
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_obj, nrows=nrows)
    if suffix == ".tsv":
        return pd.read_csv(path_obj, sep="\t", nrows=nrows)
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path_obj)
            chunks = []
            rows_loaded = 0
            for batch in parquet_file.iter_batches(batch_size=min(50000, max(nrows, 1))):
                batch_df = batch.to_pandas()
                chunks.append(batch_df)
                rows_loaded += len(batch_df)
                if rows_loaded >= nrows:
                    break
            if chunks:
                return pd.concat(chunks, ignore_index=True).head(nrows)
            return pd.DataFrame()
        except Exception:
            return pd.read_parquet(path_obj).head(nrows)
    raise ValueError(f"Unsupported file type: {suffix}")


def render_dataset_editor():
    """Dataset Editor - Browse and review datasets with richer controls."""
    st.markdown("### Dataset Explorer")

    defaults = {
        "dataset_preview_path": None,
        "dataset_search": "",
        "dataset_folder_filter": "(root)",
        "dataset_browser_folder": "",
        "dataset_sort_mode": "Updated (newest)",
        "dataset_scan_scope": "All tabular files (recommended)",
        "dataset_page": 1,
        "dataset_page_size": 12,
        "dataset_show_upload": False,
        "dataset_upload_folder": "datasets",
        "dataset_preview_rows": 500,
        "dataset_view_mode": "Head",
        "dataset_chart_type": "Histogram",
        "dataset_last_preview_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    storage_path = Path(STORAGE_DIR)

    controls = st.columns([3.4, 1.5, 1.1, 1.8, 0.8, 0.8])
    controls[0].text_input(
        "Search",
        key="dataset_search",
        placeholder="Search files in current folder...",
        label_visibility="collapsed",
    )
    controls[1].selectbox(
        "Sort",
        options=["Updated (newest)", "Size (largest)", "Name (A-Z)"],
        key="dataset_sort_mode",
        label_visibility="collapsed",
    )
    controls[2].selectbox(
        "Per page",
        options=[8, 12, 20, 40],
        key="dataset_page_size",
        label_visibility="collapsed",
    )
    controls[3].selectbox(
        "Scope",
        options=["All tabular files (recommended)", "Only /datasets folders"],
        key="dataset_scan_scope",
        label_visibility="collapsed",
    )
    if controls[4].button("Upload", use_container_width=True):
        st.session_state.dataset_show_upload = True
    if controls[5].button("Refresh", use_container_width=True):
        _scan_dataset_inventory.clear()
        _load_dataset_preview_cached.clear()
        st.rerun()

    scan_scope_key = "datasets_only" if st.session_state.dataset_scan_scope.startswith("Only /datasets") else "all"
    all_datasets = _scan_dataset_inventory(STORAGE_DIR, scan_scope_key)

    if not all_datasets:
        hint = "`datasets/` folders" if scan_scope_key == "datasets_only" else "`app/storage`"
        st.info(f"No tabular files found under {hint}. Upload one to get started.")
        return

    def _folder_key(folder_value: str) -> str:
        return "" if folder_value in ("", "(root)") else folder_value

    folder_keys = {""}
    for item in all_datasets:
        fkey = _folder_key(item["folder"])
        if not fkey:
            continue
        parts = fkey.split("/")
        for i in range(1, len(parts) + 1):
            folder_keys.add("/".join(parts[:i]))

    current_folder = _folder_key(st.session_state.dataset_browser_folder)
    if current_folder not in folder_keys:
        current_folder = ""
        st.session_state.dataset_browser_folder = ""
    folder_options = ["(root)"] + sorted([k for k in folder_keys if k], key=lambda x: x.lower())
    if st.session_state.dataset_folder_filter not in folder_options:
        st.session_state.dataset_folder_filter = "(root)"

    st.selectbox(
        "Folder",
        options=folder_options,
        key="dataset_folder_filter",
    )
    selected_folder = st.session_state.dataset_folder_filter
    desired_folder = "" if selected_folder == "(root)" else selected_folder
    if desired_folder != current_folder:
        st.session_state.dataset_browser_folder = desired_folder
        st.session_state.dataset_page = 1
        st.rerun()

    if st.session_state.dataset_show_upload:
        with st.expander("Upload Dataset", expanded=True):
            up_cols = st.columns([2.5, 2.5, 1.2, 1.2])
            up_cols[0].text_input("Target folder", key="dataset_upload_folder")
            upload_file = up_cols[1].file_uploader(
                "Choose file",
                type=[s.lstrip(".") for s in sorted(DATASET_SUFFIXES)],
                key="dataset_upload_file",
                label_visibility="collapsed",
            )
            if up_cols[2].button("Save", type="primary", use_container_width=True):
                if upload_file is None:
                    st.warning("Choose a file first.")
                else:
                    try:
                        target_dir = storage_path / st.session_state.dataset_upload_folder.strip()
                        target_dir.mkdir(parents=True, exist_ok=True)
                        out_path = target_dir / upload_file.name
                        with open(out_path, "wb") as f:
                            f.write(upload_file.getbuffer())
                        st.success(f"Uploaded: {out_path.relative_to(storage_path)}")
                        _scan_dataset_inventory.clear()
                        _load_dataset_preview_cached.clear()
                        st.session_state.dataset_show_upload = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
            if up_cols[3].button("Close", use_container_width=True):
                st.session_state.dataset_show_upload = False
                st.rerun()

    query = st.session_state.dataset_search.strip().lower()
    files_in_folder = [
        row for row in all_datasets
        if _folder_key(row["folder"]) == current_folder
    ]
    if query:
        files_in_folder = [
            row for row in files_in_folder
            if query in f"{row['file']} {row['path']}".lower()
        ]

    sort_mode = st.session_state.dataset_sort_mode
    if sort_mode == "Updated (newest)":
        files_in_folder.sort(key=lambda x: x["modified_ts"], reverse=True)
    elif sort_mode == "Size (largest)":
        files_in_folder.sort(key=lambda x: x["size_bytes"], reverse=True)
    else:
        files_in_folder.sort(key=lambda x: x["file"].lower())

    page_size = int(st.session_state.dataset_page_size)
    total_files = len(files_in_folder)
    total_pages = max(1, (total_files + page_size - 1) // page_size)
    st.session_state.dataset_page = min(max(1, int(st.session_state.dataset_page)), total_pages)

    browser_col, preview_col = st.columns([1.1, 1.9], gap="large")

    with browser_col:
        st.markdown("##### Files")
        st.caption(f"Folder: `/{current_folder}`" if current_folder else "Folder: `/`")

        nav_cols = st.columns([1.1, 1.8, 1.1])
        if nav_cols[0].button("Prev", use_container_width=True, disabled=st.session_state.dataset_page <= 1):
            st.session_state.dataset_page -= 1
            st.rerun()
        nav_cols[1].markdown(
            f"<div style='text-align:center;padding-top:0.45rem;'>Page {st.session_state.dataset_page}/{total_pages}</div>",
            unsafe_allow_html=True,
        )
        if nav_cols[2].button("Next", use_container_width=True, disabled=st.session_state.dataset_page >= total_pages):
            st.session_state.dataset_page += 1
            st.rerun()

        st.caption(f"{total_files} files in selected folder")

        if not files_in_folder:
            st.info("No files in this folder match the current filter.")
        else:
            start_idx = (st.session_state.dataset_page - 1) * page_size
            end_idx = min(start_idx + page_size, total_files)
            visible_files = files_in_folder[start_idx:end_idx]
            file_table = pd.DataFrame(
                [
                    {
                        "File": row["file"],
                        "Size": row["size_label"],
                        "Updated": row["modified"],
                    }
                    for row in visible_files
                ]
            )
            st.dataframe(file_table, use_container_width=True, hide_index=True, height=260)

            picker_cols = st.columns([4.0, 1.3])
            picker_suffix = (current_folder or "root").replace("/", "_")
            file_pick_key = f"dataset_file_pick_{picker_suffix}_{st.session_state.dataset_page}"
            selected_file_idx = picker_cols[0].selectbox(
                "Select file",
                options=list(range(len(visible_files))),
                format_func=lambda i: f"{visible_files[i]['file']} ({visible_files[i]['size_label']})",
                key=file_pick_key,
            )
            selected_full_path = visible_files[selected_file_idx]["full_path"]
            already_selected = st.session_state.dataset_preview_path == selected_full_path
            if picker_cols[1].button("Preview", use_container_width=True, disabled=already_selected):
                st.session_state.dataset_preview_path = selected_full_path
                st.rerun()

    with preview_col:
        preview_path_raw = st.session_state.dataset_preview_path
        if not preview_path_raw:
            st.info("Choose a file from the left panel to preview.")
            return

        preview_path = Path(preview_path_raw)
        if not preview_path.exists():
            st.warning("Selected file no longer exists.")
            st.session_state.dataset_preview_path = None
            return

        # Reset chart widget state when switching to a different file.
        current_preview_key = str(preview_path)
        if st.session_state.dataset_last_preview_path != current_preview_key:
            for state_key in [
                "dataset_chart_hist_col",
                "dataset_chart_hist_bins",
                "dataset_chart_scatter_x",
                "dataset_chart_scatter_y",
                "dataset_chart_box_y",
                "dataset_chart_box_x",
                "dataset_chart_bar_col",
                "dataset_chart_bar_topn",
            ]:
                st.session_state.pop(state_key, None)
            st.session_state.dataset_last_preview_path = current_preview_key

        header_cols = st.columns([4.0, 1.4, 1.1, 1.1])
        header_cols[0].markdown(f"#### {preview_path.name}")
        try:
            rel_preview = preview_path.relative_to(storage_path)
            header_cols[0].caption(f"`{rel_preview}`")
        except ValueError:
            header_cols[0].caption(f"`{preview_path}`")

        header_cols[1].selectbox(
            "Rows",
            options=[200, 500, 1000, 2000],
            key="dataset_preview_rows",
            label_visibility="collapsed",
        )
        if header_cols[2].button("Delete", use_container_width=True):
            try:
                os.remove(preview_path)
                st.toast(f"Deleted {preview_path.name}")
                _scan_dataset_inventory.clear()
                _load_dataset_preview_cached.clear()
                st.session_state.dataset_preview_path = None
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")
        if header_cols[3].button("Close", use_container_width=True):
            st.session_state.dataset_preview_path = None
            st.rerun()

        try:
            file_stat = preview_path.stat()
            df = _load_dataset_preview_cached(
                str(preview_path),
                int(st.session_state.dataset_preview_rows),
                float(file_stat.st_mtime),
            )
        except Exception as e:
            st.error(f"Could not load preview: {e}")
            return

        numeric_cols = list(df.select_dtypes(include="number").columns)
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
        missing_cells = int(df.isna().sum().sum())

        metrics = st.columns(5)
        metrics[0].metric("Rows loaded", f"{len(df):,}")
        metrics[1].metric("Columns", len(df.columns))
        metrics[2].metric("Numeric cols", len(numeric_cols))
        metrics[3].metric("Missing cells", f"{missing_cells:,}")
        metrics[4].metric("File size", _format_dataset_size(file_stat.st_size))

        tab_preview, tab_columns, tab_chart = st.tabs(["Data Preview", "Column Review", "Quick Chart"])

        with tab_preview:
            mode_cols = st.columns([1.5, 1.5, 3.0])
            mode_cols[0].radio(
                "Mode",
                options=["Head", "Tail", "Sample"],
                key="dataset_view_mode",
                horizontal=True,
                label_visibility="collapsed",
            )
            rows_to_show = mode_cols[1].slider(
                "Rows to show",
                min_value=10,
                max_value=max(10, min(200, len(df))),
                value=min(50, max(10, len(df))),
                step=10,
                label_visibility="collapsed",
            )

            if st.session_state.dataset_view_mode == "Tail":
                view_df = df.tail(rows_to_show)
            elif st.session_state.dataset_view_mode == "Sample":
                sample_n = min(rows_to_show, len(df))
                view_df = df.sample(sample_n, random_state=42) if sample_n > 0 else df
            else:
                view_df = df.head(rows_to_show)

            st.dataframe(view_df, use_container_width=True, height=420)

        with tab_columns:
            profile_df = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Type": [str(df[c].dtype) for c in df.columns],
                    "Non-Null": [int(df[c].notna().sum()) for c in df.columns],
                    "Missing": [int(df[c].isna().sum()) for c in df.columns],
                    "Missing %": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
                    "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
                }
            ).sort_values(["Missing", "Unique"], ascending=[False, False])
            st.dataframe(profile_df, use_container_width=True, height=420)

        with tab_chart:
            try:
                import plotly.express as px
            except Exception:
                st.warning("Plotly is not available for interactive charts.")
                return

            if df.empty:
                st.info("No rows available to chart.")
                return

            chart_type = st.selectbox(
                "Chart type",
                options=["Histogram", "Scatter", "Box", "Bar (Top Categories)"],
                key="dataset_chart_type",
            )

            if chart_type == "Histogram":
                if not numeric_cols:
                    st.info("Histogram needs at least one numeric column.")
                else:
                    c1, c2 = st.columns([2, 1])
                    hist_col = c1.selectbox("Column", options=numeric_cols, key="dataset_chart_hist_col")
                    bins = c2.slider("Bins", 10, 80, 30, key="dataset_chart_hist_bins")
                    fig = px.histogram(df, x=hist_col, nbins=bins, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Scatter":
                if len(numeric_cols) < 2:
                    st.info("Scatter needs at least two numeric columns.")
                else:
                    c1, c2 = st.columns(2)
                    x_col = c1.selectbox("X axis", options=numeric_cols, key="dataset_chart_scatter_x")
                    y_default = 1 if len(numeric_cols) > 1 else 0
                    y_col = c2.selectbox("Y axis", options=numeric_cols, index=y_default, key="dataset_chart_scatter_y")
                    fig = px.scatter(df, x=x_col, y=y_col, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Box":
                if not numeric_cols:
                    st.info("Box plot needs at least one numeric column.")
                else:
                    c1, c2 = st.columns(2)
                    y_col = c1.selectbox("Value", options=numeric_cols, key="dataset_chart_box_y")
                    if non_numeric_cols:
                        x_col = c2.selectbox("Group (optional)", options=["(none)"] + non_numeric_cols, key="dataset_chart_box_x")
                        x_arg = None if x_col == "(none)" else x_col
                    else:
                        x_arg = None
                    fig = px.box(df, x=x_arg, y=y_col, points="outliers", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

            else:
                cat_options = non_numeric_cols if non_numeric_cols else list(df.columns)
                c1, c2 = st.columns([2, 1])
                cat_col = c1.selectbox("Category column", options=cat_options, key="dataset_chart_bar_col")
                top_n = c2.slider("Top N", 5, 30, 12, key="dataset_chart_bar_topn")
                counts = (
                    df[cat_col]
                    .astype(str)
                    .value_counts(dropna=False)
                    .head(top_n)
                    .rename_axis(cat_col)
                    .reset_index(name="count")
                )
                fig = px.bar(counts, x=cat_col, y="count", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)


def render_graph_visualizer():
    """Graph Visualizer using PyVis with Data Flow Analysis."""
    st.markdown("### 🕸️ Pipeline Flow Visualizer")
    
    # --- Pipeline Source Selector ---
    pipeline_options = ["📝 Current Pipeline"]
    saved_pipelines = {}
    
    # Load saved pipelines from Knowledge Base
    try:
        from kb import KnowledgeBase
        kb_path = os.path.join(os.path.dirname(__file__), "kb.sqlite")
        if os.path.exists(kb_path):
            kb = KnowledgeBase(kb_path)
            pipelines_list = kb.list_pipelines()
            for p in pipelines_list:
                name = p.get("name", "Unnamed")
                display_name = f"💾 {name}"
                pipeline_options.append(display_name)
                saved_pipelines[display_name] = p
            kb.close()
    except Exception as e:
        st.warning(f"Could not load saved pipelines: {e}")
    
    # Pipeline selector
    selected_source = st.selectbox(
        "Pipeline Source",
        pipeline_options,
        index=0,
        help="Select which pipeline to visualize",
        key="graph_pipeline_source"
    )
    
    # Determine which steps to use
    if selected_source == "📝 Current Pipeline":
        steps = st.session_state.pipeline_steps
        if not steps:
            st.info("📭 Pipeline is empty. Add steps in the **Pipeline Editor** to visualize the data flow.")
            return
    else:
        # Load from saved pipeline
        pipeline_data = saved_pipelines.get(selected_source)
        if not pipeline_data:
            st.error("Could not load selected pipeline.")
            return
        
        spec = pipeline_data.get("specification", [])
        if isinstance(spec, str):
            import json
            spec = json.loads(spec)
        
        # Handle both formats: dict with 'steps' key or list of steps directly
        if isinstance(spec, dict) and 'steps' in spec:
            steps = spec['steps']
        else:
            steps = spec
        
        if not steps:
            st.info(f"📭 Pipeline '{pipeline_data.get('name')}' is empty.")
            return
        
        # Show pipeline info
        st.caption(f"📊 **{pipeline_data.get('name')}** • {len(steps)} steps • {pipeline_data.get('problem_type', 'N/A')}")

    try:
        from pyvis.network import Network
        import networkx as nx
        import tempfile
    except ImportError:
        st.error("PyVis is not installed. Run: `pip install pyvis`")
        return

    # --- Horizontal Toolbar ---
    st.markdown("##### ⚙️ View Settings")
    col0, col1, col2, col3, col4 = st.columns([1.7, 1.4, 1.4, 1, 1])

    with col0:
        node_view = st.selectbox(
            "Node View",
            ["Node by Data", "Node by Services"],
            index=0,
            label_visibility="collapsed",
            key="graph_node_view",
            help="Choose whether nodes represent data files or services."
        )

    with col1:
        layout_type = st.selectbox(
            "Layout",
            ["Hierarchical (DAG)", "Organic (Free)"],
            index=0,
            label_visibility="collapsed",
            key="graph_layout_type"
        )

    with col2:
        if layout_type == "Hierarchical (DAG)":
            direction = st.selectbox(
                "Direction",
                ["LR", "UD", "RL", "DU"],
                index=0,
                format_func=lambda x: {"LR": "→ Left to Right", "UD": "↓ Top to Bottom", "RL": "← Right to Left", "DU": "↑ Bottom to Top"}[x],
                label_visibility="collapsed",
                key="graph_direction"
            )
        else:
            direction = "UD"
            st.empty()

    with col3:
        node_sep = st.slider(
            "Spacing",
            150,
            600,
            350,
            label_visibility="collapsed",
            help="Node spacing",
            key="graph_node_spacing"
        )

    with col4:
        level_sep = (
            st.slider(
                "Layers",
                200,
                800,
                450,
                label_visibility="collapsed",
                help="Layer separation",
                key="graph_layer_spacing"
            )
            if layout_type == "Hierarchical (DAG)"
            else 300
        )

    st.divider()

    # --- Build Dependency Graph ---
    G = nx.DiGraph()
    start_node_id = "START"
    end_node_id = "END"

    def _normalized_paths(io_map: Dict[str, Any]) -> List[str]:
        return [
            str(path).strip()
            for path in (io_map or {}).values()
            if isinstance(path, str) and path.strip()
        ]

    def _add_or_merge_edge(src: Any, dst: Any, label: str, title: str) -> None:
        if G.has_edge(src, dst):
            current_label = G[src][dst].get("label", "")
            current_title = G[src][dst].get("title", "")
            labels = [x for x in current_label.split(" | ") if x]
            titles = [x for x in current_title.split("\n") if x]
            if label and label not in labels:
                labels.append(label)
            if title and title not in titles:
                titles.append(title)
            G[src][dst]["label"] = " | ".join(labels)
            G[src][dst]["title"] = "\n".join(titles)
        else:
            G.add_edge(src, dst, label=label, title=title)

    all_outputs = set()
    all_inputs = set()
    for step in steps:
        all_outputs.update(_normalized_paths(step.get("outputs", {})))
        all_inputs.update(_normalized_paths(step.get("inputs", {})))

    external_inputs = all_inputs - all_outputs  # Inputs not produced by any step
    final_outputs = all_outputs - all_inputs    # Outputs not consumed by any step

    if node_view == "Node by Services":
        file_producers = {}

        # First pass: Register file producers
        for i, step in enumerate(steps):
            for out_path in _normalized_paths(step.get("outputs", {})):
                file_producers[out_path] = i

        # Second pass: Create service nodes + dependency edges
        for i, step in enumerate(steps):
            svc_name = step.get("service", f"service_{i + 1}")
            G.add_node(i, service=svc_name)

            for in_path in _normalized_paths(step.get("inputs", {})):
                producer_idx = file_producers.get(in_path)
                if producer_idx is not None and producer_idx != i:
                    fname = os.path.basename(in_path) or in_path
                    _add_or_merge_edge(producer_idx, i, fname, f"Data: {in_path}")

        # START node for external inputs
        if external_inputs:
            start_tooltip = "📥 External Inputs\n\n" + "\n".join([f"• {p}" for p in sorted(external_inputs)])
            G.add_node(start_node_id, label="📥 START\nExternal Data", title=start_tooltip, service="START", is_special=True)

            for i, step in enumerate(steps):
                for in_path in _normalized_paths(step.get("inputs", {})):
                    if in_path in external_inputs:
                        fname = os.path.basename(in_path) or in_path
                        _add_or_merge_edge(start_node_id, i, fname, f"External: {in_path}")

        # END node for final outputs
        if final_outputs:
            end_tooltip = "📤 Final Outputs\n\n" + "\n".join([f"• {p}" for p in sorted(final_outputs)])
            G.add_node(end_node_id, label="📤 END\nFinal Results", title=end_tooltip, service="END", is_special=True)

            for i, step in enumerate(steps):
                for out_path in _normalized_paths(step.get("outputs", {})):
                    if out_path in final_outputs:
                        fname = os.path.basename(out_path) or out_path
                        _add_or_merge_edge(i, end_node_id, fname, f"Final: {out_path}")

        # Order service nodes by DAG topological order when possible
        try:
            execution_order = list(nx.topological_sort(G))
            step_counter = 1
            exec_step_map = {}
            for node_id in execution_order:
                if node_id not in [start_node_id, end_node_id]:
                    exec_step_map[node_id] = step_counter
                    step_counter += 1
        except nx.NetworkXUnfeasible:
            exec_step_map = {i: i + 1 for i in range(len(steps))}

        for i, step in enumerate(steps):
            svc_name = step.get("service", f"service_{i + 1}")
            exec_step_num = exec_step_map.get(i, i + 1)
            display_name = svc_name[:25] + "..." if len(svc_name) > 25 else svc_name
            label = f"Step {exec_step_num}\n{display_name}"

            inputs_list = [f"  • {k}: {v}" for k, v in (step.get("inputs", {}) or {}).items() if v]
            outputs_list = [f"  • {k}: {v}" for k, v in (step.get("outputs", {}) or {}).items() if v]
            params = step.get("params", {}) or {}
            params_list = [f"  • {k}: {v}" for k, v in params.items()]

            title_parts = [f"Step {exec_step_num}: {svc_name}", ""]
            title_parts.append("Inputs:")
            title_parts.extend(inputs_list if inputs_list else ["  (none)"])
            title_parts.append("")
            title_parts.append("Outputs:")
            title_parts.extend(outputs_list if outputs_list else ["  (none)"])
            title_parts.append("")
            title_parts.append("Parameters:")
            title_parts.extend(params_list if params_list else ["  (none)"])

            G.nodes[i]["label"] = label
            G.nodes[i]["title"] = "\n".join(title_parts)
    else:
        # Node by Data: data files are nodes, services are edge labels.
        all_data_paths = sorted(all_inputs | all_outputs)
        for path in all_data_paths:
            producers = []
            consumers = []
            for i, step in enumerate(steps):
                if path in _normalized_paths(step.get("outputs", {})):
                    producers.append(f"  • Step {i + 1}: {step.get('service', 'unknown')}")
                if path in _normalized_paths(step.get("inputs", {})):
                    consumers.append(f"  • Step {i + 1}: {step.get('service', 'unknown')}")

            if path in external_inputs:
                node_kind = "external"
            elif path in final_outputs:
                node_kind = "final"
            else:
                node_kind = "intermediate"

            filename = os.path.basename(path) or path
            display_name = filename[:32] + "..." if len(filename) > 32 else filename
            title_parts = [f"Data File: {path}", ""]
            title_parts.append("Produced by:")
            title_parts.extend(producers if producers else ["  (external input)"])
            title_parts.append("")
            title_parts.append("Consumed by:")
            title_parts.extend(consumers if consumers else ["  (final output)"])

            G.add_node(
                path,
                label=f"📄 {display_name}",
                title="\n".join(title_parts),
                node_kind=node_kind,
            )

        add_start = False
        add_end = False
        for i, step in enumerate(steps):
            svc_name = step.get("service", f"service_{i + 1}")
            svc_label = svc_name[:20] + "..." if len(svc_name) > 20 else svc_name
            edge_label = f"S{i + 1}: {svc_label}"
            edge_title = f"Step {i + 1}: {svc_name}"

            step_inputs = _normalized_paths(step.get("inputs", {}))
            step_outputs = _normalized_paths(step.get("outputs", {}))

            if step_inputs and step_outputs:
                for in_path in step_inputs:
                    for out_path in step_outputs:
                        if in_path != out_path:
                            _add_or_merge_edge(in_path, out_path, edge_label, edge_title)
            elif step_outputs:
                add_start = True
                for out_path in step_outputs:
                    _add_or_merge_edge(start_node_id, out_path, edge_label, edge_title)
            elif step_inputs:
                add_end = True
                for in_path in step_inputs:
                    _add_or_merge_edge(in_path, end_node_id, edge_label, edge_title)

        if add_start:
            G.add_node(
                start_node_id,
                label="📥 START\nGenerated",
                title="Services that emit outputs without explicit inputs",
                is_special=True
            )
        if add_end:
            G.add_node(
                end_node_id,
                label="📤 END\nTerminal",
                title="Services that consume inputs without explicit outputs",
                is_special=True
            )

    # --- Calculate Positions (Static Layout) ---
    positions = {}
    if layout_type == "Hierarchical (DAG)":
        try:
            layers = list(nx.topological_generations(G))
            for layer_idx, nodes_in_layer in enumerate(layers):
                num_nodes = len(nodes_in_layer)
                for node_idx, node_id in enumerate(nodes_in_layer):
                    offset = (node_idx - (num_nodes - 1) / 2.0) * node_sep
                    layer_pos = layer_idx * level_sep
                    
                    if direction == "LR":
                        x, y = layer_pos, offset
                    elif direction == "UD":
                        x, y = offset, layer_pos
                    elif direction == "RL":
                        x, y = -layer_pos, offset
                    elif direction == "DU":
                        x, y = offset, -layer_pos
                    else:
                        x, y = layer_pos, offset
                    
                    positions[node_id] = {'x': x, 'y': y}
        except Exception:
            pos = nx.kamada_kawai_layout(G)
            scale = max(node_sep * 3, 800)
            for n, p in pos.items():
                positions[n] = {'x': p[0] * scale, 'y': p[1] * scale}
    else:
        pos = nx.kamada_kawai_layout(G)
        scale = max(node_sep * 4, 1000)
        for n, p in pos.items():
            positions[n] = {'x': p[0] * scale, 'y': p[1] * scale}

    # --- Create PyVis Network ---
    net = Network(
        height="850px", 
        width="100%", 
        directed=True, 
        bgcolor="#f8f9fa",  # Light gray background
        font_color="#1a1a2e"
    )
    
    # Color palette for nodes
    node_colors = [
        "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7", 
        "#dfe6e9", "#74b9ff", "#a29bfe", "#fd79a8"
    ]
    
    # Add Nodes with improved styling
    data_node_styles = {
        "external": {"background": "#81ecec", "border": "#00a8a8"},
        "intermediate": {"background": "#dfe6e9", "border": "#636e72"},
        "final": {"background": "#ffeaa7", "border": "#e1b12c"},
    }

    for idx, n_id in enumerate(G.nodes()):
        node_attr = G.nodes[n_id]
        x = positions.get(n_id, {}).get('x')
        y = positions.get(n_id, {}).get('y')
        
        # Special styling for START and END nodes
        if n_id == "START":
            net.add_node(
                n_id, 
                label=node_attr['label'], 
                title=node_attr['title'], 
                shape="ellipse",
                color={
                    "background": "#00b894",  # Green
                    "border": "#00695c",
                    "highlight": {"background": "#55efc4", "border": "#00695c"}
                },
                font={"size": 16, "face": "Arial", "color": "#ffffff", "bold": True},
                borderWidth=3,
                borderWidthSelected=5,
                shadow=True,
                margin=20,
                x=x, 
                y=y
            )
        elif n_id == "END":
            net.add_node(
                n_id, 
                label=node_attr['label'], 
                title=node_attr['title'], 
                shape="ellipse",
                color={
                    "background": "#e17055",  # Orange-red
                    "border": "#d63031",
                    "highlight": {"background": "#fab1a0", "border": "#d63031"}
                },
                font={"size": 16, "face": "Arial", "color": "#ffffff", "bold": True},
                borderWidth=3,
                borderWidthSelected=5,
                shadow=True,
                margin=20,
                x=x, 
                y=y
            )
        else:
            if node_view == "Node by Data":
                node_kind = node_attr.get("node_kind", "intermediate")
                style = data_node_styles.get(node_kind, data_node_styles["intermediate"])
                net.add_node(
                    n_id,
                    label=node_attr["label"],
                    title=node_attr["title"],
                    shape="box",
                    color={
                        "background": style["background"],
                        "border": style["border"],
                        "highlight": {"background": "#dfe6e9", "border": "#2d3436"}
                    },
                    font={"size": 14, "face": "Arial", "color": "#2d3436", "bold": True},
                    borderWidth=2,
                    borderWidthSelected=4,
                    shadow=True,
                    margin=12,
                    x=x,
                    y=y
                )
            else:
                # Regular service nodes
                color = node_colors[idx % len(node_colors)]
                net.add_node(
                    n_id,
                    label=node_attr["label"],
                    title=node_attr["title"],
                    shape="box",
                    color={
                        "background": color,
                        "border": "#2d3436",
                        "highlight": {"background": "#00cec9", "border": "#0984e3"}
                    },
                    font={"size": 18, "face": "Arial", "color": "#2d3436", "bold": True},
                    borderWidth=2,
                    borderWidthSelected=4,
                    shadow=True,
                    margin=15,
                    x=x,
                    y=y
                )
        
    # Add Edges with improved styling
    for u, v, data in G.edges(data=True):
        net.add_edge(
            u, v, 
            label=data.get('label', ''), 
            title=data.get('title', ''),
            arrows={"to": {"enabled": True, "scaleFactor": 1.2}},
            color={"color": "#636e72", "highlight": "#0984e3"},
            width=2.5,
            smooth={"type": "curvedCW", "roundness": 0.2},
            font={"size": 14, "face": "Arial", "color": "#2d3436", "strokeWidth": 3, "strokeColor": "#ffffff"}
        )

    # --- PyVis Options ---
    options_dict = {
        "physics": {"enabled": False},
        "interaction": {
            "dragNodes": True,
            "zoomView": True,
            "dragView": True,
            "hover": True,
            "tooltipDelay": 50
        },
        "nodes": {
            "shapeProperties": {"borderRadius": 8}
        }
    }

    net.set_options(json.dumps(options_dict))
    
    # --- Render Graph ---
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                html_content = f.read()
        
        # Display with full width
        components.html(html_content, height=850, scrolling=False)
        
        # Stats bar
        if node_view == "Node by Data":
            data_nodes = [n for n in G.nodes() if n not in [start_node_id, end_node_id]]
            st.caption(f"📊 **{len(data_nodes)}** data nodes • **{len(G.edges())}** service transformations")
        else:
            st.caption(f"📊 **{len(G.nodes())}** steps • **{len(G.edges())}** data connections")
        
    except Exception as e:
        st.error(f"Error generating graph: {e}")

# =============================================================================
# MAIN LAYOUT
# =============================================================================

def main():
    # Keep JSON editors in sync with the current pipeline state (must run before widgets render).
    maybe_sync_pipeline_json_views()

    # Sidebar Navigation
    with st.sidebar:
        st.title("🧱 Contract-Composable Analytics")
        nav_mode = st.radio(
            "Navigation", 
            [
                "Pipeline Editor",
                "Graph Visualizer",
                "Microservice Editor",
                "Dataset Editor",
                "Knowledge Base Editor", 
                "Software Introduction", 
                "About the Creator"
            ]
        )
        st.markdown("---")
        
        # Panel visibility controls (only for Pipeline Editor)
        if nav_mode == "Pipeline Editor":
            st.markdown("##### 📐 Panels")
            show_resources = st.checkbox("🗄️ Resources", value=True, key="show_resources")
            show_workspace = st.checkbox("🚧 Workspace", value=True, key="show_workspace")
            show_ai_recommender = st.checkbox("🧠 AI Recommender", value=True, key="show_ai_recommender")
            with st.popover("📏 Layout"):
                res_width = st.slider("Resources Width", 0.5, 5.0, 1.2, 0.1) if show_resources else 0
                ws_width = st.slider("Workspace Width", 0.5, 10.0, 2.5, 0.1) if show_workspace else 0
                ai_width = st.slider("AI Recommender Width", 0.5, 5.0, 1.5, 0.1) if show_ai_recommender else 0
            st.markdown("---")
        
        st.info(f"Mode: {nav_mode}")
    
    # Main Content Area
    if nav_mode == "Pipeline Editor":
        # Determine visible panels
        visible_panels = []
        if st.session_state.get("show_resources", True):
            visible_panels.append("resources")
        if st.session_state.get("show_workspace", True):
            visible_panels.append("workspace")
        if st.session_state.get("show_ai_recommender", True):
            visible_panels.append("ai_recommender")
        
        if not visible_panels:
            st.warning("All panels hidden. Use sidebar to enable panels.")
            return
        
        # Calculate column ratios using slider values
        ratio = []
        if "resources" in visible_panels: ratio.append(res_width)
        if "workspace" in visible_panels: ratio.append(ws_width)
        if "ai_recommender" in visible_panels: ratio.append(ai_width)
        
        cols = st.columns(ratio, gap="medium")
        col_idx = 0
        
        # 1. Resources Panel (Left)
        if "resources" in visible_panels:
            with cols[col_idx]:
                 # No expander needed if we want it to feel like a sidebar, or use one for consistency
                 with st.container(): # Use container to keep scroll scope
                    render_resources_panel()
            col_idx += 1
        
        # 2. Workspace Panel (Center)
        if "workspace" in visible_panels:
            with cols[col_idx]:
                with st.container():
                    render_pipeline_builder()
            col_idx += 1
        
        # 3. AI Recommender Panel (Right)
        if "ai_recommender" in visible_panels:
            with cols[col_idx]:
                with st.container():
                    render_right_panel()

    elif nav_mode == "Graph Visualizer":
        render_graph_visualizer()

    # elif nav_mode == "Snake":
    #     render_snake_game()  # Easter egg, removed for publication

    elif nav_mode == "Microservice Editor":
        render_microservice_editor()
    
    elif nav_mode == "Dataset Editor":
        render_dataset_editor()
        
    elif nav_mode == "Knowledge Base Editor":
        render_kb_editor()
        
    elif nav_mode == "Software Introduction":
        st.header("Introduction")
        st.markdown(
            """
            **Contract-Composable Analytics** (Service-LEGO) is a modular architecture for data analytics.
            It solves the "3 Tribes" problem by separating concerns:
            - **Tech Experts** build reusable microservices.
            - **Domain Experts** compose pipelines using these blocks.
            - **End Users** consume the results.
            """
        )
        
    elif nav_mode == "About the Creator":
        st.header("About")
        st.info("Created by Antigravity.")

if __name__ == "__main__":
    main()
