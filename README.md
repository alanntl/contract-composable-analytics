# Contract-Composable Analytics

A contract-driven framework for composable analytics pipelines, built on the Contract-Composable Analytics (Service LEGo Orchestration) system. This repository accompanies the conference paper submission.

## Overview

Contract-Composable Analytics enables non-expert users to assemble analytics pipelines from reusable, contract-verified microservices. The system features:

- **Contract System (G2, G6):** Machine-checkable data interface contracts via Python decorators, ensuring type-safe service composition
- **Knowledge Base:** SQLite-backed service/pipeline registry with FAISS vector embeddings for semantic search
- **AI Pipeline Composer:** LangGraph-based recommender that retrieves similar pipelines via FAISS and uses LLM to compose new ones
- **DAG Execution Engine:** Topologically-ordered pipeline runner with pre-flight contract validation
- **Formal Verification:** Dafny proofs for contract properties (type safety, composability)
- **100 Composable Microservices** spanning 107 Kaggle competitions across 16 domain categories

## Repository Structure

```
contract-composible-analytics/
├── README.md
├── DATASETS.md                  # 107 Kaggle competitions with download instructions
├── requirements.txt             # Python dependencies
│
├── paper/                       # Conference paper source
│   ├── main.tex                 # LaTeX source (IEEE format)
│   ├── main.pdf                 # Compiled paper
│   ├── Contract-Composable AnalyticsContracts.dfy       # Dafny formal verification
│   ├── IEEEtran.cls             # IEEE template
│   ├── algorithm.sty / algorithmicx.sty / algpseudocode.sty
│   └── figure/                  # Paper figures
│
├── app/                         # Runnable Streamlit application
│   ├── contract_app_streamlit.py   # Main UI
│   ├── contract_contract.py        # Contract decorator system
│   ├── pipeline_runner.py       # DAG execution engine
│   ├── contract_kb.py              # Knowledge base ORM
│   ├── recommender.py           # FAISS + LLM pipeline composer
│   ├── kb.sqlite          # Pre-built knowledge base
│   ├── .env.example             # API key template
│   ├── style.css
│   ├── services/                # 100 composable microservices
│   ├── tests/                   # Pytest test suite
│   └── storage/                 # Runtime data directory
│
└── evaluation/                  # Evaluation scripts and results
    ├── EVALUATION_METHODOLOGY.md
    ├── phase1_generate.py / .csv
    ├── phase2_execute.py / .csv
    ├── run_evaluation_compose.py
    ├── run_evaluation_with_execution.py
    ├── analyze_reuse.py
    ├── *.xlsx / *.csv           # Evaluation results
    └── experiments/             # Baseline comparisons
        ├── bm25_baseline/
        ├── tfidf_baseline/
        ├── random_baseline/
        ├── faiss recommendation/
        ├── llm_rerank/
        └── data_and_reusability/
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys (optional, for AI recommender)

```bash
cd app
cp .env.example .env
# Edit .env with your OpenAI and/or Gemini API keys
```

### 3. Run the app

```bash
cd app
streamlit run contract_app_streamlit.py
```

The app will open at `http://localhost:8501`.

### 4. Download datasets (optional)

See [DATASETS.md](DATASETS.md) for the full list of 107 Kaggle competitions and a bulk download script. Datasets go into `app/storage/<competition-slug>/datasets/`.

```bash
pip install kaggle
kaggle competitions download -c <competition-slug>
```

## Key Components

| File | Description |
|------|-------------|
| `contract_contract.py` | `@contract` decorator, `IOManager`, `ServiceRegistry` — implements Guidelines G2 and G6 |
| `pipeline_runner.py` | DAG-based pipeline execution with topological ordering and contract validation |
| `recommender.py` | FAISS retrieval + LangGraph + LLM composition for pipeline recommendation |
| `contract_kb.py` | Knowledge base storing services, pipelines, embeddings, and execution history |
| `Contract-Composable AnalyticsContracts.dfy` | Dafny formal verification of contract properties |

## Evaluation

The `evaluation/` directory contains all scripts and results for reproducibility:

- **Baselines:** Random, BM25, TF-IDF, FAISS, LLM-rerank
- **End-to-end:** Pipeline generation (Phase 1) and execution (Phase 2)
- **Methodology:** See `EVALUATION_METHODOLOGY.md`

## License

This repository is provided for conference review purposes.
