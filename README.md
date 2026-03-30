# Contract-Composable Analytics

A contract-driven framework for composable analytics pipelines. Non-expert users can assemble analytics workflows from reusable, contract-verified microservices through a Streamlit web interface.

## Overview

- **Contract System:** Machine-checkable data interface contracts via Python decorators, ensuring type-safe service composition
- **Knowledge Base:** SQLite-backed service/pipeline registry with FAISS vector embeddings for semantic search
- **AI Pipeline Composer:** LangGraph-based recommender that retrieves similar pipelines via FAISS and uses LLM to compose new ones
- **DAG Execution Engine:** Topologically-ordered pipeline runner with pre-flight contract validation
- **100+ Composable Microservices** spanning 107 Kaggle competitions across 16 domain categories

## Repository Structure

```
contract-composible-analytics/
├── README.md
├── DATASETS.md                  # 107 Kaggle competitions with download instructions
├── requirements.txt             # Python dependencies
│
├── app/                         # Runnable Streamlit application
│   ├── app_streamlit.py         # Main UI
│   ├── contract.py              # Contract decorator system
│   ├── pipeline_runner.py       # DAG execution engine
│   ├── kb.py                    # Knowledge base ORM
│   ├── recommender.py           # FAISS + LLM pipeline composer
│   ├── kb.sqlite                # Pre-built knowledge base
│   ├── .env.example             # API key template
│   ├── style.css
│   ├── services/                # 100+ composable microservices
│   ├── tests/                   # Pytest test suite
│   └── storage/                 # Runtime data directory
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
streamlit run app_streamlit.py
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
| `contract.py` | `@contract` decorator, `IOManager`, `ServiceRegistry` |
| `pipeline_runner.py` | DAG-based pipeline execution with topological ordering and contract validation |
| `recommender.py` | FAISS retrieval + LangGraph + LLM composition for pipeline recommendation |
| `kb.py` | Knowledge base storing services, pipelines, embeddings, and execution history |

## License

See [LICENSE](LICENSE).
