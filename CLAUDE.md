# AutoViz — Claude Code Guide

## Project overview

AutoViz is a pure CLI tool (no server, no database) that takes a CSV file and produces AI-generated visualisation charts (PNG) and an optional written report. It uses:

- **pydantic-ai** with **DeepSeek** (`deepseek-chat`, OpenAI-compatible) for all LLM calls
- **Modal** remote sandboxes for executing generated Python/matplotlib code safely
- **Click** for the CLI
- **uv** for dependency management

Entry point: `autoviz analyze run data.csv`

---

## Package layout

```
autoviz/
├── cli.py              # Click CLI + run_analyze_pipeline() + run_report_only()
├── config.py           # Pydantic BaseSettings, reads .env
├── events.py           # VizResult dataclass, message bus helpers
├── agents/
│   ├── context.py      # Builds LLM model/API-key/language context
│   ├── prompts.py      # All system prompt strings
│   ├── feature_engineer.py  # pydantic-ai agent: decides FE transformations
│   ├── planner.py      # pydantic-ai agent: plans list[VizTaskSpec]
│   ├── coder.py        # pydantic-ai agent: writes matplotlib code per task
│   ├── critic.py       # pydantic-ai agent: text-only chart QC (digest, no image)
│   ├── annotator.py    # pydantic-ai agent: text-only insight annotation
│   ├── chart_digest.py # deterministic text digest of a chart (task + data + code)
│   ├── selector.py     # pydantic-ai agent: feasibility + IG filter
│   ├── reporter.py     # pydantic-ai agent: generates DataReport
│   ├── summarizer.py   # executive summary per analysis batch (folder report)
│   └── translator.py   # English report → Chinese report_chn.md
├── store/
│   ├── embeddings.py   # sentence-transformers (BGE) + hash backend for tests
│   ├── schema.py       # pgvector DDL (runs / charts / embeddings, HNSW)
│   ├── repository.py   # psycopg repository + cosine search + auto_prune_dead_links
│   └── persist.py      # ingest a run directory, build embedding text
└── pipeline/
    ├── csv.py          # preprocess_csv_with_data() → CsvSummary
    ├── features.py     # run_feature_engineering() — calls Modal sandbox
    ├── charts.py       # run_visualization() — multi-agent orchestration
    ├── sandbox.py      # Modal sandbox wrapper (run_in_sandbox)
    └── runner.py       # SubAgentManager — concurrent task execution
```

---

## Key data flow

```
CLI (cli.py)
  → preprocess_csv_with_data()          # csv.py  → CsvSummary + cleaned CSV string
  → run_feature_engineering()           # features.py → enhanced CSV string (Modal)
  → run_visualization()                 # charts.py
      → plan_visualizations()           # planner.py → list[VizTaskSpec]
      → select_tasks()                  # selector.py (optional, off by default)
      → SubAgentManager.run_all()       # runner.py
          → generate_code()             # coder.py → Python code string
          → run_in_sandbox()            # sandbox.py → PNG bytes or error
          → review_chart()              # critic.py → accept / code-retry / skip
          → annotate_chart()            # annotator.py → adds text annotation
  → generate_report()                   # reporter.py → DataReport
```

---

## Running the tool

```bash
uv sync                          # install deps
cp .env.example .env             # add your keys
uv run autoviz analyze run sample/order.csv --report
```

Required `.env` keys: `DEEPSEEK_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.

---

## Config reference (`autoviz/config.py`)

| Key | Default | Purpose |
|-----|---------|---------|
| `AI_MODEL` | `deepseek-chat` | DeepSeek model name |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | Documented base URL (agents pin it in code) |
| `AI_TEMPERATURE` | `0.7` | LLM temperature |
| `DISABLE_SSL_VERIFY` | `False` | Local-dev only: bypass TLS verification for httpx/grpclib clients |
| `CSV_MAX_ANALYSIS_ROWS` | `2000` | Random-sample cap for large CSVs (0 = no limit) |
| `VIZ_TARGET_CHARTS` | `10` | Charts to generate per run |
| `VIZ_CONCURRENCY` | `10` | Concurrent sub-agent tasks |
| `VIZ_TURNS` | `1` | Planning iterations |
| `VIZ_CRITIC_MAX_CODE_ROUNDS` | `3` | Max code-fix retries per chart |
| `VIZ_CRITIC_MAX_SEMANTIC_REPLANS` | `2` | Max semantic-skip replacements |
| `VIZ_SELECTOR_ENABLED` | `False` | Enable feasibility/IG filter |
| `DATABASE_URL` | `""` | PostgreSQL DSN; empty disables persistence/search |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | Local sentence-transformers model |
| `EMBEDDING_DIM` | `512` | Must match `vector(dim)` and the model |
| `RAG_TOP_K` | `5` | Default semantic-search result count |
| `RAG_RECALL_K` | `20` | Vector candidates recalled before reranking |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | CrossEncoder reranker |
| `HF_ENDPOINT` | empty | Optional HuggingFace mirror (e.g. hf-mirror.com) |

---

## Conventions

- All LLM calls go through `autoviz/agents/context.py` helpers (`get_effective_model`, `get_effective_api_key`). Never hard-code model names or keys.
- DeepSeek models are text-only: never pass images/`BinaryContent` to an agent. The critic and annotator reason over `agents/chart_digest.py` output (task + data profile + statically parsed axes/scales/aggregation from the generated code).
- All code execution happens inside Modal sandboxes via `autoviz/pipeline/sandbox.py`. No `exec()` or `subprocess` locally.
- `VizResult` (from `autoviz/events.py`) is the canonical result type flowing from `runner.py` → `charts.py` → `cli.py`.
- Output directory layout is always `<base>/<csv-stem>/<timestamp>/` — set in `cli.py:analyze_run`.
- `csv_summary.json` and `charts_metadata.json` in the output dir are the contract between `analyze run` and `analyze report`.

---

## Adding a new agent

1. Create `autoviz/agents/<name>.py` with a pydantic-ai `Agent` and an `async def` entry function.
2. Use `get_effective_model()` / `get_effective_api_key()` from `context.py` for the model.
3. Wire it into the relevant pipeline step in `pipeline/charts.py`, `pipeline/features.py`, or `cli.py`.
4. Add any new prompts to `agents/prompts.py`.

---

## Dependencies

Managed via `pyproject.toml` + `uv.lock`. Key packages:

- `pydantic-ai-slim[openai]` — agent framework (DeepSeek via OpenAI-compatible endpoint)
- `modal` — remote sandbox execution
- `pandas` — CSV parsing
- `click` — CLI
- `pydantic-settings` — config from `.env`
- `python-docx`, `reportlab` — report export (used by reporter agent)
- `psycopg`, `pgvector`, `sentence-transformers` — pgvector persistence + local BGE embeddings (core deps)
