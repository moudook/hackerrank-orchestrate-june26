# Multi-Modal Evidence Review Pipeline

A production-grade system for the **HackerRank Orchestrate** hackathon that verifies visual evidence for damage claims across **cars**, **laptops**, and **packages** using multi-modal LLMs. Supports any provider (Gemini, OpenAI, Anthropic, etc.) via LiteLLM.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Overview                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  claims.csv ──► Loader ──► Preprocessor ──► Evidence Filter     │
│                      │                │              │           │
│                      ▼                ▼              ▼           │
│                 [Safety Gate]   [Image QC]     [Rule Match]      │
│                  (fraud/tamper    (corrupt/      (determine       │
│                   detection)      tiny/large)    min evidence)    │
│                      │                │              │           │
│                      └────────────────┴──────────────┘           │
│                                       │                          │
│                                       ▼                          │
│                          Vision Analyzer                         │
│                     ┌─────────────────────────┐                  │
│                     │  LiteLLM Router         │                  │
│                     │  │                      │                  │
│                     │  ├─ gemini/* (primary)   │                  │
│                     │  ├─ openai/* (fallback)  │                  │
│                     │  └─ anthropic/* (fallbk) │                  │
│                     │  Response Cache (disk)   │                  │
│                     │  Structured JSON output  │                  │
│                     └─────────────────────────┘                  │
│                                       │                          │
│                                       ▼                          │
│  Postprocessor ──► Validator ──► output.csv                      │
│  (risk flags,     (enum check,   (14 columns, 1 row/claim)       │
│   trust detect,    flag dedup,                                    │
│   decision logic)  ID cleanup)                                    │
│                                                                  │
│  ┌─────────────────────────────┐                                 │
│  │  Evaluation (sample data)   │                                 │
│  │  Strategy A vs B comparison │                                 │
│  │  precision/recall/F1        │                                 │
│  └─────────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Provider-agnostic** — Switch between Gemini, OpenAI, Anthropic, xAI, Groq, etc. via `.env` vars
- **Automatic fallback chain** — If primary provider fails (429/5xx), tries fallback providers
- **Safety-first** — Trust manipulation detection (7 patterns), risk flag auto-promotion, pre-VLM safety gate
- **Structured JSON output** — Enforced via LiteLLM's response schema, parsed with fallback
- **Response caching** — SHA256-keyed disk+memory cache avoids redundant API calls
- **Checkpoint/resume** — Survives crashes, saves progress every 10 claims
- **Token bucket rate limiter** — Tracks real RPM/TPM with dynamic wait
- **Structured logging** — JSON format with request IDs and per-stage timing
- **64 unit/integration tests** — 0 ruff errors, deterministic where possible

## Quickstart

```bash
# 1. Clone and enter
git clone git@github.com:interviewstreet/hackerrank-orchestrate-june26.git
cd hackerrank-orchestrate-june26

# 2. Install dependencies
cd code
pip install -r requirements.txt
cd ..

# 3. Set your API key (copy .env.example and edit)
cp code/.env.example code/.env
# Edit code/.env with your LLM_API_KEY

# 4. Run the full pipeline
cd code && python main.py --output ../output.csv

# 5. Run evaluation on sample data
python code/evaluation/main.py

# 6. Run demo (end-to-end)
python demo/run_demo.py
```

## Environment Variables

All configurable via `code/.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | Primary provider (`gemini`, `openai`, `anthropic`, etc.) |
| `LLM_MODEL` | `gemini/gemini-2.0-flash` | Full LiteLLM model string for text |
| `VISION_MODEL` | `gemini/gemini-2.0-flash` | Full LiteLLM model string for vision |
| `LLM_API_KEY` | — | API key for primary provider |
| `LLM_FALLBACK_CHAIN` | — | Comma-separated fallback models (e.g. `openai/gpt-4o,anthropic/claude-sonnet-4-20250514`) |
| `MAX_IMAGES_PER_CALL` | `4` | Max images per claim |
| `RATE_LIMIT_RPM` | `2000` | Requests per minute cap |
| `RATE_LIMIT_TPM` | `4000000` | Tokens per minute cap |
| `CACHE_ENABLED` | `true` | Enable response caching |
| `JSON_LOGGING` | `false` | Enable JSON-structured log output |

## Adding a New Provider

Set the appropriate env var. LiteLLM reads per-provider keys automatically:

| Provider | Env Var | Example Model String |
|---|---|---|
| Gemini | `GEMINI_API_KEY` or `LLM_API_KEY` | `gemini/gemini-2.0-flash` |
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4-20250514` |
| xAI | `XAI_API_KEY` | `xai/grok-2-vision` |
| Groq | `GROQ_API_KEY` | `groq/llama-3.2-90b-vision` |

No code changes needed. Just update `.env` and run.

## Project Structure

```text
.
├── AGENTS.md                # AI coding tool rules + transcript logging
├── problem_statement.md     # Full task spec and I/O schema
├── README.md                # You are here
├── pyproject.toml           # Ruff + pytest config
├── demo/
│   └── run_demo.py          # End-to-end demo script
├── code/
│   ├── main.py              # Production pipeline (CLI with argparse)
│   ├── config.py            # Central config (env vars, schema, validation)
│   ├── requirements.txt     # Python dependencies
│   ├── .env.example         # Template for environment variables
│   ├── pipeline/
│   │   ├── loader.py        # CSV loading with multi-encoding fallback
│   │   ├── preprocessor.py  # Path normalization, image QC, history lookup
│   │   ├── evidence_filter.py # Keyword-to-issue mapping (multi-language)
│   │   ├── vision_analyzer.py # VLM caller with cache + fallback
│   │   ├── llm_router.py    # LiteLLM abstraction + fallback chain
│   │   ├── postprocessor.py # Risk flags, trust detection, decision logic
│   │   └── validator.py     # Enum cleanup, flag dedup, ID formatting
│   ├── utils/
│   │   ├── logger.py        # JSON formatter, request IDs, stage decorator
│   │   ├── cache.py         # SHA256-keyed disk+memory cache
│   │   ├── checkpoint.py    # Crash recovery, periodic save
│   │   ├── rate_limiter.py  # Token bucket with RPM/TPM tracking
│   │   ├── token_tracker.py # Token counting + cost estimation
│   │   └── image_utils.py   # Image resizing for API limits
│   ├── evaluation/
│   │   ├── main.py          # Dual-strategy comparison (A vs B)
│   │   ├── metrics.py       # Accuracy, precision/recall/F1
│   │   └── report_template.md # Fillable evaluation report
│   ├── prompts/
│   │   ├── system_vision.txt # System prompt for VLM
│   │   └── json_schema.txt  # Response JSON schema reference
│   └── tests/
│       ├── test_pipeline.py # 20 unit tests (preprocessor, filter, postprocessor)
│       ├── test_llm_router.py # 14 tests (router, fallback, JSON extraction)
│       ├── test_logger.py   # 14 tests (JSON format, request IDs, stage decorator)
│       ├── test_integration.py # 10 integration tests (real fixtures)
│       └── fixtures/        # 5 generated test images
└── dataset/
    ├── claims.csv           # 44 test claims for final output
    ├── sample_claims.csv    # 20 claims with ground truth for evaluation
    ├── user_history.csv     # Historical claim data
    ├── evidence_requirements.csv # Minimum image evidence rules
    └── images/
        ├── sample/          # Sample claim images
        └── test/            # Test claim images
```

## Running Tests

```bash
# From repo root
python -m pytest code/tests/ -v

# With coverage style
python -m pytest code/tests/ -q --tb=short

# Lint check
pip install ruff
ruff check code/
```

## CLI Usage

```bash
# Basic run
cd code && python main.py

# With options
python main.py --model openai/gpt-4o --output ../output.csv --verbose
python main.py --reset-checkpoint
```

## Submission

1. **Code zip**: `cd code && python -m zipfile -c ../code.zip . -x ".env" "__pycache__/*" ".pytest_cache/*" "*.pyc"`
2. **Predictions CSV**: `output.csv` (generated by `python main.py --output ../output.csv`)
3. **Chat transcript**: `C:\Users\<you>\hackerrank_orchestrate\log.txt` (auto-logged per AGENTS.md)

## Judge Interview Prep

Be ready to discuss:
- Why LiteLLM instead of a single provider
- How the fallback chain handles provider failures
- Trust manipulation detection patterns and their limitations
- Trade-off between structured JSON output vs free-form
- Determinism strategies (seeded RNG, cache, fallback)
- Why not to use VLM for every claim (safety gate + rule layer)
- Evaluation methodology (Strategy A vs B, per-class metrics)
