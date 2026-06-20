# Orchestrate — Multi-Modal Evidence Review Pipeline

Submission entry point for the HackerRank Orchestrate hackathon.

## Quickstart

```bash
cd code
pip install -r requirements.txt
cp .env.example .env   # Add your API key
python main.py --output ../output.csv
```

## CLI

```bash
python main.py                      # Run full pipeline on claims.csv
python main.py --eval               # Evaluate on sample_claims.csv
python main.py --model openai/gpt-4o --verbose
python main.py --reset-checkpoint   # Clear cache and re-run all claims
python main.py --dry-run            # Safety gate only, no API calls
python main.py --skip-checkpoint    # Ignore checkpoint, re-process all
```

## Evaluation

```bash
python evaluation/main.py           # Dual-strategy comparison (A vs B)
python evaluation/main.py --quick   # Baseline only
python evaluation/main.py --output ../eval_report.md
```

## Tests

```bash
pip install -e ".[test]"
pytest tests/ -q --tb=short
ruff check .
```

## Provider Setup

Set API keys in `.env` (copy from `.env.example`):

| Provider   | Env Var              | Example Model String                           |
|------------|----------------------|------------------------------------------------|
| Groq       | `GROQ_API_KEY`       | `groq/qwen/qwen3.6-27b`                       |
| Gemini     | `GEMINI_API_KEY`     | `gemini/gemini-2.5-flash-lite`                 |
| OpenAI     | `OPENAI_API_KEY`     | `openai/gpt-4o`                               |
| Anthropic  | `ANTHROPIC_API_KEY`  | `anthropic/claude-sonnet-4-20250514`           |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/meta-llama/llama-3.2-90b-vision-instruct` |

No code changes needed — just update `.env`.

## Output

`output.csv` — 14 columns, one row per claim:

```
user_id,image_paths,user_claim,claim_object,evidence_standard_met,
evidence_standard_met_reason,risk_flags,issue_type,object_part,
claim_status,claim_status_justification,supporting_image_ids,valid_image,severity
```

## Architecture

```
Input (claims.csv + images)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Loader (load_all)                                  │
│  • claims.csv, user_history.csv, evidence_rules.csv │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Preprocessor (preprocess_claim)                    │
│  • Image validation, path resolution, history lookup│
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Pre-VLM Forensics Scan (image_forensics)           │
│  • OCR text-in-image detection                      │
│  • EXIF metadata forensics                          │
│  • Screenshot/editing software detection            │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Safety Gate (evaluate_safety_gate)                 │
│  • Text injection pattern detection                 │
│  • Trust manipulation detection                     │
│  • Coercion keyword detection                       │
│  • Rubric steering detection                        │
│  • User history risk assessment                     │
│  • Forensics flag integration                       │
│  • BLOCKS if text_instruction_present detected       │
└─────────────────────────────────────────────────────┘
    │
    ▼ (if not blocked)
┌─────────────────────────────────────────────────────┐
│  Vision Analyzer (run_vision_analysis)              │
│  • LiteLLM provider-agnostic VLM calls             │
│  • Adaptive rate limiter (auto-backoff on 429)      │
│  • Response caching                                 │
│  • Hardened system prompt (anti-jailbreak)          │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Postprocessor (apply_claim_decision)               │
│  • Gate + VLM flag merging                          │
│  • Post-VLM jailbreak output detection              │
│  • Claim-object consistency check                   │
│  • Output anomaly detection                         │
│  • Claim status decision logic                      │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Validator (validate_output)                        │
│  • Value clamping (issue_type, object_part, etc.)   │
│  • Boolean formatting (true/false strings)          │
│  • Risk flag auto-promotion                         │
│  • Supporting image ID cleanup                      │
└─────────────────────────────────────────────────────┘
    │
    ▼
  output.csv
```

## Jailbreak Defenses (Layered)

| Layer | Component | What It Catches |
|-------|-----------|-----------------|
| 1 | Pre-VLM OCR scan | Text-in-image injection, sticky notes, watermarks |
| 2 | EXIF forensics | Screenshots, edited images, stripped metadata |
| 3 | Safety gate (text) | Prompt injection, trust manipulation, coercion |
| 4 | Safety gate (rubric) | COMET-style evaluation dashboards |
| 5 | Hardened system prompt | In-image instruction following |
| 6 | VLM analysis | Visual damage assessment |
| 7 | Post-VLM output scan | Jailbreak patterns in VLM response |
| 8 | Object consistency | Car claim vs laptop image mismatch |
| 9 | Output anomalies | High confidence + uncertainty language |

## Key Files

```
code/
├── main.py                          # Pipeline entry point
├── config.py                        # Env config, allowed values
├── requirements.txt                 # Dependencies
├── .env.example                     # API key template
├── prompts/
│   └── system_vision.txt            # Hardened VLM system prompt
├── pipeline/
│   ├── preprocessor.py              # Image validation, path resolution
│   ├── safety_gate.py               # Multi-layer safety checks
│   ├── image_forensics.py           # OCR + EXIF pre-scanning
│   ├── vision_analyzer.py           # VLM calls via LiteLLM
│   ├── postprocessor.py             # Decision logic + output validation
│   ├── validator.py                 # Value clamping + formatting
│   ├── evidence_filter.py           # Evidence rule matching
│   ├── llm_router.py                # LiteLLM provider fallback
│   └── loader.py                    # Data loading
├── utils/
│   ├── cache.py                     # Response caching
│   ├── checkpoint.py                # Processing checkpoint
│   ├── image_utils.py               # Image resize/enhance
│   ├── logger.py                    # Structured logging
│   ├── rate_limiter.py              # Adaptive rate limiter
│   └── token_tracker.py             # Token usage tracking
├── tests/                           # 159 tests
│   ├── test_pipeline.py
│   ├── test_safety_gate.py
│   ├── test_integration.py
│   └── ...
└── evaluation/
    ├── main.py                      # Evaluation runner
    ├── metrics.py                   # Scoring metrics
    └── vlm_jailbreak_research.md    # Security research report
```
