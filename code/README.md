# Orchestrate — Multi-Modal Evidence Review Pipeline

Submission entry point for the HackerRank Orchestrate hackathon.

## Quickstart

```bash
cd code
pip install -r requirements.txt
python main.py --output ../output.csv
```

## CLI

```bash
python main.py                      # Run full pipeline on claims.csv
python main.py --eval               # Evaluate on sample_claims.csv
python main.py --model openai/gpt-4o --verbose
python main.py --reset-checkpoint   # Clear cache and re-run all claims
python main.py --dry-run            # Safety gate only, no API calls
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

# Or from repo root:
python -m pytest code/tests/ -q
```

## Provider Setup

Set API keys in `.env` (copy from `.env.example`):

| Provider   | Env Var              | Example Model String                           |
|------------|----------------------|------------------------------------------------|
| Gemini     | `GEMINI_API_KEY`     | `gemini/gemini-2.5-flash-lite`                 |
| OpenAI     | `OPENAI_API_KEY`     | `openai/gpt-4o`                               |
| Anthropic  | `ANTHROPIC_API_KEY`  | `anthropic/claude-sonnet-4-20250514`           |
| Groq       | `GROQ_API_KEY`       | `groq/llama-3.2-90b-vision-preview`           |
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
Loader → Preprocessor → Safety Gate → Vision Analyzer (LiteLLM) → Postprocessor → Validator → output.csv
```

See root `README.md` for full architecture diagram and judge interview prep.
