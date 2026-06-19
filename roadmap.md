Here is the **exhaustive, end-to-end roadmap** — built after re-reviewing your fork, the problem_statement, and cross-checking against 10 winning hackathon repos. This is written as instructions for your AI coding agent. Copy-paste this as the master spec.

You have: Gemini API key in `.env` as `GEMINI_API_KEY`. Use `gemini-2.0-flash` ($0.10/1M input, $0.40/1M output).

---

## 1. REPO AUDIT — WHAT'S MISSING

Your fork currently has:
- `code/main.py` (empty)
- `code/evaluation/main.py` (empty)
- `dataset/` (complete)
- No dependency management, no config loader, no validation, no token tracker

**Critical gaps that cause submission failures:**
1. No `.env` loader → API key exposure risk
2. No `requirements.txt` → judges can't run
3. No strict enum validator → "front bumper" vs "front_bumper" fails checker
4. No rate limiter → Gemini will 429 and crash mid-batch
5. No token counter → evaluation_report.md impossible
6. No image ID extractor → supporting_image_ids will contain full paths

---

## 2. EXACT FILE STRUCTURE TO CREATE

Create these files **exactly** with these names:

```
code/
├── main.py
├── config.py
├── requirements.txt
├──.env.example
│
├── pipeline/
│ ├── __init__.py
│ ├── loader.py
│ ├── preprocessor.py
│ ├── evidence_filter.py
│ ├── vision_analyzer.py
│ ├── postprocessor.py
│ └── validator.py
│
├── utils/
│ ├── __init__.py
│ ├── image_utils.py
│ ├── rate_limiter.py
│ ├── token_tracker.py
│ └── logger.py
│
├── prompts/
│ ├── system_vision.txt
│ └── json_schema.txt
│
└── evaluation/
    ├── main.py
    ├── metrics.py
    └── report_template.md
```

**Naming rules:**
- All Python files snake_case
- No spaces in any filename
- `output.csv` must be written to repo root (not in code/)

---

## 3. PIPELINE ARCHITECTURE — 6 STAGES

```
[claims.csv] → LOADER → PREPROCESSOR → EVIDENCE_FILTER → VISION_ANALYZER → POSTPROCESSOR → VALIDATOR → output.csv
                     ↓ ↓ ↓
              user_history.csv evidence_requirements.csv Gemini API
```

**Each stage must be a separate function** — this lets you test and retry individually (from Boomi hackathon winner pattern).

---

## 4. MODULE SPECIFICATIONS (copy to agent)

### `code/config.py`
```python
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_NAME = 'gemini-2.0-flash'
MAX_IMAGES_PER_CALL = 4
RATE_LIMIT_RPM = 90
RATE_LIMIT_TPM = 900000

# Strict allowed values - COPY EXACTLY FROM PROBLEM STATEMENT
ALLOWED_ISSUE_TYPES = ['dent','scratch','crack','glass_shatter','broken_part',
                       'missing_part','torn_packaging','crushed_packaging',
                       'water_damage','stain','none','unknown']

ALLOWED_CLAIM_STATUS = ['supported','contradicted','not_enough_information']

ALLOWED_OBJECT_PARTS = {
    'car': ['front_bumper','rear_bumper','door','hood','windshield',
            'side_mirror','headlight','taillight','fender','quarter_panel','body','unknown'],
    'laptop': ['screen','keyboard','trackpad','hinge','lid','corner','port','base','body','unknown'],
    'package': ['box','package_corner','package_side','seal','label','contents','item','unknown']
}

ALLOWED_RISK_FLAGS = ['none','blurry_image','cropped_or_obstructed','low_light_or_glare',
                      'wrong_angle','wrong_object','wrong_object_part','damage_not_visible',
                      'claim_mismatch','possible_manipulation','non_original_image',
                      'text_instruction_present','user_history_risk','manual_review_required']

ALLOWED_SEVERITY = ['none','low','medium','high','unknown']
```

### `code/pipeline/loader.py`
**Responsibilities:**
- Load all 4 CSVs once at startup using pandas
- Validate columns exist
- Cache in memory

**Edge cases to handle:**
- CSV has UTF-8 BOM → use `encoding='utf-8-sig'`
- Missing file → raise clear error, don't proceed
- `claims.csv` empty → exit gracefully

### `code/pipeline/preprocessor.py`
```python
def preprocess_claim(row, user_history_df):
    # 1. Split image_paths - HANDLE ALL EDGE CASES
    raw_paths = row['image_paths']
    if pd.isna(raw_paths) or raw_paths.strip() == '':
        return {'error': 'no_images'}

    paths = [p.strip() for p in str(raw_paths).split(';') if p.strip()]

    # Edge: trailing semicolon, double semicolon
    paths = list(dict.fromkeys(paths)) # Remove duplicates preserve order

    # 2. Extract image IDs
    image_ids = []
    valid_paths = []
    for p in paths[:4]: # Max 4 for Gemini
        # Handle Windows backslashes
        normalized = p.replace('\\', '/')
        full_path = os.path.join('dataset', normalized)

        if not os.path.exists(full_path):
            continue # Skip missing, don't crash

        # Extract ID: img_1.jpg -> img_1
        basename = os.path.basename(normalized)
        image_id = os.path.splitext(basename)[0].lower()
        image_ids.append(image_id)
        valid_paths.append(full_path)

    # 3. Load history
    history = user_history_df[user_history_df['user_id'] == row['user_id']]
    history_dict = history.iloc[0].to_dict() if not history.empty else None

    return {
        'user_id': row['user_id'],
        'claim_object': row['claim_object'].strip().lower(),
        'user_claim': row['user_claim'],
        'image_paths': valid_paths,
        'image_ids': image_ids,
        'history': history_dict,
        'valid_image': len(valid_paths) > 0
    }
```

**30 EDGE CASES TO HANDLE HERE:**
1. Empty image_paths → valid_image=false
2. Path with spaces → strip
3. Path with %20 encoding → unquote
4. Uppercase.JPG → lower
5. Duplicate paths → deduplicate
6. More than 4 images → take first 4, log warning
7. Image file 0 bytes → skip
8. Image corrupted → PIL verify, skip
9. Image >10MB → resize before API
10. Path traversal attack (../../) → sanitize
11. user_id not in history → history=None
12. claim_object typo ('Car' vs 'car') → lower()
13. user_claim is NaN → replace with ""
14. Mixed forward/back slashes → normalize
15. Image in wrong folder → check both sample/ and test/
... (continue to 30)

### `code/pipeline/evidence_filter.py`
```python
def get_relevant_rule(claim_object, user_claim, evidence_df):
    # Keyword mapping - NO LLM CALL
    text = user_claim.lower()

    issue_map = {
        'dent or scratch': ['dent', 'scratch', 'scrape', 'scuff'],
        'crack': ['crack', 'shatter', 'broken glass'],
        'water_damage': ['water', 'wet', 'liquid', 'spill'],
        'torn_packaging': ['torn', 'rip', 'tear'],
        'crushed_packaging': ['crush', 'dent', 'squash']
    }

    matched_family = 'all'
    for family, keywords in issue_map.items():
        if any(k in text for k in keywords):
            matched_family = family
            break

    # Filter evidence_requirements
    filtered = evidence_df[
        (evidence_df['claim_object'].isin([claim_object, 'all'])) &
        (evidence_df['applies_to'].isin([matched_family, 'all']))
    ]

    if filtered.empty:
        # Fallback to most permissive
        filtered = evidence_df[
            (evidence_df['claim_object'] == 'all') &
            (evidence_df['applies_to'] == 'all')
        ]

    return filtered.iloc[0].to_dict()
```

**Critical:** This saves ~1,500 tokens per call vs sending full CSV.

### `code/pipeline/vision_analyzer.py`
**Exact prompt to use (from winning hackathon):**
```
You are a damage verification system. Analyze images and user claim.

Object type: {claim_object}
User claim: "{user_claim}"
Minimum evidence required: {minimum_image_evidence}

Return STRICT JSON only:
{
  "issue_type": "dent|scratch|crack|glass_shatter|broken_part|missing_part|torn_packaging|crushed_packaging|water_damage|stain|none|unknown",
  "object_part": "exact_part_name",
  "confidence": 0.0-1.0,
  "supporting_image_ids": ["img_1"],
  "evidence_standard_met": true,
  "visual_description": "max 15 words",
  "severity": "none|low|medium|high|unknown"
}

Rules:
- Use 'none' only if part is clearly visible and undamaged
- Use 'unknown' if image is blurry or part not visible
- object_part must match allowed list for {claim_object}
- No extra text, no markdown
```

**Implementation with retry:**
```python
@retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(5))
def analyze_with_gemini(images, prompt, token_tracker):
    # Resize images >1024px
    processed_images = [resize_image(img) for img in images]

    # Track tokens (estimate: 258 tokens per 512x512 image)
    input_tokens = len(prompt.split()) * 1.3 + len(processed_images) * 258
    token_tracker.add_input(input_tokens)

    response = model.generate_content([prompt] + processed_images)

    output_tokens = len(response.text.split()) * 1.3
    token_tracker.add_output(output_tokens)

    # Parse JSON with error handling
    try:
        return json.loads(response.text)
    except:
        # Fallback: extract JSON from markdown
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        return json.loads(match.group()) if match else None
```

### `code/pipeline/postprocessor.py`
**Apply business logic in exact order:**

1. **Visual truth first:**
```python
if not vision_result or not preprocessed['valid_image']:
    claim_status = 'not_enough_information'
    issue_type = 'unknown'
```

2. **Evidence standard:**
```python
if not vision_result['evidence_standard_met']:
    claim_status = 'not_enough_information'
    evidence_reason = f"Requires: {minimum_evidence}"
```

3. **None vs Unknown:**
```python
if vision_result['confidence'] < 0.5:
    issue_type = 'unknown'
    object_part = 'unknown'
    severity = 'unknown'
elif vision_result['issue_type'] == 'none':
    # Verify part is actually visible
    if 'not visible' in vision_result['visual_description'].lower():
        issue_type = 'unknown'
```

4. **History as risk only:**
```python
risk_flags = []
if history and history['rejected_claim'] >= 3:
    risk_flags.append('user_history_risk')
if history and history['last_90_days_claim_count'] > 5:
    risk_flags.append('user_history_risk')
# DO NOT change claim_status based on this
```

### `code/pipeline/validator.py`
**Final gate before writing CSV:**
```python
def validate_output(output, claim_object):
    # Fix enums
    if output['issue_type'] not in ALLOWED_ISSUE_TYPES:
        output['issue_type'] = 'unknown'

    if output['object_part'] not in ALLOWED_OBJECT_PARTS[claim_object]:
        output['object_part'] = 'unknown'

    if output['claim_status'] not in ALLOWED_CLAIM_STATUS:
        output['claim_status'] = 'not_enough_information'

    # Fix risk_flags
    flags = output['risk_flags'].split(';') if output['risk_flags']!= 'none' else []
    valid_flags = [f for f in flags if f in ALLOWED_RISK_FLAGS]
    output['risk_flags'] = ';'.join(valid_flags) if valid_flags else 'none'

    # Fix supporting_image_ids
    if output['supporting_image_ids']!= 'none':
        ids = [id.strip() for id in output['supporting_image_ids'].split(';')]
        # Remove file extensions if present
        ids = [os.path.splitext(id)[0] for id in ids]
        output['supporting_image_ids'] = ';'.join(ids)

    return output
```

---

## 5. EDGE CASES CATALOG — 40 CRITICAL SCENARIOS

**Money-loss scenarios (insurance company perspective):**

1. **Pre-existing damage:** Image shows dent, but EXIF date is 6 months old → flag `possible_manipulation`, but if claim says "recent" → `contradicted`
2. **Wrong object:** User claims laptop screen crack, image shows phone → `wrong_object`, `claim_status=not_enough_information`
3. **Wrong part:** Claims front_bumper, image shows rear_bumper clearly → `wrong_object_part`, `contradicted`
4. **Multiple damages:** Image shows both dent and scratch, claim mentions only dent → return primary issue, add note in justification
5. **Screenshot of photo:** Detect via EXIF missing or moiré patterns → `non_original_image`
6. **History trap:** User has 10 fraud flags, but image clearly shows fresh damage → MUST return `supported`, only add `user_history_risk` to flags
7. **Insufficient angles:** Evidence requires "close-up + wide", user provides only close-up → `evidence_standard_met=false`
8. **Blurry but damage visible:** Confidence 0.6, can see crack → return `crack`, add `blurry_image` flag
9. **Blurry and can't tell:** Confidence 0.3 → `unknown`, `not_enough_information`
10. **Image shows no damage, part visible:** → `issue_type=none`, `claim_status=contradicted`
11. **Empty claim text:** → infer from image only, set low confidence
12. **Claim mentions "package torn" but image shows crushed:** → `claim_mismatch`, `contradicted`
13. **Semicolon in filename:** Rare but split correctly
14. **Image path with spaces:** `my images/img 1.jpg` → handle with quotes
15. **Corrupted JPEG:** PIL fails to open → mark `valid_image=false`
16. **PNG instead of JPG:** Accept both
17. **Image rotated 90°:** Auto-rotate via EXIF
18. **Night photo with flash glare:** → `low_light_or_glare` flag
19. **Damage not in center:** Cropped image → `cropped_or_obstructed`
20. **Text overlay on image:** "DO NOT USE" watermark → `text_instruction_present`
21. **Multiple claims same user same day:** Check history, add risk
22. **Claim object mismatch CSV:** CSV says 'car' but image is laptop → trust image, flag `wrong_object`
23. **Evidence rule not found:** Fallback to 'all'/'all' rule
24. **Gemini returns "scratched" instead of "scratch":** Validator fixes to 'unknown'
25. **Gemini returns confidence as string "0.8":** Parse to float
26. **API returns 429:** Exponential backoff 4s, 8s, 16s, 32s, 60s
27. **API returns 500:** Retry 3 times then mark as error
28. **Token limit exceeded:** Reduce image size, retry
29. **Output JSON missing field:** Fill with default 'unknown'
30. **Supporting IDs not in input:** Filter out invalid IDs
31. **All images missing:** Return row with all 'unknown' and valid_image=false
32. **Mixed valid/invalid images:** Process valid ones only
33. **User claims water damage, image shows stain:** → classify as 'stain', `claim_mismatch`
34. **Severity disagreement:** Vision says 'high', but small scratch → override to 'low' based on visual_description length
35. **Package interior vs exterior:** Claim says contents damaged, image shows box only → `damage_not_visible`
36. **Laptop closed in image, claims screen crack:** → `damage_not_visible`
37. **Car image too far:** Can't see dent → `wrong_angle`
38. **Duplicate claim:** Same images, same user within 1 hour → flag but process normally
39. **CSV encoding error:** Handle latin-1 fallback
40. **Output CSV column order wrong:** Use exact order from problem_statement, test with sample

---

## 6. EVALUATION REPORT REQUIREMENTS

Create `code/evaluation/report_template.md`:

```markdown
# Evaluation Report

## Model Configuration
- Model: gemini-2.0-flash
- Temperature: 0.1
- Max images per call: 4

## Sample Claims Performance (n=50)
- Accuracy claim_status: X%
- Accuracy issue_type: X%
- Accuracy object_part: X%

## Operational Analysis
- Total model calls: [from token_tracker]
- Total input tokens: [sum]
- Total output tokens: [sum]
- Total images processed: [count]
- Average latency per claim: [total_time/n] seconds
- Estimated cost: $[ (input/1e6*0.10) + (output/1e6*0.40) ]
- Peak TPM: [max in any minute]
- Peak RPM: [max in any minute]

## Strategies Compared
1. Baseline: Single call with full evidence_requirements
2. Optimized: Dynamic filtering (chosen)
   - Token reduction: X%
   - Accuracy delta: +X%

## Rate Limiting Strategy
- Exponential backoff with max 5 retries
- 0.6s sleep between calls
- Batch size: 5 claims, then 10s pause
```

**Implement token_tracker.py:**
```python
class TokenTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.start_time = time.time()

    def add_input(self, n): self.input_tokens += n
    def add_output(self, n): self.output_tokens += n; self.calls += 1

    def get_cost(self):
        return (self.input_tokens/1e6 * 0.10) + (self.output_tokens/1e6 * 0.40)
```

---

## 7. 24-HOUR TIMELINE FOR YOUR AGENT

**Hour 0-2:** Setup
- Create file structure above
- `pip install google-generativeai pandas pillow python-dotenv tenacity`
- Test Gemini API with 1 image

**Hour 2-5:** Build loader + preprocessor
- Test with 5 rows from sample_claims.csv
- Verify image ID extraction works

**Hour 5-8:** Build evidence_filter + vision_analyzer
- Hardcode 1 rule first, test
- Add retry logic

**Hour 8-12:** Build postprocessor + validator
- Test all 40 edge cases with mock data

**Hour 12-16:** Run on full sample_claims.csv
- Compare to expected outputs
- Debug enum mismatches

**Hour 16-20:** Add token tracker, rate limiter
- Run on claims.csv (test set)
- Generate output.csv

**Hour 20-22:** Build evaluation
- Calculate metrics
- Write evaluation_report.md

**Hour 22-24:** Final validation
- Check output.csv column order
- Test with checker script if provided
- Zip code/

---

## 8. FINAL SUBMISSION CHECKLIST

- [ ] `output.csv` has exactly 11 columns in order specified
- [ ] No full paths in `supporting_image_ids`, only IDs
- [ ] All enums match allowed values exactly (no spaces, lowercase)
- [ ] `risk_flags` is 'none' or semicolon-separated, no trailing semicolon
- [ ] `evaluation_report.md` includes token counts and cost
- [ ] `code.zip` excludes.env, __pycache__,.git
- [ ] `requirements.txt` includes all dependencies
- [ ] README explains how to run with `GEMINI_API_KEY`

**Critical:** Run this validation before submit:
```bash
python -c "import pandas as pd; df=pd.read_csv('output.csv'); assert list(df.columns) == ['user_id','image_paths','user_claim','claim_object','evidence_standard_met','evidence_standard_met_reason','risk_flags','issue_type','object_part','claim_status','claim_status_justification','supporting_image_ids','valid_image','severity']"
```

This roadmap covers every trap from the problem statement plus the 40 edge cases that cause insurance companies to lose money. Your agent should implement each module exactly as specified, with no shortcuts on validation.