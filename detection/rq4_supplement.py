"""
RQ4 Supplement Experiments (Section 6 of action.md)

1. Multi-LLM validation: Claude (claude-sonnet-4-20250514) on A/E/C groups
2. BM25 baseline: Traditional IR for function-level localization
3. Top-3 for E/F1/F2 (补齐 Hit@1/Hit@3)
4. RC score preserved
5. Unified v4 patterns + strict matching throughout

Usage:
  python rq4_supplement.py pilot          # 10-bug pilot for Claude A/E/C
  python rq4_supplement.py claude_full    # 207-bug full Claude run
  python rq4_supplement.py bm25          # BM25 baseline (207 bugs)
  python rq4_supplement.py top3_eff      # Top-3 for E/F1/F2
  python rq4_supplement.py analyze       # Unified analysis
"""

import os, sys, json, time, re, hashlib, math
import pandas as pd
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GPT_TOKEN = os.environ["OPENAI_API_KEY"]
claude_token = os.environ.get("ANTHROPIC_API_KEY", "")

from svl_common import save_result, load_done_keys
from rq4_fullfile import (
    get_full_buggy_files, format_full_files, extract_gold_functions,
    match_location, compute_4level, parse_response,
    OUTPUT_FORMAT, load_oracle_pattern
)
from rq4_new_experiments import (
    load_all_patterns_pool, sample_flat_patterns, get_c_pattern_size,
    load_structured_patterns, load_cleaned_reports,
    build_prompt_e, build_prompt_f1, build_prompt_f2
)
from rq4_fullfile import build_prompt_a, build_prompt_c

PATTERN_DIR = 'aggregated_patterns_v4'


# ── LLM callers ──────────────────────────────────────────────────

def call_claude(prompt, model="claude-sonnet-4-20250514"):
    """Call Claude API via direct HTTP (no anthropic SDK needed)."""
    resp = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={
            'x-api-key': claude_token,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        },
        json={
            'model': model,
            'max_tokens': 1024,
            'temperature': 0,
            'messages': [{'role': 'user', 'content': prompt}],
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise Exception(f"Claude API error {resp.status_code}: {resp.text[:300]}")
    return resp.json()['content'][0]['text']


def call_gpt(prompt, model="gpt-4o-mini"):
    """Call GPT via OpenAI SDK."""
    from openai import OpenAI
    client = OpenAI(api_key=GPT_TOKEN)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── Strict matching (from rq4_fullfile_top3.py) ──────────────────

def match_location_strict(predicted_loc, gold_functions):
    """Strict matching: empty gold_funcs -> (True, False), not (True, True)."""
    if not predicted_loc or not gold_functions:
        return False, False

    pred_loc = predicted_loc.strip()
    pred_file = ''
    pred_func = ''
    if ':' in pred_loc:
        parts = pred_loc.rsplit(':', 1)
        pred_file = parts[0].strip()
        pred_func = parts[1].strip()
    else:
        pred_file = pred_loc

    pred_basename = os.path.basename(pred_file).lower()

    file_match = False
    matched_file = None
    for gold_file in gold_functions:
        gold_basename = os.path.basename(gold_file).lower()
        if pred_basename == gold_basename:
            file_match = True
            matched_file = gold_file
            break
        if pred_file.lower().endswith(gold_file.lower()) or gold_file.lower().endswith(pred_file.lower()):
            file_match = True
            matched_file = gold_file
            break

    if not file_match:
        return False, False

    gold_funcs = gold_functions.get(matched_file, set())
    if not gold_funcs:
        # STRICT: can't confirm function match → False
        return True, False

    pred_func_clean = re.sub(r'[\d\-:()]+', '', pred_func).strip()
    if not pred_func_clean:
        return True, False

    for gf in gold_funcs:
        if pred_func_clean.lower() == gf.lower():
            return True, True
        if pred_func_clean.lower() in gf.lower() or gf.lower() in pred_func_clean.lower():
            return True, True

    return True, False


# ── Top-3 format ─────────────────────────────────────────────────

OUTPUT_FORMAT_TOP3 = """Output exactly this format. Provide your top 3 most likely bug locations, ranked from most to least likely:

BUG_LOCATION_1: [file_path:function_name]
ROOT_CAUSE_1: [1-2 sentences: what is wrong and why]
CONFIDENCE_1: [HIGH / MEDIUM / LOW]

BUG_LOCATION_2: [file_path:function_name]
ROOT_CAUSE_2: [1-2 sentences: what is wrong and why]
CONFIDENCE_2: [HIGH / MEDIUM / LOW]

BUG_LOCATION_3: [file_path:function_name]
ROOT_CAUSE_3: [1-2 sentences: what is wrong and why]
CONFIDENCE_3: [HIGH / MEDIUM / LOW]"""


def parse_response_top3(response):
    """Parse top-3 response into list of predictions."""
    preds = []
    for k in range(1, 4):
        loc = ''
        rc = ''
        conf = ''
        for line in response.split('\n'):
            line = line.strip()
            upper = line.upper()
            if upper.startswith(f'BUG_LOCATION_{k}:'):
                loc = line.split(':', 1)[1].strip() if ':' in line else ''
            elif upper.startswith(f'ROOT_CAUSE_{k}:'):
                rc = line.split(':', 1)[1].strip() if ':' in line else ''
            elif upper.startswith(f'CONFIDENCE_{k}:'):
                conf = line.split(':', 1)[1].strip() if ':' in line else ''
        if loc:
            preds.append({'bug_location': loc, 'root_cause': rc, 'confidence': conf})
    # Fallback: try single format
    if not preds:
        p = parse_response(response)
        if p['bug_location']:
            preds.append(p)
    return preds


def compute_hit_at_k(predictions, gold_funcs, k):
    """Any of top-k predictions hit a gold function? Strict matching."""
    for pred in predictions[:k]:
        _, func_match = match_location_strict(pred['bug_location'], gold_funcs)
        if func_match:
            return True
    return False


def compute_first_rank(predictions, gold_funcs):
    """First rank that hits gold (1-based). 4 if no hit in top-3."""
    for i, pred in enumerate(predictions[:3]):
        _, func_match = match_location_strict(pred['bug_location'], gold_funcs)
        if func_match:
            return i + 1
    return 4


# ── Top-3 prompt builders ────────────────────────────────────────

def build_prompt_a_top3(files_content):
    return f"""You are a code reviewer for a VR/XR project. This code contains a known bug.

## Source Code
{files_content}

## Task
Find the VR/XR bug in this code. Identify the specific file and function where the bug is,
and explain what is wrong.

{OUTPUT_FORMAT_TOP3}
"""

def build_prompt_c_top3(files_content, patterns, symptom, layer):
    hints = ""
    for i, p in enumerate(patterns, 1):
        hints += f"{i}. {p}\n\n"
    return f"""You are a VR/XR bug detection expert. This code contains a known bug.

## VR Bug Patterns ({symptom} category, {layer} layer)
The following are specific bug patterns for {symptom}-type bugs at the {layer} layer:

{hints}

## Source Code
{files_content}

## Task
Check this code against the patterns above. Identify which pattern is violated,
the specific file and function, and explain what is wrong.

{OUTPUT_FORMAT_TOP3}
"""

def build_prompt_e_top3(files_content, flat_patterns):
    return f"""You are a VR/XR bug detection expert. This code contains a known bug.

## VR Bug Patterns (General Reference)
The following are known VR/XR bug patterns collected from real projects:

{flat_patterns}

## Source Code
{files_content}

## Task
Check this code against the patterns above. Identify which pattern is violated,
the specific file and function, and explain what is wrong.

{OUTPUT_FORMAT_TOP3}
"""

def build_prompt_f1_top3(files_content, cleaned_report):
    return f"""You are a code reviewer for a VR/XR project. This code contains a known bug.

## Bug Report (Symptom)
{cleaned_report}

## Source Code
{files_content}

## Task
Based on the bug report above, find the bug in this code. Identify the specific file and function
where the bug is, and explain the root cause.

{OUTPUT_FORMAT_TOP3}
"""

def build_prompt_f2_top3(files_content, cleaned_report, patterns, symptom, layer):
    hints = ""
    for i, p in enumerate(patterns, 1):
        hints += f"{i}. {p}\n\n"
    return f"""You are a VR/XR bug detection expert. This code contains a known bug.

## Bug Report (Symptom)
{cleaned_report}

## VR Bug Patterns ({symptom} category, {layer} layer)
The following are specific bug patterns for {symptom}-type bugs at the {layer} layer.
Use these patterns to guide your analysis of the code based on the symptom described above.

{hints}

## Source Code
{files_content}

## Task
Based on the symptom in the bug report and the patterns above, find the bug in this code.
Identify which pattern is violated, the specific file and function, and explain the root cause.

{OUTPUT_FORMAT_TOP3}
"""


# ── Data loaders ─────────────────────────────────────────────────

def load_common_data():
    """Load test set, dimension map, gold root causes."""
    test_df = pd.read_csv('test_set.csv')
    all_dim = pd.read_csv('ALL_DIMENSION.csv')

    dim_map = {}
    for _, row in all_dim.iterrows():
        dim_map[(row['repo'], int(row['id']))] = {
            'symptom': row['symptom_cate'],
            'layer': row['entity_layer'],
        }

    gold_rc_map = {}
    with open('ALL_summaries.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            gold_rc_map[(r['repo'], r['id'])] = r.get('root_cause', '')

    # Use existing results to get the 207 valid bugs
    existing = []
    with open('rq4_fullfile_results.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                existing.append(json.loads(line))

    return test_df, dim_map, gold_rc_map, existing


def get_bug_seed(repo, bug_id):
    """Deterministic seed for per-bug sampling."""
    h = hashlib.md5(f"{repo}#{bug_id}".encode()).hexdigest()
    return int(h, 16) % (2**31)


# ── BM25 Baseline ────────────────────────────────────────────────

def bm25_score(query_terms, doc_terms, df_map, n_docs, k1=1.5, b=0.75, avgdl=100):
    """Compute BM25 score for a single document."""
    score = 0.0
    dl = len(doc_terms)
    for qt in query_terms:
        if qt not in doc_terms:
            continue
        tf = doc_terms.count(qt)
        df = df_map.get(qt, 1)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        score += idf * tf_norm
    return score


def tokenize(text):
    """Simple tokenizer for BM25."""
    return re.findall(r'[a-zA-Z_]\w{2,}', text.lower())


def run_bm25_baseline(existing, dim_map, gold_rc_map, output_file):
    """BM25: for each bug, rank functions in the buggy files by similarity to patterns."""
    done_keys = load_done_keys(output_file)
    print(f"BM25 baseline: {len(existing)} bugs, {len(done_keys)} already done")

    # Load all patterns as query corpus
    all_patterns = {}  # {(sym, layer): pattern_text}
    for fname in sorted(os.listdir(PATTERN_DIR)):
        if not fname.endswith('_patterns.json'):
            continue
        sym_layer = fname.replace('_patterns.json', '')
        parts = sym_layer.split('_')
        sym, layer = parts[0], parts[1]
        path = os.path.join(PATTERN_DIR, fname)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text = ""
        for key, val in data.items():
            text += f"{key}\n{val['summary']}\n\n"
        all_patterns[(sym, layer)] = text

    count = 0
    for ex in existing:
        repo = ex['repo']
        bug_id = ex['issue_id']
        if (repo, bug_id) in done_keys:
            continue

        # Get buggy files
        files = get_full_buggy_files(repo, bug_id)
        if not files:
            continue

        gold_funcs = extract_gold_functions(repo, bug_id)
        gold_rc = gold_rc_map.get((repo, bug_id), '')

        info = dim_map.get((repo, bug_id))
        if not info:
            continue
        symptom, layer = info['symptom'], info['layer']

        # Get matched pattern as query
        pattern_text = all_patterns.get((symptom, layer), '')
        if not pattern_text:
            # Fallback: use all patterns
            pattern_text = ' '.join(all_patterns.values())

        query_tokens = tokenize(pattern_text)

        # Extract functions from buggy files and score each
        func_scores = []  # [(file, func_name, score, body_snippet)]
        for fname, content in files.items():
            # Simple function extraction: split by function-like patterns
            # Find function boundaries
            func_pattern = r'(?:function\s+(\w+)|def\s+(\w+)|(?:public|private|protected|static|void|int|float|bool|string|IEnumerator|Task|override|virtual|async)\s+\w+\s+(\w+)\s*\(|(\w+)\s*[=:]\s*function\s*\(|fn\s+(\w+)\s*[<(]|func\s+(\w+)\s*\()'
            matches = list(re.finditer(func_pattern, content))

            if not matches:
                # Treat whole file as one "function"
                doc_tokens = tokenize(content[:5000])
                # Build simple df
                df_map = {}
                for t in set(doc_tokens):
                    df_map[t] = 1
                score = bm25_score(query_tokens, doc_tokens, df_map, 1)
                func_scores.append((fname, '(file-level)', score, content[:200]))
                continue

            for idx, m in enumerate(matches):
                func_name = next(g for g in m.groups() if g)
                start = m.start()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else min(start + 3000, len(content))
                body = content[start:end]
                doc_tokens = tokenize(body)
                df_map = {}
                for t in set(doc_tokens):
                    df_map[t] = 1
                score = bm25_score(query_tokens, doc_tokens, df_map, max(len(matches), 1))
                func_scores.append((fname, func_name, score, body[:200]))

        # Sort by score descending
        func_scores.sort(key=lambda x: -x[2])

        # Take top-1 prediction
        if func_scores:
            top = func_scores[0]
            pred_loc = f"{top[0]}:{top[1]}"
        else:
            pred_loc = ''

        # Match
        file_m, func_m = match_location_strict(pred_loc, gold_funcs)

        # BM25 has no root cause prediction — score RC as 0
        rc_score = 0
        level = compute_4level(rc_score, file_m, func_m)

        # Top-3 hit
        hit1 = False
        hit3 = False
        mfr = 4
        for i, fs in enumerate(func_scores[:3]):
            ploc = f"{fs[0]}:{fs[1]}"
            _, fm = match_location_strict(ploc, gold_funcs)
            if fm and not hit1 and i == 0:
                hit1 = True
            if fm and mfr == 4:
                mfr = i + 1
                hit3 = True

        result = {
            'repo': repo, 'issue_id': bug_id,
            'symptom': symptom, 'layer': layer,
            'bm25_location': pred_loc,
            'bm25_score': func_scores[0][2] if func_scores else 0,
            'bm25_file_match': file_m, 'bm25_func_match': func_m,
            'bm25_rc_score': 0,
            'bm25_level': level,
            'bm25_hit1': hit1, 'bm25_hit3': hit3, 'bm25_mfr': mfr,
            'bm25_top3': [f"{fs[0]}:{fs[1]}" for fs in func_scores[:3]],
        }
        save_result(output_file, result)
        count += 1
        if count % 20 == 0:
            print(f"  BM25: {count} done")

    print(f"BM25 done: {count} new results")


# ── Claude Multi-LLM Experiment ──────────────────────────────────

def run_claude_experiment(existing, dim_map, gold_rc_map, output_file, limit=None):
    """Run A/E/C groups on Claude for multi-LLM validation."""
    pool = load_all_patterns_pool()
    done_keys = load_done_keys(output_file)
    bugs = existing[:limit] if limit else existing
    print(f"Claude experiment: {len(bugs)} bugs, {len(done_keys)} already done")

    count = 0
    for ex in bugs:
        repo = ex['repo']
        bug_id = ex['issue_id']
        if (repo, bug_id) in done_keys:
            continue

        files = get_full_buggy_files(repo, bug_id)
        if not files:
            print(f"  SKIP {repo}#{bug_id}: no files")
            continue

        files_content = format_full_files(files)
        gold_funcs = extract_gold_functions(repo, bug_id)
        gold_rc = gold_rc_map.get((repo, bug_id), '')

        info = dim_map.get((repo, bug_id))
        if not info:
            continue
        symptom, layer = info['symptom'], info['layer']

        result = {'repo': repo, 'issue_id': bug_id, 'symptom': symptom, 'layer': layer}

        try:
            # Group A: bare Claude
            prompt_a = build_prompt_a(files_content)
            resp_a = call_claude(prompt_a)
            parsed_a = parse_response(resp_a)
            fm_a, fnm_a = match_location_strict(parsed_a['bug_location'], gold_funcs)
            rc_a = 0  # Root-cause scoring done manually
            result.update({
                'claude_a_location': parsed_a['bug_location'],
                'claude_a_root_cause': parsed_a['root_cause'],
                'claude_a_file_match': fm_a, 'claude_a_func_match': fnm_a,
                'claude_a_rc_score': rc_a,
                'claude_a_level': compute_4level(rc_a, fm_a, fnm_a),
            })

            # Group E: unstructured pattern
            target_chars = get_c_pattern_size(symptom, layer)
            seed = get_bug_seed(repo, bug_id)
            flat_patterns = sample_flat_patterns(pool, target_chars, seed)
            prompt_e = build_prompt_e(files_content, flat_patterns)
            resp_e = call_claude(prompt_e)
            parsed_e = parse_response(resp_e)
            fm_e, fnm_e = match_location_strict(parsed_e['bug_location'], gold_funcs)
            rc_e = 0  # Root-cause scoring done manually
            result.update({
                'claude_e_location': parsed_e['bug_location'],
                'claude_e_root_cause': parsed_e['root_cause'],
                'claude_e_file_match': fm_e, 'claude_e_func_match': fnm_e,
                'claude_e_rc_score': rc_e,
                'claude_e_level': compute_4level(rc_e, fm_e, fnm_e),
            })

            # Group C: structured pattern
            patterns = load_structured_patterns(symptom, layer)
            prompt_c = build_prompt_c(files_content, patterns, symptom, layer)
            resp_c = call_claude(prompt_c)
            parsed_c = parse_response(resp_c)
            fm_c, fnm_c = match_location_strict(parsed_c['bug_location'], gold_funcs)
            rc_c = 0  # Root-cause scoring done manually
            result.update({
                'claude_c_location': parsed_c['bug_location'],
                'claude_c_root_cause': parsed_c['root_cause'],
                'claude_c_file_match': fm_c, 'claude_c_func_match': fnm_c,
                'claude_c_rc_score': rc_c,
                'claude_c_level': compute_4level(rc_c, fm_c, fnm_c),
            })

        except Exception as e:
            print(f"  ERROR {repo}#{bug_id}: {e}")
            time.sleep(10)
            continue

        save_result(output_file, result)
        count += 1
        print(f"  [{count}] {repo}#{bug_id}: A={result['claude_a_level']} E={result['claude_e_level']} C={result['claude_c_level']}")

        # Rate limit: Claude has tighter limits
        time.sleep(1)

    print(f"Claude done: {count} new results")


# ── Top-3 for E/F1/F2 ────────────────────────────────────────────

def run_top3_eff(existing, dim_map, gold_rc_map, output_file):
    """Run top-3 experiment for E, F1, F2 groups (GPT)."""
    pool = load_all_patterns_pool()
    cleaned_reports = load_cleaned_reports()
    done_keys = load_done_keys(output_file)
    print(f"Top-3 E/F1/F2: {len(existing)} bugs, {len(done_keys)} already done")

    count = 0
    for ex in existing:
        repo = ex['repo']
        bug_id = ex['issue_id']
        if (repo, bug_id) in done_keys:
            continue

        files = get_full_buggy_files(repo, bug_id)
        if not files:
            continue

        files_content = format_full_files(files)
        gold_funcs = extract_gold_functions(repo, bug_id)
        gold_rc = gold_rc_map.get((repo, bug_id), '')
        info = dim_map.get((repo, bug_id))
        if not info:
            continue
        symptom, layer = info['symptom'], info['layer']

        result = {'repo': repo, 'issue_id': bug_id, 'symptom': symptom, 'layer': layer}

        try:
            # E group top-3
            target_chars = get_c_pattern_size(symptom, layer)
            seed = get_bug_seed(repo, bug_id)
            flat_patterns = sample_flat_patterns(pool, target_chars, seed)
            prompt_e = build_prompt_e_top3(files_content, flat_patterns)
            resp_e = call_gpt(prompt_e)
            preds_e = parse_response_top3(resp_e)
            rc_e = 0  # Root-cause scoring done manually
            result.update({
                'e_preds': [p['bug_location'] for p in preds_e],
                'e_rc1_score': rc_e,
                'e_hit1': compute_hit_at_k(preds_e, gold_funcs, 1),
                'e_hit3': compute_hit_at_k(preds_e, gold_funcs, 3),
                'e_mfr': compute_first_rank(preds_e, gold_funcs),
            })

            # F1 group top-3
            cleaned = cleaned_reports.get((repo, bug_id), '')
            prompt_f1 = build_prompt_f1_top3(files_content, cleaned)
            resp_f1 = call_gpt(prompt_f1)
            preds_f1 = parse_response_top3(resp_f1)
            rc_f1 = 0  # Root-cause scoring done manually
            result.update({
                'f1_preds': [p['bug_location'] for p in preds_f1],
                'f1_rc1_score': rc_f1,
                'f1_hit1': compute_hit_at_k(preds_f1, gold_funcs, 1),
                'f1_hit3': compute_hit_at_k(preds_f1, gold_funcs, 3),
                'f1_mfr': compute_first_rank(preds_f1, gold_funcs),
            })

            # F2 group top-3
            patterns = load_structured_patterns(symptom, layer)
            prompt_f2 = build_prompt_f2_top3(files_content, cleaned, patterns, symptom, layer)
            resp_f2 = call_gpt(prompt_f2)
            preds_f2 = parse_response_top3(resp_f2)
            rc_f2 = 0  # Root-cause scoring done manually
            result.update({
                'f2_preds': [p['bug_location'] for p in preds_f2],
                'f2_rc1_score': rc_f2,
                'f2_hit1': compute_hit_at_k(preds_f2, gold_funcs, 1),
                'f2_hit3': compute_hit_at_k(preds_f2, gold_funcs, 3),
                'f2_mfr': compute_first_rank(preds_f2, gold_funcs),
            })

        except Exception as e:
            print(f"  ERROR {repo}#{bug_id}: {e}")
            time.sleep(5)
            continue

        save_result(output_file, result)
        count += 1
        if count % 10 == 0:
            print(f"  Top-3 E/F1/F2: {count} done")

    print(f"Top-3 E/F1/F2 done: {count} new results")


# ── Analysis ─────────────────────────────────────────────────────

def analyze_all():
    """Unified analysis across all experiments."""
    print("=" * 80)
    print("UNIFIED ANALYSIS — RQ4 Supplement")
    print("=" * 80)

    # --- GPT results (existing) ---
    gpt_results = {}
    with open('rq4_fullfile_results.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                key = (r['repo'], r['issue_id'])
                gpt_results[key] = r

    cv4_results = {}
    if os.path.exists('rq4_fullfile_c_v4_results.jsonl'):
        with open('rq4_fullfile_c_v4_results.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cv4_results[(r['repo'], r['issue_id'])] = r

    e_results = {}
    if os.path.exists('rq4_e_v2_unstructured_results.jsonl'):
        with open('rq4_e_v2_unstructured_results.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    e_results[(r['repo'], r['issue_id'])] = r

    # --- Claude results ---
    claude_results = {}
    claude_file = 'rq4_claude_results.jsonl'
    if os.path.exists(claude_file):
        with open(claude_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    claude_results[(r['repo'], r['issue_id'])] = r

    # --- BM25 results ---
    bm25_results = {}
    bm25_file = 'rq4_bm25_results.jsonl'
    if os.path.exists(bm25_file):
        with open(bm25_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    bm25_results[(r['repo'], r['issue_id'])] = r

    # --- Top-3 E/F results ---
    top3_ef_results = {}
    top3_ef_file = 'rq4_top3_ef_results.jsonl'
    if os.path.exists(top3_ef_file):
        with open(top3_ef_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    top3_ef_results[(r['repo'], r['issue_id'])] = r

    n = len(gpt_results)
    print(f"\nDataset: n={n}")
    print(f"Claude results: {len(claude_results)}")
    print(f"BM25 results: {len(bm25_results)}")
    print(f"Top-3 E/F results: {len(top3_ef_results)}")

    def compute_metrics(results, prefix, use_strict=True):
        """Compute standard metrics for a group."""
        func_match = sum(1 for r in results if r.get(f'{prefix}_func_match', False))
        rc0 = sum(1 for r in results if r.get(f'{prefix}_rc_score', -1) == 0)
        rc1 = sum(1 for r in results if r.get(f'{prefix}_rc_score', -1) >= 1)
        avg_rc = sum(r.get(f'{prefix}_rc_score', 0) for r in results) / max(len(results), 1)
        return {
            'n': len(results),
            'func_match': func_match,
            'func_match_pct': func_match / max(len(results), 1) * 100,
            'wrong_rate': rc0 / max(len(results), 1) * 100,
            'partial_plus': rc1 / max(len(results), 1) * 100,
            'avg_rc': avg_rc,
        }

    # ── GPT metrics (re-compute with strict matching from existing results) ──
    print("\n" + "─" * 60)
    print("TABLE 1: GPT-4o-mini Results (strict matching, n=207)")
    print("─" * 60)

    # Re-score existing results with strict matching
    groups = {'A': [], 'E': [], 'C_v4': []}
    for key in gpt_results:
        r = gpt_results[key]
        gold_funcs = extract_gold_functions(r['repo'], r['issue_id'])

        # A
        fm, fnm = match_location_strict(r.get('a_location', ''), gold_funcs)
        groups['A'].append({'func_match': fnm, 'rc_score': r.get('a_rc_score', 0)})

        # C_v4
        cv4 = cv4_results.get(key, {})
        if cv4:
            fm_c, fnm_c = match_location_strict(cv4.get('cv4_location', ''), gold_funcs)
            groups['C_v4'].append({'func_match': fnm_c, 'rc_score': cv4.get('cv4_rc_score', 0)})

        # E
        er = e_results.get(key, {})
        if er:
            fm_e, fnm_e = match_location_strict(er.get('e_location', ''), gold_funcs)
            groups['E'].append({'func_match': fnm_e, 'rc_score': er.get('e_rc_score', 0)})

    print(f"\n{'Metric':<20} {'A':>10} {'E':>10} {'C_v4':>10}")
    print("-" * 52)
    for name, data in groups.items():
        nn = len(data)
        fm = sum(1 for d in data if d['func_match'])
        wrong = sum(1 for d in data if d['rc_score'] == 0)
        partial = sum(1 for d in data if d['rc_score'] >= 1)
        avg = sum(d['rc_score'] for d in data) / max(nn, 1)
        if name == 'A':
            print(f"{'Func Match':<20} {fm:>6} ({fm/nn*100:.1f}%)", end='')
            a_data = data
        elif name == 'E':
            print(f" {fm:>6} ({fm/nn*100:.1f}%)", end='')
        else:
            print(f" {fm:>6} ({fm/nn*100:.1f}%)")

    # Print full table
    for metric_name in ['Func Match', 'Wrong Rate', 'Partial+', 'Avg RC']:
        vals = []
        for name in ['A', 'E', 'C_v4']:
            data = groups[name]
            nn = len(data)
            if nn == 0:
                vals.append('N/A')
                continue
            if metric_name == 'Func Match':
                v = sum(1 for d in data if d['func_match'])
                vals.append(f"{v} ({v/nn*100:.1f}%)")
            elif metric_name == 'Wrong Rate':
                v = sum(1 for d in data if d['rc_score'] == 0)
                vals.append(f"{v} ({v/nn*100:.1f}%)")
            elif metric_name == 'Partial+':
                v = sum(1 for d in data if d['rc_score'] >= 1)
                vals.append(f"{v} ({v/nn*100:.1f}%)")
            elif metric_name == 'Avg RC':
                v = sum(d['rc_score'] for d in data) / nn
                vals.append(f"{v:.2f}")
        print(f"{metric_name:<20} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    # ── Claude metrics ──
    if claude_results:
        print("\n" + "─" * 60)
        n_claude = len(claude_results)
        print(f"TABLE 2: Claude-Sonnet Results (strict matching, n={n_claude})")
        print("─" * 60)
        for metric_name in ['Func Match', 'Wrong Rate', 'Partial+', 'Avg RC']:
            vals = []
            for grp in ['a', 'e', 'c']:
                data = list(claude_results.values())
                nn = len(data)
                if metric_name == 'Func Match':
                    v = sum(1 for d in data if d.get(f'claude_{grp}_func_match', False))
                    vals.append(f"{v} ({v/nn*100:.1f}%)")
                elif metric_name == 'Wrong Rate':
                    v = sum(1 for d in data if d.get(f'claude_{grp}_rc_score', 0) == 0)
                    vals.append(f"{v} ({v/nn*100:.1f}%)")
                elif metric_name == 'Partial+':
                    v = sum(1 for d in data if d.get(f'claude_{grp}_rc_score', 0) >= 1)
                    vals.append(f"{v} ({v/nn*100:.1f}%)")
                elif metric_name == 'Avg RC':
                    v = sum(d.get(f'claude_{grp}_rc_score', 0) for d in data) / nn
                    vals.append(f"{v:.2f}")
            print(f"{metric_name:<20} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    # ── BM25 metrics ──
    if bm25_results:
        print("\n" + "─" * 60)
        n_bm25 = len(bm25_results)
        print(f"TABLE 3: BM25 Baseline (n={n_bm25})")
        print("─" * 60)
        data = list(bm25_results.values())
        hit1 = sum(1 for d in data if d.get('bm25_hit1', False))
        hit3 = sum(1 for d in data if d.get('bm25_hit3', False))
        func_m = sum(1 for d in data if d.get('bm25_func_match', False))
        mfr = sum(d.get('bm25_mfr', 4) for d in data) / n_bm25
        print(f"  Func Match: {func_m} ({func_m/n_bm25*100:.1f}%)")
        print(f"  Hit@1: {hit1} ({hit1/n_bm25*100:.1f}%)")
        print(f"  Hit@3: {hit3} ({hit3/n_bm25*100:.1f}%)")
        print(f"  MFR: {mfr:.2f}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")


# ── Main ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rq4_supplement.py [pilot|claude_full|bm25|top3_eff|analyze]")
        sys.exit(1)

    cmd = sys.argv[1]
    test_df, dim_map, gold_rc_map, existing = load_common_data()

    if cmd == 'pilot':
        print("=== PILOT: 10 bugs on Claude A/E/C ===")
        run_claude_experiment(existing, dim_map, gold_rc_map,
                            'rq4_claude_pilot_results.jsonl', limit=10)
        # Also pilot BM25
        print("\n=== PILOT: BM25 (10 bugs) ===")
        run_bm25_baseline(existing[:10], dim_map, gold_rc_map,
                         'rq4_bm25_pilot_results.jsonl')

    elif cmd == 'claude_full':
        print("=== FULL: Claude A/E/C (207 bugs) ===")
        run_claude_experiment(existing, dim_map, gold_rc_map,
                            'rq4_claude_results.jsonl')

    elif cmd == 'bm25':
        print("=== BM25 Baseline (207 bugs) ===")
        run_bm25_baseline(existing, dim_map, gold_rc_map,
                         'rq4_bm25_results.jsonl')

    elif cmd == 'top3_eff':
        print("=== Top-3 for E/F1/F2 (207 bugs) ===")
        run_top3_eff(existing, dim_map, gold_rc_map,
                    'rq4_top3_ef_results.jsonl')

    elif cmd == 'analyze':
        analyze_all()

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
