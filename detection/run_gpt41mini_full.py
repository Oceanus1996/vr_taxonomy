"""
GPT-4.1-mini Experiment for Table 5: Taxonomy-Guided Fault Localization.

Single-pass top-3 experiment. Each condition outputs 3 ranked predictions.
  - L_bare (A): Only buggy source code, no fix patterns.
  - L_random (B): Randomly sampled fix patterns (token-matched to L_full).
  - L_full (C): Patterns matched by symptom x entity layer.

Metrics derived from the same run:
  - Localization: Hit@1, Hit@3, MFR (from top-3 predictions)
  - Diagnosis: Wrong%, Partial%, Correct%, AvgRC (from top-1 root cause, scored manually)

BM25 is model-independent (reuse existing data).

Usage:
  python run_gpt41mini_full.py run         # Run experiment
  python run_gpt41mini_full.py analyze     # Print results
"""
import os, sys, json, time
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svl_common import call_llm, save_result, load_done_keys
from rq4_fullfile import (
    get_full_buggy_files, format_full_files, extract_gold_functions,
    compute_4level, parse_response,
    build_prompt_a,
)
from rq4_supplement import (
    match_location_strict, parse_response_top3, compute_hit_at_k,
    compute_first_rank, OUTPUT_FORMAT_TOP3,
    load_all_patterns_pool, sample_flat_patterns, get_c_pattern_size,
    load_structured_patterns, get_bug_seed,
    build_prompt_a_top3,
)

MODEL = 'gpt-4.1-mini'


# ── Prompt builders ───────────────────────────────────────────

def build_prompt_bare_top3(files_content):
    return f"""You are a code reviewer for a VR/XR project. This code contains a known bug.

## Source Code
{files_content}

## Task
Find the VR/XR bug in this code. Identify the specific file and function where the bug is,
and explain the root cause in detail.

{OUTPUT_FORMAT_TOP3}
"""


def build_prompt_random_top3(files_content, flat_patterns):
    return f"""You are a VR/XR bug detection expert. This code contains a known bug.

## VR Bug Patterns (General Reference)
The following are known VR/XR bug patterns collected from real projects:

{flat_patterns}

## Source Code
{files_content}

## Task
Check this code against the patterns above. Identify which pattern is violated,
the specific file and function, and explain the root cause in detail.

{OUTPUT_FORMAT_TOP3}
"""


def build_prompt_full_top3(files_content, patterns, label):
    hints = ""
    for i, p in enumerate(patterns, 1):
        hints += f"{i}. {p}\n\n"
    return f"""You are a VR/XR bug detection expert. This code contains a known bug.

## VR Bug Patterns ({label})
The following are bug patterns for VR/XR projects:

{hints}

## Source Code
{files_content}

## Task
Check this code against the patterns above. Identify which pattern is violated,
the specific file and function, and explain the root cause in detail.

{OUTPUT_FORMAT_TOP3}
"""


# ── Data loading ──────────────────────────────────────────────

def load_common_data():
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

    existing = []
    with open('rq4_fullfile_results.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                existing.append(json.loads(line))

    return dim_map, gold_rc_map, existing


# ── Run ───────────────────────────────────────────────────────

def run():
    dim_map, gold_rc_map, existing = load_common_data()
    pool = load_all_patterns_pool()
    output_file = 'rq4_gpt41mini_all_top3_results.jsonl'
    done_keys = load_done_keys(output_file)
    print(f"GPT-4.1-mini: {len(existing)} bugs, {len(done_keys)} already done")

    count = 0
    for ex in existing:
        repo = ex['repo']
        bug_id = ex['issue_id']
        if (repo, bug_id) in done_keys:
            continue

        info = dim_map.get((repo, bug_id))
        if not info:
            continue
        symptom, layer = info['symptom'], info['layer']

        files = get_full_buggy_files(repo, bug_id)
        if not files:
            continue

        files_content = format_full_files(files)
        gold_funcs = extract_gold_functions(repo, bug_id)

        result = {'repo': repo, 'issue_id': bug_id, 'symptom': symptom, 'layer': layer}

        try:
            # --- A (L_bare) ---
            resp = call_llm(build_prompt_bare_top3(files_content), model=MODEL)
            preds = parse_response_top3(resp)
            result.update({
                'a_hit1': compute_hit_at_k(preds, gold_funcs, 1),
                'a_hit3': compute_hit_at_k(preds, gold_funcs, 3),
                'a_mfr': compute_first_rank(preds, gold_funcs),
                'a_root_cause': preds[0]['root_cause'][:500] if preds else '',
                'a_rc_score': 0,  # To be scored manually
            })

            # --- B (L_random) ---
            target_chars = get_c_pattern_size(symptom, layer)
            seed = get_bug_seed(repo, bug_id)
            flat_patterns = sample_flat_patterns(pool, target_chars, seed)
            resp = call_llm(build_prompt_random_top3(files_content, flat_patterns), model=MODEL)
            preds = parse_response_top3(resp)
            result.update({
                'b_hit1': compute_hit_at_k(preds, gold_funcs, 1),
                'b_hit3': compute_hit_at_k(preds, gold_funcs, 3),
                'b_mfr': compute_first_rank(preds, gold_funcs),
                'b_root_cause': preds[0]['root_cause'][:500] if preds else '',
                'b_rc_score': 0,  # To be scored manually
            })

            # --- C (L_full) ---
            pats_full = load_structured_patterns(symptom, layer)
            if pats_full:
                resp = call_llm(build_prompt_full_top3(files_content, pats_full, f"{symptom} x {layer}"), model=MODEL)
                preds = parse_response_top3(resp)
            else:
                preds = []
            result.update({
                'c_hit1': compute_hit_at_k(preds, gold_funcs, 1),
                'c_hit3': compute_hit_at_k(preds, gold_funcs, 3),
                'c_mfr': compute_first_rank(preds, gold_funcs),
                'c_root_cause': preds[0]['root_cause'][:500] if preds else '',
                'c_rc_score': 0,  # To be scored manually
            })

        except Exception as e:
            print(f"  ERROR {repo}#{bug_id}: {e}")
            time.sleep(5)
            continue

        save_result(output_file, result)
        count += 1
        h1 = [result['a_hit1'], result['b_hit1'], result['c_hit1']]
        h1_str = '/'.join(['Y' if x else 'N' for x in h1])
        print(f"  [{count}] {repo}#{bug_id}: Hit@1 A/B/C = {h1_str}")

    print(f"\nDone: {count} new results")


# ── Analyze ───────────────────────────────────────────────────

def analyze():
    def load_jsonl(path):
        data = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        return data

    def pct(num, den):
        return f'{num/den*100:.1f}' if den > 0 else '-'

    results = load_jsonl('rq4_gpt41mini_all_top3_results.jsonl')
    bm25 = load_jsonl('rq4_bm25_results.jsonl')

    print(f"\n{'='*70}")
    print(f"GPT-4.1-mini Results (Table 5)")
    print(f"Bugs: {len(results)}, BM25: {len(bm25)}")
    print(f"{'='*70}")
    print(f"{'Cond':<15} {'Hit@1':>7} {'Hit@3':>7} {'MFR':>6} {'Wrong%':>8} {'Part%':>8} {'Corr%':>8} {'AvgRC':>7}")
    print('-' * 70)

    # BM25
    if bm25:
        n_bm = len(bm25)
        h1_bm = sum(1 for r in bm25 if r.get('bm25_hit1'))
        h3_bm = sum(1 for r in bm25 if r.get('bm25_hit3'))
        mfr_bm = sum(r.get('bm25_mfr', 4) for r in bm25) / n_bm
        print(f"{'BM25':<15} {pct(h1_bm, n_bm):>7} {pct(h3_bm, n_bm):>7} {mfr_bm:>6.2f} {'-':>8} {'-':>8} {'-':>8} {'-':>7}")

    n = len(results)
    for label, prefix in [('L_bare (A)', 'a'), ('L_random (B)', 'b'), ('L_full (C)', 'c')]:
        h1 = sum(1 for r in results if r.get(f'{prefix}_hit1'))
        h3 = sum(1 for r in results if r.get(f'{prefix}_hit3'))
        mfr = sum(r.get(f'{prefix}_mfr', 4) for r in results) / n

        rc_scores = [r.get(f'{prefix}_rc_score', 0) for r in results]
        wrong = sum(1 for s in rc_scores if s == 0)
        partial = sum(1 for s in rc_scores if s == 1)
        correct = sum(1 for s in rc_scores if s == 2)
        avg_rc = sum(rc_scores) / n

        print(f"{label:<15} {pct(h1, n):>7} {pct(h3, n):>7} {mfr:>6.2f} "
              f"{pct(wrong, n):>8} {pct(partial, n):>8} {pct(correct, n):>8} {avg_rc:>7.2f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_gpt41mini_full.py [run|analyze]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'run':
        run()
    elif cmd == 'analyze':
        analyze()
    else:
        print(f"Unknown command: {cmd}")
