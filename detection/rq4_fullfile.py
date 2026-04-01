"""
Shared utilities for Taxonomy-Guided Fault Localization (Section 6.1).

Provides: source file retrieval, gold extraction, prompt construction,
location matching, and 4-level scoring.

Used by: run_gpt41mini_full.py, run_gemini_flash_full.py
"""

import os, sys, json, time, re, subprocess
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from svl_common import call_llm, save_result, load_done_keys

TEMP_REPOS = os.path.join('..', 'temp_repos')
DIFF_DIR = os.path.join('..', 'record', 'diff')

# Source code extensions (skip binary/config/docs)
SOURCE_EXTS = {
    '.js', '.ts', '.jsx', '.tsx', '.cs', '.cpp', '.c', '.h', '.hpp',
    '.py', '.gd', '.rs', '.java', '.kt', '.swift', '.lua', '.rb',
    '.glsl', '.hlsl', '.shader', '.cginc', '.compute',
}

OUTPUT_FORMAT = """Output exactly this format:
BUG_LOCATION: [file_path:function_name]
ROOT_CAUSE: [1-2 sentences: what is wrong and why]
CONFIDENCE: [HIGH / MEDIUM / LOW]"""


# ── File retrieval ───────────────────────────────────────────────

def get_repo_dir(repo):
    """Get local repo directory path."""
    repo_dir = repo.replace('/', '__')
    path = os.path.join(TEMP_REPOS, repo_dir)
    if os.path.isdir(path):
        return path
    # Try other naming conventions
    for d in os.listdir(TEMP_REPOS):
        if d.replace('_', '__') == repo_dir or d == repo_dir:
            return os.path.join(TEMP_REPOS, d)
    return None


def git_cmd(repo_path, cmd):
    """Run a git command in a repo directory, return stdout or None."""
    try:
        result = subprocess.run(
            ['git'] + cmd,
            cwd=repo_path,
            capture_output=True, text=True, timeout=30,
            encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


def get_full_buggy_files(repo, bug_id):
    """Get complete buggy file contents (pre-fix) from git."""
    repo_path = get_repo_dir(repo)
    if not repo_path:
        return {}

    # Read diff CSV to get commit_sha and filenames
    repo_short = repo.replace('/', '__')
    diff_path = os.path.join(DIFF_DIR, f'{repo_short}_issue{bug_id}_code_diff.csv')
    if not os.path.exists(diff_path):
        return {}

    df = pd.read_csv(diff_path)
    buggy_files = {}
    seen_commits = {}

    for _, row in df.iterrows():
        filename = row['filename']
        commit_sha = str(row['commit_sha']).strip()

        # Skip non-source files
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SOURCE_EXTS:
            continue

        # Get parent commit (cache per commit_sha)
        if commit_sha not in seen_commits:
            parent = git_cmd(repo_path, ['rev-parse', f'{commit_sha}^'])
            seen_commits[commit_sha] = parent.strip() if parent else None
        parent_sha = seen_commits[commit_sha]
        if not parent_sha:
            continue

        # Get full file content at parent commit
        content = git_cmd(repo_path, ['show', f'{parent_sha}:{filename}'])
        if content and len(content) > 10:
            # Truncate very large files
            if len(content) > 30000:
                content = content[:30000] + '\n... (truncated)'
            buggy_files[filename] = content

    return buggy_files


# ── Gold extraction ──────────────────────────────────────────────

def extract_gold_functions(repo, bug_id):
    """Extract gold file + function names from diff @@ lines."""
    repo_short = repo.replace('/', '__')
    diff_path = os.path.join(DIFF_DIR, f'{repo_short}_issue{bug_id}_code_diff.csv')
    if not os.path.exists(diff_path):
        return {}

    df = pd.read_csv(diff_path)
    gold = {}  # {filename: set(function_names)}

    for _, row in df.iterrows():
        filename = row['filename']
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SOURCE_EXTS:
            continue

        patch = str(row['patch'])
        if patch == 'nan':
            continue

        functions = set()
        for line in patch.split('\n'):
            if not line.startswith('@@'):
                continue
            m = re.search(r'@@.*?@@\s*(.*)', line)
            if not m:
                continue
            context = m.group(1).strip()
            if not context:
                continue
            # Try various function patterns
            patterns = [
                # C#/Java/C++: access_modifier return_type FuncName(
                r'(?:public|private|protected|internal|static|virtual|override|abstract|async|sealed|extern|unsafe|volatile|partial|readonly)\s+.*?(\w+)\s*\(',
                # C#/Java: void/int/string FuncName(
                r'(?:void|int|float|double|string|bool|var|IEnumerator|Task)\s+(\w+)\s*\(',
                # Python: def func_name(
                r'def\s+(\w+)\s*\(',
                # JS/TS: function funcName( or funcName = function(
                r'function\s+(\w+)\s*\(',
                r'(\w+)\s*[=:]\s*function\s*\(',
                # JS/TS: methodName( in object/class
                r'(\w+)\s*\([^)]*\)\s*\{',
                # Rust: fn func_name(
                r'fn\s+(\w+)\s*[<(]',
                # GDScript: func func_name(
                r'func\s+(\w+)\s*\(',
                # Lua: function Class:method( or function name(
                r'function\s+\w*[:.]*(\w+)\s*\(',
                # Generic: word followed by (
                r'(\w{3,})\s*\(',
            ]
            for pat in patterns:
                fm = re.search(pat, context)
                if fm:
                    func_name = fm.group(1)
                    # Skip common non-function matches
                    if func_name.lower() not in {'if', 'for', 'while', 'switch',
                                                  'catch', 'foreach', 'elif', 'class',
                                                  'struct', 'enum', 'namespace', 'interface',
                                                  'import', 'from', 'return', 'new', 'var',
                                                  'using', 'with', 'try', 'except',
                                                  'function', 'require', 'target', 'module',
                                                  'exports', 'define', 'include', 'pragma',
                                                  'typedef', 'extern', 'const', 'let',
                                                  'async', 'await', 'yield', 'super',
                                                  'this', 'self', 'null', 'none', 'true',
                                                  'false', 'elif', 'else', 'case', 'default',
                                                  'break', 'continue', 'pass', 'raise',
                                                  'throw', 'delete', 'typeof', 'instanceof',
                                                  'sizeof', 'assert', 'print', 'echo'}:
                        functions.add(func_name)
                    break

        if functions:
            gold[filename] = functions
        elif ext in SOURCE_EXTS:
            # File was modified but no function extracted — record file at least
            gold[filename] = set()

    return gold


# ── Format files──────────────────────────────────

def format_full_files(files_dict, max_total=60000):
    """Format complete file contents for prompt. No @@ lines."""
    content = ""
    total = 0
    for fname, body in files_dict.items():
        remaining = max_total - total
        if remaining < 200:
            content += f"\n### File: {fname}\n(skipped — token budget reached)\n"
            continue
        if len(body) > remaining:
            body = body[:remaining] + '\n... (truncated)'
        content += f"\n### File: {fname}\n```\n{body}\n```\n"
        total += len(body)
    return content


# ── Pattern loaders ──────────────────────────────────────────────

def load_symptom_layer_patterns(symptom, layer):
    """Load aggregated_patterns/{sym}_{layer}_patterns.json."""
    path = f'aggregated_patterns/{symptom}_{layer}_patterns.json'
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    patterns = []
    for key, val in data.items():
        patterns.append(f"[{key}] (n={val['count']})\n{val['summary']}")
    return patterns


# ── Prompt builders ──────────────────────────────────────────────

def build_prompt_a(files_content):
    """L_bare (A): bare LLM, no patterns."""
    return f"""You are a code reviewer for a VR/XR project. This code contains a known bug.

## Source Code
{files_content}

## Task
Find the VR/XR bug in this code. Identify the specific file and function where the bug is,
and explain what is wrong.

{OUTPUT_FORMAT}
"""


def build_prompt_c(files_content, patterns, symptom, layer):
    """L_full (C): symptom x entity layer matched patterns."""
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

{OUTPUT_FORMAT}
"""


# ── Evaluation ───────────────────────────────────────────────────


def match_location(predicted_loc, gold_functions):
    """Match predicted location against gold file+functions.

    Returns: (file_match: bool, func_match: bool)
    """
    if not predicted_loc or not gold_functions:
        return False, False

    pred_loc = predicted_loc.strip()

    # Parse predicted: "file_path:function_name" or "file_path:line_range"
    pred_file = ''
    pred_func = ''
    if ':' in pred_loc:
        parts = pred_loc.rsplit(':', 1)
        pred_file = parts[0].strip()
        pred_func = parts[1].strip()
    else:
        pred_file = pred_loc

    # Extract just filename (no path) for comparison
    pred_basename = os.path.basename(pred_file).lower()

    # Check file match
    file_match = False
    matched_file = None
    for gold_file in gold_functions:
        gold_basename = os.path.basename(gold_file).lower()
        if pred_basename == gold_basename:
            file_match = True
            matched_file = gold_file
            break
        # Also try partial path match
        if pred_file.lower().endswith(gold_file.lower()) or gold_file.lower().endswith(pred_file.lower()):
            file_match = True
            matched_file = gold_file
            break

    if not file_match:
        return False, False

    # Check function match
    gold_funcs = gold_functions.get(matched_file, set())
    if not gold_funcs:
        # No gold functions extracted — can't judge function, give benefit of doubt
        return True, True

    # Clean predicted function (remove line numbers, parens, etc.)
    pred_func_clean = re.sub(r'[\d\-:()]+', '', pred_func).strip()
    if not pred_func_clean:
        # Predicted only a line number, can't match function name
        return True, False

    for gf in gold_funcs:
        if pred_func_clean.lower() == gf.lower():
            return True, True
        # Partial match: predicted contains gold or vice versa
        if pred_func_clean.lower() in gf.lower() or gf.lower() in pred_func_clean.lower():
            return True, True

    return True, False


def compute_4level(rc_score, file_match, func_match):
    """Combine root cause score + location into 4-level rating."""
    if rc_score == 2:
        return 'correct'
    elif rc_score == 1:
        if file_match and func_match:
            return '1a'  # function right, mechanism wrong
        else:
            return '1b'  # function wrong
    else:
        return 'wrong'


def parse_response(response):
    """Parse LLM detection response."""
    fields = {'bug_location': '', 'root_cause': '', 'confidence': ''}
    for line in response.split('\n'):
        line = line.strip()
        upper = line.upper()
        if upper.startswith('BUG_LOCATION:'):
            fields['bug_location'] = line[13:].strip()
        elif upper.startswith('ROOT_CAUSE:'):
            fields['root_cause'] = line[11:].strip()
        elif upper.startswith('CONFIDENCE:'):
            fields['confidence'] = line[11:].strip()
    return fields


