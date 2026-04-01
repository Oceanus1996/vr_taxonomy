"""
Shared utilities for Taxonomy-Guided Fault Localization (Section 6.1).
Provides: LLM API calls, diff file lookup, result I/O.

Used by: rq4_fullfile.py, run_gpt41mini_full.py, run_gemini_flash_full.py
"""

import pandas as pd
import json
import os
import sys
import re
import time
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GPT_TOKEN = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=GPT_TOKEN)


def call_llm(prompt, model="gpt-4.1-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024
    )
    return response.choices[0].message.content


def load_done_keys(output_file):
    """Load already-processed keys from output file."""
    done_keys = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_keys.add((r['repo'], r['issue_id']))
    return done_keys


def save_result(output_file, result):
    """Append one result to JSONL output."""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def find_diff_file(repo, bug_id):
    """Find diff CSV file for a bug."""
    diff_dir = os.path.join('..', 'data', 'diffs')
    repo_short = repo.replace('/', '__')
    diff_file = os.path.join(diff_dir, f'{repo_short}_issue{bug_id}_code_diff.csv')
    if os.path.isfile(diff_file):
        return diff_file
    return None
