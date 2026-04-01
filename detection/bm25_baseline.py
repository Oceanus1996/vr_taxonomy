"""
BM25 Baseline for VR Bug Fault Localization

This implements a classic BM25 information retrieval baseline for fault localization.
Input: Bug report text
Output: Ranked list of candidate buggy files

This is equivalent to BLIZZARD's IR core without query reformulation.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import math
import re
import os
from typing import List, Dict, Tuple
import json


class BM25:
    """BM25 ranking function for fault localization"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # term frequency saturation parameter
        self.b = b    # length normalization parameter
        self.doc_freqs = Counter()
        self.doc_lens = {}
        self.avgdl = 0
        self.N = 0
        self.docs = {}

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase + split on non-alphanumeric"""
        text = text.lower()
        # Split camelCase and snake_case
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'_', ' ', text)
        # Extract tokens
        tokens = re.findall(r'[a-z0-9]+', text)
        # Filter short tokens
        tokens = [t for t in tokens if len(t) >= 2]
        return tokens

    def fit(self, documents: Dict[str, str]):
        """
        Fit BM25 on document corpus
        documents: dict of {doc_id: doc_text}
        """
        self.docs = documents
        self.N = len(documents)

        # Compute document frequencies and lengths
        for doc_id, text in documents.items():
            tokens = self.tokenize(text)
            self.doc_lens[doc_id] = len(tokens)

            # Count unique terms in this document
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1

        # Compute average document length
        self.avgdl = sum(self.doc_lens.values()) / self.N if self.N > 0 else 0

    def score(self, query: str, doc_id: str) -> float:
        """
        Compute BM25 score for a query-document pair
        """
        query_tokens = self.tokenize(query)
        doc_text = self.docs.get(doc_id, "")
        doc_tokens = self.tokenize(doc_text)
        doc_len = self.doc_lens.get(doc_id, 0)

        # Count term frequencies in document
        doc_term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in self.doc_freqs:
                continue

            # IDF component
            df = self.doc_freqs[term]
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

            # TF component
            tf = doc_term_freqs.get(term, 0)

            # Length normalization
            norm = 1 - self.b + self.b * (doc_len / self.avgdl)

            # BM25 formula
            term_score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
            score += term_score

        return score

    def rank(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Rank all documents by BM25 score for the query
        Returns list of (doc_id, score) tuples
        """
        scores = []
        for doc_id in self.docs.keys():
            score = self.score(query, doc_id)
            scores.append((doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k] if top_k > 0 else scores


def load_vr_bugs(csv_path: str) -> pd.DataFrame:
    """Load the 50 VR bug samples"""
    return pd.read_csv(csv_path)


def load_file_corpus(bug_df: pd.DataFrame, diff_dir: str) -> Dict[str, str]:
    """
    Build a corpus of all changed files across all bugs
    Returns: {file_id: file_content_placeholder}

    Note: We don't have actual source code, so we use filenames + paths as proxy
    This simulates the file corpus that BLIZZARD would use
    """
    corpus = {}
    file_to_bugs = defaultdict(list)

    for idx, row in bug_df.iterrows():
        repo = row['repo'].replace('/', '__')
        issue_id = row['id']
        diff_file = f"{diff_dir}/{repo}_issue{issue_id}_code_diff.csv"

        if os.path.exists(diff_file):
            diff_df = pd.read_csv(diff_file)
            if 'filename' in diff_df.columns:
                for _, diff_row in diff_df.iterrows():
                    filename = diff_row['filename']
                    file_id = f"{repo}::{filename}"

                    # Track which bugs touch this file (for evaluation)
                    file_to_bugs[file_id].append((repo, issue_id))

                    # Create pseudo-content from filename
                    # In real BLIZZARD, this would be actual source code
                    if file_id not in corpus:
                        # Extract tokens from file path
                        path_tokens = filename.replace('/', ' ').replace('.', ' ')
                        corpus[file_id] = path_tokens

    return corpus, file_to_bugs


def load_goldset(bug_df: pd.DataFrame, diff_dir: str) -> Dict[int, List[str]]:
    """
    Load ground truth buggy files for each bug
    Returns: {bug_index: [list of buggy file_ids]}
    """
    goldset = {}

    for idx, row in bug_df.iterrows():
        repo = row['repo'].replace('/', '__')
        issue_id = row['id']
        diff_file = f"{diff_dir}/{repo}_issue{issue_id}_code_diff.csv"

        buggy_files = []
        if os.path.exists(diff_file):
            diff_df = pd.read_csv(diff_file)
            if 'filename' in diff_df.columns:
                for filename in diff_df['filename'].unique():
                    file_id = f"{repo}::{filename}"
                    buggy_files.append(file_id)

        goldset[idx] = buggy_files

    return goldset


def evaluate_ranking(ranked_files: List[str], gold_files: List[str], k_values=[1, 5, 10]) -> Dict:
    """
    Evaluate ranking against gold standard
    Returns: {Hit@K, MRR, MAP}
    """
    gold_set = set(gold_files)

    # Hit@K
    hits = {}
    for k in k_values:
        top_k = set(ranked_files[:k])
        hits[f'Hit@{k}'] = 1 if len(top_k & gold_set) > 0 else 0

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, file_id in enumerate(ranked_files, 1):
        if file_id in gold_set:
            mrr = 1.0 / rank
            break

    # MAP (Mean Average Precision)
    ap = 0.0
    hits_so_far = 0
    for rank, file_id in enumerate(ranked_files, 1):
        if file_id in gold_set:
            hits_so_far += 1
            precision_at_k = hits_so_far / rank
            ap += precision_at_k

    if len(gold_set) > 0:
        ap = ap / len(gold_set)

    result = hits.copy()
    result['MRR'] = mrr
    result['MAP'] = ap

    return result


def run_bm25_baseline(bugs_csv: str, diff_dir: str, output_file: str):
    """
    Run BM25 baseline on VR bugs
    """
    print("Loading VR bugs...")
    bug_df = load_vr_bugs(bugs_csv)
    print(f"Loaded {len(bug_df)} bugs")

    print("\nBuilding file corpus...")
    corpus, file_to_bugs = load_file_corpus(bug_df, diff_dir)
    print(f"Corpus size: {len(corpus)} unique files")

    print("\nLoading goldset...")
    goldset = load_goldset(bug_df, diff_dir)

    print("\nFitting BM25 model...")
    bm25 = BM25()
    bm25.fit(corpus)

    print("\nRanking files for each bug...")
    all_results = []
    metrics_by_bug = []

    for idx, row in bug_df.iterrows():
        query = row['text']
        repo = row['repo'].replace('/', '__')

        # Filter corpus to only files from this repo
        # (cross-repo ranking doesn't make sense)
        repo_corpus = {fid: content for fid, content in corpus.items()
                      if fid.startswith(f"{repo}::")}

        if len(repo_corpus) == 0:
            print(f"  Bug {idx}: No files in corpus for repo {repo}")
            continue

        # Fit BM25 on repo-specific corpus
        repo_bm25 = BM25()
        repo_bm25.fit(repo_corpus)

        # Rank files
        ranked = repo_bm25.rank(query, top_k=0)  # Get all rankings
        ranked_files = [file_id for file_id, score in ranked]

        # Evaluate
        gold_files = goldset.get(idx, [])
        metrics = evaluate_ranking(ranked_files, gold_files, k_values=[1, 5, 10])
        metrics['bug_id'] = idx
        metrics['repo'] = row['repo']
        metrics['symptom'] = row['symptom_cate']
        metrics['num_gold_files'] = len(gold_files)
        metrics_by_bug.append(metrics)

        # Save top-10 results
        for rank, (file_id, score) in enumerate(ranked[:10], 1):
            all_results.append({
                'bug_id': idx,
                'rank': rank,
                'file': file_id,
                'score': score,
                'is_buggy': file_id in gold_files
            })

        if idx % 10 == 0:
            print(f"  Processed {idx+1}/{len(bug_df)} bugs")

    # Aggregate metrics
    metrics_df = pd.DataFrame(metrics_by_bug)

    print("\n" + "="*60)
    print("BM25 Baseline Results")
    print("="*60)
    print(f"\nOverall Performance:")
    print(f"  Hit@1:  {metrics_df['Hit@1'].mean():.3f}")
    print(f"  Hit@5:  {metrics_df['Hit@5'].mean():.3f}")
    print(f"  Hit@10: {metrics_df['Hit@10'].mean():.3f}")
    print(f"  MRR:    {metrics_df['MRR'].mean():.3f}")
    print(f"  MAP:    {metrics_df['MAP'].mean():.3f}")

    print(f"\nPerformance by Symptom:")
    symptom_metrics = metrics_df.groupby('symptom')[['Hit@1', 'Hit@5', 'Hit@10', 'MRR', 'MAP']].mean()
    print(symptom_metrics.to_string())

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)

    metrics_file = output_file.replace('.csv', '_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)

    print(f"\nResults saved to:")
    print(f"  Rankings: {output_file}")
    print(f"  Metrics:  {metrics_file}")

    return metrics_df


if __name__ == '__main__':
    bugs_csv = 'blizzard_50_bugs.csv'
    diff_dir = '../record/diff_c'
    output_file = 'bm25_baseline_results.csv'

    metrics = run_bm25_baseline(bugs_csv, diff_dir, output_file)
