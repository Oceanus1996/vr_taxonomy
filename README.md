# Seeing the Bugs: Mining and Characterizing Virtual Reality Visualization Bugs

Artifact repository for the paper *"Seeing the Bugs: Mining and Characterizing Virtual Reality Visualization Bugs"*.

## Dataset

**1,991 confirmed VR Visualization Bugs (VVBs)** mined from **538 open-source VR repositories** and **112,517 issue reports** on GitHub. Each VVB is annotated along three dimensions:

| Dimension | Categories |
|-----------|-----------|
| Visual Symptom (6) | Asset Fault (AF), Spatial Disorder (SD), Interaction Conflict (IC), Rendering Artifact (RA), Viewpoint Fault (VF), Runtime Collapse (RS) |
| Root Cause (7) | ICFM, UPF, GCE, RLM, IIV, IAC, TDI |
| Visual Element (5) | E1 Interaction Proxy, E2 Embodied Avatar, E3 Interface Entity, E4 Scene Entity, E5 View Configuration |

Each bug is linked to a complete issue-commit-code change chain.

## Repository Structure

```
vr_taxonomy/
├── README.md
├── requirements.txt
│
├── data/
│   ├── ALL_DIMENSION.csv                 1,991 VVBs with full taxonomy annotations
│   ├── ALL_with_diff.csv                 Same as above + code_diff column
│   ├── ALL_summaries.jsonl               LLM-extracted causal chains (root_cause, fix, vr_concept)
│   ├── train_set.csv                     Training split: 1,745 bugs from 196 repositories
│   ├── test_set.csv                      Test split: 246 bugs from 30 repositories (zero repo overlap)
│   ├── labele_1200_labeled.csv           Ground-truth for VVB identification (1,200 samples, 5 annotators)
│   ├── encode_sbert_1200.npy             SBERT embeddings for ground-truth samples
│   ├── codebert_embeddings_1200.npy      CodeBERT embeddings for ground-truth samples
│   ├── codebert_logreg_model.pkl         Trained CodeBERT + Logistic Regression model
│   ├── fix_complexity_all.csv            Repair complexity metrics (n_files, LoC) for all 1,991 bugs
│   ├── annotations/
│   │   ├── vvb_identification_annotations.csv
│   │   │     5-annotator labels for VVB identification (1,200 samples).
│   │   │     Columns: repo, id, annotator_1..5, majority_vote, final_label.
│   │   │     Fleiss' kappa = 0.83.
│   │   └── three_dimension_annotations.csv
│   │         5-annotator labels for taxonomy categorization (1,991 VVBs).
│   │         Symptom (kappa=0.85), Root Cause (kappa=0.80), Visual Element (kappa=0.83).
│   ├── diffs/                            1,991 code diff files (one per VVB)
│   └── pipeline/
│       └── 2_all_repos_issues.csv        Raw crawled issues: 112,517 records from 538 repos
│
├── patterns/
│   └── v4/                               Fix pattern library: 30 JSON files
│         {symptom}_{layer}_patterns.json (e.g., AF_L1_patterns.json)
│         Aggregated from training set only. 259 pattern entries total.
│         Each entry: {"VR Concept": {"count": N, "summary": "..."}}
│
├── identification/                       Section 3.2: VVB Identification Pipeline
│   ├── sbert_xgboost.py                  Classifier 1: SBERT + XGBoost with SMOTE, threshold tuning
│   ├── codebert_linear.py                Classifier 2: CodeBERT + Logistic Regression
│   └── gpt41_ensemble.py                 Classifier 3: GPT-4.1 + unanimous ensemble decision
│
└── detection/                            Section 6.1: Taxonomy-Guided Fault Localization
    ├── rq4_fullfile.py                   Shared utilities: source file retrieval, gold extraction,
    │                                     prompt construction, location matching, 4-level scoring
    ├── svl_common.py                     LLM API calls, result I/O, diff file lookup
    ├── rq4_supplement.py                 BM25 baseline, strict matching, top-3 parsing
    ├── run_gpt41mini_full.py             GPT-4.1-mini experiment: L_bare / L_random / L_full
    ├── run_gemini_flash_full.py          Gemini 2.5 Flash-Lite experiment: same 3 conditions
    ├── bm25_baseline.py                  BM25 information retrieval baseline
    └── results/                          Experiment results (Table 5)
        ├── rq4_gpt41mini_all_top1_results.jsonl      GPT-4.1-mini: Wrong%, Partial%, Correct%, AvgRC
        ├── rq4_gpt41mini_all_top3_results.jsonl      GPT-4.1-mini: Hit@1, Hit@3, MFR
        ├── rq4_gemini25_flash_lite_top1_results.jsonl Gemini: Wrong%, Partial%, Correct%, AvgRC
        ├── rq4_gemini25_flash_lite_top3_results.jsonl Gemini: Hit@1, Hit@3, MFR
        └── rq4_bm25_results.jsonl                    BM25 baseline: Hit@1, Hit@3, MFR
```

## Data Schema

### ALL_DIMENSION.csv

| Column | Description |
|--------|-------------|
| repo | GitHub repository (owner/name) |
| id | Issue ID |
| text | Bug report text (title + description) |
| repo_level | VR software stack: C1 (XR Runtime), C2 (Graphics), C3 (Engine), C4 (Middleware), C5 (Application) |
| symptom_cate | AF, SD, IC, RA, VF, RS |
| entity_layer | L1 (Interaction Proxy), L2 (Embodied Avatar), L3 (Interface Entity), L4 (Scene Entity), L5 (View Configuration) |
| root_cause | ICFM, UPF, GCE, RLM, IIV, IAC, TDI |
| root_cause_sub | Pipeline Misordering, Scope Mismatch, Unbounded Loop, etc. |

### ALL_summaries.jsonl

Each line: `{repo, id, root_cause, fix, causal_chain, vr_concept}`

### Experiment Results (detection/results/)

**top1 files** — per-bug diagnosis quality (scored manually):
- `{prefix}_rc_score`: 0 (wrong), 1 (partial), 2 (correct)
- `{prefix}_file_match`, `{prefix}_func_match`: localization accuracy
- `{prefix}_level`: correct / 1a / 1b / wrong

**top3 files** — per-bug localization ranking:
- `{prefix}_hit1`, `{prefix}_hit3`: whether top-1/top-3 predictions hit gold function
- `{prefix}_mfr`: Mean First Rank (1=best, 4=miss)

Prefixes: `a` = L_bare, `e` = L_random, `full` = L_full

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+. API keys via environment variables:
- `OPENAI_API_KEY` for GPT-4.1-mini experiments
- `GEMINI_API_KEY` for Gemini experiments

## Reproducing Results

### VVB Identification (Section 3.2)

```bash
python identification/sbert_xgboost.py
python identification/codebert_linear.py
python identification/gpt41_ensemble.py
```

### Fault Localization (Section 6.1, Table 5)

```bash
python detection/run_gpt41mini_full.py run
python detection/run_gemini_flash_full.py run
python detection/run_gpt41mini_full.py analyze
python detection/run_gemini_flash_full.py analyze
```

## Inter-Annotator Agreement

| Dimension | Fleiss' Kappa |
|-----------|:------------:|
| VVB Identification | 0.83 |
| Visual Symptoms | 0.85 |
| Root Causes | 0.80 |
| Visual Elements | 0.83 |

## License

Released for research purposes. Please cite our paper if you use this dataset or code.
